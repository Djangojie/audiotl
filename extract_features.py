import argparse
from torch.utils.data import DataLoader
import yaml
import logging
import torch
import torchaudio
import torch.nn as nn
from torchvision import transforms
import numpy as np
import random
from ppacls.utils.utils import dict_to_object, plot_confusion_matrix, print_arguments
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from campplus import CAMPPlus
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group, get_world_size, get_rank
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import json
import time
import shutil
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioSegment:
    @staticmethod
    def from_file(file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        return AudioSegment(waveform, sample_rate)

    def __init__(self, waveform, sample_rate):
        self.samples = waveform
        self.sample_rate = sample_rate
        self.duration = waveform.shape[1] / sample_rate

    def vad(self):
        # VAD implementation
        pass

    def resample(self, target_sample_rate):
        self.samples = torchaudio.transforms.Resample(self.sample_rate, target_sample_rate)(self.samples)
        self.sample_rate = target_sample_rate

    def normalize(self, target_db):
        gain = target_db - self.samples.mean().item()
        self.samples = self.samples + gain

    def crop(self, duration, mode='train'):
        max_samples = int(duration * self.sample_rate)
        if self.samples.shape[1] > max_samples:
            if mode == 'train':
                start = random.randint(0, self.samples.shape[1] - max_samples)
                self.samples = self.samples[:, start:start + max_samples]
            else:
                self.samples = self.samples[:, :max_samples]

    def change_speed(self, speed_rate):
        self.samples = torchaudio.transforms.Resample(self.sample_rate, int(self.sample_rate / speed_rate))(
            self.samples)
        self.sample_rate = int(self.sample_rate / speed_rate)

    def gain_db(self, gain):
        self.samples = self.samples + gain

class PPAClsDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 audio_featurizer,
                 do_vad=True,
                 max_duration=3,
                 min_duration=0.5,
                 mode='train',
                 sample_rate=16000,
                 aug_conf={},
                 use_dB_normalization=True,
                 target_dB=-20):
        """音频数据加载器

        Args:
            data_list_path: 包含音频路径和标签的数据列表文件的路径
            audio_featurizer: 声纹特征提取器
            do_vad: 是否对音频进行语音活动检测（VAD）来裁剪静音部分
            max_duration: 最长的音频长度，大于这个长度会裁剪掉
            min_duration: 过滤最短的音频长度
            aug_conf: 用于指定音频增强的配置
            mode: 数据集模式。在训练模式下，数据集可能会进行一些数据增强的预处理
            sample_rate: 采样率
            use_dB_normalization: 是否对音频进行音量归一化
            target_dB: 音量归一化的大小
        """
        super(PPAClsDataset, self).__init__()
        assert mode in ['train', 'eval', 'extract_feature']
        self.data_list_path = data_list_path
        self.do_vad = do_vad
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.aug_conf = aug_conf
        self.noises_path = None
        self.audio_featurizer = audio_featurizer
        self.max_feature_len = self.get_crop_feature_len()


        # 获取数据列表
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        # 评估模式下，数据列表需要排序
        if self.mode == 'eval':
            self.sort_list()

    def __getitem__(self, idx):
        # 分割数据文件路径和标签
        data_path, label = self.lines[idx].strip().split('\t')
        label = torch.tensor(int(label), dtype=torch.long)
        # 如果后缀名为.npy的文件，那么直接读取
        if data_path.endswith('.npy'):
            feature = np.load(data_path)
            if feature.shape[0] > self.max_feature_len:
                crop_start = random.randint(0, feature.shape[0] - self.max_feature_len) if self.mode == 'train' else 0
                feature = feature[crop_start:crop_start + self.max_feature_len, :]
            feature = torch.tensor(feature, dtype=torch.float32)
        else:
            # 读取音频
            audio_segment = AudioSegment.from_file(data_path)
            # 裁剪静音
            if self.do_vad:
                audio_segment.vad()
            # 数据太短不利于训练
            if self.mode == 'train' and audio_segment.duration < self.min_duration:
                return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)
            # 重采样
            if audio_segment.sample_rate != self._target_sample_rate:
                audio_segment.resample(self._target_sample_rate)
            # 音频增强
            if self.mode == 'train':
                audio_segment = self.augment_audio(audio_segment, **self.aug_conf)
            # 音量归一化
            if self._use_dB_normalization:
                audio_segment.normalize(target_db=self._target_dB)
            # 裁剪需要的数据
            if self.mode != 'extract_feature' and audio_segment.duration > self.max_duration:
                audio_segment.crop(duration=self.max_duration, mode=self.mode)
            samples = audio_segment.samples
            feature = self.audio_featurizer(samples)
            feature = feature.squeeze(0)
        return feature, label

    def __len__(self):
        return len(self.lines)

    def get_crop_feature_len(self):
        samples = torch.randn((1, self.max_duration * self._target_sample_rate))
        feature = self.audio_featurizer(samples).squeeze(0)
        freq_len = feature.shape[0]
        return freq_len

    # 数据列表需要排序
    def sort_list(self):
        lengths = []
        for line in tqdm(self.lines, desc=f"对列表[{self.data_list_path}]进行长度排序"):
            # 分割数据文件路径和标签
            data_path, _ = line.split('\t')
            if data_path.endswith('.npy'):
                feature = np.load(data_path)
                length = feature.shape[0]
                lengths.append(length)
            else:
                # 读取音频
                audio_segment = AudioSegment.from_file(data_path)
                length = audio_segment.duration
                lengths.append(length)
        # 对长度排序并获取索引
        sorted_indexes = np.argsort(lengths)
        self.lines = [self.lines[i] for i in sorted_indexes]

    # 音频增强
    def augment_audio(self,
                      audio_segment,
                      speed_perturb=False,
                      volume_perturb=False,
                      volume_aug_prob=0.2,
                      noise_dir=None,
                      noise_aug_prob=0.2):
        # 语速增强
        if speed_perturb:
            speeds = [1.0, 0.9, 1.1]
            speed_idx = random.randint(0, 2)
            speed_rate = speeds[speed_idx]
            if speed_rate != 1.0:
                audio_segment.change_speed(speed_rate)
        # 音量增强
        if volume_perturb and random.random() < volume_aug_prob:
            min_gain_dBFS, max_gain_dBFS = -15, 15
            gain = random.uniform(min_gain_dBFS, max_gain_dBFS)
            audio_segment.gain_db(gain)

        return audio_segment

class AudioFeaturizer(nn.Module):
    """音频特征器

    :param feature_method: 所使用的预处理方法
    :type feature_method: str
    :param method_args: 预处理方法的参数
    :type method_args: dict
    """

    def __init__(self, feature_method='MelSpectrogram', method_args={}):
        super().__init__()
        self._method_args = method_args
        self._feature_method = feature_method
        if feature_method == 'LogMelSpectrogram':
            self.feat_fun = torchaudio.transforms.MelSpectrogram(**method_args)
            self.log_transform = torchaudio.transforms.AmplitudeToDB()
        elif feature_method == 'MelSpectrogram':
            self.feat_fun = torchaudio.transforms.MelSpectrogram(**method_args)
        elif feature_method == 'Spectrogram':
            self.feat_fun = torchaudio.transforms.Spectrogram(**method_args)
        elif feature_method == 'MFCC':
            self.feat_fun = torchaudio.transforms.MFCC(**method_args)
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')

    def forward(self, waveforms, input_lens_ratio=None):
        """从AudioSegment中提取音频特征

        :param waveforms: 要提取特征的音频波形
        :type waveforms: Tensor
        :param input_lens_ratio: 输入长度比例
        :type input_lens_ratio: tensor
        :return: 2D数组形式的谱图音频特征
        :rtype: Tensor
        """
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)
        feature = self.feat_fun(waveforms)
        if self._feature_method == 'LogMelSpectrogram':
            feature = self.log_transform(feature)
        feature = feature.transpose(1, 2)  # 转置使得特征在最后一维

        # 归一化
        feature = feature - feature.mean(dim=1, keepdim=True)

        if input_lens_ratio is not None:
            # 计算实际长度
            input_lens = (input_lens_ratio * feature.shape[1]).to(torch.int32)
            mask_lens = input_lens.unsqueeze(1)
            # 创建掩码
            idxs = torch.arange(feature.shape[1], device=feature.device).expand(feature.shape[0], -1)
            mask = idxs < mask_lens
            mask = mask.unsqueeze(-1)
            # 掩码特征
            feature = torch.where(mask, feature, torch.zeros_like(feature))

        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'LogMelSpectrogram':
            return self._method_args.get('n_mels', 128)
        elif self._feature_method == 'MelSpectrogram':
            return self._method_args.get('n_mels', 64)
        elif self._feature_method == 'Spectrogram':
            return self._method_args.get('n_fft', 512) // 2 + 1
        elif self._feature_method == 'MFCC':
            return self._method_args.get('n_mfcc', 40)
        else:
            raise Exception(f'没有 {self._feature_method} 预处理方法')

class PPAClsTrainer:
    def __init__(self, configs, use_gpu=True):
        """ PPACls 集成工具类

        :param configs: 配置字典
        :param use_gpu: 是否使用GPU训练模型
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.safe_load(f)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        self.model = None
        self.audio_featurizer = None
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.class_labels = self._load_class_labels()

    def _load_class_labels(self):
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [l.strip() for l in lines]

    def __setup_dataloader(self, is_train=False):
        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))

        if is_train:
            self.train_dataset = PPAClsDataset(data_list_path=self.configs.dataset_conf.train_list,
                                               audio_featurizer=self.audio_featurizer,
                                               do_vad=self.configs.dataset_conf.do_vad,
                                               max_duration=self.configs.dataset_conf.max_duration,
                                               min_duration=self.configs.dataset_conf.min_duration,
                                               aug_conf=self.configs.dataset_conf.aug_conf,
                                               sample_rate=self.configs.dataset_conf.sample_rate,
                                               use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                               target_dB=self.configs.dataset_conf.target_dB,
                                               mode='train')
            train_sampler = None
            if torch.cuda.device_count() > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           num_workers=self.configs.dataset_conf.dataLoader.num_workers)

        self.test_dataset = PPAClsDataset(data_list_path=self.configs.dataset_conf.test_list,
                                          audio_featurizer=self.audio_featurizer,
                                          do_vad=self.configs.dataset_conf.do_vad,
                                          max_duration=self.configs.dataset_conf.eval_conf.max_duration,
                                          min_duration=self.configs.dataset_conf.min_duration,
                                          sample_rate=self.configs.dataset_conf.sample_rate,
                                          use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                          target_dB=self.configs.dataset_conf.target_dB,
                                          mode='eval',)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.configs.dataset_conf.eval_conf.batch_size,
                                      shuffle=False,
                                      num_workers=self.configs.dataset_conf.dataLoader.num_workers)

    def extract_features(self, save_dir='dataset/features'):
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        for i, data_list in enumerate([self.configs.dataset_conf.train_list, self.configs.dataset_conf.test_list]):
            # 获取数据集
            dataset = PPAClsDataset(data_list_path=data_list,
                                    audio_featurizer=self.audio_featurizer,
                                    do_vad=self.configs.dataset_conf.do_vad,
                                    sample_rate=self.configs.dataset_conf.sample_rate,
                                    use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                    target_dB=self.configs.dataset_conf.target_dB,
                                    mode='extract_feature')

            save_data_list = data_list.replace('.txt', '_features.txt')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for i in tqdm(range(len(dataset))):
                    feature, label = dataset[i]
                    feature = feature.numpy()
                    label = int(label)
                    save_path = os.path.join(save_dir, str(label), f'{int(time.time() * 1000)}.npy').replace('\\', '/')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, feature)
                    f.write(f'{save_path}\t{label}\n')
            logger.info(f'{data_list}列表中的数据已提取特征完成，新列表为：{save_data_list}')


    def __setup_model(self, input_size, is_train=False):
        # 自动获取类别数量
        if self.configs.model_conf.num_class is None:
            self.configs.model_conf.num_class = len(self.class_labels)

        # 获取模型
        if self.configs.use_model == 'CAMPPlus':
            self.model = CAMPPlus(**self.configs.model_conf)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')

        # 打印模型摘要
        print(self.model)

        # 获取损失函数
        weight = torch.tensor(self.configs.train_conf.loss_weight, dtype=torch.float32) \
            if self.configs.train_conf.loss_weight is not None else None
        self.loss = nn.CrossEntropyLoss(weight=weight)

        if is_train:
            if self.configs.train_conf.enable_amp:
                # 自动混合精度训练
                self.amp_scaler = GradScaler(init_scale=1024)

            optimizer_name = self.configs.optimizer_conf.optimizer
            if optimizer_name == 'Adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001,
                                            weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer_name == 'AdamW':
                self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001,
                                             weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer_name == 'Momentum':
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.01,
                                           momentum=self.configs.optimizer_conf.get('momentum', 0.9),
                                           weight_decay=self.configs.optimizer_conf.weight_decay)
            else:
                raise Exception(f'不支持优化方法：{optimizer_name}')

            # 设置学习率调度器
            scheduler_args = self.configs.optimizer_conf.get('scheduler_args', {})
            if self.configs.optimizer_conf.scheduler == 'CosineAnnealingLR':
                max_step = int(self.configs.train_conf.max_epoch * 1.2) * len(self.train_loader)
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_step, **scheduler_args)
            else:
                raise Exception(f'不支持学习率衰减函数：{self.configs.optimizer_conf.scheduler}')



    def __load_pretrained(self, pretrained_model):
        # 加载预训练模型
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pth')
            assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
            model_state_dict = torch.load(pretrained_model, map_location=self.device)
            self.model.load_state_dict(model_state_dict, strict=False)
            logger.info(f'成功加载预训练模型：{pretrained_model}')


    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_acc = 0.0
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'last_model')

        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth')) and os.path.exists(
                os.path.join(last_model_dir, 'optimizer.pth'))):
            if resume_model is None:
                resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pth')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pth')), "优化方法参数文件不存在！"

            self.model.load_state_dict(torch.load(os.path.join(resume_model, 'model.pth')))
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pth')))

            if self.amp_scaler is not None and os.path.exists(os.path.join(resume_model, 'scaler.pth')):
                self.amp_scaler.load_state_dict(torch.load(os.path.join(resume_model, 'scaler.pth')))

            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_acc = json_data['accuracy']

            logger.info(f'成功恢复模型参数和优化方法参数：{resume_model}')

        return last_epoch, best_acc


    def __save_checkpoint(self, save_model_path, epoch_id, best_acc=0., best_model=False):
        if best_model:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'best_model')
        else:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      f'epoch_{epoch_id}')

        os.makedirs(model_path, exist_ok=True)
        try:
            torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
            torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))
            if self.amp_scaler is not None:
                torch.save(self.amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pth'))
        except Exception as e:
            logger.error(f'保存模型时出现错误，错误信息：{e}')
            return

        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            data = {"last_epoch": epoch_id, "accuracy": best_acc}
            f.write(json.dumps(data))

        if not best_model:
            last_model_path = os.path.join(save_model_path,
                                           f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                           'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)

            old_model_path = os.path.join(save_model_path,
                                          f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                          f'epoch_{epoch_id - 3}')
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)

        logger.info(f'已保存模型：{model_path}')

    def __train_epoch(self, epoch_id, local_rank, writer):
        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        for batch_id, (features, label) in enumerate(self.train_loader):

            # 特征增强
            if self.configs.dataset_conf.use_spec_aug:
                features = self.spec_aug(features)

            # 将输入数据和标签移动到 GPU (如果可用)
            features, label = features.to(self.device), label.to(self.device)

            # 执行模型计算，是否开启自动混合精度
            with torch.cuda.amp.autocast(enabled=self.configs.train_conf.enable_amp):
                output = self.model(features)
                # 计算损失值
                los = self.loss(output, label)

            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                self.amp_scaler.scale(los).backward()
            else:
                los.backward()

            if self.configs.train_conf.enable_amp:
                # 更新参数（参数梯度先除系数loss_scaling再更新参数）
                self.amp_scaler.step(self.optimizer)
                # 基于动态loss_scaling策略更新loss_scaling系数
                self.amp_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # 计算准确率
            _, preds = torch.max(output, 1)
            acc = (preds == label).float().mean().item()
            accuracies.append(acc)
            loss_sum.append(los.item())
            train_times.append((time.time() - start) * 1000)
            self.train_step += 1

            # 多卡训练只使用一个进程打印
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                batch_id = batch_id + 1
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.dataLoader.batch_size / (
                            sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                self.train_eta_sec = (sum(train_times) / len(train_times)) * (self.max_step - self.train_step) / 1000
                eta_str = str(timedelta(seconds=int(self.train_eta_sec)))
                self.train_loss = sum(loss_sum) / len(loss_sum)
                self.train_acc = sum(accuracies) / len(accuracies)
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {self.train_loss:.5f}, accuracy: {self.train_acc:.5f}, '
                            f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', self.train_loss, self.train_log_step)
                writer.add_scalar('Train/Accuracy', self.train_acc, self.train_log_step)
                # 记录学习率
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.train_log_step)
                train_times, accuracies, loss_sum = [], [], []
                self.train_log_step += 1

            self.scheduler.step()
            start = time.time()

    def train(self, save_model_path='models/', resume_model=None, pretrained_model=None):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        """
        torch.manual_seed(1000)

        # 初始化分布式训练环境
        if torch.cuda.device_count() > 1:
            init_process_group(backend='nccl')
            local_rank = get_rank()
            world_size = get_world_size()
        else:
            local_rank = 0
            world_size = 1

        writer = None
        if local_rank == 0:
            writer = SummaryWriter(log_dir='log')

        if world_size > 1:
            torch.cuda.set_device(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

        # 获取数据
        self.__setup_dataloader(is_train=True)
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=True)

        # 多卡训练
        if world_size > 1:
            train_sampler = DistributedSampler(self.train_dataset)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler,
                                                            batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                                            num_workers=self.configs.dataset_conf.dataLoader.num_workers)

        logger.info(f'训练数据：{len(self.train_dataset)}')

        self.__load_pretrained(pretrained_model=pretrained_model)
        last_epoch, best_acc = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        self.train_loss, self.train_acc = None, None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        last_epoch += 1

        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], last_epoch)

        # 最大步数
        self.max_step = len(self.train_loader) * self.configs.train_conf.max_epoch
        self.train_step = max(last_epoch, 0) * len(self.train_loader)

        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            epoch_id += 1
            start_epoch = time.time()

            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer)

            if local_rank == 0:
                if self.stop_eval: continue
                logger.info('=' * 70)
                self.eval_loss, self.eval_acc = self.evaluate()
                logger.info(
                    f'Test epoch: {epoch_id}, time/epoch: {str(timedelta(seconds=(time.time() - start_epoch)))}, loss: {self.eval_loss:.5f}, accuracy: {self.eval_acc:.5f}')
                logger.info('=' * 70)
                writer.add_scalar('Test/Accuracy', self.eval_acc, self.test_log_step)
                writer.add_scalar('Test/Loss', self.eval_loss, self.test_log_step)
                self.test_log_step += 1
                self.model.train()

                # 保存最优模型
                if self.eval_acc >= best_acc:
                    best_acc = self.eval_acc
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=self.eval_acc,
                                           best_model=True)

                # 保存模型
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=self.eval_acc)


    def evaluate(self, resume_model=None, save_matrix_path=None):
        """
        评估模型
        :param resume_model: 所使用的模型
        :param save_matrix_path: 保存混合矩阵的路径
        :return: 评估结果
        """
        if self.test_loader is None:
            self.__setup_dataloader(is_train=False)
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=False)

        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model, map_location=self.device)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')

        self.model.eval()

        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (features, label) in enumerate(tqdm(self.test_loader)):
                if self.stop_eval: break
                features, label = features.to(self.device), label.to(self.device)
                output = self.model(features)
                loss = self.loss(output, label)

                # 计算准确率
                _, preds_batch = torch.max(output, 1)
                acc = (preds_batch == label).float().mean().item()
                accuracies.append(acc)
                losses.append(loss.item())

                preds.extend(preds_batch.cpu().numpy())
                labels.extend(label.cpu().numpy())

        loss = float(np.mean(losses)) if len(losses) > 0 else -1
        acc = float(np.mean(accuracies)) if len(accuracies) > 0 else -1

        # 保存混合矩阵
        if save_matrix_path is not None:
            try:
                cm = confusion_matrix(labels, preds)
                plot_confusion_matrix(cm=cm, save_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'),
                                      class_labels=self.class_labels)
            except Exception as e:
                logger.error(f'保存混淆矩阵失败：{e}')

        self.model.train()
        return loss, acc


    def export(self, save_model_path='models/', resume_model='models/EcapaTdnn_Fbank/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=False)

        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pth')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = torch.load(resume_model, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        logger.info(f'成功恢复模型参数和优化方法参数：{resume_model}')

        self.model.eval()

        # 导出为 TorchScript 模型
        infer_model_path = os.path.join(save_model_path,
                                        f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}', 'infer')
        os.makedirs(infer_model_path, exist_ok=True)

        scripted_model = torch.jit.script(self.model)
        scripted_model.save(os.path.join(infer_model_path, 'model.pt'))

        logger.info(f"预测模型已保存：{os.path.join(infer_model_path, 'model.pt')}")


def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Feature Extraction Script")
    default_configs = os.getenv('CONFIG_FILE', '/home/ubuntu/ZX/dataset/cam++.yml')
    default_save_dir = os.getenv('SAVE_DIR', '/home/ubuntu/ZX/dataset/feature')

    parser.add_argument('--configs', type=str, default=default_configs, help='配置文件')
    parser.add_argument('--save_dir', type=str, default=default_save_dir, help='保存特征的路径')

    return parser

def main(args):
    print(f"Using config file: {args.configs}")
    print(f"Saving features to: {args.save_dir}")
    trainer = PPAClsTrainer(configs=args.configs)
    trainer.extract_features(args.save_dir)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)