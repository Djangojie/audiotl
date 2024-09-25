import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import librosa
import pandas as pd
import random
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, train=False, test=False, val=False, transform=None):
        self.data_dir = data_dir
        self.MFCC_data_dir = os.path.join(self.data_dir, 'MFCC_Output')
        self.Label_data_dir = os.path.join(self.data_dir, 'Label')

        self.MFCC_file_list = os.listdir(self.MFCC_data_dir)
        self.Label_file_list = os.listdir(self.Label_data_dir)

        combined = list(zip(self.MFCC_file_list, self.Label_file_list))
        random.shuffle(combined)
        self.MFCC_file_list[:], self.Label_file_list[:] = zip(*combined)

        self.train = train
        self.test = test
        self.val = val
        self.listLen = len(self.MFCC_file_list)
        self.transform = transform

        if self.train:
            self.MFCC_file_list = self.MFCC_file_list[:int(0.8 * self.listLen)]
            self.Label_file_list = self.Label_file_list[:int(0.8 * self.listLen)]
        elif self.val:
            self.MFCC_file_list = self.MFCC_file_list[int(0.8 * self.listLen):]
            self.Label_file_list = self.Label_file_list[int(0.8 * self.listLen):]

    def __len__(self):
        return len(self.MFCC_file_list)

    def __getitem__(self, idx):
        MFCC_file_name = self.MFCC_file_list[idx]
        MFCC_file_path = os.path.join(self.MFCC_data_dir, MFCC_file_name)

        MFCC_data = pd.read_excel(MFCC_file_path, header=None, engine="openpyxl")
        MFCC_data = MFCC_data.values.astype(float)
        MFCC_data = torch.tensor(MFCC_data, dtype=torch.float32)
        MFCC_data = MFCC_data.unsqueeze(0)
        MFCC_data = MFCC_data.repeat(3, 1, 1)

        Label_file_name = self.Label_file_list[idx]
        Label_file_path = os.path.join(self.Label_data_dir, Label_file_name)

        Label_dataframe = pd.read_excel(Label_file_path)
        label = Label_dataframe.values[:, 1:]
        label = label.astype(float)
        skill_label = label[:10, :].ravel()

        Label_data = torch.tensor(skill_label - 1, dtype=torch.float32)

        if self.transform:
            MFCC_data = self.transform(MFCC_data)

        return MFCC_data, Label_data

class CRNN(nn.Module):
    def __init__(self, cnn_out_channels, rnn_hidden_size, num_classes, pretrained=True):
        super(CRNN, self).__init__()

        # 加载预训练的ResNet18作为特征提取器
        resnet = models.resnet18(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        self.rnn = nn.LSTM(cnn_out_channels, rnn_hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)

        # reshape x to (batch_size, time_steps, cnn_out_channels)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, x.size(1) * x.size(2), -1)

        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return x


class UrbanSoundDataset(data.Dataset):
    def __init__(self, file_list, label_list, data_dir, transform=None, n_mfcc=40):
        self.file_list = file_list
        self.labels = self.load_labels(label_list)
        self.data_dir = data_dir
        self.transform = transform
        self.n_mfcc = n_mfcc

    def load_labels(self, label_list):
        labels = {}
        with open(label_list, 'r') as f:
            for line in f:
                label, idx = line.strip().split(',')
                labels[int(idx)] = int(idx)
        return labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])


        file_name = os.path.basename(file_path)
        class_id = int(file_name.split('-')[1])

        if class_id not in self.labels:
            raise KeyError(f"Class ID {class_id} not found in labels.")


        audio, sr = librosa.load(file_path, sr=16000)
        signal_length = len(audio)
        n_fft = min(2048, signal_length)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_mels=128, fmax=sr/2, n_fft=n_fft)


        mfcc_resized = np.resize(mfcc, (224, 224))
        mfcc_resized = np.expand_dims(mfcc_resized, axis=0)


        mfcc_rgb = np.repeat(mfcc_resized, 3, axis=0)

        mfcc_rgb = torch.tensor(mfcc_rgb, dtype=torch.float32)

        if self.transform:
            mfcc_rgb = self.transform(mfcc_rgb)

        return mfcc_rgb, torch.tensor(class_id, dtype=torch.long)


class Train(object):
    """Training Pipeline"""

    def __init__(self, cfg):
        self.device = "cuda:{}".format(cfg.gpu_id) if torch.cuda.is_available() else "cpu"
        self.num_epoch = cfg.num_epoch
        self.net_type = cfg.net_type
        self.work_dir = os.path.join(cfg.work_dir, self.net_type)
        self.model_dir = os.path.join(self.work_dir, "model")
        self.log_dir = os.path.join(self.work_dir, "log")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tensorboard = SummaryWriter(self.log_dir)
        self.train_loader, self.test_loader = self.build_dataset(cfg)

        # Initialize CRNN model and move to GPU if available
        self.model = CRNN(cnn_out_channels=512, rnn_hidden_size=256, num_classes=cfg.num_classes, pretrained=True)
        self.model.to(self.device)  # Move model to the device (GPU or CPU)

        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=cfg.learning_rate,
                                    weight_decay=5e-4)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 80], gamma=0.1)

        self.losses = nn.CrossEntropyLoss()

    def build_dataset(self, cfg):
        """构建训练数据和测试数据"""
        # train_file_list = [line.strip() for line in open(cfg.train_data)]
        # test_file_list = [line.strip() for line in open(cfg.test_data)]

        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        train_dataset = CustomDataset(cfg.data_dir, train=True, transform=transform)
        test_dataset = CustomDataset(cfg.data_dir, val=True, transform=transform)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=cfg.num_workers)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                      num_workers=cfg.num_workers)

        # cfg.num_classes = len(train_dataset.labels)
        cfg.num_classes = 50
        print("train nums:{}".format(len(train_dataset)))
        print("test  nums:{}".format(len(test_dataset)))
        return train_loader, test_loader

    def build_model(self, cfg):
        """构建模型并加载预训练权重"""
        if cfg.net_type == "CRNN_pre_image":
            model = CRNN(cnn_out_channels=512, rnn_hidden_size=256, num_classes=cfg.num_classes, pretrained=False)
        elif cfg.net_type == "mbv2":
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.last_channel, cfg.num_classes)
        elif cfg.net_type == "resnet34":
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
        elif cfg.net_type == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
        else:
            raise Exception("Error: Unsupported net_type {}".format(cfg.net_type))

        # 加载自定义预训练权重
        if cfg.pretrained_weights_path and os.path.exists(cfg.pretrained_weights_path):
            print(f"Loading custom pretrained weights from {cfg.pretrained_weights_path}")
            model.load_state_dict(torch.load(cfg.pretrained_weights_path))
            print("I get it.")

        model.to(self.device)
        return model

    def epoch_test(self, epoch):
        """模型测试"""
        loss_sum = []
        accuracies = []
        self.model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(self.test_loader)):
                inputs = inputs.to(self.device)  # Move inputs to GPU
                labels = labels.to(self.device)  # Move labels to GPU
                output = self.model(inputs)
                output = output.view(output.shape[0], 5, 10)

                loss = self.losses(output.float(), labels.long())

                output = output.cpu().numpy()
                output = np.argmax(output, axis=1)
                # loss = self.losses(output, labels)

                # output = torch.nn.functional.softmax(output, dim=1)
                # output = output.cpu().numpy()
                # output = np.argmax(output, axis=1)
                labels = labels.cpu().numpy()
                acc = np.mean((output == labels).astype(int))
                accuracies.append(acc)
                loss_sum.append(loss.item())
        acc = sum(accuracies) / len(accuracies)
        loss = sum(loss_sum) / len(loss_sum)
        print("Test epoch:{:3.3f},Acc:{:3.3f},loss:{:3.3f}".format(epoch, acc, loss))
        print('=' * 70)
        return acc, loss

    def epoch_train(self, epoch):
        """模型训练"""
        loss_sum = []
        accuracies = []
        self.model.train()
        for step, (inputs, labels) in enumerate(tqdm(self.train_loader)):
            inputs = inputs.to(self.device)  # Move inputs to GPU
            labels = labels.to(self.device)  # Move labels to GPU
            output = self.model(inputs)
            output = output.view(output.shape[0], 5, 10)

            loss = self.losses(output.float(), labels.long())
            # loss = self.losses(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # output = torch.nn.functional.softmax(output, dim=1)
            output = output.cpu().detach().numpy()
            output = np.argmax(output, axis=1)
            labels = labels.cpu().numpy()
            acc = np.mean((output == labels).astype(int))
            accuracies.append(acc)
            loss_sum.append(loss.item())
            if step % 50 == 0:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f，lr:%f' % (
                    datetime.now(), epoch, step, len(self.train_loader), sum(loss_sum) / len(loss_sum),
                    sum(accuracies) / len(accuracies), lr))
        acc = sum(accuracies) / len(accuracies)
        loss = sum(loss_sum) / len(loss_sum)
        print("Train epoch:{:3.3f},Acc:{:3.3f},loss:{:3.3f}".format(epoch, acc, loss))
        print('=' * 70)
        return acc, loss

    def run(self):

        for epoch in range(self.num_epoch):
            train_acc, train_loss = self.epoch_train(epoch)
            test_acc, test_loss = self.epoch_test(epoch)
            self.tensorboard.add_scalar("train_acc", train_acc, epoch)
            self.tensorboard.add_scalar("train_loss", train_loss, epoch)
            self.tensorboard.add_scalar("test_acc", test_acc, epoch)
            self.tensorboard.add_scalar("test_loss", test_loss, epoch)
            self.scheduler.step()
            self.save_model(epoch, test_acc)

    def save_model(self, epoch, acc):
        """保存模型"""
        model_path = os.path.join(self.model_dir, 'model_{:0=3d}_{:.3f}.pth'.format(epoch, acc))
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.model.state_dict(), model_path)


def get_parser():
    data_dir = "mezzo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size', type=int, default=64, help='训练的批量大小')
    parser.add_argument('--num_workers', type=int, default=8, help='读取数据的线程数量')
    parser.add_argument('--num_epoch', type=int, default=100, help='训练的轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='初始学习率的大小')
    parser.add_argument('--input_shape', type=str, default='(None, 3, 224, 224)', help='数据输入的形状')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--net_type', type=str, default="CRNN_pre_image", help='backbone')
    parser.add_argument('--data_dir', type=str, default=data_dir, help='数据路径')
    parser.add_argument('--train_data', type=str, default='/home/ubuntu/ZX/UrbanSound8K/train_list.txt', help='训练数据列表路径')
    parser.add_argument('--test_data', type=str, default='/home/ubuntu/ZX/UrbanSound8K/test_list.txt', help='测试数据列表路径')
    parser.add_argument('--class_name', type=str, default='/home/ubuntu/ZX/UrbanSound8K/label_list.txt', help='类别文件路径')
    parser.add_argument('--work_dir', type=str, default='/home/ubuntu/ZX/UrbanSound8K/work', help='模型训练和保存的路径')
    parser.add_argument('--pretrained_weights_path', type=str, default=None, help='自定义预训练权重文件路径')

    return parser



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    t = Train(args)
    t.run()
