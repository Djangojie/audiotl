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
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data_dir, train=False, test=False, val=False):
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
        MFCC_data = MFCC_data.unsqueeze(0).unsqueeze(0)
        MFCC_data = MFCC_data.repeat(1, 3, 1, 1)
        MFCC_data = nn.BatchNorm2d(3)(MFCC_data).squeeze(0)

        Label_file_name = self.Label_file_list[idx]
        Label_file_path = os.path.join(self.Label_data_dir, Label_file_name)

        Label_dataframe = pd.read_excel(Label_file_path)
        label = Label_dataframe.values[:, 1:]
        label = label.astype(float)
        skill_label = label[:10, :].ravel()

        Label_data = torch.tensor(skill_label - 1, dtype=torch.float32)

        return MFCC_data, Label_data

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

        self.model = self.build_model(cfg)

        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=cfg.learning_rate,
                                    weight_decay=5e-4)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 80], gamma=0.1)

        self.losses = nn.CrossEntropyLoss()

    def build_dataset(self, cfg):
        """构建训练数据和测试数据"""
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        train_dataset = CustomDataset(cfg.data_dir, train=True)
        test_dataset = CustomDataset(cfg.data_dir, val=True)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=cfg.num_workers)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                      num_workers=cfg.num_workers)

        cfg.num_classes = 50
        print("train nums:{}".format(len(train_dataset)))
        print("test  nums:{}".format(len(test_dataset)))
        return train_loader, test_loader

    def build_model(self, cfg):
        """构建模型并加载预训练权重"""
        if cfg.net_type == "mbv2_baseline":
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.last_channel, cfg.num_classes)
        elif cfg.net_type == "resnet34":
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
        elif cfg.net_type == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
        else:
            raise Exception("Error: Unsupported net_type {}".format(cfg.net_type))

        # 加载自定义预训练权重并跳过分类器
        if cfg.pretrained_weights_path and os.path.exists(cfg.pretrained_weights_path):
            print(f"Loading custom pretrained weights from {cfg.pretrained_weights_path}")
            pretrained_dict = torch.load(cfg.pretrained_weights_path)

            # 获取当前模型的 state_dict
            model_dict = model.state_dict()

            # 过滤掉预训练权重中不匹配的部分（如分类器层）
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and model_dict[k].shape == v.shape}

            # 更新现有的 model_dict
            model_dict.update(pretrained_dict)

            # 加载新的 state_dict
            model.load_state_dict(model_dict)


        model.to(self.device)
        return model

    def epoch_test(self, epoch):
        """模型测试"""
        loss_sum = []
        accuracies = []
        self.model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(self.test_loader)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                output = self.model(inputs)
                output = output.view(output.shape[0], 5, 10)

                loss = self.losses(output.float(), labels.long())

                output = output.cpu().numpy()
                output = np.argmax(output, axis=1)
                labels = labels.cpu().numpy()
                acc = np.mean((output == labels).astype(int))
                accuracies.append(acc)
                loss_sum.append(loss.item())
        acc = sum(accuracies) / len(accuracies)
        loss = sum(loss_sum) / len(loss_sum)
        print("Test epoch:{:3.3f},Acc:{:3.3f},loss:{:3.3f}".format(epoch, acc, loss))
        print('=' * 100)
        return acc, loss

    def epoch_train(self, epoch):
        """模型训练"""
        loss_sum = []
        accuracies = []
        self.model.train()
        for step, (inputs, labels) in enumerate(tqdm(self.train_loader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            output = self.model(inputs)
            output = output.view(output.shape[0], 5, 10)

            loss = self.losses(output.float(), labels.long())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
        print('=' * 100)
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
    parser.add_argument('--num_workers', type=int, default=0, help='读取数据的线程数量')
    parser.add_argument('--num_epoch', type=int, default=100, help='训练的轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='初始学习率的大小')
    parser.add_argument('--input_shape', type=str, default='(None, 3, 224, 224)', help='数据输入的形状')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--net_type', type=str, default="mbv2_baseline", help='backbone')
    parser.add_argument('--data_dir', type=str, default=data_dir, help='数据路径')
    parser.add_argument('--work_dir', type=str, default='/home/ubuntu/ZX/UrbanSound8K/work', help='模型训练和保存的路径')
    parser.add_argument('--pretrained_weights_path', type=str, default=None, help='自定义预训练权重文件路径')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    t = Train(args)
    t.run()
