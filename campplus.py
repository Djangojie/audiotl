import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
import logging
import shutil


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (state['log_dir'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (state['log_dir']) + 'model_best.pth')


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
            self.MFCC_file_list = self.MFCC_file_list[:int(0.7 * self.listLen)]
            self.Label_file_list = self.Label_file_list[:int(0.7 * self.listLen)]
        elif self.val:
            self.MFCC_file_list = self.MFCC_file_list[int(0.7 * self.listLen):int(0.8 * self.listLen)]
            self.Label_file_list = self.Label_file_list[int(0.7 * self.listLen):int(0.8 * self.listLen)]
        elif self.test:
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
        MFCC_data = torch.nn.BatchNorm2d(1)(MFCC_data).squeeze(0)

        Label_file_name = self.Label_file_list[idx]
        Label_file_path = os.path.join(self.Label_data_dir, Label_file_name)

        Label_dataframe = pd.read_excel(Label_file_path)
        label = Label_dataframe.values[:, 1:].astype(float)
        label = label[:10, :].ravel()
        Label_data = torch.tensor(label-1)

        return MFCC_data, Label_data

def train_epoch(net, train_loader, criterion, optimizer, device, lr_scheduler):
    net.train()
    train_loss = []
    correct = 0
    total = 0
    for im, label in train_loader:
        im, label = im.to(device), label.to(device)
        optimizer.zero_grad()

        output = net(im)
        output = output.view(output.shape[0], 5, 10)

        loss = criterion(output.float(), label.long())
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss.append(loss.item())
        total += label.size(0) * label.size(1)
        correct += (label == output.argmax(dim=1)).sum().item()
    return np.mean(train_loss), correct / total


def validate(net, val_loader, criterion, device):
    net.eval()
    val_loss = []
    correct = 0
    total = 0
    with torch.no_grad():
        for im, label in val_loader:
            im, label = im.to(device), label.to(device)
            output = net(im)
            output = output.view(output.shape[0], 5, 10)

            loss = criterion(output.float(), label.long())
            val_loss.append(loss.item())
            total += label.size(0) * label.size(1)
            correct += (label == output.argmax(dim=1)).sum().item()

    return np.mean(val_loss), correct / total


def train_model(net, train_loader, val_loader, num_epochs, optimizer, criterion, lr_scheduler, device, log_dir, pretrained_weights=None):
    if pretrained_weights and os.path.exists(pretrained_weights):
        net.load_state_dict(torch.load(pretrained_weights))
        logging.info(f"Loaded pretrained weights from {pretrained_weights}")

    writer = SummaryWriter(log_dir=log_dir)
    best_val_loss = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, device, lr_scheduler)
        val_loss, val_acc = validate(net, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Val_Loss', val_loss, epoch)
        writer.add_scalar('Val_Accuracy', val_acc, epoch)


        state = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'log_dir': log_dir
        }
        save_checkpoint(state, False, filename='checkpoint_latest.pth')


        if val_loss > best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(state, True, filename='checkpoint_best.pth')

    writer.close()

def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU())
        elif name == 'sigmod':
            nonlinear.add_module('sigmod', nn.Sigmoid())
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm2d(channels))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear

def statistics_pooling(x, axis=1, keepdim=True, unbiased=True):
    mean = x.mean(dim=axis)
    std = x.std(dim=axis, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=1)
    if keepdim:
        stats = stats.unsqueeze(dim=axis)
    return stats

class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)

class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x

class CAMLayer(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv2d(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.linear1 = nn.Conv2d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Conv2d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        if stype == 'avg':
            seg = F.avg_pool2d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool2d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand((*shape, seg_len)).reshape((*shape[:-1], -1))
        seg = seg[..., :x.shape[-1]]
        return seg

class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, bias=False, config_str='batchnorm-relu', memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv2d(in_channels, bn_channels, 1)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x

class CAMDenseTDNNBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, bias=False, config_str='batchnorm-relu', memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels, out_channels=out_channels,
                                      bn_channels=bn_channels, kernel_size=kernel_size, stride=stride,
                                      dilation=dilation, bias=bias, config_str=config_str, memory_efficient=memory_efficient)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x

class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv2d(in_channels, out_channels, 1)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x

class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FCM(nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=64, in_channels=1):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(in_channels, m_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(m_channels)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        return out

class CAMPPlus(nn.Module):
    def __init__(self, num_class, input_size, embd_dim=384, growth_rate=32, bn_size=4, in_channels=64, init_channels=128, config_str='batchnorm-relu', memory_efficient=True):
        super(CAMPPlus, self).__init__()

        self.head = FCM(block=BasicResBlock, num_blocks=[2, 2], m_channels=64, in_channels=input_size)

        self.xvector = nn.Sequential(TDNNLayer(in_channels, init_channels, 5, stride=2, dilation=1, padding=-1, config_str=config_str),
                                     CAMDenseTDNNBlock(num_layers=3, in_channels=init_channels, out_channels=growth_rate, bn_channels=growth_rate * bn_size,
                                                       kernel_size=3, stride=1, dilation=1, bias=False, config_str=config_str, memory_efficient=memory_efficient),
                                     TransitLayer(in_channels=init_channels + growth_rate * 3, out_channels=init_channels, bias=False, config_str=config_str))
        self.xvector_1 = get_nonlinear(config_str, init_channels)
        self.xvector_2 = StatsPool()
        self.xvector_3 = DenseLayer(1, 1, config_str='batchnorm-relu')
        self.output_1 = nn.Linear(embd_dim, num_class)

    def forward(self, x):
        x = self.head(x)
        x = self.xvector(x)
        x = self.xvector_1(x)
        x = self.xvector_2(x)
        x = self.xvector_3(x)
        x = x.view(x.size(0), -1)
        x = self.output_1(x)

        return x

if __name__ == '__main__':
    data_dir = r"mezzo"
    train_batch_size = 128
    val_batch_size = 128
    num_workers = 0
    num_classes = 50
    num_epochs = 400
    learning_rate = 1e-4
    pretrained_weights = None

    train_dataset = CustomDataset(data_dir, train=True)
    val_dataset = CustomDataset(data_dir, val=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAMPPlus(num_class=num_classes,
                     input_size=1,
                     embd_dim=384,
                     growth_rate=32,
                     bn_size=4,
                     init_channels=128,
                     config_str='batchnorm-relu',
                     memory_efficient=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, lr_scheduler, device, log_dir,
                pretrained_weights)