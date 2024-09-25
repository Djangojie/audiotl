import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
import logging
import shutil

# 设置随机数种子
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


# 自定义数据集
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
        MFCC_data = nn.BatchNorm2d(1)(MFCC_data).squeeze(0)

        Label_file_name = self.Label_file_list[idx]
        Label_file_path = os.path.join(self.Label_data_dir, Label_file_name)

        Label_dataframe = pd.read_excel(Label_file_path)
        label = Label_dataframe.values[:, 1:]
        label = label.astype(float)
        skill_label = label[:10, :].ravel()

        Label_data = torch.tensor(skill_label - 1, dtype=torch.float32)

        return MFCC_data, Label_data


# CRNN模型定义
class CRNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.rnn = nn.LSTM(input_size=128 * 16, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        self.bn = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.dropout(x, 0.1)
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.dropout(x, 0.1)
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.dropout(x, 0.1)

        x = x.permute(0, 2, 1, 3)
        batch_size, time_steps, channels, freq_bins = x.size()
        x = x.reshape(batch_size, time_steps, channels * freq_bins)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = F.dropout(x, 0.1)
        x = self.fc(x)

        return x


# 训练和验证函数
def train_epoch(net, train_loader, criterion, optimizer, device, lr_scheduler):
    net.train()
    running_loss = 0.
    train_loss = []
    correct = 0
    total = 0
    total_samples = 0
    for im, label in train_loader:
        im, label = im.to(device), label.to(device)

        optimizer.zero_grad()

        output = net(im)
        output = output.view(output.shape[0], 5, 10)

        loss_2 = criterion(output.float(), label.long())
        running_loss += loss_2.item() * im.size(0)

        loss_2.requires_grad_(True)
        loss_2.backward()
        optimizer.step()
        lr_scheduler.step()

        total = output.size(0) * output.size(2)
        total_samples += im.size(0)

        correct += (label == output.argmax(dim=1)).sum().item()
    epoch_loss = running_loss / total_samples

    return epoch_loss, correct / total


def validate(net, val_loader, criterion, device):
    net.eval()
    running_loss = 0.
    val_loss = []
    correct = 0
    total = 0
    total_samples = 0
    with torch.no_grad():
        for im, label in val_loader:
            im, label = im.to(device), label.to(device)
            output = net(im)
            output = output.view(output.shape[0], 5, 10)
            # output = output.argmax(dim=1)
            loss_1 = criterion(output.float(), label.long())
            # loss_2 = torch.sqrt(criterion(output, label))
            # val_loss.append(loss_2.cpu())
            running_loss += loss_1.item() * im.size(0)

            total = label.size(0) * label.size(1)
            total_samples += im.size(0)

            correct += int((label == output.argmax(dim=1)).sum())
    epoch_loss = running_loss / total_samples

    return epoch_loss, correct / total


def train_model(net, train_loader, val_loader, num_epochs, optimizer, criterion, lr_scheduler, device, log_dir, pretrained_weights=None):
    if pretrained_weights and os.path.exists(pretrained_weights):
        net.load_state_dict(torch.load(pretrained_weights), strict=False)
        print(f"Loaded pretrained weights from {pretrained_weights}")

    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, device, lr_scheduler)
        val_loss, val_acc = validate(net, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Val_Loss', val_loss, epoch)
        writer.add_scalar('Val_Accuracy', val_acc, epoch)
        # writer.add_scalar('Train_mse', train_mse, epoch)
        # writer.add_scalar('Val_mse', val_mse, epoch)

        # 保存检查点
        state = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'log_dir': log_dir
        }
        save_checkpoint(state, False, filename='checkpoint_latest.pth')

        # 保存最优检查点
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(state, True, filename='checkpoint_best.pth')

    writer.close()


# 主函数
if __name__ == '__main__':
    data_dir = r"mezzo"
    train_batch_size = 128
    val_batch_size = 128
    num_workers = 0
    num_classes = 50
    num_epochs = 600
    learning_rate = 1e-4
    pretrained_weights = "/home/ubuntu/ZX/AST/runs/logs/2024-08-14_19-39-46/model_best.pth"

    train_dataset = CustomDataset(data_dir, train=True)
    val_dataset = CustomDataset(data_dir, val=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(1, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, lr_scheduler, device, log_dir,
                pretrained_weights)
