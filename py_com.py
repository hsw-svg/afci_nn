import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
 
# exit()

def Get_FileList(folder_path):

    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths

#从csv文件中读取数据.并保存到列表中
data_list = []
label_list = []
predict_list = []
arc_file_paths = Get_FileList(r'E:\AFCI\HSW\AFCI二阶段\原始数据-5&24\训练数据（ARC_Normal）\ARC\工况2')
normal_file_paths = Get_FileList(r'E:\AFCI\HSW\AFCI二阶段\原始数据-5&24\训练数据（ARC_Normal）\Normal\工况2')
for file_path in arc_file_paths:
    file_data = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            file_data.append(int(row[0]))
            if(len(file_data) == 1024):
                data_list.append(file_data)
                file_data = []
                label_list.append(1)

for file_path in normal_file_paths:
    file_data = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            file_data.append(int(row[0]))
            if(len(file_data) == 1024):
                data_list.append(file_data)
                file_data = []
                label_list.append(0)


# 定义 CustomDataset 类（确保在代码开头）
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 数据转换为Tensor并调整维度
data = torch.tensor(data_list, dtype=torch.float32)
data = data.unsqueeze(2)  # 添加特征维度 → 形状：(num_samples, seq_length, 1)
labels = torch.tensor(label_list, dtype=torch.long)

# 数据集划分（使用NumPy数组）
train_data_np, test_data_np, train_labels_np, test_labels_np = train_test_split(
    data.numpy(), labels.numpy(), test_size=0.2
)

# 转换回Tensor
train_data = torch.from_numpy(train_data_np).float()
test_data = torch.from_numpy(test_data_np).float()
train_labels = torch.from_numpy(train_labels_np).long()
test_labels = torch.from_numpy(test_labels_np).long()

# 定义数据加载器
batch_size = 32
train_loader = DataLoader(
    dataset=CustomDataset(train_data, train_labels),
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    dataset=CustomDataset(test_data, test_labels),
    batch_size=batch_size,
    shuffle=False
)

# 定义NN模型
class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NNModel, self).__init__()
# 卷积层1：输入通道=input_size，输出通道=hidden_size，kernel_size=3
        self.conv1 = nn.Conv1d(in_channels=input_size, 
                              out_channels=16, 
                              kernel_size=3, 
                              padding=1)  # padding=1保证输出长度不变
        # 卷积层2：输入通道=hidden_size，输出通道=hidden_size
        self.conv2 = nn.Conv1d(in_channels=16, 
                              out_channels=32, 
                              kernel_size=3, 
                              padding=1)
                # 卷积层3：输入通道=hidden_size，输出通道=hidden_size
        self.conv3 = nn.Conv1d(in_channels=32, 
                              out_channels=64, 
                              kernel_size=3, 
                              padding=1)
        # 池化层：使用最大池化，步长2，将序列长度减半
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 全局平均池化：将序列长度压缩到1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 全连接层：将特征转换为输出维度
        self.fc = nn.Linear(64, 32)
    
    def forward(self, x):
        # 调整输入维度为 (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)  
        
        # 第一层卷积 + 激活函数 + 池化
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)  # 输出形状变为 (batch, hidden_size, seq_len/2)
        
        # 第二层卷积 + 激活函数 + 池化
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)  # 输出形状变为 (batch, hidden_size, seq_len/4)
        # 第三层卷积 + 激活函数 + 池化
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)  # 输出形状变为 (batch, hidden_size, seq_len/8)
        
        # 全局平均池化：将序列长度压缩到1，形状变为 (batch, hidden_size, 1)
        x = self.avg_pool(x)
        
        # 展平为 (batch_size, hidden_size)
        x = x.view(x.size(0), -1)
        
        # 全连接层输出
        return self.fc(x)

# 初始化模型、损失函数和优化器
input_size = data.shape[2]  # 正确获取特征维度（1）
print(input_size)
hidden_size = 128
output_size = len(np.unique(labels))
model = NNModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
num_epochs = 100
print('开始训练')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

# 测试
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device)
        outputs = model(batch_data)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels.to(device)).sum().item()
print(f'Test Accuracy: {100 * correct / total:.2f}%')

