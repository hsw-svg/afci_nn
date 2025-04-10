import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from TSTransformerEncoder import TSTransformerEncoder
 
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


# 初始化模型、损失函数和优化器
input_size = data.shape[2]  # 正确获取特征维度（1）
print(input_size)
hidden_size = 128
output_size = len(np.unique(labels))
# model = NNModel(input_size, hidden_size, output_size)
model = TSTransformerEncoder(            
            feat_dim=input_size,
            max_len= 1024,
            d_model=hidden_size,
            n_heads=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1
            )
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
        batch_labels = batch_labels.unsqueeze(1)
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

