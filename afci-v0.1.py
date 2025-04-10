import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.amp import autocast
import ctypes
import sys

# 管理员权限检查（网页2硬件适配要求）
if sys.platform.startswith('win'):
    def is_admin():
        try: return ctypes.windll.shell32.IsUserAnAdmin()
        except: return False
    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
        sys.exit()

# 硬件加速配置（网页6大模型训练优化）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# 安全数据加载（网页2数据采集规范）
class AFCI_DataLoader:
    def __init__(self, arc_paths, normal_paths):
        self.arc_files = arc_paths
        self.normal_files = normal_paths
    
    
    def _read_signal(self, file_path):
        """安全读取CSV信号（网页2数据校验）"""
        signal = []
        try:
            if not os.access(file_path, os.R_OK):
                print(f"权限不足跳过: {file_path}")
                return None
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 1: continue
                    signal.append(float(row[0]))
                    if len(signal) == 1024: break
            return np.array(signal) if len(signal)==1024 else None
        except Exception as e:
            print(f"读取异常{file_path}: {str(e)}")
            return None
    
    def load_dataset(self):
        """加载数据集（网页3数据校验）"""
        arc_data = [d for f in self.arc_files if (d := self._read_signal(f)) is not None]
        normal_data = [d for f in self.normal_files if (d := self._read_signal(f)) is not None]
        
        # 空数据集保护（网页5异常处理）
        if not arc_data and not normal_data:
            raise RuntimeError("未加载到有效数据，请检查：\n1.文件路径\n2.数据格式\n3.读取权限")
        
        # 合并数据集时自动填充空数组
        arc_data = arc_data if arc_data else [np.zeros(1024)]
        normal_data = normal_data if normal_data else [np.zeros(1024)]
        
        X = np.vstack(arc_data + normal_data)
        y = np.concatenate([np.ones(len(arc_data)), np.zeros(len(normal_data))])
        return train_test_split(X, y, test_size=0.2, stratify=y)

# 数据增强（网页1频域特征处理）
class AFCI_Dataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels
        
    def __len__(self): return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx].copy().astype(np.float32)  # 强制转换为float32
        # 频域噪声增强（网页1电弧特征）
        if np.random.rand() > 0.5:
            fft = np.fft.fft(signal)
            freq_mask = (np.random.rand(*fft.shape) > 0.2)  # 屏蔽20%频段
            signal = np.fft.ifft(fft * freq_mask).real
        # 时域增强（网页2数据质量要求）
        if np.random.rand() > 0.5:
            signal += np.random.normal(0, 0.02, size=signal.shape)
        # 标准化（网页6输入规范化）
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        return torch.FloatTensor(signal), torch.tensor(self.labels[idx], dtype=torch.float32)

# 轻量化模型（网页6部署优化）
class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积（参数量减少75%）"""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, 
                                  padding=kernel_size//2, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class AFCI_Detector(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取（网页1 30-100kHz特征捕获）
        self.feature_extractor = nn.Sequential(
            DepthwiseSeparableConv(1, 16, 15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            DepthwiseSeparableConv(16, 32, 11),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            DepthwiseSeparableConv(32, 64, 7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # 分类器（网页5轻量分类）
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, 1024]
        features = self.feature_extractor(x).squeeze()
        return self.classifier(features)

# 混合精度训练（网页6训练优化）
def train_model(model, train_loader, test_loader, epochs=100):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler(device='cuda') 
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        # 训练阶段
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type = 'cuda'):
                outputs = model(signals).squeeze()
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        # 验证阶段
        model.eval()
        correct = 0
        with torch.no_grad():
            for signals, labels in test_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals).squeeze()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%')
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_afci_model.pth')
        scheduler.step()
    print(f'训练完成,最高准确率: {best_acc:.2f}%')

def Get_FileList(folder_path):

    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths

if __name__ == "__main__":
    # 数据路径（网页2采集规范）
    arc_file_paths = Get_FileList(r'E:\AFCI\HSW\AFCI二阶段\原始数据-5&24\训练数据（ARC_Normal）\ARC\工况2')
    normal_file_paths = Get_FileList(r'E:\AFCI\HSW\AFCI二阶段\原始数据-5&24\训练数据（ARC_Normal）\Normal\工况2')
    
    # 加载数据
    loader = AFCI_DataLoader(arc_file_paths, normal_file_paths)
    X_train, X_test, y_train, y_test = loader.load_dataset()
    
    # 创建DataLoader（网页7批量处理）
    batch_size = 64
    train_loader = DataLoader(
        AFCI_Dataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if sys.platform.startswith('win') else os.cpu_count()
    )
    test_loader = DataLoader(
        AFCI_Dataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if sys.platform.startswith('win') else os.cpu_count()
    )
    
    # 初始化模型
    model = AFCI_Detector()
    
    # 开始训练
    train_model(model, train_loader, test_loader, epochs=100)
    
    # 模型导出（网页2部署要求）
    model.load_state_dict(torch.load('best_afci_model.pth'))
    dummy_input = torch.randn(1, 1024).to(device)
    torch.onnx.export(model, dummy_input, "afci_model.onnx",
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'])