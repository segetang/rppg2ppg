'''
#epoch200일때 노이즈가 더 많아짐 한 177정도에서 로스값이 튐
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 데이터셋 클래스 정의
class RPPGDataset(Dataset):
    def __init__(self, data, labels, sequence_length=300, feature_dim=1):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.sequence_length].reshape(self.sequence_length, self.feature_dim), dtype=torch.float32),
            torch.tensor(self.labels[idx:idx+self.sequence_length], dtype=torch.float32)  # 전체 시퀀스 예측
        )

# LSTM 모델 정의 (Many-to-Many)
class LSTMManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMManyToManyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 모든 시점에서 1개 출력
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out)  # (batch, seq_len, 1)
        return output.squeeze(-1)  # (batch, seq_len)


# 데이터 로드
pred_data = np.load('/code/rppg/real data/PURE_POS_prediction.npy')
label_data = np.load('/code/rppg/real data/PURE_label.npy')

# 입력 차원 설정
feature_dim = 1 if len(pred_data.shape) == 1 else pred_data.shape[1]

# 데이터 8:2 분할
split_idx = int(len(pred_data) * 0.8)
train_pred, test_pred = pred_data[:split_idx], pred_data[split_idx:]
train_label, test_label = label_data[:split_idx], label_data[split_idx:]

# 데이터셋 및 데이터로더 생성
sequence_length = 300
batch_size = 32

train_dataset = RPPGDataset(train_pred, train_label, sequence_length, feature_dim)
test_dataset = RPPGDataset(test_pred, test_label, sequence_length, feature_dim)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMManyToManyModel(input_size=feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 과정 저장용 리스트
train_losses = []
val_losses = []

# 모델 학습
num_epochs = 130
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    
    # 검증 데이터 손실 계산
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_losses.append(val_loss / len(test_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

# 학습 과정 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# 학습된 모델 저장
torch.save(model.state_dict(), "/code/rppg/newdata/lstm_many_to_many_model.pth")

# 예측 수행 및 저장
model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, len(pred_data) - sequence_length + 1, sequence_length):
        input_seq = torch.tensor(
            pred_data[i:i+sequence_length].reshape(1, sequence_length, feature_dim),
            dtype=torch.float32
        ).to(device)
        
        output_seq = model(input_seq).cpu().numpy().squeeze()  # (seq_len,)
        
        predictions.extend(output_seq)

predictions = np.array(predictions)

# PSNR 계산 함수
def psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(y_true)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Pearson Correlation 계산
pearson_corr, _ = pearsonr(label_data.flatten(), predictions.flatten())

# 평가 지표 출력
print("PSNR:", psnr(label_data, predictions))
print("Pearson Correlation:", pearson_corr)

print("Predictions shape:", predictions.shape)
print("Labels shape:", label_data.shape)
np.save("/code/rppg/newdata/LSTM_ManyToMany_rPPG_prediction.npy", predictions)
'''
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 데이터셋 클래스 정의
class RPPGDataset(Dataset):
    def __init__(self, data, labels, sequence_length=300, feature_dim=1, stride=5):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.stride = stride

        # 슬라이딩 윈도우 적용 (stride만큼 간격을 두고 시작 인덱스 계산)
        self.indices = [
            i for i in range(0, len(data) - sequence_length, stride)
        ]
    
    def __len__(self):
        # 슬라이딩 윈도우가 적용된 데이터의 개수
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 슬라이딩 윈도우에 해당하는 인덱스를 가져와서 데이터와 라벨을 반환
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        return (
            torch.tensor(self.data[start_idx:end_idx].reshape(self.sequence_length, self.feature_dim), dtype=torch.float32),
            torch.tensor(self.labels[start_idx:end_idx], dtype=torch.float32)  # 전체 시퀀스 예측
        )

# LSTM 모델 정의 (Many-to-Many)
class LSTMManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMManyToManyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 모든 시점에서 1개 출력
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out)  # (batch, seq_len, 1)
        return output.squeeze(-1)  # (batch, seq_len)

# 데이터 로드
pred_data = np.load('/code/rppg/real data/PURE_POS_prediction.npy')
label_data = np.load('/code/rppg/real data/PURE_label.npy')

# 입력 차원 설정
feature_dim = 1 if len(pred_data.shape) == 1 else pred_data.shape[1]

# 데이터 8:2 분할
split_idx = int(len(pred_data) * 0.8)
train_pred, test_pred = pred_data[:split_idx], pred_data[split_idx:]
train_label, test_label = label_data[:split_idx], label_data[split_idx:]

# 데이터셋 및 데이터로더 생성
sequence_length = 300
stride = 5  # 슬라이딩 윈도우 간격 설정
batch_size = 32

train_dataset = RPPGDataset(train_pred, train_label, sequence_length, feature_dim, stride)
test_dataset = RPPGDataset(test_pred, test_label, sequence_length, feature_dim, stride)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMManyToManyModel(input_size=feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.6f}")

# 학습된 모델 저장
torch.save(model.state_dict(), "/code/rppg/newdata/lstm_many_to_many_model.pth")

# 예측 수행 및 저장
model.eval()
predictions = []


with torch.no_grad():
    for i in range(0, len(pred_data) - sequence_length + 1, sequence_length):
        input_seq = torch.tensor(
            pred_data[i:i+sequence_length].reshape(1, sequence_length, feature_dim),
            dtype=torch.float32
        ).to(device)
        
        output_seq = model(input_seq).cpu().numpy().squeeze()  # (seq_len,)
        
        predictions.extend(output_seq)

predictions = np.array(predictions)

print("Predictions shape:", predictions.shape)
print("Labels shape:", label_data.shape)
np.save("/code/rppg/newdata/LSTM_ManyToMany_rPPG_prediction.npy", predictions)
"""

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 데이터셋 클래스 정의
class RPPGDataset(Dataset):
    def __init__(self, data, labels, sequence_length=300, feature_dim=1):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.sequence_length].reshape(self.sequence_length, self.feature_dim), dtype=torch.float32),
            torch.tensor(self.labels[idx:idx+self.sequence_length], dtype=torch.float32)
        )

# BiLSTM 모델 정의 (Many-to-Many)
class BiLSTMManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(BiLSTMManyToManyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM 정의
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 양방향이므로 hidden_size * 2

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 초기 hidden state와 cell state 생성
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # (num_layers * 2, batch, hidden_size)
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # (num_layers * 2, batch, hidden_size)

        # LSTM Forward
        lstm_out, (h_n, c_n) = self.bilstm(x, (h_0, c_0))  # (batch, seq_len, hidden_size * 2)
        
        output = self.fc(lstm_out)  # (batch, seq_len, 1)
        return output.squeeze(-1)  # (batch, seq_len)

# 데이터 로드
pred_data = np.load('/code/rppg/real data/PURE_POS_prediction.npy')
label_data = np.load('/code/rppg/real data/PURE_label.npy')

# 입력 차원 설정
feature_dim = 1 if len(pred_data.shape) == 1 else pred_data.shape[1]

# 데이터 8:2 분할
split_idx = int(len(pred_data) * 0.8)
train_pred, test_pred = pred_data[:split_idx], pred_data[split_idx:]
train_label, test_label = label_data[:split_idx], label_data[split_idx:]

# 데이터셋 및 데이터로더 생성
sequence_length = 300
batch_size = 32

train_dataset = RPPGDataset(train_pred, train_label, sequence_length, feature_dim)
test_dataset = RPPGDataset(test_pred, test_label, sequence_length, feature_dim)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMManyToManyModel(input_size=feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 과정 저장용 리스트
train_losses = []
val_losses = []

# 모델 학습
num_epochs = 130
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    
    # 검증 데이터 손실 계산
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_losses.append(val_loss / len(test_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

# 학습 과정 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# 학습된 모델 저장
torch.save(model.state_dict(), "/code/rppg/newdata/bilstm_many_to_many_model.pth")

# 예측 수행 및 저장
model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, len(pred_data) - sequence_length + 1, sequence_length):
        input_seq = torch.tensor(
            pred_data[i:i+sequence_length].reshape(1, sequence_length, feature_dim),
            dtype=torch.float32
        ).to(device)
        
        output_seq = model(input_seq).cpu().numpy().squeeze()  # (seq_len,)
        
        predictions.extend(output_seq)

predictions = np.array(predictions)

# PSNR 계산 함수
def psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(y_true)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Pearson Correlation 계산
pearson_corr, _ = pearsonr(label_data.flatten(), predictions.flatten())

# 평가 지표 출력
print("PSNR:", psnr(label_data, predictions))
print("Pearson Correlation:", pearson_corr)

print("Predictions shape:", predictions.shape)
print("Labels shape:", label_data.shape)
np.save("/code/rppg/newdata/BiLSTM_ManyToMany_rPPG_prediction.npy", predictions)

"""
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 데이터셋 클래스 정의
class RPPGDataset(Dataset):
    def __init__(self, data, labels, sequence_length=300, feature_dim=1):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.sequence_length].reshape(self.sequence_length, self.feature_dim), dtype=torch.float32),
            torch.tensor(self.labels[idx:idx+self.sequence_length], dtype=torch.float32)  # 전체 시퀀스 예측
        )

# LSTM 모델 정의 (Many-to-Many)
class LSTMManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMManyToManyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 모든 시점에서 1개 출력
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out)  # (batch, seq_len, 1)
        return output.squeeze(-1)  # (batch, seq_len)


# 데이터 로드
pred_data = np.load('/code/rppg2ppg/real data/PURE_POS_prediction.npy')
label_data = np.load('/code/rppg2ppg/real data/PURE_label.npy')

# 입력 차원 설정
feature_dim = 1 if len(pred_data.shape) == 1 else pred_data.shape[1]

# 데이터 8:2 분할
split_idx = int(len(pred_data) * 0.8)
train_pred, test_pred = pred_data[:split_idx], pred_data[split_idx:]
train_label, test_label = label_data[:split_idx], label_data[split_idx:]

# 데이터셋 및 데이터로더 생성
sequence_length = 300
batch_size = 32

train_dataset = RPPGDataset(train_pred, train_label, sequence_length, feature_dim)
test_dataset = RPPGDataset(test_pred, test_label, sequence_length, feature_dim)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMManyToManyModel(input_size=feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----- 추가: mixup 하이퍼파라미터 -----
mixup_alpha = 0.2  # 0.0이면 mixup 비활성화

# 학습 과정 저장용 리스트
train_losses = []
val_losses = []

# 모델 학습
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        # inputs: (batch, seq_len, feat), targets: (batch, seq_len)
        inputs, targets = inputs.to(device), targets.to(device)

        # ----- mixup 적용 (배치 수준) -----
        if mixup_alpha > 0 and inputs.size(0) > 1:
            # 배치 내에서 무작위로 섞기
            indices = torch.randperm(inputs.size(0)).to(device)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]
            # lambda 샘플링 (Beta 분포)
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam_t = torch.tensor(lam, dtype=inputs.dtype).to(device)
            # 선형 혼합
            inputs = lam_t * inputs + (1.0 - lam_t) * inputs_shuffled
            targets = lam_t * targets + (1.0 - lam_t) * targets_shuffled
        # batch_size==1 이거나 mixup_alpha==0이면 mixup 생략

        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    
    # 검증 데이터 손실 계산
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_losses.append(val_loss / len(test_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

# 학습 과정 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# 학습된 모델 저장
torch.save(model.state_dict(), "/code/rppg2ppg/newdata/LSTM_mixup.pth")

# 예측 수행 및 저장
model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, len(pred_data) - sequence_length + 1, sequence_length):
        input_seq = torch.tensor(
            pred_data[i:i+sequence_length].reshape(1, sequence_length, feature_dim),
            dtype=torch.float32
        ).to(device)
        
        output_seq = model(input_seq).cpu().numpy().squeeze()  # (seq_len,)
        
        predictions.extend(output_seq)

predictions = np.array(predictions)

# PSNR 계산 함수
def psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(y_true)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Pearson Correlation 계산
pearson_corr, _ = pearsonr(label_data.flatten(), predictions.flatten())

# 평가 지표 출력
print("PSNR:", psnr(label_data, predictions))
print("Pearson Correlation:", pearson_corr)

print("Predictions shape:", predictions.shape)
print("Labels shape:", label_data.shape)
np.save("/code/rppg2ppg/newdata/LSTM_mixup.npy", predictions)
'''
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import copy

# -------------------------
# CutMix for Time-Series
# -------------------------
def cutmix_data(x, y, alpha=1.0):
    """
    x: (batch, seq_len, feature_dim)
    y: (batch, seq_len)
    Returns: mixed_x, mixed_y, lam
    """
    # sample lambda from Beta
    lam = np.random.beta(alpha, alpha)

    batch_size, seq_len, _ = x.size()
    index = torch.randperm(batch_size).to(x.device)

    # decide cut length (number of timesteps to replace)
    cut_len = int(seq_len * (1 - lam))

    # clamp cut_len to valid range [0, seq_len]
    cut_len = max(0, min(cut_len, seq_len))

    # if nothing to cut, return originals (lam ~= 1)
    if cut_len == 0:
        return x, y, 1.0

    # compute max start position (inclusive)
    max_start = seq_len - cut_len
    # choose start in [0, max_start] (inclusive)
    start = np.random.randint(0, max_start + 1)

    # create mixed copies
    mixed_x = x.clone()
    mixed_y = y.clone()

    mixed_x[:, start:start+cut_len, :] = x[index, start:start+cut_len, :]
    mixed_y[:, start:start+cut_len] = y[index, start:start+cut_len]

    # actual lam is fraction kept from original
    lam = 1.0 - (cut_len / seq_len)

    return mixed_x, mixed_y, lam



# -------------------------
# EarlyStopping
# -------------------------
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# -------------------------
# Dataset
# -------------------------
class RPPGDataset(Dataset):
    def __init__(self, data, labels, sequence_length=300, feature_dim=1):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        seq = torch.tensor(
            self.data[idx:idx+self.sequence_length].reshape(self.sequence_length, self.feature_dim),
            dtype=torch.float32
        )
        label = torch.tensor(
            self.labels[idx:idx+self.sequence_length],
            dtype=torch.float32
        )
        return seq, label


# -------------------------
# Model
# -------------------------
class LSTMManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)


# -------------------------
# Data Load
# -------------------------
pred_data = np.load('/code/rppg2ppg/real data/PURE_POS_prediction.npy')
label_data = np.load('/code/rppg2ppg/real data/PURE_label.npy')

feature_dim = 1 if len(pred_data.shape) == 1 else pred_data.shape[1]

split_idx = int(len(pred_data) * 0.8)
train_pred, test_pred = pred_data[:split_idx], pred_data[split_idx:]
train_label, test_label = label_data[:split_idx], label_data[split_idx:]

sequence_length = 300
batch_size = 32

train_dataset = RPPGDataset(train_pred, train_label, sequence_length, feature_dim)
test_dataset = RPPGDataset(test_pred, test_label, sequence_length, feature_dim)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMManyToManyModel(input_size=feature_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
early_stopper = EarlyStopping(patience=12, min_delta=1e-5)


# -------------------------
# Training Loop
# -------------------------
num_epochs = 130
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # ----- ★ CutMix 적용 -----
        mixed_inputs, mixed_targets, lam = cutmix_data(inputs, targets, alpha=0.2)

        optimizer.zero_grad()
        outputs = model(mixed_inputs)

        loss = criterion(outputs, mixed_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # ----- Validation -----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]  Train: {train_loss:.6f}  Val: {val_loss:.6f}")

    # ----- Early Stopping -----
    if early_stopper.step(val_loss, model):
        print(f"Early Stopping triggered at epoch {epoch+1}!")
        break

# 베스트 모델 복구
model.load_state_dict(early_stopper.best_state)


# -------------------------
# Plot Loss
# -------------------------
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# -------------------------
# Prediction
# -------------------------
model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, len(pred_data) - sequence_length + 1, sequence_length):
        input_seq = torch.tensor(
            pred_data[i:i+sequence_length].reshape(1, sequence_length, feature_dim),
            dtype=torch.float32
        ).to(device)

        pred_seq = model(input_seq).cpu().numpy().squeeze()
        predictions.extend(pred_seq)

predictions = np.array(predictions)

# -------------------------
# Metrics
# -------------------------
def psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = np.max(y_true)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

pearson_corr, _ = pearsonr(label_data.flatten(), predictions.flatten())

print("PSNR:", psnr(label_data, predictions))
print("Pearson Corr:", pearson_corr)

# Save
np.save("/code/rppg2ppg/newdata/LSTM_cutmix.npy", predictions)
torch.save(model.state_dict(), "/code/rppg2ppg/newdata/LSTM_cutmix.pth")
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import copy

# -------------------------
# EarlyStopping
# -------------------------
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# -------------------------
# Dataset
# -------------------------
class RPPGDataset(Dataset):
    def __init__(self, data, labels, sequence_length=300, feature_dim=1):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        seq = torch.tensor(
            self.data[idx:idx+self.sequence_length].reshape(self.sequence_length, self.feature_dim),
            dtype=torch.float32
        )
        label = torch.tensor(
            self.labels[idx:idx+self.sequence_length],
            dtype=torch.float32
        )
        return seq, label


# -------------------------
# Main Model (LSTM many-to-many)
# -------------------------
class LSTMManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)


# -------------------------
# --- NEW: latent-space augmenter (encoder -> augment -> decoder)
# -------------------------
class EncoderAug(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        # x: (B, T, feat)
        h, _ = self.lstm(x)
        # use last time-step hidden to form latent
        z = self.fc(h[:, -1, :])  # (B, latent_dim)
        return z

class PhysiologyLatentAug(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # learnable direction vectors (initialized small)
        self.b_HR = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.b_Amp = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.b_Notch = nn.Parameter(torch.randn(latent_dim) * 0.01)

    def forward(self, z, device, alpha_std=0.05, beta_std=0.10, gamma_std=0.02, clamp=True):
        """
        z: (B, latent_dim)
        returns z_aug: (B, latent_dim)
        """
        if z is None:
            return z
        B = z.size(0)
        # sample small scalars per sample
        alpha = torch.randn(B, 1, device=device) * alpha_std   # HR-like
        beta  = torch.randn(B, 1, device=device) * beta_std    # amplitude-like
        gamma = torch.randn(B, 1, device=device) * gamma_std   # notch-like

        # optionally clamp to keep perturbations moderate
        if clamp:
            alpha = torch.clamp(alpha, -0.2, 0.2)
            beta  = torch.clamp(beta, -0.3, 0.3)
            gamma = torch.clamp(gamma, -0.1, 0.1)

        b_HR = self.b_HR.unsqueeze(0).to(device)     # (1, latent_dim)
        b_Amp = self.b_Amp.unsqueeze(0).to(device)
        b_Notch = self.b_Notch.unsqueeze(0).to(device)

        z_aug = z + alpha * b_HR + beta * b_Amp + gamma * b_Notch
        return z_aug

class DecoderAug(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, output_dim=1, seq_len=300, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    def forward(self, z):
        # z: (B, latent_dim)
        h0 = self.fc_in(z).unsqueeze(1)  # (B,1,H)
        h0 = h0.repeat(1, self.seq_len, 1)  # (B, T, H)
        h, _ = self.lstm(h0)
        out = self.fc_out(h)  # (B, T, output_dim)
        return out  # do not squeeze here; keep shape (B,T,1)


class LatentAugmenter(nn.Module):
    def __init__(self, seq_len=300, latent_dim=32, device='cpu'):
        super().__init__()
        self.encoder = EncoderAug(input_dim=1, hidden_dim=64, latent_dim=latent_dim).to(device)
        self.augment = PhysiologyLatentAug(latent_dim=latent_dim).to(device)
        self.decoder = DecoderAug(latent_dim=latent_dim, hidden_dim=64, output_dim=1, seq_len=seq_len).to(device)
        self.device = device
    def forward(self, x):
        # x: (B, T, feat)
        z = self.encoder(x)                      # (B, latent)
        z_aug = self.augment(z, device=self.device)   # (B, latent)
        out = self.decoder(z_aug)                # (B, T, 1)
        # return augmented inputs in same shape as original x
        return out.squeeze(-1)                   # (B, T)


# -------------------------
# Data Load
# -------------------------
pred_data = np.load('/code/rppg2ppg/real data/PURE_POS_prediction.npy')
label_data = np.load('/code/rppg2ppg/real data/PURE_label.npy')

feature_dim = 1 if len(pred_data.shape) == 1 else pred_data.shape[1]

split_idx = int(len(pred_data) * 0.8)
train_pred, test_pred = pred_data[:split_idx], pred_data[split_idx:]
train_label, test_label = label_data[:split_idx], label_data[split_idx:]

sequence_length = 300
batch_size = 32

train_dataset = RPPGDataset(train_pred, train_label, sequence_length, feature_dim)
test_dataset = RPPGDataset(test_pred, test_label, sequence_length, feature_dim)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMManyToManyModel(input_size=feature_dim).to(device)

# instantiate augmenter and include its params in optimizer so the latent direction vectors can be learned
augmenter = LatentAugmenter(seq_len=sequence_length, latent_dim=32, device=device)
augmenter.to(device)

# combine parameters of main model + augmenter (augmenter will be trained jointly)
optimizer = optim.Adam(list(model.parameters()) + list(augmenter.parameters()), lr=0.001)

criterion = nn.MSELoss()

train_losses, val_losses = [], []
early_stopper = EarlyStopping(patience=12, min_delta=1e-5)


# -------------------------
# Training Loop (CutMix replaced by latent augment)
# -------------------------
num_epochs = 130
print(" Training start (CutMix -> latent physiology augmentation)")
for epoch in range(num_epochs):
    model.train()
    augmenter.train()
    total_loss = 0

    for inputs, targets in train_loader:
        # inputs: (B, T, feat); targets: (B, T)
        inputs, targets = inputs.to(device), targets.to(device)

        # ----- ★ Latent-space augmentation applied HERE -----
        # augmented_inputs has shape (B, T)
        with torch.no_grad():
            # We will NOT freeze augmenter entirely; we want its direction vectors to be trainable,
            # but decoder/encoder parts could be noisy; here we do forward without grad for encoder/decoder to
            # avoid overly large gradients initially. If you prefer full joint-train, remove no_grad context.
            pass

        # Option A: allow gradients for augmenter (joint training)
        # augmented_inputs = augmenter(inputs)  # (B, T)
        # Option B: keep encoder/decoder frozen while allowing b_* to be trained is more involved.
        # For simplicity we'll allow full augmenter gradients (joint training).
        augmented_inputs = augmenter(inputs)  # (B, T)

        # reshape to (B, T, feat) to feed main model (feature_dim=1)
        aug_inputs_3d = augmented_inputs.unsqueeze(-1)

        optimizer.zero_grad()
        outputs = model(aug_inputs_3d)  # (B, T)

        # NOTE: targets are kept unchanged (we simulate input-side variations). If you prefer label-warping
        # for certain augment types (time-warp), modify targets accordingly.
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # ----- Validation (no augmentation) -----
    model.eval()
    augmenter.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)   # original clean inputs
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]  Train: {train_loss:.6f}  Val: {val_loss:.6f}")

    # ----- Early Stopping -----
    if early_stopper.step(val_loss, model):
        print(f"Early Stopping triggered at epoch {epoch+1}!")
        break

# 베스트 모델 복구
model.load_state_dict(early_stopper.best_state)


# -------------------------
# Plot Loss
# -------------------------
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Training & Validation Loss (latent-aug)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# -------------------------
# Prediction (same as before)
# -------------------------
model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, len(pred_data) - sequence_length + 1, sequence_length):
        input_seq = torch.tensor(
            pred_data[i:i+sequence_length].reshape(1, sequence_length, feature_dim),
            dtype=torch.float32
        ).to(device)

        pred_seq = model(input_seq).cpu().numpy().squeeze()
        predictions.extend(pred_seq)

predictions = np.array(predictions)

# -------------------------
# Metrics
# -------------------------
def psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(y_true)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

pearson_corr, _ = pearsonr(label_data.flatten(), predictions.flatten())

print("PSNR:", psnr(label_data, predictions))
print("Pearson Corr:", pearson_corr)

# Save
np.save("/code/rppg2ppg/newdata/LSTM_latent.npy", predictions)
torch.save(model.state_dict(), "/code/rppg2ppg/newdata/LSTM_latent.pth")

print(" Done. CutMix was replaced by latent-space physiology augmentation.")
