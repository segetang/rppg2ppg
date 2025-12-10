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
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  

        
        lstm_out, (h_n, c_n) = self.bilstm(x, (h_0, c_0))  
        
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
num_epochs = 90
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
'''
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
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  

        
        lstm_out, (h_n, c_n) = self.bilstm(x, (h_0, c_0))  
        
        output = self.fc(lstm_out)  # (batch, seq_len, 1)
        return output.squeeze(-1)  # (batch, seq_len)

# MixUp 함수 (배치 단위)
def apply_mixup(inputs, targets, alpha=0.8):
    """
    inputs: (batch, seq_len, feat)
    targets: (batch, seq_len)
    returns mixed_inputs, mixed_targets
    """
    if inputs.size(0) <= 1 or alpha <= 0.0:
        return inputs, targets

    indices = torch.randperm(inputs.size(0)).to(inputs.device)
    inputs_shuffled = inputs[indices]
    targets_shuffled = targets[indices]

    lam = np.random.beta(alpha, alpha)
    lam_t = torch.tensor(lam, dtype=inputs.dtype).to(inputs.device)

    mixed_inputs = lam_t * inputs + (1.0 - lam_t) * inputs_shuffled
    mixed_targets = lam_t * targets + (1.0 - lam_t) * targets_shuffled

    return mixed_inputs, mixed_targets

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
model = BiLSTMManyToManyModel(input_size=feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# MixUp 하이퍼파라미터
mixup_alpha = 0.2  # 0.0이면 MixUp 비활성화

# 학습 과정 저장용 리스트
train_losses = []
val_losses = []

# 모델 학습
num_epochs = 90
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # --- MixUp 적용 ---
        inputs_mixed, targets_mixed = apply_mixup(inputs, targets, alpha=mixup_alpha)

        optimizer.zero_grad()
        outputs = model(inputs_mixed)  # (batch, seq_len)
        loss = criterion(outputs, targets_mixed)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    
    # 검증 데이터 손실 계산 (MixUp 적용하지 않음)
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
torch.save(model.state_dict(), "/code/rppg2ppg/newdata/bilstm_mixup.pth")

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
np.save("/code/rppg2ppg/newdata/bilstm_mixup.npy", predictions)

'''
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
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  

        lstm_out, (h_n, c_n) = self.bilstm(x, (h_0, c_0))  
        
        output = self.fc(lstm_out)  # (batch, seq_len, 1)
        return output.squeeze(-1)  # (batch, seq_len)

# -------------------------
# CutMix (시계열용 안정화된 구현)
# -------------------------
def cutmix_batch(inputs, targets, alpha=0.2):
    """
    inputs: (batch, seq_len, feat)
    targets: (batch, seq_len)
    returns: mixed_inputs, mixed_targets

    안전 처리:
      - cut_len이 0이면 CutMix를 건너뜀
      - start 인덱스 선택 범위를 안전하게 처리
    """
    B = inputs.size(0)
    T = inputs.size(1)
    if B <= 1 or T <= 1 or alpha <= 0.0:
        return inputs, targets

    perm = torch.randperm(B).to(inputs.device)
    lam = np.random.beta(alpha, alpha)
    # 대체할 길이 계산 (정수)
    cut_len = int(round(T * (1.0 - lam)))
    # clamp
    cut_len = max(0, min(cut_len, T))
    if cut_len == 0:
        return inputs, targets

    max_start = T - cut_len
    # start 범위를 안전하게 설정 (inclusive)
    start = np.random.randint(0, max_start + 1) if max_start >= 0 else 0

    mixed_inputs = inputs.clone()
    mixed_targets = targets.clone()

    mixed_inputs[:, start:start+cut_len, :] = inputs[perm, start:start+cut_len, :]
    mixed_targets[:, start:start+cut_len] = targets[perm, start:start+cut_len]

    return mixed_inputs, mixed_targets

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
model = BiLSTMManyToManyModel(input_size=feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# CutMix 하이퍼파라미터 (권장: 0.1 ~ 0.3)
cutmix_alpha = 1.0

# 학습 과정 저장용 리스트
train_losses = []
val_losses = []

# 모델 학습
num_epochs = 90
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # --- CutMix 적용 (훈련에만) ---
        inputs_mixed, targets_mixed = cutmix_batch(inputs, targets, alpha=cutmix_alpha)

        optimizer.zero_grad()
        outputs = model(inputs_mixed)  # (batch, seq_len)
        loss = criterion(outputs, targets_mixed)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    
    # 검증 데이터 손실 계산 (CutMix 적용하지 않음)
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
torch.save(model.state_dict(), "/code/rppg2ppg/newdata/bilstm_cutmix.pth")

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
# 주의: predictions 길이와 label_data 길이가 다를 수 있음 (원래 코드 동작 유지)
pearson_corr, _ = pearsonr(label_data.flatten(), predictions.flatten())

# 평가 지표 출력
print("PSNR:", psnr(label_data, predictions))
print("Pearson Correlation:", pearson_corr)

print("Predictions shape:", predictions.shape)
print("Labels shape:", label_data.shape)
np.save("/code/rppg2ppg/newdata/bilstm_cutmix.npy", predictions)
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -------------------------
# Dataset 클래스
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
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  

        lstm_out, (h_n, c_n) = self.bilstm(x, (h_0, c_0))  
        
        output = self.fc(lstm_out)  # (batch, seq_len, 1)
        return output.squeeze(-1)  # (batch, seq_len)


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
        return out  # (B, T, 1)


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
        return out.squeeze(-1)                   # (B, T)


# -------------------------
# 데이터 로드
# -------------------------
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

# -------------------------
# 모델/augmenter/손실/옵티마이저 설정
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMManyToManyModel(input_size=feature_dim).to(device)

augmenter = LatentAugmenter(seq_len=sequence_length, latent_dim=32, device=device)
augmenter.to(device)

# combine parameters so augmenter learns jointly
optimizer = optim.Adam(list(model.parameters()) + list(augmenter.parameters()), lr=0.001)
criterion = nn.MSELoss()

# CutMix 하이퍼변수는 더이상 사용되지 않지만 원본과의 비교를 위해 남겨둠
cutmix_alpha = 1.0

# 학습 기록
train_losses = []
val_losses = []

# -------------------------
# 학습 루프 (CutMix 대신 latent augmentation 사용)
# -------------------------
num_epochs = 90
for epoch in range(num_epochs):
    model.train()
    augmenter.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # ----- latent-space augmentation 적용 (훈련 중만) -----
        # augmented_inputs: (B, T)
        augmented_inputs = augmenter(inputs)  # gradients flow to augmenter & model via optimizer
        
        # reshape to (B, T, 1) for BiLSTM input
        aug_inputs_3d = augmented_inputs.unsqueeze(-1)

        optimizer.zero_grad()
        outputs = model(aug_inputs_3d)  # (batch, seq_len)
        loss = criterion(outputs, targets)  # targets kept unchanged (input-side variation)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    
    # 검증 (augmentation 적용하지 않음)
    model.eval()
    augmenter.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_losses.append(val_loss / len(test_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

# 학습 과정 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss (latent-aug)')
plt.show()

# 학습된 모델 저장
torch.save(model.state_dict(), "/code/rppg2ppg/newdata/bilstm_latent.pth")

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
np.save("/code/rppg2ppg/newdata/bilstm_latent.npy", predictions)

