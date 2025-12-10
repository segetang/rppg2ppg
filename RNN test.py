''' orginal code
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

class RNNManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RNNManyToManyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN 정의
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        batch_size = x.size(0)
        # 초기 hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # RNN 통과
        out, _ = self.rnn(x, h0)                # out: (batch, seq_len, hidden_size)
        out = self.fc(out)                      # out: (batch, seq_len, 1)
        return out.squeeze(-1)                  # -> (batch, seq_len)

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
model = RNNManyToManyModel(input_size=feature_dim).to(device)   # BiLSTM이 아닌 RNN 모델로 변경
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 학습 과정 저장용 리스트
train_losses = []
val_losses = []

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

class RNNManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RNNManyToManyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN 정의
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        batch_size = x.size(0)
        # 초기 hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # RNN 통과
        out, _ = self.rnn(x, h0)                # out: (batch, seq_len, hidden_size)
        out = self.fc(out)                      # out: (batch, seq_len, 1)
        return out.squeeze(-1)                  # -> (batch, seq_len)

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
model = RNNManyToManyModel(input_size=feature_dim).to(device)   # BiLSTM이 아닌 RNN 모델로 변경
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 추가: mixup 하이퍼파라미터 ---
mixup_alpha = 0.2  # 0.0 이면 mixup 사용 안 함

# 학습 과정 저장용 리스트
train_losses = []
val_losses = []

# --- 추가: 얼리스탑 설정 ---
num_epochs = 100
early_stopping_patience = 0
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = "/code/rppg2ppg/newdata/rnn_mixup.pth"  # 개선 시 이 경로로 저장

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # inputs: (batch, seq_len, feat), targets: (batch, seq_len)

        # --- 추가: mixup 적용 (배치 수준) ---
        if mixup_alpha > 0 and inputs.size(0) > 1:
            # shuffle within the batch
            indices = torch.randperm(inputs.size(0)).to(device)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]
            # sample lambda
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam_tensor = torch.tensor(lam, dtype=inputs.dtype).to(device)
            # mix inputs and targets
            inputs = lam_tensor * inputs + (1.0 - lam_tensor) * inputs_shuffled
            targets = lam_tensor * targets + (1.0 - lam_tensor) * targets_shuffled
        # (배치 크기 1 이거나 mixup_alpha == 0 이면 mixup 생략)

        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_loss_avg = total_loss / len(train_loader)
    train_losses.append(train_loss_avg)
    
    # 검증 데이터 손실 계산
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss_avg = val_loss / len(test_loader)
    val_losses.append(val_loss_avg)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

    # --- 추가: 얼리스탑 로직 ---
    if val_loss_avg < best_val_loss - 1e-8:
        best_val_loss = val_loss_avg
        epochs_no_improve = 0
        # 개선된 모델 저장
        torch.save(model.state_dict(), best_model_path)
        print(f"  Validation loss improved. Saved best model to {best_model_path}")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break

# 학습 과정 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# 학습된 모델 저장 (최종적으로도 best 모델을 같은 경로에 저장해둠)
# 이미 best 모델이 저장되어 있을 수 있음 — 여기서는 안전 차원에서 현재 state도 저장
torch.save(model.state_dict(), "/code/rppg2ppg/newdata/rnn_mixup.pth")

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
# (predictions 길이가 label_data 길이와 다를 수 있음 — 원래 코드와 동일하게 계산)
pearson_corr, _ = pearsonr(label_data.flatten(), predictions.flatten())

# 평가 지표 출력
print("PSNR:", psnr(label_data, predictions))
print("Pearson Correlation:", pearson_corr)

print("Predictions shape:", predictions.shape)
print("Labels shape:", label_data.shape)
np.save("/code/rppg2ppg/newdata/rnn_mixup.npy", predictions)
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

class RNNManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RNNManyToManyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN 정의
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        batch_size = x.size(0)
        # 초기 hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # RNN 통과
        out, _ = self.rnn(x, h0)                # out: (batch, seq_len, hidden_size)
        out = self.fc(out)                      # out: (batch, seq_len, 1)
        return out.squeeze(-1)                  # -> (batch, seq_len)

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
model = RNNManyToManyModel(input_size=feature_dim).to(device)   # BiLSTM이 아닌 RNN 모델로 변경
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 추가: CutMix 하이퍼파라미터 (1D 시계열용) ---
cutmix_alpha = 0.2  # 0.0 이면 CutMix 사용 안 함

# 학습 과정 저장용 리스트
train_losses = []
val_losses = []

# --- 추가: 얼리스탑 설정 ---
num_epochs = 100
early_stopping_patience = 0
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = "/code/rppg2ppg/newdata/rnn_cutmix.pth"  # 개선 시 이 경로로 저장

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        # inputs: (batch, seq_len, feat), targets: (batch, seq_len)
        inputs, targets = inputs.to(device), targets.to(device)

        # --- 추가: CutMix 적용 (시계열용, 연속된 time segment 교체) ---
        if cutmix_alpha > 0 and inputs.size(0) > 1:
            # shuffle within the batch
            indices = torch.randperm(inputs.size(0)).to(device)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]

            # sample lambda
            lam = np.random.beta(cutmix_alpha, cutmix_alpha)
            seq_len = inputs.size(1)
            # patch length: 비율 (1 - lam)을 잘라서 교체 (정수로)
            cut_len = int(seq_len * (1.0 - lam))
            # ensure at least 1 if lam very close to 1
            if cut_len > 0:
                # random start index for the cut segment
                start = np.random.randint(0, seq_len - cut_len + 1)
                end = start + cut_len
                # create mask: 1 for keep original, 0 for replaced segment
                mask = torch.ones((inputs.size(0), seq_len), dtype=inputs.dtype).to(device)
                mask[:, start:end] = 0.0  # replaced region
                # apply mask to inputs and targets (inputs has feature dim)
                mask_inputs = mask.unsqueeze(-1)  # (batch, seq_len, 1)
                inputs = inputs * mask_inputs + inputs_shuffled * (1.0 - mask_inputs)
                targets = targets * mask + targets_shuffled * (1.0 - mask)
                # actual lambda (proportion of original kept)
                lam = 1.0 - (cut_len / seq_len)
            # if cut_len == 0, we skip CutMix for this batch (effectively lam ~= 1)
        # else: batch size 1 or cutmix_alpha == 0 -> no CutMix

        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_loss_avg = total_loss / len(train_loader)
    train_losses.append(train_loss_avg)
    
    # 검증 데이터 손실 계산
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss_avg = val_loss / len(test_loader)
    val_losses.append(val_loss_avg)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

    # --- 추가: 얼리스탑 로직 ---
    if val_loss_avg < best_val_loss - 1e-8:
        best_val_loss = val_loss_avg
        epochs_no_improve = 0
        # 개선된 모델 저장
        torch.save(model.state_dict(), best_model_path)
        print(f"  Validation loss improved. Saved best model to {best_model_path}")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break

# 학습 과정 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# 학습된 모델 저장 (최종적으로도 best 모델을 같은 경로에 저장해둠)
# 이미 best 모델이 저장되어 있을 수 있음 — 여기서는 안전 차원에서 현재 state도 저장
torch.save(model.state_dict(), "/code/rppg2ppg/newdata/rnn_cutmix.pth")

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
np.save("/code/rppg2ppg/newdata/rnn_cutmix.npy", predictions)
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math

# -----------------------------
# Dataset 클래스
# -----------------------------
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

# -----------------------------
# 간단 RNN many-to-many 모델
# -----------------------------
class RNNManyToManyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RNNManyToManyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN 정의
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        batch_size = x.size(0)
        # 초기 hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # RNN 통과
        out, _ = self.rnn(x, h0)                # out: (batch, seq_len, hidden_size)
        out = self.fc(out)                      # out: (batch, seq_len, 1)
        return out.squeeze(-1)                  # -> (batch, seq_len)

# -----------------------------
# 데이터 로드
# -----------------------------
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

# -----------------------------
# 모델, 손실 함수, 옵티마이저 설정
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNManyToManyModel(input_size=feature_dim).to(device)   # BiLSTM이 아닌 RNN 모델로 변경
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- (이전 CutMix 대체) physiology-inspired augmentation 하이퍼파라미터 ---
alpha_std = 0.05   # HR shift 표준편차: 상대적 비율 (예: 0.05 => ±5% 시간 스케일)
beta_std  = 0.10   # amplitude 스케일 표준편차
gamma_std = 0.02   # notch artifact 크기 (신호 std에 대한 비율)
notch_freq_range = (0.5, 2.5)  # 추가하는 간섭성 사인파 주파수 범위 (Hz)
sampling_rate = 30.0  # rPPG/PPG 신호의 샘플링 주파수(너의 데이터에 맞춰 조정해라). 중요!

# 학습 과정 저장용 리스트
train_losses = []
val_losses = []

# --- 얼리스탑 설정 ---
num_epochs = 100
early_stopping_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = "/code/rppg2ppg/newdata/rnn_latent.pth"  # 개선 시 이 경로로 저장

# -----------------------------
# augmentation 함수 정의
# - HR shift (time-warp)과 amplitude scaling은 inputs와 targets에 동일하게 적용
# - notch artifact는 inputs(관측치)에만 추가
# -----------------------------
def time_warp_resample(signal, scale):
    """
    signal: 1D numpy array (length = seq_len)
    scale: float (1.0 => no warp). >1 -> stretch, <1 -> compress
    Return: warped and resampled to original length (1D numpy array)
    """
    seq_len = len(signal)
    if scale <= 0:
        scale = 1.0
    # create scaled time grid, then resample back to original grid
    orig_t = np.linspace(0.0, 1.0, seq_len)
    scaled_len = max(2, int(np.round(seq_len * scale)))
    scaled_t = np.linspace(0.0, 1.0, scaled_len)
    # interpolate signal to scaled grid
    warped = np.interp(scaled_t, orig_t, signal)
    # resample warped back to original grid
    resampled = np.interp(orig_t, scaled_t, warped)
    return resampled

def augment_physiology_batch(inputs, targets, alpha_std=0.05, beta_std=0.1, gamma_std=0.02, notch_freq_range=(0.5,2.5), fs=30.0, device='cpu'):
    """
    inputs: tensor (B, seq_len, feat)  -- feat usually 1
    targets: tensor (B, seq_len)
    Returns augmented inputs, targets (both tensors on same device)
    """
    # move to cpu numpy for interpolation
    inputs_np = inputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    B, seq_len, feat = inputs_np.shape
    assert feat == 1 or feat == inputs_np.shape[2]  # feature dim assumed 1

    aug_inputs = np.empty_like(inputs_np)
    aug_targets = np.empty_like(targets_np)

    t = np.arange(seq_len) / fs  # time axis for notch sine

    for i in range(B):
        inp = inputs_np[i,:,0].astype(np.float64)
        tgt = targets_np[i,:].astype(np.float64)

        # sample augmentation factors
        alpha = np.random.normal(0.0, alpha_std)  # relative time-scale change
        beta  = np.random.normal(0.0, beta_std)   # amplitude scale (relative)
        gamma = np.random.normal(0.0, gamma_std)  # notch artifact strength (relative to signal std)

        scale = 1.0 + alpha
        # 1) time warp both input and target the same way (보존적 변형)
        inp_w = time_warp_resample(inp, scale)
        tgt_w = time_warp_resample(tgt, scale)

        # 2) amplitude scaling (same factor for input and target)
        amp_scale = 1.0 + beta
        inp_w = inp_w * amp_scale
        tgt_w = tgt_w * amp_scale

        # 3) notch-like interference: add a small sinusoid to input only
        # freq in Hz sampled from notch_freq_range
        f_notch = np.random.uniform(notch_freq_range[0], notch_freq_range[1])
        phase = np.random.uniform(0, 2*np.pi)
        # amplitude relative to signal std
        std_signal = np.std(inp_w) if np.std(inp_w) > 1e-8 else 1.0
        notch_amp = gamma * std_signal
        notch_wave = notch_amp * np.sin(2 * np.pi * f_notch * t + phase)
        inp_w = inp_w + notch_wave

        # store (reshape to keep dims)
        aug_inputs[i,:,0] = inp_w
        aug_targets[i,:] = tgt_w

    # convert back to torch tensors on target device
    aug_inputs_t = torch.tensor(aug_inputs, dtype=inputs.dtype).to(device)
    aug_targets_t = torch.tensor(aug_targets, dtype=targets.dtype).to(device)

    return aug_inputs_t, aug_targets_t

# -----------------------------
# 모델 학습 (CutMix 대체된 버전)
# -----------------------------
print(" Training start (with physiology-inspired augmentation replacing CutMix)")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        # inputs: (batch, seq_len, feat), targets: (batch, seq_len)
        inputs, targets = inputs.to(device), targets.to(device)

        # --- 대체된 augmentation 적용 ---
        # 각 배치마다 무작위로 augmentation 적용 (확률로 제어하려면 여기에 조건문 추가)
        inputs, targets = augment_physiology_batch(inputs, targets,
                                                   alpha_std=alpha_std,
                                                   beta_std=beta_std,
                                                   gamma_std=gamma_std,
                                                   notch_freq_range=notch_freq_range,
                                                   fs=sampling_rate,
                                                   device=device)

        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_loss_avg = total_loss / len(train_loader)
    train_losses.append(train_loss_avg)
    
    # 검증 데이터 손실 계산 (검증 시에는 augmentation 적용 안 함)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss_avg = val_loss / len(test_loader)
    val_losses.append(val_loss_avg)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

    # --- 얼리스탑 로직 ---
    if val_loss_avg < best_val_loss - 1e-8:
        best_val_loss = val_loss_avg
        epochs_no_improve = 0
        # 개선된 모델 저장
        torch.save(model.state_dict(), best_model_path)
        print(f"  Validation loss improved. Saved best model to {best_model_path}")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break

# 학습 과정 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss (physio-aug)')
plt.show()

# 학습된 모델 저장 (최종적으로도 best 모델을 같은 경로에 저장해둠)
torch.save(model.state_dict(), "/code/rppg2ppg/newdata/rnn_latent.pth")

# -----------------------------
# 예측 수행 및 저장 (슬라이딩/블록 방식)
# -----------------------------
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

# -----------------------------
# PSNR 계산 함수
# -----------------------------
def psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(y_true)
    # 안전: if max_pixel==0 fallback
    if max_pixel == 0:
        max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Pearson Correlation 계산
# shapes 맞추기: label_data 전체 길이에 맞춰라 (슬라이딩/스트라이드에 따라 다름)
# 여기서는 기존 코드와 동일하게 predictions는 block 단위로 이어붙인 형태로 가정
try:
    pearson_corr, _ = pearsonr(label_data.flatten(), predictions.flatten())
except Exception as e:
    pearson_corr = float('nan')
    print("Warning: Pearson calculation failed:", e)

# 평가 지표 출력
print("PSNR:", psnr(label_data, predictions))
print("Pearson Correlation:", pearson_corr)

print("Predictions shape:", predictions.shape)
print("Labels shape:", label_data.shape)
np.save("/code/rppg2ppg/newdata/rnn_latent.npy", predictions)

print(" Done. Augmentation used: physiology-inspired time-warp + amplitude scale + notch artifact.")
