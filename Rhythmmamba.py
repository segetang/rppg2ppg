# RhythmMamba 기반 rPPG 예측 전체 코드 (PURE POS 기반, 중복 제거 및 용량 최적화)
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from mamba_ssm import Mamba

# ── Low-pass filter
def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

# ── Dataset
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.x = torch.from_numpy(data).float().unsqueeze(-1)  # (T, 1)
        self.y = torch.from_numpy(labels).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return self.x[idx:idx+self.seq_len], self.y[idx:idx+self.seq_len]

# ── Model
class MambaRPPGModel(nn.Module):
    def __init__(self, d_model=128, d_state=32, d_conv=8, n_blocks=6, dropout=0.1, conv_kernel=7):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pre_act = nn.GELU()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
                nn.Dropout(dropout)
            ) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pre_conv(x)
        x = self.pre_act(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            res = x
            x = block(x)
            x = x + res
        x = self.output_proj(x)
        return x.squeeze(-1)

# ── PSNR
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(y_true.max() / np.sqrt(mse))

# ── Main

def main():
    pred_np  = np.load('/mnt/c/code/rppg/real data/PURE_POS_prediction.npy')
    label_np = np.load('/mnt/c/code/rppg/real data/PURE_label.npy')

    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(pred_np.reshape(-1,1)).flatten()
    y_all = sy.fit_transform(label_np.reshape(-1,1)).flatten()

    split = int(len(x_all) * 0.8)
    train_x, test_x = x_all[:split], x_all[split:]
    train_y, test_y = y_all[:split], y_all[split:]

    seq_len, bs = 300, 32
    tr_ds = RPPGDataset(train_x, train_y, seq_len)
    te_ds = RPPGDataset(test_x,  test_y, seq_len)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True)
    te_ld = DataLoader(te_ds, bs, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaRPPGModel().to(device)
    criterion = nn.SmoothL1Loss()
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    train_losses, val_losses = [], []
    for ep in range(1, 61):
        model.train()
        total = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        train_losses.append(total / len(tr_ld))

        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                total_val += criterion(model(xb), yb).item()
        val_losses.append(total_val / len(te_ld))

        print(f"Epoch {ep}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

    # 결과 예측
    model.eval()
    L = len(x_all)
    step = seq_len // 2
    preds_accum = np.zeros(L, dtype=np.float32)
    weight_accum = np.zeros(L, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, L - seq_len + 1, step):
            clip = torch.tensor(x_all[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(clip).cpu().numpy().flatten()
            preds_accum[i:i+seq_len] += pred
            weight_accum[i:i+seq_len] += 1
    preds = preds_accum / np.maximum(weight_accum, 1e-6)

    preds_scaled = sy.inverse_transform(preds.reshape(-1,1)).flatten()
    preds_filt = lowpass_filter(preds_scaled).astype(np.float32)

    print("Predictions shape:", preds_filt.shape)
    print("Labels shape:", label_np.shape)

    corr = pearsonr(label_np[:len(preds_filt)], preds_filt)[0]
    snr = psnr(label_np[:len(preds_filt)], preds_filt)
    print(f"Final Pearson: {corr:.4f}, PSNR: {snr:.2f} dB")

    # 내보 저장 (유적화)
    np.save('/mnt/c/code/rppg/newdata/rhythmmamba_rppg_final.npy', preds_filt.astype(np.float32))

    # 그림
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/mnt/c/code/rppg/loss_plot.png')

if __name__ == '__main__':
    main()

'''
'''
# RhythmMamba 기반 rPPG 예측 성능 개선 버전 (PURE POS 기반)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from mamba_ssm import Mamba

# ────────────────────── 필터 함수 ──────────────────────
def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

# ────────────────────── 데이터셋 ──────────────────────
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.x = torch.from_numpy(data).float().unsqueeze(-1)
        self.y = torch.from_numpy(labels).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return self.x[idx:idx+self.seq_len], self.y[idx:idx+self.seq_len]

# ────────────────────── 모델 정의 ──────────────────────
class MambaRPPGModel(nn.Module):
    def __init__(self, d_model=160, d_state=64, d_conv=12, n_blocks=8, dropout=0.1, conv_kernel=9):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pre_act = nn.GELU()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
                nn.Dropout(dropout)
            ) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pre_conv(x)
        x = self.pre_act(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            res = x
            x = block(x)
            x = x + res
        x = self.output_proj(x)
        return x.squeeze(-1)

# ────────────────────── PSNR ──────────────────────
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(y_true.max() / np.sqrt(mse))

# ────────────────────── 메인 함수 ──────────────────────
def main():
    # 데이터 로드 및 정규화
    pred_np  = np.load('/mnt/c/code/rppg2ppg/real data/PURE_POS_prediction.npy')
    label_np = np.load('/mnt/c/code/rppg2ppg/real data/PURE_label.npy')

    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(pred_np.reshape(-1,1)).flatten()
    y_all = sy.fit_transform(label_np.reshape(-1,1)).flatten()

    # 학습/테스트 분할
    split = int(len(x_all) * 0.8)
    train_x, test_x = x_all[:split], x_all[split:]
    train_y, test_y = y_all[:split], y_all[split:]

    seq_len, bs = 300, 32
    tr_ds = RPPGDataset(train_x, train_y, seq_len)
    te_ds = RPPGDataset(test_x,  test_y, seq_len)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True)
    te_ld = DataLoader(te_ds, bs, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaRPPGModel().to(device)
    criterion = nn.SmoothL1Loss()
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    train_losses, val_losses = [], []
    for ep in range(1, 91):
        model.train()
        total = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        train_losses.append(total / len(tr_ld))

        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                total_val += criterion(model(xb), yb).item()
        val_losses.append(total_val / len(te_ld))
        scheduler.step()

        print(f"Epoch {ep}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

    # 슬라이딩 윈도우 예측
    model.eval()
    L = len(x_all)
    step = seq_len // 2
    preds_accum = np.zeros(L, dtype=np.float32)
    weight_accum = np.zeros(L, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, L - seq_len + 1, step):
            clip = torch.tensor(x_all[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(clip).cpu().numpy().flatten()
            preds_accum[i:i+seq_len] += pred
            weight_accum[i:i+seq_len] += 1
    preds = preds_accum / np.maximum(weight_accum, 1e-6)

    preds_scaled = sy.inverse_transform(preds.reshape(-1,1)).flatten()
    preds_filt = lowpass_filter(preds_scaled).astype(np.float32)

    print("Predictions shape:", preds_filt.shape)
    print("Labels shape:", label_np.shape)

    corr = pearsonr(label_np[:len(preds_filt)], preds_filt)[0]
    snr = psnr(label_np[:len(preds_filt)], preds_filt)
    print(f"Final Pearson: {corr:.4f}, PSNR: {snr:.2f} dB")

    np.save('/mnt/c/code/rppg2ppg/newdata/rhythmmamba_rppg_final.npy', preds_filt)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/mnt/c/code/rppg2ppg/loss_plot.png')

if __name__ == '__main__':
    main()
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from mamba_ssm import Mamba
import os
import random

# ────────────────────── EarlyStopping ──────────────────────
class EarlyStopping:
    def __init__(self, patience=12, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


# ────────────────────── 필터 함수 ──────────────────────
def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

# ────────────────────── 데이터셋 ──────────────────────
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.x = torch.from_numpy(data).float().unsqueeze(-1)
        self.y = torch.from_numpy(labels).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return self.x[idx:idx+self.seq_len], self.y[idx:idx+self.seq_len]

# ────────────────────── 모델 정의 ──────────────────────
class MambaRPPGModel(nn.Module):
    def __init__(self, d_model=160, d_state=64, d_conv=12, n_blocks=8, dropout=0.1, conv_kernel=9):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pre_act = nn.GELU()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
                nn.Dropout(dropout)
            ) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pre_conv(x)
        x = self.pre_act(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            res = x
            x = block(x)
            x = x + res
        x = self.output_proj(x)
        return x.squeeze(-1)

# ────────────────────── PSNR ──────────────────────
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(y_true.max() / np.sqrt(mse))

# ────────────────────── Augmentations ──────────────────────
def mixup_1d(x, y, alpha=0.2):
    if alpha <= 0: return x, y
    B = x.size(0)
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(B, device=x.device)
    return lam * x + (1-lam)*x[idx], lam * y + (1-lam)*y[idx]

def cutmix_1d(x, y, alpha=0.2):
    if alpha <= 0: return x, y
    B, L, C = x.shape
    if C != 1: raise ValueError("cutmix_1d expects (B,L,1)")

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(B, device=x.device)
    cut_len = max(1, int(round(L * lam)))
    cut_len = min(cut_len, L)
    start = random.randint(0, L - cut_len)
    end = start + cut_len

    x2 = x[idx]
    y2 = y[idx]

    x_mix = x.clone()
    y_mix = y.clone()
    x_mix[:, start:end, :] = x2[:, start:end, :]
    y_mix[:, start:end] = y2[:, start:end]
    return x_mix, y_mix

# ────────────────────── Main ──────────────────────
def main(augment_mode=None, alpha=0.2):

    pred_path = '/mnt/c/code/rppg2ppg/real data/PURE_POS_prediction.npy'
    label_path = '/mnt/c/code/rppg2ppg/real data/PURE_label.npy'
    out_dir = '/mnt/c/code/rppg2ppg/newdata'
    os.makedirs(out_dir, exist_ok=True)

    pred_np = np.load(pred_path)
    label_np = np.load(label_path)

    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(pred_np.reshape(-1,1)).flatten()
    y_all = sy.fit_transform(label_np.reshape(-1,1)).flatten()

    split = int(len(x_all)*0.8)
    train_x, test_x = x_all[:split], x_all[split:]
    train_y, test_y = y_all[:split], y_all[split:]

    seq_len = 300
    bs = 32
    tr_ld = DataLoader(RPPGDataset(train_x, train_y, seq_len), batch_size=bs, shuffle=True)
    te_ld = DataLoader(RPPGDataset(test_x, test_y, seq_len), batch_size=bs, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaRPPGModel().to(device)

    criterion = nn.SmoothL1Loss()
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    train_losses, val_losses = [], []

    # ★ EarlyStopping 추가
    early = EarlyStopping(patience=15, min_delta=1e-4)
    best_model_path = os.path.join(out_dir, "mamba_rppg_best.pth")

    epochs = 90
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)

            if augment_mode == 'mixup':
                xb, yb = mixup_1d(xb, yb, alpha)
            elif augment_mode == 'cutmix':
                xb, yb = cutmix_1d(xb, yb, alpha)

            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        train_loss = total / len(tr_ld)
        train_losses.append(train_loss)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                total_val += criterion(model(xb), yb).item()
        val_loss = total_val / len(te_ld)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"Epoch {ep:03d}/{epochs} — Train {train_loss:.6f}, Val {val_loss:.6f}")

        # ★ 조기 종료 체크 + best 모델 저장
        if val_loss == min(val_losses):
            torch.save(model.state_dict(), best_model_path)

        if early.step(val_loss):
            print("Early stopping triggered!")
            break

    print("Best model saved to:", best_model_path)

    # 이후 inference 동일 (생략 없이 그대로 진행)
    # ─────────────────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    L = len(x_all)
    step = seq_len // 2
    preds_accum = np.zeros(L, dtype=np.float32)
    weight_accum = np.zeros(L, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, L - seq_len + 1, step):
            clip = torch.tensor(x_all[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(clip).cpu().numpy().flatten()
            preds_accum[i:i+seq_len] += out
            weight_accum[i:i+seq_len] += 1.0

        last_start = L - seq_len
        if weight_accum[last_start:].min() == 0:
            clip = torch.tensor(x_all[last_start:]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(clip).cpu().numpy().flatten()
            preds_accum[last_start:] += out
            weight_accum[last_start:] += 1.0

    preds = preds_accum / np.maximum(weight_accum, 1e-6)
    preds_scaled = sy.inverse_transform(preds.reshape(-1,1)).flatten()
    preds_filt = lowpass_filter(preds_scaled, fs=30, cutoff=3.0).astype(np.float32)

    n_eval = len(preds_filt)
    corr = pearsonr(label_np[:n_eval], preds_filt[:n_eval])[0]
    snr = psnr(label_np[:n_eval], preds_filt[:n_eval])
    print(f"Final Pearson: {corr:.4f}, PSNR: {snr:.2f} dB")

    suffix = f"_{augment_mode}" if augment_mode else "_baseline"
    np.save(os.path.join(out_dir, f"Rhythmmamba{suffix}.npy"), preds_filt)

    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"loss{suffix}.png"))
    plt.close()


if __name__ == "__main__":
    main(augment_mode='cutmix', alpha=0.2)
