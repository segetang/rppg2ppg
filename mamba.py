'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from mamba_ssm import Mamba  # 핵심 블록

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        # data, labels: 1D numpy arrays of equal length
        self.data = torch.from_numpy(data).float().unsqueeze(-1)   # (T,1)
        self.labels = torch.from_numpy(labels).float()             # (T,)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]      # (seq_len,1)
        y = self.labels[idx : idx + self.seq_len]    # (seq_len,)
        return x, y

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class RPPGRegressor(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=4, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
            for _ in range(n_blocks)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, L, 1)
        x = self.input_proj(x)  # → (B, L, d_model)
        for norm, block in zip(self.norms, self.blocks):
            x = norm(x)
            res = x
            x = block(x)        # → (B, L, d_model)
            x = x + res
        x = self.output_proj(x)  # → (B, L, 1)
        return x.squeeze(-1)     # → (B, L)

# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(y_true)
    return 20 * np.log10(max_val / np.sqrt(mse))

def run_training():
    # Load raw data
    pred_np = np.load('/mnt/c/code/rppg/real data/PURE_POS_prediction.npy')
    label_np = np.load('/mnt/c/code/rppg/real data/PURE_label.npy')

    # Scale to [0,1]
    sx, sy = MinMaxScaler(), MinMaxScaler()
    pred = sx.fit_transform(pred_np.reshape(-1,1)).flatten()
    label = sy.fit_transform(label_np.reshape(-1,1)).flatten()

    # Split
    split = int(len(pred) * 0.8)
    train_x, test_x = pred[:split], pred[split:]
    train_y, test_y = label[:split], label[split:]

    # Datasets & Loaders
    seq_len, bs = 300, 32
    train_ds = RPPGDataset(train_x, train_y, seq_len)
    test_ds  = RPPGDataset(test_x,  test_y,  seq_len)
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_ld  = DataLoader(test_ds,  batch_size=bs)

    # Model / Loss / Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RPPGRegressor().to(device)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 100
    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        model.train()
        tloss = 0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            tloss += loss.item()
        train_losses.append(tloss/len(train_ld))

        model.eval()
        vloss = 0
        with torch.no_grad():
            for xb, yb in test_ld:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vloss += criterion(out, yb).item()
        val_losses.append(vloss/len(test_ld))

        print(f"Epoch {ep}/{epochs} — Train: {train_losses[-1]:.6f}, Val: {val_losses[-1]:.6f}")

    # Plot losses
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig('/mnt/c/code/rppg/loss_plot.png')

    # Save model
    torch.save(model.state_dict(), '/mnt/c/code/rppg/newdata/mamba_rppg_multi.pth')

    # Full-sequence predictions
    model.eval()
    full_preds = []
    with torch.no_grad():
        for i in range(0, len(pred)-seq_len+1, seq_len):
            chunk = torch.from_numpy(pred[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(chunk).cpu().numpy().flatten()
            full_preds.append(out)
    full_preds = np.concatenate(full_preds)

    # Inverse scale
    full_preds = sy.inverse_transform(full_preds.reshape(-1,1)).flatten()

    # Metrics
    p_corr = pearsonr(label_np[:len(full_preds)], full_preds)[0]
    p_snr = psnr(label_np[:len(full_preds)], full_preds)
    print("Pearson:", p_corr, "PSNR:", p_snr)

    # Save predictions
    np.save('/mnt/c/code/rppg/newdata/Mamba_rPPG_prediction.npy', full_preds)

if __name__ == '__main__':
    run_training()

'''

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

# ── Low-pass Filter for rPPG Post-processing ───────────────────────────────────
def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

# ── Dataset ───────────────────────────────────────────────────────────────────
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.x = torch.from_numpy(data).float().unsqueeze(-1)  # (T,1)
        self.y = torch.from_numpy(labels).float()             # (T,)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return (self.x[idx:idx+self.seq_len],
                self.y[idx:idx+self.seq_len])

# ── Model ─────────────────────────────────────────────────────────────────────
class ImprovedRPPGModel(nn.Module):
    def __init__(self,
                 d_model=128,
                 d_state=32,
                 d_conv=8,
                 n_blocks=6,
                 dropout=0.1,
                 conv_kernel=7):
        super().__init__()
        # Initial learnable low-pass conv
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pre_act  = nn.GELU()

        # Stacked Mamba blocks with LayerNorm + Dropout + Residual
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
                nn.Dropout(dropout)
            ))

        # Final regression head
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, L, 1)
        B, L, _ = x.shape

        # 1) Learnable low-pass convolution
        x = x.permute(0, 2, 1)           # → (B, 1, L)
        x = self.pre_conv(x)            # → (B, d_model, L)
        x = self.pre_act(x)
        x = x.permute(0, 2, 1)          # → (B, L, d_model)

        # 2) Mamba blocks with residual connections
        for block in self.blocks:
            res = x
            x = block(x)               # LayerNorm → Mamba → Dropout
            x = x + res

        # 3) Regression projection
        x = self.output_proj(x)        # → (B, L, 1)
        return x.squeeze(-1)           # → (B, L)

# ── Training & Evaluation ─────────────────────────────────────────────────────
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(y_true.max() / np.sqrt(mse))

def main():
    # 1) Load and normalize
    raw_pred  = np.load('/mnt/c/code/rppg/real data/PURE_POS_prediction.npy')
    raw_label = np.load('/mnt/c/code/rppg/real data/PURE_label.npy')
    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(raw_pred.reshape(-1,1)).flatten()
    y_all = sy.fit_transform(raw_label.reshape(-1,1)).flatten()

    # 2) Train/Test split
    n = len(x_all)
    split = int(n * 0.8)
    train_x, test_x = x_all[:split], x_all[split:]
    train_y, test_y = y_all[:split], y_all[split:]

    # 3) DataLoaders
    seq_len, bs = 300, 32
    tr_ds = RPPGDataset(train_x, train_y, seq_len)
    te_ds = RPPGDataset(test_x,  test_y, seq_len)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True,  num_workers=4, pin_memory=True)
    te_ld = DataLoader(te_ds, bs, shuffle=False, num_workers=4, pin_memory=True)

    # 4) Model / Loss / Optimizer / Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedRPPGModel().to(device)
    criterion = nn.SmoothL1Loss()  # robust to outliers
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 5) Training loop
    epochs = 65
    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        model.train()
        total_tr = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_tr += loss.item()
        train_losses.append(total_tr / len(tr_ld))
        scheduler.step()

        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                total_val += criterion(model(xb), yb).item()
        val_losses.append(total_val / len(te_ld))

        print(f"Epoch {ep}/{epochs} — Train {train_losses[-1]:.4f}, Val {val_losses[-1]:.4f}")

    # 6) Plot losses
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig('/mnt/c/code/rppg/loss_best.png')

    # 7) Save model
    torch.save(model.state_dict(), '/mnt/c/code/rppg/newdata/mamba_rppg_best.pth')

    # 8) Sliding-window prediction with 50% overlap
    model.eval()
    L = len(x_all)
    step = seq_len // 2
    preds_accum = np.zeros(L, dtype=np.float32)
    weight_accum = np.zeros(L, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, L - seq_len + 1, step):
            chunk = torch.tensor(x_all[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(chunk).cpu().numpy().flatten()
            preds_accum[i:i+seq_len] += out
            weight_accum[i:i+seq_len] += 1

        # 마지막 남은 구간 보완 (길이 안 맞을 경우)
        last_start = L - seq_len
        if weight_accum[last_start:].min() == 0:
            chunk = torch.tensor(x_all[last_start:]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(chunk).cpu().numpy().flatten()
            preds_accum[last_start:] += out
            weight_accum[last_start:] += 1

    # 평균 + 나눗셈 안정화
    preds_norm = preds_accum / np.maximum(weight_accum, 1e-8)

    # 9) Inverse-scale & filter
    preds_scaled = sy.inverse_transform(preds_norm[:len(raw_label)].reshape(-1, 1)).flatten()
    preds_filt   = lowpass_filter(preds_scaled, fs=30, cutoff=3.0)

    # 10) Metrics
    n_eval = len(preds_filt)
    corr = pearsonr(raw_label[:n_eval], preds_filt[:n_eval])[0]
    snr  = psnr(raw_label[:n_eval], preds_filt[:n_eval])
    print(f"Final Pearson: {corr:.4f}, PSNR: {snr:.2f} dB")

    # 11) Save predictions (float32 for compact size)
    np.save('/mnt/c/code/rppg/newdata/mamba_rppg_final.npy', preds_filt.astype(np.float32))


if __name__ == '__main__':
    main()
'''
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
import random

# ─────────────────────────────────────────────────────────
# 1) Mixup
# ─────────────────────────────────────────────────────────
def apply_mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    perm = torch.randperm(bs)
    x2 = x[perm]
    y2 = y[perm]
    x_mix = lam * x + (1 - lam) * x2
    y_mix = lam * y + (1 - lam) * y2
    return x_mix, y_mix


# ─────────────────────────────────────────────────────────
# 2) CutMix (1D version)
# ─────────────────────────────────────────────────────────
def apply_cutmix(x, y, alpha=0.2):
    bs, seq_len, _ = x.size()
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(bs)
    x2 = x[perm]
    y2 = y[perm]

    cut_len = int(seq_len * lam)
    start = random.randint(0, seq_len - cut_len)
    end = start + cut_len

    x_cut = x.clone()
    x_cut[:, start:end, :] = x2[:, start:end, :]
    y_cut = y.clone()
    y_cut[:, start:end] = y2[:, start:end]

    return x_cut, y_cut


# ── Low-pass Filter ───────────────────────────────────────
def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

# ── Dataset ───────────────────────────────────────────────
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.x = torch.from_numpy(data).float().unsqueeze(-1)
        self.y = torch.from_numpy(labels).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return (self.x[idx:idx+self.seq_len],
                self.y[idx:idx+self.seq_len])

# ── Model ─────────────────────────────────────────────────
class ImprovedRPPGModel(nn.Module):
    def __init__(self,
                 d_model=128,
                 d_state=32,
                 d_conv=8,
                 n_blocks=6,
                 dropout=0.1,
                 conv_kernel=7):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pre_act  = nn.GELU()

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
                nn.Dropout(dropout)
            ))

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, _ = x.shape
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

# ── PSNR ──────────────────────────────────────────────────
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(y_true.max() / np.sqrt(mse))

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():

    # ★ 여기서 선택: 'mixup', 'cutmix', None
    augment_mode = 'mixup'   # ← 필요한 부분 이것만 바꿔라 (mixup / cutmix / None)

    # 파일 이름 자동 변경
    save_suffix = f"_{augment_mode}" if augment_mode else "_baseline"

    # Load
    raw_pred  = np.load('/mnt/c/code/rppg2ppg/real data/PURE_POS_prediction.npy')
    raw_label = np.load('/mnt/c/code/rppg2ppg/real data/PURE_label.npy')

    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(raw_pred.reshape(-1,1)).flatten()
    y_all = sy.fit_transform(raw_label.reshape(-1,1)).flatten()

    # Split
    n = len(x_all)
    split = int(n * 0.8)
    train_x, test_x = x_all[:split], x_all[split:]
    train_y, test_y = y_all[:split], y_all[split:]

    # Loader
    seq_len, bs = 300, 32
    tr_ld = DataLoader(RPPGDataset(train_x, train_y, seq_len), bs, shuffle=True)
    te_ld = DataLoader(RPPGDataset(test_x,  test_y, seq_len), bs, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedRPPGModel().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    epochs = 65
    train_losses, val_losses = [], []
    
    best_val = float('inf')
    patience = 8
    wait = 0
    best_model_path = '/mnt/c/code/rppg2ppg/newdata/mamba_rppg_best.pth'


    # ────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────
    for ep in range(1, epochs+1):
        model.train()
        total_tr = 0

        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)

            # ★★★ Apply Mixup or CutMix
            if augment_mode == 'mixup':
                xb, yb = apply_mixup(xb, yb, alpha=0.2)

            elif augment_mode == 'cutmix':
                xb, yb = apply_cutmix(xb, yb, alpha=0.2)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_tr += loss.item()

        train_losses.append(total_tr / len(tr_ld))
        scheduler.step()

        # Validation
        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                total_val += criterion(model(xb), yb).item()
        
        val_losses.append(total_val / len(te_ld))
        val_loss = total_val / len(te_ld)

        print(f"Epoch {ep}/{epochs} - Train {train_losses[-1]:.4f}, Val {val_losses[-1]:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {ep}")
                break


    # Save loss plot
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.legend(); plt.tight_layout()
    plt.savefig(f'/mnt/c/code/rppg2ppg/loss{save_suffix}.png')

    # Save model
    torch.save(model.state_dict(), f'/mnt/c/code/rppg2ppg/newdata/mamba_rppg{save_suffix}.pth')

    # ────────────────────────────────────────────
    # Prediction (변경 없음)
    # ────────────────────────────────────────────
    model.eval()
    L = len(x_all)
    step = seq_len // 2
    preds_accum = np.zeros(L, dtype=np.float32)
    weight_accum = np.zeros(L, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, L - seq_len + 1, step):
            chunk = torch.tensor(x_all[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(chunk).cpu().numpy().flatten()
            preds_accum[i:i+seq_len] += out
            weight_accum[i:i+seq_len] += 1

        last_start = L - seq_len
        if weight_accum[last_start:].min() == 0:
            chunk = torch.tensor(x_all[last_start:]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(chunk).cpu().numpy().flatten()
            preds_accum[last_start:] += out
            weight_accum[last_start:] += 1

    preds_norm = preds_accum / np.maximum(weight_accum, 1e-8)
    preds_scaled = sy.inverse_transform(preds_norm[:len(raw_label)].reshape(-1, 1)).flatten()
    preds_filt   = lowpass_filter(preds_scaled, fs=30, cutoff=3.0)

    # Metrics
    corr = pearsonr(raw_label[:len(preds_filt)], preds_filt)[0]
    snr  = psnr(raw_label[:len(preds_filt)], preds_filt)
    print(f"Final Pearson: {corr:.4f}, PSNR: {snr:.2f} dB")

    np.save(f'/mnt/c/code/rppg2ppg/newdata/mamba_{save_suffix}.npy',
            preds_filt.astype(np.float32))


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
import random


# ─────────────────────────────────────────────────────────
# Mixup
# ─────────────────────────────────────────────────────────
def apply_mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    perm = torch.randperm(bs)
    x2, y2 = x[perm], y[perm]
    return lam * x + (1-lam)*x2, lam*y + (1-lam)*y2


# ─────────────────────────────────────────────────────────
# CutMix (1D)
# ─────────────────────────────────────────────────────────
def apply_cutmix(x, y, alpha=0.2):
    bs, seq_len, _ = x.size()
    lam = np.random.beta(alpha, alpha)

    perm = torch.randperm(bs)
    x2, y2 = x[perm], y[perm]

    cut_len = int(seq_len * lam)
    start = random.randint(0, seq_len - cut_len)
    end = start + cut_len

    x_new = x.clone()
    y_new = y.clone()
    x_new[:, start:end, :] = x2[:, start:end, :]
    y_new[:, start:end] = y2[:, start:end]

    return x_new, y_new


# ── Low-pass Filter ───────────────────────────────────────
def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)


# ── Dataset ───────────────────────────────────────────────
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.x = torch.from_numpy(data).float().unsqueeze(-1)
        self.y = torch.from_numpy(labels).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return (self.x[idx:idx+self.seq_len],
                self.y[idx:idx+self.seq_len])


# ── Mamba Model ───────────────────────────────────────────
class ImprovedRPPGModel(nn.Module):
    def __init__(self, d_model=128, d_state=32, d_conv=8, n_blocks=6, dropout=0.1, conv_kernel=7):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pre_act = nn.GELU()

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
                nn.Dropout(dropout)
            ))

        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, _ = x.shape
        x = x.permute(0,2,1)
        x = self.pre_conv(x)
        x = self.pre_act(x)
        x = x.permute(0,2,1)

        for blk in self.blocks:
            res = x
            x = blk(x)
            x = x + res

        return self.out_proj(x).squeeze(-1)


# ───────────────────────────────────────────────────────────
# PSNR
# ───────────────────────────────────────────────────────────
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(y_true.max() / np.sqrt(mse))



# ───────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────
def main():

    # augmentation 모드 (여기서는 baseline + mixup + cutmix 모두 사용)
    augment_list = ["baseline", "mixup", "cutmix"]

    raw_pred  = np.load('/mnt/c/code/rppg2ppg/real data/PURE_POS_prediction.npy')
    raw_label = np.load('/mnt/c/code/rppg2ppg/real data/PURE_label.npy')

    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(raw_pred.reshape(-1,1)).flatten()
    y_all = sy.fit_transform(raw_label.reshape(-1,1)).flatten()

    # Split
    n = len(x_all)
    split = int(n * 0.8)
    train_x, test_x = x_all[:split], x_all[split:]
    train_y, test_y = y_all[:split], y_all[split:]

    seq_len = 300
    bs = 32

    tr_ld = DataLoader(RPPGDataset(train_x, train_y, seq_len), bs, shuffle=True)
    te_ld = DataLoader(RPPGDataset(test_x,  test_y, seq_len), bs, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedRPPGModel().to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    epochs = 65
    best_val = float("inf")
    patience = 12
    wait = 0
    best_model_path = "/mnt/c/code/rppg2ppg/newdata/mamba_rppg_best.pth"

    train_losses, val_losses = [], []


    # ───────────────────────────────────────────────────────────
    # Training
    # ───────────────────────────────────────────────────────────
    for ep in range(1, epochs+1):
        model.train()
        total_tr = 0

        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)

            # baseline
            xb_base, yb_base = xb, yb

            # mixup
            xb_mix, yb_mix = apply_mixup(xb, yb, alpha=0.2)

            # cutmix
            xb_cut, yb_cut = apply_cutmix(xb, yb, alpha=0.2)

            # 최종 3배 확장된 batch 구성
            xb_all = torch.cat([xb_base, xb_mix, xb_cut], dim=0)
            yb_all = torch.cat([yb_base, yb_mix, yb_cut], dim=0)

            optimizer.zero_grad()
            out = model(xb_all)
            loss = criterion(out, yb_all)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_tr += loss.item()

        train_losses.append(total_tr / len(tr_ld))
        scheduler.step()

        # Validation
        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                total_val += criterion(model(xb), yb).item()

        val_loss = total_val / len(te_ld)
        val_losses.append(val_loss)
        print(f"Epoch {ep} / {epochs} - Train {train_losses[-1]:.4f}, Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"EARLY STOP at epoch {ep}")
                break

    # Loss plot
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend(); plt.tight_layout()
    plt.savefig('/mnt/c/code/rppg2ppg/loss_aug3.png')

    torch.save(model.state_dict(), '/mnt/c/code/rppg2ppg/mamba_rppg_aug3.pth')

    # Prediction
    model.eval()
    L = len(x_all)
    step = seq_len // 2
    preds = np.zeros(L)
    w = np.zeros(L)

    with torch.no_grad():
        for i in range(0, L - seq_len + 1, step):
            chunk = torch.tensor(x_all[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(chunk).cpu().numpy().flatten()
            preds[i:i+seq_len] += out
            w[i:i+seq_len] += 1

    preds_norm = preds / np.maximum(w, 1e-8)
    preds_scaled = sy.inverse_transform(preds_norm.reshape(-1,1)).flatten()
    preds_filt = lowpass_filter(preds_scaled, fs=30, cutoff=3.0)

    corr = pearsonr(raw_label[:len(preds_filt)], preds_filt)[0]
    snr = psnr(raw_label[:len(preds_filt)], preds_filt)

    print(f"FINAL Pearson = {corr:.4f},  PSNR = {snr:.2f} dB")

    np.save('/mnt/c/code/rppg2ppg/mamba_pred_aug3.npy', preds_filt.astype(np.float32))


if __name__ == "__main__":
    main()
