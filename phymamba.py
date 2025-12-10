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

# ── Low-pass Filter for rPPG Post-processing ─────────────────────────────
def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

# ── Dataset ───────────────────────────────────────────────────────────────
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.x = torch.from_numpy(data).float().unsqueeze(1)  # (T, 1)
        self.y = torch.from_numpy(labels).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return (self.x[idx:idx+self.seq_len], self.y[idx:idx+self.seq_len])

# ── Bi-directional Mamba Block ────────────────────────────────────────────
class BiMambaBlock(nn.Module):
    def __init__(self, d_model=128, d_state=32, d_conv=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.mamba_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        x_norm = self.norm(x)
        x_rev = torch.flip(x_norm, dims=[1])
        f_out = self.mamba_f(x_norm)
        b_out = self.mamba_b(x_rev)
        b_out = torch.flip(b_out, dims=[1])
        fused = torch.cat([f_out, b_out], dim=-1)
        gated = self.gate(fused)
        return x + self.dropout(gated)

# ── PhysMamba Full Architecture (Bi-Mamba version for 1D Input) ───────────
class PhysMamba1D(nn.Module):
    def __init__(self, d_model=128, d_state=32, d_conv=8, n_blocks=6, dropout=0.1, conv_kernel=7):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pre_act  = nn.GELU()

        self.blocks = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model, 1),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pre_act(self.pre_conv(x))
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        x_att = x.permute(0, 2, 1)
        attn = self.se(x_att)
        x = (x_att * attn).permute(0, 2, 1)

        x = self.output_proj(x)
        return x.squeeze(-1)

# ── Metrics ───────────────────────────────────────────────────────────────
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(y_true.max() / np.sqrt(mse))

# ── Main Training & Evaluation ────────────────────────────────────────────
def main():
    raw_pred  = np.load('/mnt/c/code/rppg/real data/PURE_POS_prediction.npy')
    raw_label = np.load('/mnt/c/code/rppg/real data/PURE_label.npy')
    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(raw_pred.reshape(-1,1)).flatten()
    y_all = sy.fit_transform(raw_label.reshape(-1,1)).flatten()

    n = len(x_all)
    split = int(n * 0.8)
    train_x, test_x = x_all[:split], x_all[split:]
    train_y, test_y = y_all[:split], y_all[split:]

    seq_len, bs = 300, 32
    tr_ds = RPPGDataset(train_x, train_y, seq_len)
    te_ds = RPPGDataset(test_x, test_y, seq_len)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True,  num_workers=4, pin_memory=True)
    te_ld = DataLoader(te_ds, bs, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhysMamba1D().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    epochs = 55
    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        model.train(); total_tr = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_tr += loss.item()
        train_losses.append(total_tr / len(tr_ld))
        scheduler.step()

        model.eval(); total_val = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                total_val += criterion(model(xb), yb).item()
        val_losses.append(total_val / len(te_ld))

        print(f"Epoch {ep}/{epochs} — Train {train_losses[-1]:.4f}, Val {val_losses[-1]:.4f}")

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig('/mnt/c/code/rppg/loss_physmamba_full.png')

    torch.save(model.state_dict(), '/mnt/c/code/rppg/newdata/physmamba1d_best.pth')

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
    preds_filt = lowpass_filter(preds_scaled, fs=30, cutoff=3.0)

    n_eval = len(preds_filt)
    corr = pearsonr(raw_label[:n_eval], preds_filt[:n_eval])[0]
    snr  = psnr(raw_label[:n_eval], preds_filt[:n_eval])
    print(f"Final Pearson: {corr:.4f}, PSNR: {snr:.2f} dB")

    np.save('/mnt/c/code/rppg/newdata/physmamba_rppg_final.npy', preds_filt.astype(np.float32))

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
import os

# ── Low-pass Filter for rPPG Post-processing ─────────────────────────────
def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

# ── Dataset ───────────────────────────────────────────────────────────────
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.x = torch.from_numpy(data).float().unsqueeze(1)  # (T, 1)
        self.y = torch.from_numpy(labels).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return (self.x[idx:idx+self.seq_len], self.y[idx:idx+self.seq_len])

# ── Bi-directional Mamba Block ────────────────────────────────────────────
class BiMambaBlock(nn.Module):
    def __init__(self, d_model=128, d_state=32, d_conv=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.mamba_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        x_norm = self.norm(x)
        x_rev = torch.flip(x_norm, dims=[1])
        f_out = self.mamba_f(x_norm)
        b_out = self.mamba_b(x_rev)
        b_out = torch.flip(b_out, dims=[1])
        fused = torch.cat([f_out, b_out], dim=-1)
        gated = self.gate(fused)
        return x + self.dropout(gated)

# ── PhysMamba Full Architecture (Bi-Mamba version for 1D Input) ───────────
class PhysMamba1D(nn.Module):
    def __init__(self, d_model=128, d_state=32, d_conv=8, n_blocks=6, dropout=0.1, conv_kernel=7):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pre_act  = nn.GELU()

        self.blocks = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model, 1),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pre_act(self.pre_conv(x))
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        x_att = x.permute(0, 2, 1)
        attn = self.se(x_att)
        x = (x_att * attn).permute(0, 2, 1)

        x = self.output_proj(x)
        return x.squeeze(-1)
# ── Negative Pearson Loss ─────────────────────────────────────────────────
def neg_pearson_loss(y_pred, y_true):
    y_pred = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true = y_true - y_true.mean(dim=1, keepdim=True)
    num = (y_pred * y_true).sum(dim=1)
    den = torch.sqrt((y_pred**2).sum(dim=1) * (y_true**2).sum(dim=1)) + 1e-8
    corr = num / den
    return 1 - corr.mean()
    
# ── Metrics ───────────────────────────────────────────────────────────────
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(y_true.max() / np.sqrt(mse))

# ── Main Training & Evaluation ────────────────────────────────────────────
def main():
    raw_pred  = np.load('/mnt/c/code/rppg/real data/PURE_POS_prediction.npy')
    raw_label = np.load('/mnt/c/code/rppg/real data/PURE_label.npy')
    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(raw_pred.reshape(-1,1)).flatten()
    y_all = sy.fit_transform(raw_label.reshape(-1,1)).flatten()

    n = len(x_all)
    split = int(n * 0.8)
    train_x, test_x = x_all[:split], x_all[split:]
    train_y, test_y = y_all[:split], y_all[split:]

    seq_len, bs = 300, 32
    tr_ds = RPPGDataset(train_x, train_y, seq_len)
    te_ds = RPPGDataset(test_x, test_y, seq_len)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True,  num_workers=4, pin_memory=True)
    te_ld = DataLoader(te_ds, bs, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhysMamba1D().to(device)
    criterion = neg_pearson_loss 
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    epochs = 60
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        model.train(); total_tr = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_tr += loss.item()
        train_losses.append(total_tr / len(tr_ld))
        scheduler.step()

        model.eval(); total_val = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                total_val += criterion(model(xb), yb).item()
        val_loss = total_val / len(te_ld)
        val_losses.append(val_loss)

        print(f"Epoch {ep}/{epochs} — Train {train_losses[-1]:.4f}, Val {val_loss:.4f}")
        
        # 저장조건 강화
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/mnt/c/code/rppg/newdata/physmamba1d_best.pth')

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig('/mnt/c/code/rppg/loss_physmamba_full.png')

    torch.save(model.state_dict(), '/mnt/c/code/rppg/newdata/physmamba1d_best.pth')

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
    preds_filt = lowpass_filter(preds_scaled, fs=30, cutoff=3.0)

    n_eval = len(preds_filt)
    corr = pearsonr(raw_label[:n_eval], preds_filt[:n_eval])[0]
    snr  = psnr(raw_label[:n_eval], preds_filt[:n_eval])
    print(f"Final Pearson: {corr:.4f}, PSNR: {snr:.2f} dB")

    np.save('/mnt/c/code/rppg/newdata/physmamba_rppg_final.npy', preds_filt.astype(np.float32))

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

# ───────────────────────────────────────────────
# 0) Early Stopping
# ───────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # store CPU tensors to avoid GPU/serialization issues later
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

# ───────────────────────────────────────────────
# 1) Low-pass Filter
# ───────────────────────────────────────────────
def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

# ───────────────────────────────────────────────
# 2) Dataset
# ───────────────────────────────────────────────
class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        # data, labels are 1D numpy arrays (normalized)
        # store as (T,1) and (T,)
        self.x = torch.from_numpy(data).float().unsqueeze(1)  # (T,1)
        self.y = torch.from_numpy(labels).float()            # (T,)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        # returns (seq_len, 1), (seq_len,)
        return (self.x[idx:idx+self.seq_len],
                self.y[idx:idx+self.seq_len])

# ───────────────────────────────────────────────
# 3) Mixup & CutMix (safe for shape (B, seq_len, 1))
# ───────────────────────────────────────────────
def mixup_1d(x, y, alpha=0.2):
    """
    x: (B, L, 1)
    y: (B, L)
    returns mixed x,y and metadata (lam, idx, region=None)
    """
    if alpha <= 0:
        return x, y, None, None, None

    B, L, C = x.shape
    assert C == 1, "mixup_1d expects shape (B, L, 1)"

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(B, device=x.device)

    x2 = x[idx]              # (B, L, 1)
    y2 = y[idx]              # (B, L)

    x_mix = lam * x + (1.0 - lam) * x2
    y_mix = lam * y + (1.0 - lam) * y2

    return x_mix, y_mix, lam, idx, None

def cutmix_1d(x, y, alpha=0.2):
    """
    x: (B, L, 1)
    y: (B, L)
    Cut a contiguous segment along time axis and replace from a random example.
    Returns x_mix, y_mix, lam, idx, (start,end)
    """
    if alpha <= 0:
        return x, y, None, None, None

    B, L, C = x.shape
    assert C == 1, "cutmix_1d expects shape (B, L, 1)"

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(B, device=x.device)

    cut_len = max(1, int(round(L * lam)))
    start = random.randint(0, L - cut_len)
    end = start + cut_len

    x2 = x[idx]
    y2 = y[idx]

    x_mix = x.clone()
    y_mix = y.clone()

    # replace time segment (preserve last dim shape)
    x_mix[:, start:end, :] = x2[:, start:end, :]
    y_mix[:, start:end] = y2[:, start:end]

    # recompute effective lam as fraction of kept vs replaced (not strictly necessary)
    lam_eff = 1.0 - (cut_len / L)

    return x_mix, y_mix, lam_eff, idx, (start, end)

# ───────────────────────────────────────────────
# 4) Bi-directional Mamba block
# ───────────────────────────────────────────────
class BiMambaBlock(nn.Module):
    def __init__(self, d_model=128, d_state=32, d_conv=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.mamba_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        x_norm = self.norm(x)
        x_rev = torch.flip(x_norm, dims=[1])

        f_out = self.mamba_f(x_norm)
        b_out = self.mamba_b(x_rev)
        b_out = torch.flip(b_out, dims=[1])

        fused = torch.cat([f_out, b_out], dim=-1)
        gated = self.gate(fused)

        return x + self.dropout(gated)

# ───────────────────────────────────────────────
# 5) PhysMamba1D
# ───────────────────────────────────────────────
class PhysMamba1D(nn.Module):
    def __init__(self, d_model=128, d_state=32, d_conv=8, n_blocks=6, dropout=0.1, conv_kernel=7):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, conv_kernel, padding=conv_kernel // 2)
        self.pre_act = nn.GELU()

        self.blocks = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, dropout)
            for _ in range(n_blocks)
        ])

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model, 1),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # expects x: (B, L, 1)
        x = x.permute(0, 2, 1)               # -> (B, 1, L)
        x = self.pre_act(self.pre_conv(x))   # -> (B, d_model, L)
        x = x.permute(0, 2, 1)               # -> (B, L, d_model)

        for blk in self.blocks:
            x = blk(x)

        x_att = x.permute(0, 2, 1)           # -> (B, d_model, L)
        attn = self.se(x_att)                # -> (B, d_model, 1)
        x = (x_att * attn).permute(0, 2, 1)  # -> (B, L, d_model)

        x = self.output_proj(x)              # -> (B, L, 1)
        return x.squeeze(-1)                 # -> (B, L)

# ───────────────────────────────────────────────
# 6) PSNR
# ───────────────────────────────────────────────
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(y_true.max() / np.sqrt(mse))

# ───────────────────────────────────────────────
# 7) Main
# ───────────────────────────────────────────────
def main(mode="mixup", alpha=0.2):
    """
    mode: "none", "mixup", "cutmix"
    alpha: augmentation strength
    """
    # files (change paths if needed)
    pred_file = '/mnt/c/code/rppg2ppg/real data/PURE_POS_prediction.npy'
    label_file = '/mnt/c/code/rppg2ppg/real data/PURE_label.npy'
    out_dir = '/mnt/c/code/rppg2ppg/newdata'
    os.makedirs(out_dir, exist_ok=True)

    raw_pred = np.load(pred_file)
    raw_label = np.load(label_file)

    # normalize
    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(raw_pred.reshape(-1,1)).flatten()
    y_all = sy.fit_transform(raw_label.reshape(-1,1)).flatten()

    # split
    n = len(x_all)
    split = int(n * 0.8)
    train_x, test_x = x_all[:split], x_all[split:]
    train_y, test_y = y_all[:split], y_all[split:]

    seq_len, bs = 300, 32
    tr_ld = DataLoader(RPPGDataset(train_x, train_y, seq_len), batch_size=bs, shuffle=True)
    te_ld = DataLoader(RPPGDataset(test_x, test_y, seq_len), batch_size=bs, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysMamba1D().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    epochs = 55
    train_losses, val_losses = [], []
    early = EarlyStopping(patience=5)

    print("Train start | mode:", mode, "| alpha:", alpha)

    for ep in range(1, epochs+1):
        model.train()
        total_tr = 0.0

        for xb, yb in tr_ld:
            # xb: (B, L, 1), yb: (B, L)
            xb, yb = xb.to(device), yb.to(device)

            if mode == "mixup":
                xb, yb, _, _, _ = mixup_1d(xb, yb, alpha=alpha)
            elif mode == "cutmix":
                xb, yb, _, _, _ = cutmix_1d(xb, yb, alpha=alpha)
            # else none

            optimizer.zero_grad()
            pred = model(xb)   # model expects (B, L, 1)
            loss = criterion(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_tr += loss.item()

        train_loss = total_tr / len(tr_ld)
        train_losses.append(train_loss)
        scheduler.step()

        # validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                total_val += criterion(model(xb), yb).item()
        val_loss = total_val / len(te_ld)
        val_losses.append(val_loss)

        print(f"[{mode}] Epoch {ep}/{epochs} - Train {train_loss:.6f}, Val {val_loss:.6f}")

        early.step(val_loss, model)
        if early.early_stop:
            print(f"Early stopping triggered at epoch {ep}")
            break

    # load best state into model
    if early.best_state is not None:
        model.load_state_dict(early.best_state)
    # save best model file
    best_model_path = os.path.join(out_dir, f"physmamba1d_{mode}.pth")
    torch.save(model.state_dict(), best_model_path)
    print("Best model saved to:", best_model_path)

    # Plot losses
    try:
        plt.figure()
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.title(f"loss_{mode}")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"loss_{mode}.png"))
        plt.close()
    except Exception:
        pass

    # ───────────────────────────────────────────────
    # Prediction (sliding window, 50% overlap)
    # ───────────────────────────────────────────────
    model.eval()
    L_total = len(x_all)
    step = seq_len // 2
    preds_acc = np.zeros(L_total, dtype=np.float32)
    w_acc = np.zeros(L_total, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, L_total - seq_len + 1, step):
            # create chunk with shape (1, seq_len, 1)
            chunk = torch.tensor(x_all[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            # chunk shape: (1, L, 1) -> model will permute to (1,1,L)
            out = model(chunk).cpu().numpy().flatten()
            preds_acc[i:i+seq_len] += out
            w_acc[i:i+seq_len] += 1.0

        # tail handling (if any leftover)
        last_start = L_total - seq_len
        if w_acc[last_start:].min() == 0:
            chunk = torch.tensor(x_all[last_start:]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(chunk).cpu().numpy().flatten()
            preds_acc[last_start:] += out
            w_acc[last_start:] += 1.0

    preds_norm = preds_acc / np.maximum(w_acc, 1.0)

    # inverse scale and low-pass
    preds_scaled = sy.inverse_transform(preds_norm[:len(raw_label)].reshape(-1,1)).flatten()
    preds_filt = lowpass_filter(preds_scaled, fs=30, cutoff=3.0)

    # metrics
    corr = pearsonr(raw_label[:len(preds_filt)], preds_filt)[0]
    snr = psnr(raw_label[:len(preds_filt)], preds_filt)
    print(f"FINAL Pearson: {corr:.4f}, PSNR: {snr:.2f} dB")

    # save prediction
    pred_path = os.path.join(out_dir, f"pred_physmamba1d_{mode}.npy")
    np.save(pred_path, preds_filt.astype(np.float32))
    print("Saved prediction:", pred_path)

if __name__ == "__main__":
    # choose mode: "none", "mixup", "cutmix"
    main(mode="mixup", alpha=0.2)
