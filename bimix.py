import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from mamba_ssm import Mamba

def lowpass_filter(signal, fs=30, cutoff=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

class RPPGDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.x = torch.from_numpy(data).float().unsqueeze(-1)
        self.y = torch.from_numpy(labels).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len + 1

    def __getitem__(self, idx):
        return self.x[idx:idx+self.seq_len], self.y[idx:idx+self.seq_len]

class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, dropout):
        super().__init__()
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_rev = torch.flip(x, dims=[1])
        f = self.fwd(x)
        b = torch.flip(self.bwd(x_rev), dims=[1])
        out = (f + b) / 2
        return self.norm(self.drop(out) + x)

class BiMambaRPPGModel(nn.Module):
    def __init__(self, d_model=128, d_state=32, d_conv=8, n_blocks=6, dropout=0.1, conv_kernel=7):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        self.pre_act = nn.GELU()
        self.blocks = nn.ModuleList([BiMambaBlock(d_model, d_state, d_conv, dropout) for _ in range(n_blocks)])
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.pre_act(self.pre_conv(x.permute(0,2,1))).permute(0,2,1)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x).squeeze(-1)

def mixup_batch(x, y, alpha=1.0):
    B = x.size(0)
    perm = torch.randperm(B, device=x.device)
    lam = np.random.beta(alpha, alpha)
    return lam*x + (1-lam)*x[perm], lam*y + (1-lam)*y[perm]

def cutmix_batch(x, y, alpha=1.0):
    B, T = x.size(0), x.size(1)
    if T <= 1:
        return x, y
    perm = torch.randperm(B, device=x.device)
    lam = np.random.beta(alpha, alpha)
    L = int((1-lam)*T)
    start = np.random.randint(0, T - L + 1)
    xm = x.clone()
    xm[:, start:start+L] = x[perm, start:start+L]
    ym = lam*y + (1-lam)*y[perm]
    return xm, ym

def psnr(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse==0 else 20*np.log10(y_true.max()/np.sqrt(mse))

def train_and_eval(name, aug_fn, x_all, y_all, seq_len, device, save_dir):
    print(f" [{name.upper()}] 시작")
    train_n = int(0.8*len(x_all))
    tr_x, te_x = x_all[:train_n], x_all[train_n:]
    tr_y, te_y = y_all[:train_n], y_all[train_n:]

    tr_ds = RPPGDataset(tr_x, tr_y, seq_len)
    te_ds = RPPGDataset(te_x, te_y, seq_len)
    tr_ld = DataLoader(tr_ds, 32, shuffle=True, pin_memory=True)
    te_ld = DataLoader(te_ds, 32, shuffle=False, pin_memory=True)

    model = BiMambaRPPGModel().to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    crit = nn.SmoothL1Loss()

    best_loss, wait = float('inf'), 0
    tr_hist, va_hist = [], []
    for ep in range(1, 101):
        model.train()
        tr_l = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            xb, yb = aug_fn(xb, yb)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_l += loss.item()
        sched.step()
        tr_hist.append(tr_l/len(tr_ld))

        model.eval()
        va_l = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                va_l += crit(model(xb), yb).item()
        va_hist.append(va_l/len(te_ld))
        print(f"[{name}][{ep}/80] Train: {tr_hist[-1]:.4f} | Val: {va_hist[-1]:.4f}")

        if va_hist[-1] < best_loss - 1e-6:
            best_loss, wait = va_hist[-1], 0
            torch.save(model.state_dict(), f"{save_dir}/{name}.pth")
        else:
            wait += 1
            if wait > 10:
                print(" 조기 종료")
                break

    # Loss 플롯 저장
    plt.figure()
    plt.plot(tr_hist, label='Train')
    plt.plot(va_hist, label='Val')
    plt.title(name)
    plt.legend()
    plt.savefig(f"{save_dir}/{name}_loss.png")
    plt.close()

    # 예측 병합
    model.load_state_dict(torch.load(f"{save_dir}/{name}.pth", weights_only=True))
    model.eval()
    L = len(x_all)
    step = seq_len // 2
    accum = np.zeros(L)
    wsum = np.zeros(L)

    with torch.no_grad():
        for i in range(0, L - seq_len + 1, step):
            chunk = torch.tensor(x_all[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(chunk).cpu().numpy().flatten()
            accum[i:i+seq_len] += out
            wsum[i:i+seq_len] += 1

    # tail 처리
    if wsum[-seq_len:].min() == 0:
        i = L - seq_len
        chunk = torch.tensor(x_all[i:]).float().unsqueeze(0).unsqueeze(-1).to(device)
        out = model(chunk).cpu().numpy().flatten()
        accum[i:] += out
        wsum[i:] += 1

    pred_norm = accum / np.maximum(wsum, 1e-6)
    # 마지막 테스트 길이에 맞게 자르기
    pred_f = sy.inverse_transform(pred_norm[:len(y_all)].reshape(-1,1)).flatten()
    pred_f = lowpass_filter(pred_f)
    raw_label = sy.inverse_transform(y_all.reshape(-1,1)).flatten()

    corr = pearsonr(raw_label[:len(pred_f)], pred_f)[0]
    snr = psnr(raw_label[:len(pred_f)], pred_f)
    np.save(f"{save_dir}/{name}_pred.npy", pred_f.astype(np.float32))

    print(f"{name}: Pearson {corr:.4f}, PSNR {snr:.2f} dB")
    return corr, snr

if __name__ == "__main__":
    raw_x = np.load('/mnt/c/code/rppg/real data/PURE_POS_prediction.npy').reshape(-1,1)
    raw_y = np.load('/mnt/c/code/rppg/real data/PURE_label.npy').reshape(-1,1)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    x_all = sx.fit_transform(raw_x).flatten()
    y_all = sy.fit_transform(raw_y).flatten()

    seq_len = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = '/mnt/c/code/rppg/newdata'
    os.makedirs(save_dir, exist_ok=True)

    corr_c, snr_c = train_and_eval("cutmix", cutmix_batch, x_all, y_all, seq_len, device, save_dir)
    corr_m, snr_m = train_and_eval("mixup", mixup_batch, x_all, y_all, seq_len, device, save_dir)

    print(" 비교 결과")
    print(f"CutMix → Pearson {corr_c:.4f}, PSNR {snr_c:.2f} dB")
    print(f"MixUp  → Pearson {corr_m:.4f}, PSNR {snr_m:.2f} dB")
