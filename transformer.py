'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- Dataset Definition ---
class RPPGSequenceDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.labels[idx:idx+self.seq_len]
        if x.ndim == 1:
            x = x[:, None]
        return torch.from_numpy(x), torch.from_numpy(y)

# --- Model Components ---
class SignalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim=768, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(seq_len, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))

    def forward(self, x):
        B, L, F = x.shape
        x = x.permute(0, 2, 1)  # (B, F, L)
        device = x.device
        embeds = [self.proj(x[:, i, :].to(device)).unsqueeze(1) for i in range(F)]
        tokens = torch.cat(embeds, dim=1)
        cls = self.cls_token.expand(B, -1, -1).to(device)
        pos = self.pos_embed.to(device)
        return torch.cat([cls, tokens], dim=1) + pos

class RadiantTransformer(nn.Module):
    def __init__(self, seq_len, embed_dim=768, num_heads=12, depth=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(dropout)
                )
            }))
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, seq_len)
        )

    def forward(self, x):
        for layer in self.layers:
            x_norm = layer['norm1'](x)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + layer['mlp'](layer['norm2'](x))
        cls_feat = x[:, 0, :]
        return self.head(cls_feat).unsqueeze(-1).squeeze(-1)

class RadiantModel(nn.Module):
    def __init__(self, seq_len, dropout=0.1):
        super().__init__()
        self.embedding = SignalEmbedding(seq_len, dropout=dropout)
        self.transformer = RadiantTransformer(seq_len, dropout=dropout)

    def forward(self, x):
        x = self.embedding(x)
        return self.transformer(x)

# --- Utility ---
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(np.max(y_true) / np.sqrt(mse))

# --- Training Script ---
def train_radiant(
    pred_path='/code/rppg/real data/PURE_POS_prediction.npy',
    label_path='/code/rppg/real data/PURE_label.npy',
    seq_len=300,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    epochs=80,
    clip_norm=1.0,
    dropout=0.1
):
    pred = np.load(pred_path)
    label = np.load(label_path)
    split = int(len(pred) * 0.8)
    train_x, val_x = pred[:split], pred[split:]
    train_y, val_y = label[:split], label[split:]

    train_loader = DataLoader(RPPGSequenceDataset(train_x, train_y, seq_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RPPGSequenceDataset(val_x, val_y, seq_len), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RadiantModel(seq_len, dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            tr_loss += loss.item()
        train_losses.append(tr_loss / len(train_loader))

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_loss += criterion(model(x), y).item()
        val_losses.append(v_loss / len(val_loader))

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")
        scheduler.step(val_losses[-1])
        torch.save(model.state_dict(), f'/code/rppg/newdata/radiant_epoch_{epoch}.pth')

    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.show()

    model.load_state_dict(torch.load(f'/code/rppg/newdata/radiant_epoch_{epochs}.pth'))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(pred) - seq_len + 1, seq_len):
            seg = torch.from_numpy(pred[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            preds.extend(model(seg).cpu().numpy().flatten())
    preds = np.array(preds)

    print('Final PSNR:', psnr(label, preds))
    corr, _ = pearsonr(label.flatten(), preds.flatten())
    print('Final Pearson Correlation:', corr)

    np.save('/code/rppg/newdata/Radiant_rPPG_prediction.npy', preds)

if __name__ == '__main__':
    train_radiant()
'''

'''    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch.nn.functional as F

# --- Dataset Definition ---
class RPPGSequenceDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.labels[idx:idx+self.seq_len]
        if x.ndim == 1:
            x = x[:, None]
        return torch.from_numpy(x), torch.from_numpy(y)

# --- Model Components ---
class SignalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim=768, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(seq_len, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))

    def forward(self, x):
        B, L, F = x.shape
        x = x.permute(0, 2, 1)
        device = x.device
        embeds = [self.proj(x[:, i, :].to(device)).unsqueeze(1) for i in range(F)]
        tokens = torch.cat(embeds, dim=1)
        cls = self.cls_token.expand(B, -1, -1).to(device)
        pos = self.pos_embed.to(device)
        return torch.cat([cls, tokens], dim=1) + pos

class RadiantTransformer(nn.Module):
    def __init__(self, seq_len, embed_dim=768, num_heads=12, depth=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(dropout)
                )
            }))
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, seq_len)
        )

    def forward(self, x):
        for layer in self.layers:
            x_norm = layer['norm1'](x)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + layer['mlp'](layer['norm2'](x))
        cls_feat = x[:, 0, :]
        return self.head(cls_feat).unsqueeze(-1).squeeze(-1)

class RadiantModel(nn.Module):
    def __init__(self, seq_len, dropout=0.1):
        super().__init__()
        self.embedding = SignalEmbedding(seq_len, dropout=dropout)
        self.transformer = RadiantTransformer(seq_len, dropout=dropout)

    def forward(self, x):
        x = self.embedding(x)
        return self.transformer(x)

# --- Utility Functions ---
def psnr(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(np.max(y_true) / np.sqrt(mse))

def compute_pearson(pred, target):
    vx = pred - pred.mean(dim=1, keepdim=True)
    vy = target - target.mean(dim=1, keepdim=True)
    num = (vx * vy).sum(dim=1)
    den = torch.sqrt((vx * vx).sum(dim=1) * (vy * vy).sum(dim=1))
    return (num / (den + 1e-8)).mean().item()

# --- Training and Prediction Script ---
def train_and_predict(
    pred_path='/code/rppg/real data/PURE_POS_prediction.npy',
    label_path='/code/rppg/real data/PURE_label.npy',
    save_pred_path='/code/rppg/newdata/Radiant_rPPG_prediction.npy',
    best_model_path='/code/rppg/newdata/best_radiant_model.pth',
    seq_len=300,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-4,
    epochs=100,
    clip_norm=1.0,
    dropout=0.1
):
    raw_pred = np.load(pred_path)
    raw_label = np.load(label_path)
    X = torch.from_numpy(raw_pred).float()
    Y = torch.from_numpy(raw_label).float()
    assert X.shape == Y.shape

    n = len(X)
    split = int(0.8 * n)
    train_X, val_X = X[:split], X[split:]
    train_Y, val_Y = Y[:split], Y[split:]

    train_loader = DataLoader(RPPGSequenceDataset(train_X.numpy(), train_Y.numpy(), seq_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RPPGSequenceDataset(val_X.numpy(), val_Y.numpy(), seq_len), batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RadiantModel(seq_len, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        psnr_sum = 0.0
        corr_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
                psnr_sum += psnr(y.cpu().numpy().flatten(), out.cpu().numpy().flatten()) * x.size(0)
                corr_sum += compute_pearson(out, y) * x.size(0)
        val_loss /= len(val_loader.dataset)
        psnr_avg = psnr_sum / len(val_loader.dataset)
        corr_avg = corr_sum / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | PSNR: {psnr_avg:.2f} | Corr: {corr_avg:.4f}")
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_model_path)

    # --- 학습 곡선 시각화 ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 전체 시퀀스 예측 및 저장 (Best Model 기준) ---
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, n-seq_len+1, seq_len):
            seg = torch.from_numpy(raw_pred[i:i+seq_len]).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(seg)
            preds.extend(out.cpu().numpy().flatten())
    preds = np.array(preds)
    np.save(save_pred_path, preds)
    print(f"Saved predictions to {save_pred_path}")


if __name__ == '__main__':
    train_and_predict()
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch.nn.functional as F

# --- Dataset Definition ---
class RPPGSequenceDataset(Dataset):
    def __init__(self, data, labels, seq_len=300):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.labels[idx:idx+self.seq_len]
        if x.ndim == 1:
            x = x[:, None]
        return torch.from_numpy(x), torch.from_numpy(y)

# --- Model Components ---
class SignalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim=768, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(seq_len, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos_embed shape matches cls + #features (here features typically 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))

    def forward(self, x):
        B, L, F = x.shape  # x: (B, L, F)
        x = x.permute(0, 2, 1)  # (B, F, L)
        device = x.device
        embeds = [self.proj(x[:, i, :].to(device)).unsqueeze(1) for i in range(F)]
        tokens = torch.cat(embeds, dim=1)            # (B, F, embed_dim)
        cls = self.cls_token.expand(B, -1, -1).to(device)
        pos = self.pos_embed.to(device)
        return torch.cat([cls, tokens], dim=1) + pos # (B, 1+F, embed_dim)

class RadiantTransformer(nn.Module):
    def __init__(self, seq_len, embed_dim=768, num_heads=12, depth=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(dropout)
                )
            }))
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, seq_len)
        )

    def forward(self, x):
        for layer in self.layers:
            x_norm = layer['norm1'](x)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + layer['mlp'](layer['norm2'](x))
        cls_feat = x[:, 0, :]
        return self.head(cls_feat).unsqueeze(-1).squeeze(-1)  # (B, seq_len)

class RadiantModel(nn.Module):
    def __init__(self, seq_len, dropout=0.1):
        super().__init__()
        self.embedding = SignalEmbedding(seq_len, dropout=dropout)
        self.transformer = RadiantTransformer(seq_len, dropout=dropout)

    def forward(self, x):
        x = self.embedding(x)      # (B, 1+F, embed_dim)
        return self.transformer(x) # (B, seq_len)

# --- Utilities ---
def psnr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = ((y_true - y_pred)**2).mean()
    return float('inf') if mse == 0 else 20 * np.log10(np.max(y_true) / np.sqrt(mse))

def compute_pearson(pred, target):
    # pred, target: tensors (B, seq_len)
    vx = pred - pred.mean(dim=1, keepdim=True)
    vy = target - target.mean(dim=1, keepdim=True)
    num = (vx * vy).sum(dim=1)
    den = torch.sqrt((vx * vx).sum(dim=1) * (vy * vy).sum(dim=1))
    return (num / (den + 1e-8)).mean().item()

# --- Augmentations (batch-level) ---
def mixup_batch(x, y, alpha=0.5):
    """x: (B, L, F), y: (B, L)"""
    B = x.size(0)
    if B <= 1 or alpha <= 0.0:
        return x, y
    perm = torch.randperm(B, device=x.device)
    lam = np.random.beta(alpha, alpha)
    lam_t = torch.tensor(lam, dtype=x.dtype, device=x.device)
    x_perm = x[perm]
    y_perm = y[perm]
    x_m = lam_t * x + (1.0 - lam_t) * x_perm
    y_m = lam_t * y + (1.0 - lam_t) * y_perm
    return x_m, y_m

def cutmix_batch(x, y, alpha=0.5):
    """1D CutMix on time axis. x: (B,L,F), y: (B,L)"""
    B, T = x.size(0), x.size(1)
    if B <= 1 or T <= 1 or alpha <= 0.0:
        return x, y
    perm = torch.randperm(B, device=x.device)
    lam = np.random.beta(alpha, alpha)
    cut_len = int(round(T * (1.0 - lam)))
    cut_len = max(0, min(cut_len, T))
    if cut_len == 0:
        return x, y
    max_start = T - cut_len
    start = np.random.randint(0, max_start + 1) if max_start >= 0 else 0
    x_m = x.clone()
    y_m = y.clone()
    x_m[:, start:start+cut_len, :] = x[perm, start:start+cut_len, :]
    y_m[:, start:start+cut_len] = y[perm, start:start+cut_len]
    return x_m, y_m

# --- Training + prediction per mode ---
def run_experiment(mode, raw_pred, raw_label, seq_len=300, batch_size=32, epochs=100,
                   lr=3e-4, weight_decay=1e-4, device=None, save_prefix='/code/rppg2ppg/newdata/Radiant'):
    """
    mode: 'baseline' | 'mixup' | 'cutmix'
    returns: path to saved prediction .npy and metrics
    """
    assert mode in ( 'mixup', 'cutmix')
    alpha = 0.5  # safe for rPPG
    n = len(raw_pred)
    split = int(0.8 * n)
    train_X, val_X = raw_pred[:split], raw_pred[split:]
    train_Y, val_Y = raw_label[:split], raw_label[split:]

    train_loader = DataLoader(RPPGSequenceDataset(train_X, train_Y, seq_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RPPGSequenceDataset(val_X, val_Y, seq_len), batch_size=batch_size, shuffle=False)

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RadiantModel(seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val = float('inf')
    best_path = f"{save_prefix}_best_{mode}.pth"
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)           # (B, L, F)
            yb = yb.to(device)           # (B, L)
            if mode == 'mixup':
                xb, yb = mixup_batch(xb, yb, alpha=alpha)
            elif mode == 'cutmix':
                xb, yb = cutmix_batch(xb, yb, alpha=alpha)
            optimizer.zero_grad()
            out = model(xb)              # (B, L)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        psnr_sum = 0.0
        corr_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                # per-sample psnr & pearson: use cpu numpy for psnr, compute_pearson for corr
                out_cpu = out.cpu().numpy()
                y_cpu = yb.cpu().numpy()
                # psnr: compute per-sample (flatten) then sum
                for i in range(out_cpu.shape[0]):
                    psnr_sum += psnr(y_cpu[i].flatten(), out_cpu[i].flatten()) * 1.0
                corr_sum += compute_pearson(out, yb) * xb.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        psnr_avg = psnr_sum / len(val_loader.dataset)
        corr_avg = corr_sum / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"[{mode}] Epoch {epoch}/{epochs}  Train: {train_loss:.6f}, Val: {val_loss:.6f}, PSNR: {psnr_avg:.2f}, Corr: {corr_avg:.4f}")
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    # plot losses
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title(f"Loss ({mode})")
    plt.legend()
    plt.savefig(f"{save_prefix}_loss_{mode}.png")
    plt.close()

    # load best model and predict over full raw sequence (stride = seq_len)
    model.load_state_dict(torch.load(best_path))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, n - seq_len + 1, seq_len):
            seg = torch.from_numpy(raw_pred[i:i+seq_len]).float().unsqueeze(0)
            if seg.ndim == 3 and seg.shape[-1] == 1:
                pass
            else:
                seg = seg.unsqueeze(-1)
            seg = seg.to(device)
            out = model(seg)  # (1, seq_len)
            preds.extend(out.cpu().numpy().flatten())

    preds = np.array(preds, dtype=np.float32)
    pred_path = f"{save_prefix}_pred_{mode}.npy"
    np.save(pred_path, preds)
    print(f"[{mode}] Saved prediction -> {pred_path}")

    # compute overall metrics against available labels (truncate to preds length)
    L = min(len(preds), len(raw_label))
    overall_psnr = psnr(raw_label[:L], preds[:L])
    overall_corr = pearsonr(raw_label[:L].flatten(), preds[:L].flatten())[0]
    print(f"[{mode}] Final Pearson: {overall_corr:.4f}, PSNR: {overall_psnr:.2f} dB")

    return pred_path, best_path, overall_corr, overall_psnr

# --- Main: run baseline / mixup / cutmix and produce .npy files ---
if __name__ == '__main__':
    pred_path = '/code/rppg2ppg/real data/PURE_POS_prediction.npy'
    label_path = '/code/rppg2ppg/real data/PURE_label.npy'
    save_prefix = '/code/rppg2ppg/newdata/Radiant'  # files will be Radiant_pred_{mode}.npy etc.

    raw_pred = np.load(pred_path).astype(np.float32).flatten()
    raw_label = np.load(label_path).astype(np.float32).flatten()
    # sanity: shapes equal
    if raw_pred.shape != raw_label.shape:
        raise ValueError("Prediction and label arrays must have same shape")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 300

    # mixup
    run_experiment('mixup', raw_pred, raw_label, seq_len=seq_len, batch_size=32, epochs=100,
                   lr=3e-4, weight_decay=1e-4, device=device, save_prefix=save_prefix)

    # cutmix
    run_experiment('cutmix', raw_pred, raw_label, seq_len=seq_len, batch_size=32, epochs=100,
                   lr=3e-4, weight_decay=1e-4, device=device, save_prefix=save_prefix)

    print("All experiments finished. Predictions saved with suffixes: baseline / mixup / cutmix.")
