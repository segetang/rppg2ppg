'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch.nn.functional as F

# ---- Dataset ----
class RPPGDataset(Dataset):
    def __init__(self, pred_path, label_path, seq_len=300):
        self.x = np.load(pred_path).astype(np.float32)
        self.y = np.load(label_path).astype(np.float32)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.x) - self.seq_len + 1
    def __getitem__(self, idx):
        xi = self.x[idx:idx+self.seq_len]
        yi = self.y[idx:idx+self.seq_len]
        return torch.tensor(xi).unsqueeze(-1), torch.tensor(yi)

# ---- Dynamic Disentangle ----
def disentangle(signal):
    s = signal.squeeze(-1)
    diff = s[:, 1:] - s[:, :-1]
    trend = diff / torch.sqrt(diff**2 + 1.0)
    amp = diff.abs()
    trend = torch.cat([trend, trend[:, -1:]], dim=1)
    amp = torch.cat([amp, amp[:, -1:]], dim=1)
    return trend.unsqueeze(-1), amp.unsqueeze(-1)

# ---- PhysDiff Transformer Model ----
class PhysDiffTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)
    def forward(self, x):
        trend, amp = disentangle(x)
        cond = torch.cat([x, trend, amp], dim=-1)
        h = F.relu(self.input_proj(cond))
        h = self.transformer(h)
        out = self.output_fc(h).squeeze(-1)
        return out

# ---- Training & Inference ----
def main():
    pred_path = '/code/rppg/real data/PURE_POS_prediction.npy'
    label_path = '/code/rppg/real data/PURE_label.npy'
    save_model = '/code/rppg/newdata/physdiff_transformer.pth'
    save_pred = '/code/rppg/newdata/physdiff_output.npy'
    seq_len = 300

    dataset = RPPGDataset(pred_path, label_path, seq_len)
    split = int(len(dataset) * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PhysDiffTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float('inf')
    patience = 10
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, 201):
        model.train()
        total_tr = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_tr += loss.item()
        train_loss = total_tr / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total_vl = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                total_vl += criterion(model(x), y).item()
        val_loss = total_vl / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            torch.save(model.state_dict(), save_model)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    # Inference (frame-level accumulation)
    state_dict = torch.load(save_model, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    signal_len = len(np.load(pred_path))  # e.g., 113400
    frame_preds = np.zeros(signal_len)
    frame_counts = np.zeros(signal_len)

    for i in range(len(dataset)):
        x, _ = dataset[i]  # (seq_len, 1)
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x).cpu().numpy().squeeze()  # (seq_len,)
        frame_preds[i:i+seq_len] += out
        frame_counts[i:i+seq_len] += 1

    full_preds = frame_preds / frame_counts
    labels = np.load(label_path).flatten()

    # Shape check
    print("Prediction shape:", full_preds.shape)
    print("Label shape:     ", labels.shape)
    print("Shape match:     ", full_preds.shape == labels.shape)

    # Metrics
    pearson, _ = pearsonr(labels, full_preds)
    mse = np.mean((labels - full_preds)**2)
    psnr = 20 * np.log10(np.max(labels) / np.sqrt(mse))
    print(f"PSNR={psnr:.4f}, Pearson={pearson:.4f}")

    # Save
    os.makedirs(os.path.dirname(save_pred), exist_ok=True)
    np.save(save_pred, full_preds)
    
        # Loss plot
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
'''


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch.nn.functional as F

# ---- Dataset ----
class RPPGDataset(Dataset):
    def __init__(self, pred_path, label_path, seq_len=300):
        self.x = np.load(pred_path).astype(np.float32)
        self.y = np.load(label_path).astype(np.float32)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.x) - self.seq_len + 1
    def __getitem__(self, idx):
        xi = self.x[idx:idx+self.seq_len]
        yi = self.y[idx:idx+self.seq_len]
        return torch.tensor(xi).unsqueeze(-1), torch.tensor(yi)

# ---- Dynamic Disentangle ----
def disentangle(signal):
    s = signal.squeeze(-1)
    diff = s[:, 1:] - s[:, :-1]
    trend = diff / torch.sqrt(diff**2 + 1.0)
    amp = diff.abs()
    trend = torch.cat([trend, trend[:, -1:]], dim=1)
    amp = torch.cat([amp, amp[:, -1:]], dim=1)
    return trend.unsqueeze(-1), amp.unsqueeze(-1)

# ---- PhysDiff Transformer Model ----
class PhysDiffTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)
    def forward(self, x):
        trend, amp = disentangle(x)
        cond = torch.cat([x, trend, amp], dim=-1)
        h = F.relu(self.input_proj(cond))
        h = self.transformer(h)
        out = self.output_fc(h).squeeze(-1)
        return out

# ---- Augmentations (batch-level) ----
def mixup_batch(x, y, alpha=0.5):
    """Batch-level MixUp for sequences.
       x: (B, T, 1), y: (B, T)
    """
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
    """1D CutMix on time axis for sequences.
       x: (B, T, 1), y: (B, T)
    """
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

# ---- Training + Inference helper that preserves original behavior ----
def train_with_aug(pred_path, label_path, save_model, save_pred, seq_len=300,
                   aug_fn=None, aug_name='none', epochs=200, batch_size=32, device=None,
                   patience=10, lr=1e-3):
    """
    aug_fn: function(x,y)->(x_aug,y_aug) or None
    """
    print(f"\n=== Training with augmentation: {aug_name} ===")
    dataset = RPPGDataset(pred_path, label_path, seq_len)
    split = int(len(dataset) * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhysDiffTransformer().to(dev)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        total_tr = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            if aug_fn is not None:
                x, y = aug_fn(x, y)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_tr += loss.item()
        train_loss = total_tr / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total_vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                total_vl += criterion(model(x), y).item()
        val_loss = total_vl / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            os.makedirs(os.path.dirname(save_model), exist_ok=True)
            torch.save(model.state_dict(), save_model)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    # Inference / frame-level accumulation (same logic as original)
    # load best weights
    state_dict = torch.load(save_model)
    model.load_state_dict(state_dict)
    model.eval()

    # prepare for full sequence inference
    raw_len = len(np.load(pred_path))
    frame_preds = np.zeros(raw_len, dtype=np.float32)
    frame_counts = np.zeros(raw_len, dtype=np.float32)

    # iterate over dataset windows (same ordering as dataset)
    for i in range(len(dataset)):
        x, _ = dataset[i]  # (seq_len, 1) tensor
        x = x.unsqueeze(0).to(dev)  # (1, seq_len, 1)
        with torch.no_grad():
            out = model(x).cpu().numpy().squeeze()  # (seq_len,)
        frame_preds[i:i+seq_len] += out
        frame_counts[i:i+seq_len] += 1.0

    # avoid division by zero
    full_preds = frame_preds / np.maximum(frame_counts, 1e-6)
    labels = np.load(label_path).flatten()

    # Save predictions (truncate/align to label length if different)
    L = min(len(full_preds), len(labels))
    final_preds = full_preds[:L]

    os.makedirs(os.path.dirname(save_pred), exist_ok=True)
    np.save(save_pred, final_preds.astype(np.float32))

    # Metrics
    pearson, _ = pearsonr(labels[:L], final_preds)
    mse = np.mean((labels[:L] - final_preds)**2)
    psnr = 20 * np.log10(np.max(labels[:L]) / np.sqrt(mse)) if mse > 0 else float('inf')
    print(f"[{aug_name}] PSNR={psnr:.4f}, Pearson={pearson:.4f}")
    # optional plot losses
    try:
        plt.figure()
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f"Loss ({aug_name})")
        plt.savefig(os.path.join(os.path.dirname(save_pred), f"loss_{aug_name}.png"))
        plt.close()
    except Exception:
        pass

    return save_model, save_pred, pearson, psnr

# ---- Main: run mixup and cutmix (only) ----
if __name__ == '__main__':
    pred_path = '/code/rppg2ppg/real data/PURE_POS_prediction.npy'
    label_path = '/code/rppg2ppg/real data/PURE_label.npy'
    out_dir = '/code/rppg2ppg/newdata'
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_len = 300

    # MixUp run
    mixup_model_path = os.path.join(out_dir, 'physdiff_transformer_mixup.pth')
    mixup_pred_path = os.path.join(out_dir, 'physdiff_output_mixup.npy')
    train_with_aug(pred_path, label_path, mixup_model_path, mixup_pred_path,
                   seq_len=seq_len, aug_fn=lambda x,y: mixup_batch(x,y,alpha=0.5),
                   aug_name='mixup', epochs=300, batch_size=32, device=device, patience=10, lr=1e-3)

    # CutMix run
    cutmix_model_path = os.path.join(out_dir, 'physdiff_transformer_cutmix.pth')
    cutmix_pred_path = os.path.join(out_dir, 'physdiff_output_cutmix.npy')
    train_with_aug(pred_path, label_path, cutmix_model_path, cutmix_pred_path,
                   seq_len=seq_len, aug_fn=lambda x,y: cutmix_batch(x,y,alpha=0.5),
                   aug_name='cutmix', epochs=300, batch_size=32, device=device, patience=10, lr=1e-3)

    print("Done. Predictions saved:")
    print(" - MixUp:", mixup_pred_path)
    print(" - CutMix:", cutmix_pred_path)

