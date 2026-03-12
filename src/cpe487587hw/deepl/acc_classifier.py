import torch 
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from pathlib import Path
import os
import glob
from dataclasses import dataclass
from torch.utils.data import Dataset
import numpy as np
import polars as pl

# define the convertation function from km/h to m/s
def km_to_m(x):
    return x / 3.6

# zoh technique to match the speed time and lable time, and get the corresponding label value
def zoh_labels(speed_time, label_time, label_value):
    idx = np.searchsorted(label_time, speed_time, side="right") - 1
    idx = np.clip(idx, 0, len(label_value) - 1)
    return label_value[idx]

# here is for the historical speed value in the pdf, v_t, v_{t-1}, ..., v_{t-k}
def build_historical_features(speed, k):
    feats = []
    for t in range(k, len(speed)):
        row = speed[t - k : t + 1][::-1]  # [v_t, v_{t-1}, ..., v_{t-k}]
        feats.append(row)
    return np.asarray(feats)


# match the wheel speed and acc status files, from the decoded wheel speed fl.csv and acc status.cv/csv files, to get the corresponding speed and label data for training the classifier
def find_matching_pairs(root_dir):
    wheel_files = sorted(glob.glob(os.path.join(root_dir, "*_CAN_Messages_decoded_wheel_speed_fl.csv")))
    status_files = sorted(glob.glob(os.path.join(root_dir, "*_CAN_Messages_decoded_acc_status.csv")))

    print(f"Found wheel files : {len(wheel_files)}")
    print(f"Found status files: {len(status_files)}")

    status_map = {}
    for sf in status_files:
        key = os.path.basename(sf).replace("_CAN_Messages_decoded_acc_status.csv", "")
        status_map[key] = sf

    pairs = []

    for wf in wheel_files:
        key = os.path.basename(wf).replace("_CAN_Messages_decoded_wheel_speed_fl.csv", "")
        if key in status_map:
            pairs.append((wf, status_map[key]))
    print(f"Matched pairs: {len(pairs)}")

    return pairs

@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

class ACCCruiseDataset(Dataset):
    def __init__(self, root_dir, k, split, split_ratio, normalize, random_seed, stats = None):
        super().__init__()
        assert split in {"train", "val"}

        self.root_dir = root_dir
        self.k = k
        self.split = split
        self.split_ratio = split_ratio
        self.normalize = normalize

        pairs = find_matching_pairs(root_dir)
        all_x, all_y = self.load_all_pairs(pairs, k)

        rng = np.random.default_rng(random_seed)
        idx = np.arange(len(all_x))
        rng.shuffle(idx)
        all_x = all_x[idx]
        all_y = all_y[idx]

        split_idx = int(len(all_x) * split_ratio)
        if split == "train":
            self.x = all_x[:split_idx]
            self.y = all_y[:split_idx]
        else:
            self.x = all_x[split_idx:]
            self.y = all_y[split_idx:]

        if self.normalize:
            if split == "train":
                mean = self.x.mean(axis=0)
                std = self.x.std(axis=0)
                std = np.where(std < 1e-8, 1.0, std)
                self.stats = NormalizationStats(mean=mean, std=std)
            else: 
                self.stats = stats
            self.x = (self.x - self.stats.mean) / self.stats.std

    def read_speed_csv(self, path):
        df = (
            pl.read_csv(path)
            .select(["Time", "Message"])
            .with_columns([
                pl.col("Time").cast(pl.Float64),
                pl.col("Message").cast(pl.Float32),
            ])
        )

        speed_t = df["Time"].to_numpy()
        speed_v = df["Message"].to_numpy()
        speed_v = km_to_m(speed_v.astype(np.float32))
        return speed_t, speed_v

    def read_status_csv(self, path):
        df = (
            pl.read_csv(path)
            .select(["Time", "Message"])
            .with_columns([
                pl.col("Time").cast(pl.Float64),
                pl.col("Message").cast(pl.Int64, strict=False).fill_null(-1),
            ])
        )

        label_t = df["Time"].to_numpy()
        label_raw = df["Message"].to_numpy().astype(np.int64)
        label_bin = (label_raw == 6).astype(np.float32)
        return label_t, label_bin

    def load_all_pairs(self, pairs, k):
        xs = []
        ys = []

        for speed_file, status_file in pairs:
            speed_t, speed_v = self.read_speed_csv(speed_file)
            label_t, label_bin = self.read_status_csv(status_file)

            aligned_y = zoh_labels(speed_t, label_t, label_bin)

            feat = build_historical_features(speed_v, k=k)
            target = aligned_y[k:].astype(np.float32)

            if len(feat) != len(target):
                n = min(len(feat), len(target))
                feat = feat[:n]
                target = target[:n]

            xs.append(feat)
            ys.append(target.reshape(-1, 1))

        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.float32)
        return x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ACCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block1 = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.block2 = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x


class DiceLoss(nn.Module):
    def __init__(self, smooth):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.float().view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class ACCTrainer:
    def __init__(self, model, device, loss_fn, optimizer, scheduler=None):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs

            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()

            if batch_idx % 50 == 0:
                batch_acc = (preds == y).float().mean().item()
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(loader)}, "
                    f"Loss: {loss.item():.4f}, Train Acc: {batch_acc:.4f}"
                )

        epoch_loss = total_loss / len(loader.dataset)
        epoch_acc = total_correct / total_samples
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            bs = y.size(0)
            total_loss += loss.item() * bs

            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()

        val_loss = total_loss / len(loader.dataset)
        val_acc = total_correct / total_samples
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        return val_loss, val_acc

    def fit(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            tr_loss, tr_acc = self.train_one_epoch(train_loader, epoch)
            va_loss, va_acc = self.validate(val_loader)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
                f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.4f}"
            )

            if self.scheduler is not None:
                self.scheduler.step()

    def export_onnx(self, file_name="acc_model.onnx", input_dim: int = 11):
        self.model.eval()
        dummy_input = torch.randn(1, input_dim, device=self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            file_name,
            input_names=["input"],
            output_names=["logits"],
            export_params=True,
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=12,
        )