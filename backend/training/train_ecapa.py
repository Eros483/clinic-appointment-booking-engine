# ----- Small ECAPA-TDNN training for lang ID @ backend/training/train_ecapa.py -----

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from backend.training.build_dataset import TARGET_LANGS, TARGET_SR, build_dataset
from backend.training.augment import random_augment

N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
C = 512
NUM_CLASSES = len(TARGET_LANGS)
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


# ---------------------------------------------------------------------------
# Mel-spectrogram extraction (pure numpy, used in data loading)
# ---------------------------------------------------------------------------


def _mel_filterbank(sr: int, n_mels: int, n_fft: int) -> np.ndarray:
    low = 0.0
    high = sr / 2
    mel_low = 2595.0 * math.log10(1.0 + low / 700.0)
    mel_high = 2595.0 * math.log10(1.0 + high / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / (right - center)
    return fb


class LogMelExtractor:
    _fb: np.ndarray | None = None

    @classmethod
    def _ensure_fb(cls, sr: int = TARGET_SR) -> np.ndarray:
        if cls._fb is None:
            cls._fb = _mel_filterbank(sr, N_MELS, N_FFT)
        return cls._fb

    @classmethod
    def compute(cls, audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        fb = cls._ensure_fb(sr)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        if sr != TARGET_SR:
            import scipy.signal

            audio = scipy.signal.resample_poly(audio, TARGET_SR, sr)

        n_fft = N_FFT
        hop = HOP_LENGTH
        win = WIN_LENGTH
        window = np.hanning(win).astype(np.float32)

        n_frames = 1 + (audio.shape[-1] - win) // hop
        stft = np.zeros((audio.shape[0], n_fft // 2 + 1, n_frames), dtype=np.complex64)
        for i in range(n_frames):
            start = i * hop
            segment = audio[:, start : start + win] * window[np.newaxis, :]
            spec = np.fft.rfft(segment, n=n_fft)
            stft[:, :, i] = spec

        power = np.abs(stft) ** 2
        mel = fb @ power
        log_mel = np.log(np.clip(mel, 1e-10, None))
        return log_mel.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LangIdDataset(Dataset):
    def __init__(self, samples: dict[str, list[np.ndarray]], augment: bool = True):
        self.items: list[tuple[np.ndarray, int]] = []
        self.augment = augment
        for lang_code, audios in samples.items():
            label_idx = TARGET_LANGS.index(lang_code)
            for audio in audios:
                self.items.append((audio, label_idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio, label = self.items[idx]
        if self.augment:
            audio = random_augment(audio, TARGET_SR, prob=0.6)
        mel = LogMelExtractor.compute(audio, TARGET_SR)
        mel_t = torch.from_numpy(mel[0])
        label_t = torch.tensor(label, dtype=torch.long)
        return mel_t, label_t


# ---------------------------------------------------------------------------
# ECAPA-TDNN components
# ---------------------------------------------------------------------------


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool1d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class SE_Res2Block(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, scale: int = 8, dilation: int = 1
    ):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.convs = nn.ModuleList()
        for _ in range(scale):
            self.convs.append(
                nn.Conv1d(
                    out_channels // scale,
                    out_channels // scale,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=out_channels // scale,
                )
            )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.se = SEModule(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        sc = torch.chunk(x, self.scale, dim=1)
        out = []
        for i, conv in enumerate(self.convs):
            if i == 0:
                out.append(conv(sc[i]))
            else:
                out.append(conv(sc[i] + out[i - 1]))
        x = torch.cat(out, dim=1)
        x = self.bn2(x)

        x = self.bn3(self.conv3(x))
        x = self.se(x)
        return F.relu(x + residual)


class AttentiveStatsPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.linear(x))
        mean = (attn * x).sum(dim=-1) / attn.sum(dim=-1).clamp(min=1e-10)
        std = (attn * (x - mean.unsqueeze(-1)) ** 2).sum(dim=-1).clamp(min=0).sqrt()
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, channels: int = C, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(N_MELS, channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)

        self.block1 = SE_Res2Block(channels, channels, scale=8, dilation=2)
        self.block2 = SE_Res2Block(channels, channels, scale=8, dilation=3)
        self.block3 = SE_Res2Block(channels, channels, scale=8, dilation=4)

        self.pool = AttentiveStatsPool(channels)
        self.bn_pool = nn.BatchNorm1d(channels * 2)

        self.fc = nn.Linear(channels * 2, channels)
        self.bn_fc = nn.BatchNorm1d(channels)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(mel)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.bn_pool(x)
        x = F.relu(self.bn_fc(self.fc(x)))
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mels, labels = zip(*batch)
    max_len = max(m.shape[-1] for m in mels)
    padded = []
    lengths = []
    for m in mels:
        lengths.append(m.shape[-1])
        pad = max_len - m.shape[-1]
        padded.append(F.pad(m, (0, pad), "constant", 0))
    return torch.stack(padded), torch.tensor(lengths), torch.stack(labels)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mels, lengths, labels in loader:
        mels = mels.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(mels)
        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, dict[str, float]]:
    model.eval()
    correct = 0
    total = 0
    per_lang_correct: dict[str, int] = {}
    per_lang_total: dict[str, int] = {}

    for lang in TARGET_LANGS:
        per_lang_correct[lang] = 0
        per_lang_total[lang] = 0

    for mels, lengths, labels in loader:
        mels = mels.to(device)
        labels = labels.to(device)
        logits = model(mels)
        preds = logits.argmax(dim=-1)

        for i in range(len(labels)):
            lang = TARGET_LANGS[labels[i].item()]
            per_lang_total[lang] += 1
            if preds[i] == labels[i]:
                correct += 1
                per_lang_correct[lang] += 1
            total += 1

    acc = correct / max(total, 1) * 100
    per_lang_acc = {
        lang: per_lang_correct[lang] / max(per_lang_total[lang], 1) * 100
        for lang in TARGET_LANGS
    }
    return acc, per_lang_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples-per-lang", type=int, default=800)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_label = (
        f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
    )
    print(f"Using device: {device}{cuda_label}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building dataset...")
    splits = build_dataset(
        samples_per_lang=args.samples_per_lang,
        force_rebuild=args.force_rebuild,
    )

    train_dataset = LangIdDataset(splits["train"], augment=True)
    val_dataset = LangIdDataset(splits["val"], augment=False)
    test_dataset = LangIdDataset(splits["test"], augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )
    print(f"Device: {device}")

    model = ECAPA_TDNN(channels=C, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = 5
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        epoch = step / len(train_loader)
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    patience = 7
    patience_counter = 0

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)

        lr_now = optimizer.param_groups[0]["lr"]

        val_acc, per_lang = evaluate(model, val_loader, device)

        epoch_time = time.time() - t0

        per_lang_str = " | ".join(
            f"{lang}: {per_lang[lang]:.1f}%" for lang in TARGET_LANGS
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss: {train_loss:.4f} | lr: {lr_now:.2e} | "
            f"val acc: {val_acc:.2f}% | {per_lang_str} | "
            f"{epoch_time:.1f}s"
        )

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )
            print(f"  → Saved best model ({val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val accuracy: {best_val_acc:.2f}%")

    print("\nRunning final test evaluation...")
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_acc, test_per_lang = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2f}%")
    for lang in TARGET_LANGS:
        print(f"  {lang}: {test_per_lang[lang]:.1f}%")


if __name__ == "__main__":
    main()
