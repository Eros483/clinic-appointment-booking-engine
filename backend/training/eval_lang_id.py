# ----- Evaluate lang ID model (matches Kaggle Cell 10) @ backend/training/eval_lang_id.py -----

import argparse
import math
import time
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

TARGET_LANGS = ["hi", "en", "ta", "bn", "mr"]
NUM_CLASSES = len(TARGET_LANGS)
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
TARGET_SR = 16000
C = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng(42)


# ---- telephony degradation (for --degraded flag) ----
def _mu_law_compress(x):
    mu = 255.0
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def _mu_law_expand(y):
    mu = 255.0
    return np.sign(y) * (1.0 / mu) * (np.exp(np.abs(y) * np.log1p(mu)) - 1.0)


def simulate_telephony(audio, sr=TARGET_SR):
    if sr != 8000:
        audio = scipy.signal.resample_poly(audio, 8000, sr)
    sos = scipy.signal.butter(4, [300, 3400], btype="band", fs=8000, output="sos")
    audio = scipy.signal.sosfilt(sos, audio)
    audio = _mu_law_compress(audio)
    audio = _mu_law_expand(audio)
    noise = rng.normal(0, 0.002, audio.shape)
    audio = audio + noise
    frame_len = int(0.020 * 8000)
    for start in range(0, len(audio) - frame_len + 1, frame_len):
        if rng.random() < 0.02:
            audio[start : start + frame_len] = 0.0
    return scipy.signal.resample_poly(audio, TARGET_SR, 8000)


# ---- model definition (must match training) ----
class SEModule(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool1d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class SE_Res2Block(nn.Module):
    def __init__(self, in_ch, out_ch, scale=8, dilation=1):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    out_ch // scale,
                    out_ch // scale,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=out_ch // scale,
                )
                for _ in range(scale)
            ]
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv3 = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.se = SEModule(out_ch)
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        res = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        sc = torch.chunk(x, self.scale, dim=1)
        out = []
        for i, conv in enumerate(self.convs):
            inp = sc[i] if i == 0 else sc[i] + out[i - 1]
            out.append(conv(inp))
        x = torch.cat(out, dim=1)
        x = self.bn2(x)
        x = self.bn3(self.conv3(x))
        x = self.se(x)
        return F.relu(x + res)


class AttentiveStatsPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        attn = torch.sigmoid(self.linear(x))
        mean = (attn * x).sum(dim=-1) / attn.sum(dim=-1).clamp(min=1e-10)
        std = (attn * (x - mean.unsqueeze(-1)) ** 2).sum(dim=-1).clamp(min=0).sqrt()
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, channels=C):
        super().__init__()
        self.conv1 = nn.Conv1d(N_MELS, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.block1 = SE_Res2Block(channels, channels, scale=8, dilation=2)
        self.block2 = SE_Res2Block(channels, channels, scale=8, dilation=3)
        self.block3 = SE_Res2Block(channels, channels, scale=8, dilation=4)
        self.pool = AttentiveStatsPool(channels)
        self.bn_pool = nn.BatchNorm1d(channels * 2)
        self.fc = nn.Linear(channels * 2, channels)
        self.bn_fc = nn.BatchNorm1d(channels)
        self.classifier = nn.Linear(channels, NUM_CLASSES)

    def forward(self, mel):
        x = F.relu(self.bn1(self.conv1(mel)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.bn_pool(x)
        x = F.relu(self.bn_fc(self.fc(x)))
        return self.classifier(x)


# ---- mel transform ----
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=0.0,
).to(DEVICE)


def audio_to_log_mel(audio_t):
    mel = mel_transform(audio_t)
    return torch.log(torch.clamp(mel, min=1e-10))


# ---- dataset (mirrors notebook Cell 7) ----
class LangIdEvalDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.items = [
            (a, TARGET_LANGS.index(l)) for l in TARGET_LANGS for a in samples[l]
        ]
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio, label = self.items[idx]
        if self.augment:
            audio = simulate_telephony(audio)
        audio_t = torch.from_numpy(audio).float().to(DEVICE)
        return audio_to_log_mel(audio_t), torch.tensor(label, device=DEVICE)


def collate_fn(batch):
    mels, labels = zip(*batch)
    max_len = max(m.shape[-1] for m in mels)
    padded = [F.pad(m, (0, max_len - m.shape[-1]), "constant", 0) for m in mels]
    return torch.stack(padded), torch.stack(labels)


# ---- data loading (mirrors notebook Cell 5 + 6) ----
def load_test_data(samples_per_lang: int = 200) -> dict[str, list[np.ndarray]]:
    indic_configs = {
        "hi": "hindi",
        "ta": "tamil",
        "bn": "bengali",
        "mr": "marathi",
    }
    all_samples: dict[str, list[np.ndarray]] = {l: [] for l in TARGET_LANGS}

    def _extract_audio(row):
        audio = row.get("audio") or row.get("audio_filepath")
        if audio is None:
            return None
        arr = np.asarray(audio["array"], dtype=np.float32)
        if audio["sampling_rate"] != TARGET_SR:
            arr = scipy.signal.resample_poly(arr, TARGET_SR, audio["sampling_rate"])
        if len(arr) < TARGET_SR // 2:
            return None
        return arr

    for lang_code, config in indic_configs.items():
        stream = load_dataset(
            "ai4bharat/IndicVoices", name=config, split="train", streaming=True
        )
        stream = stream.shuffle(buffer_size=512, seed=42)
        count = 0
        for row in stream:
            if count >= samples_per_lang:
                break
            audio = _extract_audio(row)
            if audio is not None:
                all_samples[lang_code].append(audio)
                count += 1

    stream = load_dataset("ai4bharat/Svarah", split="test", streaming=True)
    stream = stream.shuffle(buffer_size=512, seed=42)
    count = 0
    for row in stream:
        if count >= samples_per_lang:
            break
        audio = _extract_audio(row)
        if audio is not None:
            all_samples["en"].append(audio)
            count += 1

    # Use 10% for test (mirrors notebook Cell 6's n_test proportion)
    test_samples: dict[str, list[np.ndarray]] = {}
    for lang_code in TARGET_LANGS:
        audios = all_samples[lang_code]
        rng.shuffle(audios)
        n_test = max(1, int(len(audios) * 0.1))
        test_samples[lang_code] = audios[:n_test]

    return test_samples


# ---- main ----
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained ECAPA-TDNN (matches Kaggle notebook Cell 10)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Path to best_model.pt checkpoint",
    )
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=200,
        help="Samples to stream per language (10% used for test)",
    )
    parser.add_argument(
        "--degraded",
        action="store_true",
        help="Also evaluate with telephony degradation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading test data ({args.samples_per_lang} samples/lang)...")
    test_samples = load_test_data(args.samples_per_lang)
    total = sum(len(v) for v in test_samples.values())
    print(f"  Test samples: {total} ({total // len(TARGET_LANGS)} per lang)")

    print("Loading model from checkpoint...")
    model = ECAPA_TDNN().to(DEVICE)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    # Cell 10-style clean evaluation
    print("\n--- Clean evaluation (notebook Cell 10) ---")
    test_ds = LangIdEvalDataset(test_samples, augment=False)
    test_loader = DataLoader(
        test_ds, args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    correct, total = 0, 0
    per_lang = {l: {"c": 0, "t": 0} for l in TARGET_LANGS}
    latencies = []

    with torch.no_grad():
        for mels, labels in test_loader:
            t0 = time.perf_counter()
            preds = model(mels).argmax(dim=-1)
            latencies.append((time.perf_counter() - t0) * 1000)
            for i in range(len(labels)):
                lang = TARGET_LANGS[labels[i].item()]
                per_lang[lang]["t"] += 1
                if preds[i] == labels[i]:
                    correct += 1
                    per_lang[lang]["c"] += 1
                total += 1

    latencies.sort()
    print(f"{'Lang':<6} {'Acc':<8}")
    print("-" * 14)
    for l in TARGET_LANGS:
        d = per_lang[l]
        acc = d["c"] / max(d["t"], 1) * 100
        print(f"{l:<6} {acc:<7.1f}%")
    total_acc = correct / max(total, 1) * 100
    print(f"\nTest acc: {total_acc:.2f}%")
    print(f"p50 latency: {latencies[len(latencies) // 2]:.1f}ms")
    print(f"p95 latency: {latencies[int(len(latencies) * 0.95)]:.1f}ms")

    # Optional degraded evaluation
    if args.degraded:
        print("\n--- Degraded evaluation ---")
        deg_ds = LangIdEvalDataset(test_samples, augment=True)
        deg_loader = DataLoader(
            deg_ds, args.batch_size, shuffle=False, collate_fn=collate_fn
        )

        correct_d, total_d = 0, 0
        per_lang_d = {l: {"c": 0, "t": 0} for l in TARGET_LANGS}

        with torch.no_grad():
            for mels, labels in deg_loader:
                preds = model(mels).argmax(dim=-1)
                for i in range(len(labels)):
                    lang = TARGET_LANGS[labels[i].item()]
                    per_lang_d[lang]["t"] += 1
                    if preds[i] == labels[i]:
                        correct_d += 1
                        per_lang_d[lang]["c"] += 1
                    total_d += 1

        print(f"{'Lang':<6} {'Deg Acc':<10}")
        print("-" * 16)
        for l in TARGET_LANGS:
            d = per_lang_d[l]
            acc = d["c"] / max(d["t"], 1) * 100
            print(f"{l:<6} {acc:<9.1f}%")
        deg_acc = correct_d / max(total_d, 1) * 100
        print(f"\nDegraded test acc: {deg_acc:.2f}%")


if __name__ == "__main__":
    main()
