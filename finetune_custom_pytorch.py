"""
Fine-tune I3D (PyTorch) on a custom 5-class children activity dataset.

Compatible with:  Python 3.8+  |  PyTorch 2.x  |  CUDA 11.8 / 12.x
GPU tested on:    NVIDIA RTX 4000 Ada Generation (Ada Lovelace / sm_89)

Classes
-------
    0  normal        (1068 videos)
    1  fighting      ( 480 videos)
    2  unsafeclimb   ( 366 videos)
    3  unsafejump    ( 545 videos)
    4  unsafethrow   ( 348 videos)

Pre-trained weights
-------------------
Download rgb_imagenet.pt from the piergiaj/pytorch-i3d releases and place it
in  models/rgb_imagenet.pt  (relative to this script), OR pass --weights.

    wget -P models  https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt

Usage
-----
    # Training
    python finetune_custom_pytorch.py train \\
        --data_dir /content/drive/MyDrive/ChildSafety \\
        --weights  models/rgb_imagenet.pt

    # Prediction
    python finetune_custom_pytorch.py predict \\
        --video      clip.mp4 \\
        --checkpoint output/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── local model file (pytorch_i3d.py must be in the same directory) ──────────
sys.path.insert(0, str(Path(__file__).parent))
from pytorch_i3d import InceptionI3d

# ─────────────────────────────── constants ───────────────────────────────────

CLASSES     = ['Normal', 'fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow']
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE  = 224
CLIP_FRAMES = 64
CHANNELS    = 3

# Training defaults (all overridable via CLI)
BATCH_SIZE   = 4
EPOCHS       = 40
LR           = 1e-3
DROPOUT      = 0.5
WEIGHT_DECAY = 1e-4
MOMENTUM     = 0.9
FOCAL_GAMMA  = 2.0
NUM_WORKERS  = 4         # DataLoader workers; set 0 on Windows

# ─────────────────────────────── device detection ────────────────────────────

def detect_device() -> torch.device:
    """Auto-select GPU or CPU and print the result clearly."""
    print("=" * 60)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        props  = torch.cuda.get_device_properties(0)
        print(f"[DEVICE]  GPU detected  →  {props.name}")
        print(f"          VRAM : {props.total_memory / 1e9:.1f} GB")
        print(f"          CUDA : {torch.version.cuda}   "
              f"PyTorch : {torch.__version__}")
        # Ada Lovelace (RTX 4000 series) = compute capability 8.9
        cc = f"{props.major}.{props.minor}"
        print(f"          Compute capability : {cc}")
        if props.major >= 8:
            print("          ✓ bfloat16 / TF32 supported — enabling TF32")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("[DEVICE]  No CUDA GPU found — running on CPU")
        print(f"          PyTorch : {torch.__version__}")
    print("=" * 60)
    logging.info("Device: %s", device)
    return device


# ─────────────────────────────── Google Drive ────────────────────────────────

def mount_google_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("[INFO]  Google Drive mounted at /content/drive")
    except ImportError:
        print("[INFO]  Not in Colab — assuming Drive path is already accessible.")


# ─────────────────────────────── dataset ─────────────────────────────────────

def load_video_paths(data_dir: str):
    """
    Scan *data_dir* for one sub-folder per class.

    Returns
    -------
    paths         : list[str]
    labels        : list[int]
    class_weights : list[float]   (inverse-frequency, normalised)
    """
    paths, labels = [], []
    counts = []
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Folder '{folder}' not found for class '{cls}'.")
        vids = (glob.glob(os.path.join(folder, '*.mp4')) +
                glob.glob(os.path.join(folder, '*.avi')) +
                glob.glob(os.path.join(folder, '*.mov')) +
                glob.glob(os.path.join(folder, '*.mkv')))
        paths.extend(vids)
        labels.extend([idx] * len(vids))
        counts.append(len(vids))
        print(f"[DATA]  {cls:<15s} ({idx})  →  {len(vids):4d} videos")

    total = sum(counts)
    weights = [total / (NUM_CLASSES * c) if c > 0 else 1.0 for c in counts]
    s = sum(weights)
    weights = [w * NUM_CLASSES / s for w in weights]
    print(f"[DATA]  Total : {total}   class weights : "
          + "  ".join(f"{CLASSES[i]}={weights[i]:.3f}" for i in range(NUM_CLASSES)))
    return paths, labels, weights


def stratified_split(paths, labels, val_frac=0.15, seed=42):
    rng = random.Random(seed)
    by_class: dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
    for p, l in zip(paths, labels):
        by_class[l].append(p)
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for cls, lst in by_class.items():
        rng.shuffle(lst)
        n = max(1, int(len(lst) * val_frac))
        va_p += lst[:n];     va_l += [cls] * n
        tr_p += lst[n:];     tr_l += [cls] * (len(lst) - n)
    print(f"[SPLIT]  train={len(tr_p)}   val={len(va_p)}")
    return tr_p, tr_l, va_p, va_l


# ─────────────────────────────── video I/O ───────────────────────────────────

def read_video_clip(video_path: str,
                    n_frames: int = CLIP_FRAMES,
                    size: int = IMAGE_SIZE,
                    augment: bool = False) -> np.ndarray:
    """
    Load n_frames from video_path, resize to size×size, normalise to [-1, 1].

    Returns ndarray (C, T, H, W) float32  — PyTorch channel-first.
    """
    cap   = cv2.VideoCapture(video_path)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    if total >= n_frames:
        if augment:
            start   = random.randint(0, total - n_frames)
            indices = list(range(start, start + n_frames))
        else:
            indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
    else:
        indices = [i % total for i in range(n_frames)]

    frames = []
    last_frame = None
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = last_frame if last_frame is not None else \
                    np.zeros((size, size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            last_frame = frame
        frames.append(frame)
    cap.release()

    clip = np.stack(frames, axis=0).astype(np.float32)   # (T, H, W, 3)
    if augment and random.random() < 0.5:
        clip = clip[:, :, ::-1, :].copy()                # horizontal flip

    clip = (clip / 127.5) - 1.0                          # → [-1, 1]
    clip = clip.transpose(3, 0, 1, 2)                    # → (C, T, H, W)
    return clip


class VideoDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths   = paths
        self.labels  = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        clip  = read_video_clip(self.paths[idx], augment=self.augment)
        label = self.labels[idx]
        return torch.from_numpy(clip), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────── focal loss ──────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss with per-class alpha weights.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, class_weights: list[float], gamma: float = 2.0,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.gamma  = gamma
        self.alpha  = torch.tensor(class_weights, dtype=torch.float32,
                                   device=device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C)
        targets : (B,)  long
        """
        log_prob = F.log_softmax(logits, dim=-1)          # (B, C)
        prob     = log_prob.exp()                          # (B, C)

        # gather true-class log-prob and prob
        log_pt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt     = prob.gather(1,     targets.unsqueeze(1)).squeeze(1)  # (B,)

        # per-sample alpha
        alpha_t = self.alpha[targets]                     # (B,)

        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
        return loss.mean()


# ─────────────────────────────── model helpers ───────────────────────────────

def build_model(num_classes: int,
                weights_path: str | None,
                device: torch.device) -> nn.Module:
    """
    Load Kinetics-pretrained I3D backbone and attach a new classification head.
    """
    # ── backbone ─────────────────────────────────────────────────────────────
    backbone = InceptionI3d(num_classes=400, spatial_squeeze=True,
                            final_endpoint='Logits')

    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location='cpu')
        # piergiaj weights sometimes have 'module.' prefix
        state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"[CKPT]  Loaded backbone from: {weights_path}")
        if missing:
            print(f"        Missing keys   : {len(missing)}")
        if unexpected:
            print(f"        Unexpected keys: {len(unexpected)}")
    else:
        print(f"[CKPT]  WARNING — weights not found at '{weights_path}'. "
              "Training from random init.")

    # ── custom head ──────────────────────────────────────────────────────────
    # Replace the Kinetics 400-class logit layer with a new num_classes head
    backbone.logits = Unit3D_head(1024, num_classes)

    backbone = backbone.to(device)
    return backbone


class Unit3D_head(nn.Module):
    """Lightweight 1×1×1 conv head to replace the Kinetics logit layer."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True)
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────── training loop ───────────────────────────────

def train(args):
    # ── logging setup ────────────────────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'train.log'), mode='w'),
            logging.StreamHandler(sys.stdout),
        ])

    device = detect_device()

    # ── data ─────────────────────────────────────────────────────────────────
    if args.mount_drive:
        mount_google_drive()

    all_paths, all_labels, class_weights = load_video_paths(args.data_dir)
    tr_p, tr_l, va_p, va_l = stratified_split(
        all_paths, all_labels, val_frac=args.val_split)

    train_ds = VideoDataset(tr_p, tr_l, augment=True)
    val_ds   = VideoDataset(va_p, va_l, augment=False)

    # pin_memory gives a free speed-up when training on GPU
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)

    # ── model ────────────────────────────────────────────────────────────────
    model = build_model(NUM_CLASSES, args.weights, device)

    # ── loss ─────────────────────────────────────────────────────────────────
    criterion = FocalLoss(class_weights, gamma=FOCAL_GAMMA, device=device)

    # ── optimiser + scheduler ────────────────────────────────────────────────
    # Fine-tune with different LR for backbone vs new head
    backbone_params = [p for n, p in model.named_parameters()
                       if 'logits' not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if 'logits'     in n]

    optimizer = torch.optim.SGD(
        [{'params': backbone_params, 'lr': args.lr * 0.1},
         {'params': head_params,     'lr': args.lr}],
        momentum=MOMENTUM,
        weight_decay=args.weight_decay)

    # Cosine annealing — decays LR smoothly to near-zero
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    # ── mixed precision (free ~2× speed on Ampere / Ada GPUs) ───────────────
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── training ─────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    logging.info("Starting training  |  epochs=%d  batch=%d  device=%s",
                 args.epochs, args.batch_size, device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        t0 = time.time()

        for clips, labels in train_loader:
            clips  = clips.to(device,  non_blocking=True)   # (B, C, T, H, W)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(clips)                        # (B, NUM_CLASSES)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            preds       = logits.argmax(dim=1)
            tr_correct += (preds == labels).sum().item()
            tr_total   += labels.size(0)
            tr_loss    += loss.item() * labels.size(0)

        scheduler.step()
        epoch_time = time.time() - t0
        train_acc  = tr_correct / tr_total
        train_loss = tr_loss    / tr_total

        # ── validation ───────────────────────────────────────────────────────
        model.eval()
        va_correct = va_total = 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips  = clips.to(device,  non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(clips)
                preds       = logits.argmax(dim=1)
                va_correct += (preds == labels).sum().item()
                va_total   += labels.size(0)

        val_acc = va_correct / va_total

        lr_now = optimizer.param_groups[1]['lr']   # head LR
        logging.info(
            "Epoch %3d/%d  loss=%.4f  train_acc=%.3f  val_acc=%.3f  "
            "lr=%.2e  (%.0fs)",
            epoch, args.epochs, train_loss, train_acc, val_acc, lr_now, epoch_time)

        # ── per-class accuracy ───────────────────────────────────────────────
        if epoch % 5 == 0:
            _per_class_acc(model, val_loader, device, use_amp)

        # ── save best ────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(args.log_dir,
                                f'best_ep{epoch:03d}_acc{val_acc:.3f}.pt')
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'val_acc': val_acc,
                        'classes': CLASSES}, ckpt)
            logging.info("  ✓ Best model saved → %s", ckpt)

    # final checkpoint
    final = os.path.join(args.log_dir, 'model_final.pt')
    torch.save({'epoch': args.epochs,
                'model_state': model.state_dict(),
                'classes': CLASSES}, final)
    logging.info("Training complete. Best val_acc=%.3f  Final: %s",
                 best_val_acc, final)


def _per_class_acc(model, loader, device, use_amp):
    """Print per-class accuracy on the validation set."""
    correct = [0] * NUM_CLASSES
    total   = [0] * NUM_CLASSES
    model.eval()
    with torch.no_grad():
        for clips, labels in loader:
            clips  = clips.to(device)
            labels = labels.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                for p, l in zip(preds.cpu(), labels.cpu()):
                    total[l]   += 1
                    correct[l] += int(p == l)
    print("  Per-class val accuracy:")
    for i, cls in enumerate(CLASSES):
        acc = correct[i] / total[i] if total[i] else 0.0
        print(f"    {cls:<15s}: {acc:.3f}  ({correct[i]}/{total[i]})")


# ─────────────────────────────── prediction ──────────────────────────────────

def predict(args):
    device = detect_device()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt.get('classes', CLASSES)

    model = InceptionI3d(num_classes=len(classes), spatial_squeeze=True,
                         final_endpoint='Logits')
    model.logits = Unit3D_head(1024, len(classes))
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()

    clip = read_video_clip(args.video, augment=False)
    clip = torch.from_numpy(clip).unsqueeze(0).to(device)  # (1, C, T, H, W)

    with torch.no_grad():
        logits = model(clip)                               # (1, C)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx  = int(probs.argmax())
    pred_name = classes[pred_idx]

    print("\n── Prediction ──────────────────────────────────────────")
    for i, (cls, p) in enumerate(zip(classes, probs)):
        bar    = '█' * int(p * 40)
        marker = '  ◄ PREDICTED' if i == pred_idx else ''
        print(f"  {cls:<15s}  {p:.4f}  {bar}{marker}")
    print(f"\n  → Predicted class : {pred_name}  (index {pred_idx})")
    print("────────────────────────────────────────────────────────\n")

    return {'class_name': pred_name, 'class_index': pred_idx,
            'probabilities': {c: float(p) for c, p in zip(classes, probs)}}


# ─────────────────────────────── CLI ─────────────────────────────────────────

def parse_args():
    p   = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command')

    # train
    tr = sub.add_parser('train')
    tr.add_argument('--data_dir',    required=True,
                    help='Drive root folder with one sub-folder per class')
    tr.add_argument('--weights',     default='models/rgb_imagenet.pt',
                    help='Path to Kinetics pretrained .pt weights')
    tr.add_argument('--log_dir',     default='output/finetune_pytorch')
    tr.add_argument('--batch_size',  type=int,   default=BATCH_SIZE)
    tr.add_argument('--epochs',      type=int,   default=EPOCHS)
    tr.add_argument('--lr',          type=float, default=LR)
    tr.add_argument('--weight_decay',type=float, default=WEIGHT_DECAY)
    tr.add_argument('--val_split',   type=float, default=0.15)
    tr.add_argument('--num_workers', type=int,   default=NUM_WORKERS)
    tr.add_argument('--mount_drive', action='store_true')

    # predict
    pr = sub.add_parser('predict')
    pr.add_argument('--video',      required=True)
    pr.add_argument('--checkpoint', required=True)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if   args.command == 'train':   train(args)
    elif args.command == 'predict': predict(args)
    else:
        print(__doc__)