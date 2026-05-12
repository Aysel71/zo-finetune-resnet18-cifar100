"""
analyze.py — Full training pipeline with detailed metrics and plots.

Runs the same three checkpoints as validate.py, then produces:
  - loss_curve.png       : ZO loss per step
  - per_class_acc.png    : top-1 accuracy for each of the 100 CIFAR-100 classes
  - confusion_top20.png  : heatmap of the 20 most confused class pairs

Usage:
    python3 analyze.py --batch_size 64 --n_batches 128 --data_dir ./data
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.datasets as datasets

from augmentation import get_transforms
from model import get_model, get_model_imagenet_head
from train_data import get_train_dataset_loader
from zo_optimizer import ZeroOrderOptimizer

_MAX_BUDGET = 8192

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
    tta: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (top1_acc, all_preds, all_labels). TTA averages original + hflip logits."""
    model.eval().to(device)
    preds_list, labels_list = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"  {desc}", leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            if tta:
                logits = (logits + model(imgs.flip(-1))) / 2.0
            preds_list.append(logits.argmax(1).cpu().numpy())
            labels_list.append(lbls.numpy())
    preds = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    return float((preds == labels).mean()), preds, labels


def run_finetuning(model, train_loader, optimizer, n_batches, device, criterion):
    model.to(device)

    def _infinite(loader):
        while True:
            yield from loader

    data_iter = _infinite(train_loader)
    pbar = tqdm(range(n_batches), desc="  ZO fine-tuning", unit="step")
    for _ in pbar:
        imgs, lbls = next(data_iter)
        imgs, lbls = imgs.to(device), lbls.to(device)

        def loss_fn(_imgs=imgs, _lbls=lbls):
            model.eval()
            with torch.no_grad():
                return float(criterion(model(_imgs), _lbls).item())

        loss = optimizer.step(loss_fn)
        optimizer.loss_history.append(loss)
        pbar.set_postfix(loss=f"{loss:.4f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loss_curve(loss_history: list[float], out: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not installed — no loss curve plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, linewidth=1.5, color="steelblue")
    ax.set_xlabel("ZO step")
    ax.set_ylabel("Loss (avg of L+ and L−)")
    ax.set_title("MeZO-Adam: loss per step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_per_class_accuracy(preds: np.ndarray, labels: np.ndarray, out: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not installed — no per-class plot")
        return

    n_classes = 100
    per_class = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        per_class[c] = (preds[mask] == c).mean() if mask.sum() > 0 else 0.0

    order = np.argsort(per_class)
    names = [CIFAR100_CLASSES[i] for i in order]
    vals = per_class[order]

    fig, ax = plt.subplots(figsize=(10, 18))
    colors = ["#d9534f" if v < 0.4 else "#f0ad4e" if v < 0.6 else "#5cb85c" for v in vals]
    ax.barh(range(n_classes), vals, color=colors)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Top-1 accuracy")
    ax.set_title("Per-class accuracy (sorted)")
    ax.axvline(vals.mean(), color="navy", linestyle="--", linewidth=1, label=f"mean={vals.mean():.2%}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_confusion_top20(preds: np.ndarray, labels: np.ndarray, out: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("  [skip] matplotlib/sklearn not installed — no confusion matrix")
        return

    cm = confusion_matrix(labels, preds, labels=list(range(100)))
    np.fill_diagonal(cm, 0)

    # Find the 20 classes with most off-diagonal errors
    errors_per_class = cm.sum(1) + cm.sum(0)
    top20_idx = np.argsort(errors_per_class)[-20:][::-1]
    cm_sub = cm[np.ix_(top20_idx, top20_idx)]
    names_sub = [CIFAR100_CLASSES[i] for i in top20_idx]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_sub, cmap="Blues")
    ax.set_xticks(range(20))
    ax.set_yticks(range(20))
    ax.set_xticklabels(names_sub, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names_sub, fontsize=8)
    ax.set_title("Confusion matrix — 20 most confused classes\n(diagonal zeroed; true label = row)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def print_summary(results: dict) -> None:
    w = 38
    print("\n" + "=" * (w + 14))
    print(f"  {'Checkpoint':<{w}} {'Top-1':>8}")
    print("-" * (w + 14))
    rows = [
        ("1. Baseline (ImageNet 1000-class head)", results["top1_imagenet"]),
        ("2. Linear probe init (no ZO)",           results["top1_init"]),
        ("3. After ZO fine-tuning",                results["top1_ft"]),
    ]
    for label, acc in rows:
        print(f"  {label:<{w}} {acc*100:>7.2f}%")
    delta = results["top1_ft"] - results["top1_init"]
    print("-" * (w + 14))
    print(f"  ZO gain: {delta*100:+.2f}%")
    print("=" * (w + 14))


# ---------------------------------------------------------------------------
# Top confused pairs (text)
# ---------------------------------------------------------------------------

def print_top_confusions(preds: np.ndarray, labels: np.ndarray, k: int = 10) -> None:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds, labels=list(range(100)))
    np.fill_diagonal(cm, 0)
    flat = [(cm[i, j], i, j) for i in range(100) for j in range(100) if i != j]
    flat.sort(reverse=True)
    print(f"\n  Top-{k} confused pairs (true → predicted, count):")
    for cnt, true_cls, pred_cls in flat[:k]:
        print(f"    {CIFAR100_CLASSES[true_cls]:>15} → {CIFAR100_CLASSES[pred_cls]:<15} ({cnt})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_batches",  type=int, default=128)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out_dir",    default=".")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    total = args.n_batches * args.batch_size
    if total > _MAX_BUDGET:
        print(f"[Error] Budget {total} > {_MAX_BUDGET}", file=sys.stderr)
        sys.exit(1)

    seed_everything(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )
    print(f"[Device] {device}")

    # Data
    _, train_loader = get_train_dataset_loader(args.data_dir, args.batch_size, gen)
    val_ds = datasets.CIFAR100(root=args.data_dir, train=False, download=True,
                                transform=get_transforms(train=False))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    # --- Checkpoint 1: ImageNet head ---
    print("\n[1/3] Baseline (ImageNet head)")
    m_imnet = get_model_imagenet_head()
    top1_imnet, _, _ = evaluate_detailed(m_imnet, val_loader, device, "baseline")
    print(f"  Top-1: {top1_imnet:.2%}")
    del m_imnet

    # --- Checkpoint 2: Linear probe head (no ZO) ---
    print("\n[2/3] Linear probe init")
    model = get_model()
    top1_init, _, _ = evaluate_detailed(model, val_loader, device, "LP init")
    print(f"  Top-1: {top1_init:.2%}")

    # --- Checkpoint 3: ZO fine-tuning ---
    print(f"\n[3/3] ZO fine-tuning ({args.n_batches} steps × batch {args.batch_size})")
    optimizer = ZeroOrderOptimizer(model, T=args.n_batches)
    run_finetuning(model, train_loader, optimizer, args.n_batches, device, criterion)

    top1_ft, preds_ft, labels_ft = evaluate_detailed(model, val_loader, device, "ZO fine-tuned")
    print(f"  Top-1: {top1_ft:.2%}")

    results = dict(top1_imagenet=top1_imnet, top1_init=top1_init, top1_ft=top1_ft)
    print_summary(results)
    print_top_confusions(preds_ft, labels_ft, k=10)

    # --- Plots ---
    print("\n[Plots]")
    out = args.out_dir
    plot_loss_curve(optimizer.loss_history, os.path.join(out, "loss_curve.png"))
    plot_per_class_accuracy(preds_ft, labels_ft, os.path.join(out, "per_class_acc.png"))
    plot_confusion_top20(preds_ft, labels_ft, os.path.join(out, "confusion_top20.png"))
