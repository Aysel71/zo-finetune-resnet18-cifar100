"""Generate CIFAR-100 sample grids for README illustrations."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR = Path("./data/cifar-100-python")


def load_cifar100(split="train"):
    with open(DATA_DIR / split, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    imgs = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N,32,32,3)
    labels = np.array(d[b"fine_labels"])
    with open(DATA_DIR / "meta", "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    class_names = [n.decode() for n in meta[b"fine_label_names"]]
    return imgs, labels, class_names


def show_class_grid():
    """4×5 grid: one row per superclass theme, 5 example images each."""
    imgs, labels, class_names = load_cifar100("train")

    # Hand-picked classes that tell a story (good vs bad accuracy)
    selected = [
        # Easy — visually distinctive
        "sunflower", "motorcycle", "bicycle", "rocket", "keyboard",
        # Medium
        "elephant", "dolphin", "tiger", "lion", "bear",
        # Hard — fine-grained
        "girl", "boy", "woman", "man", "baby",
        # Hard — aquatic mammals
        "otter", "seal", "beaver", "possum", "squirrel",
    ]

    n_per_class = 5
    n_classes = len(selected)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 1.2, n_classes * 1.2))
    fig.patch.set_facecolor("#0d1117")

    for row, cls_name in enumerate(selected):
        cls_idx = class_names.index(cls_name)
        idxs = np.where(labels == cls_idx)[0]
        chosen = np.random.RandomState(42 + row).choice(idxs, n_per_class, replace=False)
        for col, img_idx in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(imgs[img_idx], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=7, color="white",
                              rotation=0, labelpad=48, va="center", fontfamily="monospace")

    # Section labels on the right
    section_labels = {
        2: "easy",
        7: "medium",
        12: "hard: people",
        17: "hard: mammals",
    }
    for row, label in section_labels.items():
        axes[row, -1].annotate(
            label, xy=(1.05, 0.5), xycoords="axes fraction",
            fontsize=7, color="#58a6ff", va="center", rotation=-90,
        )

    fig.suptitle("CIFAR-100 sample images (32×32)", color="white", fontsize=10, y=1.005)
    fig.tight_layout(pad=0.3)
    fig.savefig("cifar100_samples.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved cifar100_samples.png")


def show_confused_pairs():
    """Show the 4 most confused pairs side-by-side with 6 examples each."""
    imgs, labels, class_names = load_cifar100("train")

    pairs = [
        ("girl", "woman"),
        ("boy", "man"),
        ("otter", "beaver"),
        ("seal", "possum"),
    ]

    n_ex = 6
    fig = plt.figure(figsize=(n_ex * 1.1, len(pairs) * 2.4))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(len(pairs), 1, hspace=0.55)

    for pair_idx, (cls_a, cls_b) in enumerate(pairs):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, n_ex, subplot_spec=gs[pair_idx], hspace=0.08, wspace=0.06
        )
        for ri, cls_name in enumerate([cls_a, cls_b]):
            cls_idx = class_names.index(cls_name)
            idxs = np.where(labels == cls_idx)[0]
            chosen = np.random.RandomState(99 + pair_idx * 10 + ri).choice(
                idxs, n_ex, replace=False
            )
            for ci, img_idx in enumerate(chosen):
                ax = fig.add_subplot(inner[ri, ci])
                ax.imshow(imgs[img_idx], interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444")
                    spine.set_linewidth(0.5)
                if ci == 0:
                    color = "#79c0ff" if ri == 0 else "#ffa657"
                    ax.set_ylabel(cls_name, fontsize=8, color=color,
                                  rotation=0, labelpad=42, va="center",
                                  fontfamily="monospace")

        # Divider label
        fig.text(
            0.5,
            1 - (pair_idx + 0.05) / len(pairs),
            f"confused pair: {cls_a}  ↔  {cls_b}",
            ha="center", va="top", fontsize=8, color="#8b949e",
            transform=fig.transFigure,
        )

    fig.suptitle("Most confused class pairs — why the model struggles",
                 color="white", fontsize=10, y=1.02)
    fig.savefig("cifar100_confused_pairs.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved cifar100_confused_pairs.png")


def show_good_pairs():
    """Show the 4 best-performing classes side-by-side with 6 examples each."""
    imgs, labels, class_names = load_cifar100("train")

    # Top-accuracy pairs — visually very distinct from each other
    pairs = [
        ("sunflower", "mushroom"),
        ("motorcycle", "bicycle"),
        ("rocket", "keyboard"),
        ("elephant", "dolphin"),
    ]

    n_ex = 6
    fig = plt.figure(figsize=(n_ex * 1.1, len(pairs) * 2.4))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(len(pairs), 1, hspace=0.55)

    for pair_idx, (cls_a, cls_b) in enumerate(pairs):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, n_ex, subplot_spec=gs[pair_idx], hspace=0.08, wspace=0.06
        )
        for ri, cls_name in enumerate([cls_a, cls_b]):
            cls_idx = class_names.index(cls_name)
            idxs = np.where(labels == cls_idx)[0]
            chosen = np.random.RandomState(7 + pair_idx * 10 + ri).choice(
                idxs, n_ex, replace=False
            )
            for ci, img_idx in enumerate(chosen):
                ax = fig.add_subplot(inner[ri, ci])
                ax.imshow(imgs[img_idx], interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2ea043")
                    spine.set_linewidth(0.8)
                if ci == 0:
                    ax.set_ylabel(cls_name, fontsize=8, color="#2ea043",
                                  rotation=0, labelpad=48, va="center",
                                  fontfamily="monospace")

        fig.text(
            0.5,
            1 - (pair_idx + 0.05) / len(pairs),
            f"easy pair: {cls_a}  vs  {cls_b}   ✓  visually distinct",
            ha="center", va="top", fontsize=8, color="#3fb950",
            transform=fig.transFigure,
        )

    fig.suptitle("Best-classified class pairs — why the model gets these right",
                 color="white", fontsize=10, y=1.02)
    fig.savefig("cifar100_good_pairs.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved cifar100_good_pairs.png")


if __name__ == "__main__":
    show_class_grid()
    show_confused_pairs()
    show_good_pairs()
