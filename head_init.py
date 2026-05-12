"""
head_init.py — Final layer initialization (student-implemented).

Strategy:
  1. Extract frozen ResNet-18 features from the entire CIFAR-100 training set.
  2. ZCA-whiten the 512-d features (ridge ε=1e-2).
  3. Reduce to 99 dimensions with LDA (Linear Discriminant Analysis).
  4. Fit a logistic regression (L-BFGS) on the low-dimensional LDA features.
  5. Compose ZCA + LDA + LogReg into a single (100, 512) linear layer.

Literature basis:
  - ZCA whitening consistently improves linear/kNN probing by 1–5%
    (Kalapos & Gyires-Tóth, arXiv:2408.07519).
  - LDA preprocessing of frozen ResNet-18 features improves linear probe
    accuracy on CIFAR-100 from ~62.8% to ~66.9% by projecting into the
    99-dimensional discriminant subspace (arXiv:2604.03928).
"""

import gc
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def init_last_layer(layer: nn.Linear) -> None:
    """Fit ZCA → LDA → LogReg on frozen ResNet-18 features and copy weights in-place.

    Args:
        layer: The ``nn.Linear`` head to initialize (in_features=512, out=100).
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[head_init] Extracting features on {device} ...")

    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    # Identical to the validation pipeline.
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
    ])

    data_dir = os.environ.get("DATA_DIR", "./data")
    dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

    N = len(dataset)
    X = np.empty((N, 512), dtype=np.float32)
    y = np.empty(N, dtype=np.int64)
    idx = 0

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="  features", leave=False):
            feats = backbone(imgs.to(device)).cpu().numpy()
            bs = len(feats)
            X[idx: idx + bs] = feats
            y[idx: idx + bs] = lbls.numpy()
            idx += bs

    del backbone
    gc.collect()

    X = X.astype(np.float64)

    print("[head_init] Fitting ZCA + LDA + linear classifier ...")
    W, b = _fit_linear(X, y)

    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(W).float())
        layer.bias.copy_(torch.from_numpy(b).float())
    print("[head_init] Done.")


def _zca_whiten(X: np.ndarray, eps: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_whitened, W_zca, mu) where X_whitened = (X - mu) @ W_zca."""
    mu = X.mean(0)
    Xc = X - mu
    cov = Xc.T @ Xc / len(X)
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    W_zca = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T  # (512, 512) symmetric
    return Xc @ W_zca, W_zca, mu


def _fit_linear(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weight (100,512), bias (100,)) via ZCA→LDA→LogReg composition."""
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.linear_model import LogisticRegression

        # Step 1: ZCA whitening — decorrelates and normalises feature variance.
        X_w, W_zca, mu_raw = _zca_whiten(X, eps=1e-2)

        # Step 2: LDA — project 512-d whitened features into 99-d discriminant subspace.
        lda = LinearDiscriminantAnalysis(n_components=99, solver="svd")
        X_lda = lda.fit_transform(X_w, y)  # (N, 99)

        # Step 3: LogReg on the compact 99-d representation.
        clf = LogisticRegression(
            C=10.0,
            solver="lbfgs",
            max_iter=5000,
            tol=1e-4,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_lda, y)

        # Compose ZCA + LDA + LogReg into a single linear layer.
        #
        # For a test point x (row vector, 512-d):
        #   x_w   = (x - mu_raw) @ W_zca
        #   x_lda = (x_w - lda.xbar_) @ lda.scalings_
        #         = (x - mu_raw) @ W_zca @ scalings_ - lda.xbar_ @ scalings_
        #   score = x_lda @ coef_.T + intercept_
        #         = x @ P @ coef_.T + b_final
        #
        # where P = W_zca @ scalings_  (512, 99)
        # W_nn  = coef_ @ P.T          (100, 512)
        # b_nn  = intercept_ - (mu_raw @ P + lda.xbar_ @ scalings_) @ coef_.T
        P = W_zca @ lda.scalings_                          # (512, 99)
        W_nn = clf.coef_ @ P.T                             # (100, 512)
        correction = (mu_raw @ P + lda.xbar_ @ lda.scalings_) @ clf.coef_.T  # (100,)
        b_nn = clf.intercept_ - correction

        return W_nn.astype(np.float64), b_nn.astype(np.float64)

    except ImportError:
        return _ridge(X, y, lam=0.1)


def _ridge(
    X: np.ndarray, y: np.ndarray, lam: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge regression fallback (no sklearn required)."""
    N, d = X.shape
    K = int(y.max()) + 1
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    mu_x = X.mean(0)
    mu_y = Y.mean(0)
    Xc, Yc = X - mu_x, Y - mu_y

    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d, dtype=np.float64), Xc.T @ Yc)
    b = mu_y - mu_x @ W
    return W.T, b
