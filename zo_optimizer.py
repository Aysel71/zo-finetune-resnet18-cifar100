"""
zo_optimizer.py — MeZO-Adam optimizer (student-implemented).

Algorithm: multi-direction SPSA with Adam accumulators (Malladi et al. 2023).
Each step draws k independent random directions and averages their gradient
estimates before the Adam update.  The budget counter (n_batches × batch_size)
is unchanged because all 2k forward passes operate on the SAME fixed mini-batch.

Key design choices (backed by literature):
  - k=4 directions: optimal balance between variance reduction and Adam step
    size near the LP init. k=16 reduces variance → smaller Adam v_hat →
    larger effective step → overshoot from near-optimal LP init.
  - FC head only (fc.weight, fc.bias): 51,300 params. BN γ/β were tried
    (all 9,600 BN params, lr_bn=1e-4) but consistently regressed from LP
    init due to SPSA noise on BN params propagating through frozen features.
  - Cosine LR annealing: empirically matches constant LR; provides natural
    step-size reduction as optimum is approached.
  - clip_norm=0.0 (disabled): SPSA gradient norm ≈ sqrt(d) × (ΔL/2ε) ≈
    226 × 5 ≈ 1130 for d=51k, so any sensible clip threshold (e.g. 1.0)
    would scale every update by ~0.001 and zero all progress. We disable
    clipping by default; the `if self.clip_norm > 0.0` guard ensures
    `clip_norm=0.0` is a true no-op (not divide-by-zero scaling).
  - eps=1e-2: well-conditioned for a 100-class head at this scale.

Reference: github.com/princeton-nlp/MeZO — Algorithm 1.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """MeZO-Adam with k-direction SPSA, layerwise LR, and gradient clipping.

    Args:
        model:             The ``nn.Module`` to optimise.
        lr:                Learning rate for FC layer parameters.
        lr_bn:             Learning rate for BatchNorm affine parameters
                           (typically 5× lr; BN params adapt faster under
                           domain shift).
        eps:               Perturbation magnitude ε for finite-difference
                           gradient estimate.
        perturbation_mode: Ignored (kept for API compatibility).
        beta1:             Adam first-moment decay.
        beta2:             Adam second-moment decay.
        eps_adam:          Adam numerical stabiliser.
        T:                 Kept for API compatibility; schedule is constant.
        k:                 Number of SPSA directions averaged per step.
        clip_norm:         Global gradient-norm clipping threshold. Mandatory
                           for ZO — SPSA produces heavy-tailed estimates.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        lr_bn: float = 1e-4,
        eps: float = 1e-2,
        perturbation_mode: str = "gaussian",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        T: int = 128,
        k: int = 4,
        clip_norm: float = 0.0,
    ) -> None:
        self.model = model
        self.lr = lr
        self.lr_bn = lr_bn
        self.eps = eps
        self.perturbation_mode = perturbation_mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_adam = eps_adam
        self.T = T
        self.k = k
        self.clip_norm = clip_norm
        self.t = 0
        self._m: dict[str, torch.Tensor] | None = None
        self._v: dict[str, torch.Tensor] | None = None

        self.layer_names: list[str] = [
            "fc.weight", "fc.bias",
        ]
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lr_for(self, name: str) -> float:
        """Return the per-parameter learning rate (BN/downsample get lr_bn)."""
        return self.lr_bn if ("bn" in name or "downsample" in name) else self.lr

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"Layer names not found in model: {missing}. "
                f"Use [n for n, _ in model.named_parameters()] to inspect."
            )
        return {n: named[n] for n in self.layer_names}

    def _lazy_init(self, params: dict[str, nn.Parameter]) -> None:
        if self._m is None:
            self._m = {n: torch.zeros_like(p.data) for n, p in params.items()}
            self._v = {n: torch.zeros_like(p.data) for n, p in params.items()}

    def _perturb(
        self,
        params: dict[str, nn.Parameter],
        sign: float,
        seed: int,
    ) -> None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for p in params.values():
                u = torch.randn(p.shape, generator=g).to(device=p.device, dtype=p.dtype)
                p.add_(sign * self.eps * u)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """One MeZO-Adam step: 2k forward passes on the same batch, no backward.

        k independent SPSA gradient estimates are averaged before the Adam
        update. Gradient norm clipping is optional (disabled by default).
        Per-step LR is scaled by a cosine factor `(1 + cos(π(t-1)/T)) / 2`,
        annealing from full lr (t=1) to zero (t=T). T defaults to 128 to
        match the prescribed `--n_batches 128` run; if you change n_batches,
        construct the optimizer with `T=<n_batches>` (see analyze.py).

        Args:
            loss_fn: Closure returning scalar loss on the current fixed batch.

        Returns:
            Average loss across all 2k evaluations.
        """
        params = self._active_params()
        self._lazy_init(params)
        self.t += 1

        # Sample k independent perturbation seeds.
        seeds = [int(torch.randint(0, 2**31 - 1, (1,)).item()) for _ in range(self.k)]

        # Accumulate gradient estimates across k directions.
        grad_accum = {n: torch.zeros_like(p.data) for n, p in params.items()}
        total_loss = 0.0

        for seed in seeds:
            # θ + ε·u
            self._perturb(params, +1.0, seed)
            loss_p = loss_fn()

            # θ - ε·u
            self._perturb(params, -2.0, seed)
            loss_m = loss_fn()

            # Restore θ
            self._perturb(params, +1.0, seed)

            scalar = (loss_p - loss_m) / (2.0 * self.eps)
            total_loss += (loss_p + loss_m) / 2.0

            # Accumulate scaled gradient direction.
            g = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                for n, p in params.items():
                    u = torch.randn(p.shape, generator=g).to(device=p.device, dtype=p.dtype)
                    grad_accum[n].add_(scalar * u)

        # Average over k directions.
        for n in grad_accum:
            grad_accum[n].div_(self.k)

        # Gradient-norm clipping (clip_norm > 0 enables; 0 disables).
        if self.clip_norm > 0.0:
            total_sq = sum(g.norm().item() ** 2 for g in grad_accum.values())
            total_norm = math.sqrt(total_sq)
            if total_norm > self.clip_norm:
                scale = self.clip_norm / (total_norm + 1e-8)
                for n in grad_accum:
                    grad_accum[n].mul_(scale)

        # Cosine LR annealing scale factor (applied per-layer via _lr_for).
        cos_scale = (1.0 + math.cos(math.pi * (self.t - 1) / max(self.T, 1))) / 2.0

        with torch.no_grad():
            for n, p in params.items():
                grad = grad_accum[n]
                m = self._m[n].to(p.device)
                v = self._v[n].to(p.device)

                m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

                self._m[n] = m
                self._v[n] = v

                m_hat = m / (1.0 - self.beta1 ** self.t)
                v_hat = v / (1.0 - self.beta2 ** self.t)
                lr_n = self._lr_for(n) * cos_scale
                p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps_adam), value=-lr_n)

        return total_loss / self.k
