"""
Mean-centering and ZCA whitening for wav2vec2 embeddings.

Pre-trained audio SSL embeddings are anisotropic: most variance sits in a
narrow cone, so unrelated frames end up with uniformly high cosine similarity.
Two standard fixes applied before computing section similarity:

  center   subtract the song-mean from every embedding before normalizing.
  whiten   ZCA-whiten after centering to decorrelate dimensions.

Both transforms are fit once from full-song frame embeddings and applied
identically to section prototypes and live frames. Pipeline:

    raw window-mean -> center -> (whiten) -> L2-normalize -> cosine
"""

from __future__ import annotations

import torch

WHITEN_EPS = 1e-6


def fit_mu(frame_means: torch.Tensor) -> torch.Tensor:
    """Song-mean vector, shape [1, D]. Fit from full-song frame means."""
    return frame_means.mean(dim=0, keepdim=True)


def fit_whitening_matrix(frames_centered: torch.Tensor, eps: float = WHITEN_EPS) -> torch.Tensor:
    """ZCA whitening matrix from already-centered frames. Returns W [D, D]."""
    n = frames_centered.shape[0]
    cov = (frames_centered.T @ frames_centered) / max(n - 1, 1)
    U, S, _ = torch.linalg.svd(cov)
    return U @ torch.diag(1.0 / torch.sqrt(S + eps)) @ U.T


def apply_transform(X: torch.Tensor, mu: torch.Tensor, W: torch.Tensor | None) -> torch.Tensor:
    """Center and optionally whiten [N, D] embeddings. L2-norm is left to the caller."""
    Xc = X - mu
    if W is not None:
        Xc = Xc @ W
    return Xc


def l2_normalize_rows(X: torch.Tensor) -> torch.Tensor:
    """L2-normalize each row. Returns [N, D]."""
    return X / (X.norm(dim=1, keepdim=True) + 1e-9)


def fit_transform(
    frame_means: torch.Tensor,
    center: bool = True,
    whiten: bool = True,
    eps: float = WHITEN_EPS,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Fit centering and/or whitening parameters from full-song frame means.

    Returns:
        mu: [1, D]        centering mean (zero tensor if center=False)
        W:  [D, D] | None whitening matrix (None if whiten=False)
    """
    if whiten and not center:
        raise ValueError("whiten=True requires center=True.")
    mu = fit_mu(frame_means) if center else torch.zeros(1, frame_means.shape[1])
    W = fit_whitening_matrix(frame_means - mu, eps=eps) if whiten else None
    return mu, W
