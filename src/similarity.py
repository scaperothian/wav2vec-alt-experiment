"""Cosine similarity matrices."""

import numpy as np
import torch


def cosine_matrix(vectors: torch.Tensor) -> np.ndarray:
    """Square pairwise cosine similarity. Args: [N, D] L2-normalised rows. Returns [N, N]."""
    return (vectors @ vectors.T).detach().numpy()


def cosine_block(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    """Rectangular cosine similarity. Args: A [M, D], B [N, D] L2-normalised. Returns [M, N]."""
    return (A @ B.T).detach().numpy()
