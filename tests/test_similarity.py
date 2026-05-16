"""Tests for cosine similarity functions."""

import numpy as np
import torch

from src.similarity import cosine_block, cosine_matrix


class TestCosineMatrix:
    def test_identical_vectors_give_one(self):
        v = torch.randn(4)
        v = v / v.norm()
        M = cosine_matrix(v.unsqueeze(0).expand(3, -1))
        assert np.allclose(M, np.ones((3, 3)), atol=1e-5)

    def test_orthogonal_vectors_give_zero_off_diagonal(self):
        vectors = torch.eye(3)
        M = cosine_matrix(vectors)
        assert np.allclose(np.diag(M), 1.0, atol=1e-6)
        assert np.allclose(M - np.eye(3), 0.0, atol=1e-6)

    def test_diagonal_is_one_for_unit_vectors(self):
        torch.manual_seed(0)
        raw = torch.randn(5, 16)
        vectors = raw / raw.norm(dim=1, keepdim=True)
        assert np.allclose(np.diag(cosine_matrix(vectors)), 1.0, atol=1e-5)

    def test_matrix_is_symmetric(self):
        torch.manual_seed(1)
        raw = torch.randn(6, 32)
        vectors = raw / raw.norm(dim=1, keepdim=True)
        M = cosine_matrix(vectors)
        assert np.allclose(M, M.T, atol=1e-6)

    def test_output_is_numpy(self):
        assert isinstance(cosine_matrix(torch.eye(3)), np.ndarray)

    def test_output_shape(self):
        torch.manual_seed(2)
        raw = torch.randn(7, 64)
        vectors = raw / raw.norm(dim=1, keepdim=True)
        assert cosine_matrix(vectors).shape == (7, 7)

    def test_values_bounded(self):
        torch.manual_seed(3)
        raw = torch.randn(10, 16)
        vectors = raw / raw.norm(dim=1, keepdim=True)
        M = cosine_matrix(vectors)
        assert M.min() >= -1.0 - 1e-5
        assert M.max() <= 1.0 + 1e-5

    def test_known_2d_values(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([1.0, 1.0]) / (2 ** 0.5)
        M = cosine_matrix(torch.stack([a, b]))
        assert abs(M[0, 1] - 2 ** 0.5 / 2) < 1e-5


class TestCosineBlock:
    def _unit(self, n: int, dim: int = 16, seed: int = 0) -> torch.Tensor:
        torch.manual_seed(seed)
        raw = torch.randn(n, dim)
        return raw / raw.norm(dim=1, keepdim=True)

    def test_output_shape_rectangular(self):
        assert cosine_block(self._unit(3), self._unit(7, seed=1)).shape == (3, 7)

    def test_output_shape_square(self):
        A = self._unit(4, seed=2)
        assert cosine_block(A, A).shape == (4, 4)

    def test_self_block_diagonal_is_one(self):
        A = self._unit(5, seed=3)
        assert np.allclose(np.diag(cosine_block(A, A)), 1.0, atol=1e-5)

    def test_output_is_numpy(self):
        assert isinstance(cosine_block(self._unit(3), self._unit(4, seed=1)), np.ndarray)

    def test_values_bounded(self):
        M = cosine_block(self._unit(6, seed=6), self._unit(8, seed=7))
        assert M.min() >= -1.0 - 1e-5
        assert M.max() <= 1.0 + 1e-5

    def test_orthogonal_rows_give_zero(self):
        A = torch.tensor([[1.0, 0.0]])
        B = torch.tensor([[0.0, 1.0]])
        assert abs(cosine_block(A, B)[0, 0]) < 1e-6

    def test_known_value(self):
        A = torch.tensor([[1.0, 0.0]])
        B = torch.tensor([[1.0, 1.0]]) / (2 ** 0.5)
        assert abs(cosine_block(A, B)[0, 0] - 2 ** 0.5 / 2) < 1e-5

    def test_agrees_with_cosine_matrix_for_square_case(self):
        A = self._unit(5, seed=10)
        assert np.allclose(cosine_block(A, A), cosine_matrix(A), atol=1e-5)
