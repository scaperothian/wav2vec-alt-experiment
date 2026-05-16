"""Tests for mean-centering and ZCA whitening."""

import numpy as np
import pytest
import torch

from src.transform import (
    WHITEN_EPS,
    apply_transform,
    fit_mu,
    fit_transform,
    fit_whitening_matrix,
    l2_normalize_rows,
)


class TestFitMu:
    def test_shape(self):
        X = torch.randn(20, 16)
        assert fit_mu(X).shape == (1, 16)

    def test_equals_row_mean(self):
        X = torch.randn(10, 8)
        assert torch.allclose(fit_mu(X), X.mean(dim=0, keepdim=True), atol=1e-5)

    def test_single_row(self):
        X = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(fit_mu(X), X, atol=1e-6)

    def test_constant_rows_equal_that_value(self):
        v = torch.tensor([3.0, -1.0, 2.0])
        X = v.unsqueeze(0).expand(5, -1)
        assert torch.allclose(fit_mu(X), v.unsqueeze(0), atol=1e-6)


class TestFitWhiteningMatrix:
    def test_output_shape(self):
        X = torch.randn(50, 8)
        assert fit_whitening_matrix(X).shape == (8, 8)

    def test_approximately_decorrelates(self):
        torch.manual_seed(0)
        n, d = 200, 8
        X = torch.randn(n, d)
        X = X - X.mean(dim=0)
        W = fit_whitening_matrix(X)
        cov = (X @ W).T @ (X @ W) / (n - 1)
        assert torch.allclose(cov, torch.eye(d), atol=0.1)

    def test_eps_prevents_zero_division(self):
        X = torch.ones(10, 4)
        W = fit_whitening_matrix(X, eps=1e-4)
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()

    def test_result_is_symmetric(self):
        torch.manual_seed(1)
        X = torch.randn(30, 6) - torch.randn(30, 6).mean(dim=0)
        W = fit_whitening_matrix(X)
        assert torch.allclose(W, W.T, atol=1e-4)


class TestApplyTransform:
    def test_center_only_subtracts_mu(self):
        X = torch.randn(10, 8)
        out = apply_transform(X, torch.ones(1, 8), None)
        assert torch.allclose(out, X - 1.0, atol=1e-6)

    def test_zero_mu_identity(self):
        X = torch.randn(5, 4)
        out = apply_transform(X, torch.zeros(1, 4), None)
        assert torch.allclose(out, X, atol=1e-6)

    def test_whiten_changes_values(self):
        torch.manual_seed(2)
        X = torch.randn(20, 6)
        mu = X.mean(dim=0, keepdim=True)
        W = fit_whitening_matrix(X - mu)
        assert not torch.allclose(apply_transform(X, mu, W), X - mu, atol=1e-3)

    def test_output_shape_preserved(self):
        X = torch.randn(7, 12)
        assert apply_transform(X, torch.zeros(1, 12), None).shape == X.shape

    def test_output_not_normalized(self):
        X = torch.randn(5, 4) * 10
        out = apply_transform(X, torch.zeros(1, 4), None)
        norms = out.norm(dim=1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=0.1)


class TestL2NormalizeRows:
    def test_all_rows_unit_norm(self):
        torch.manual_seed(3)
        out = l2_normalize_rows(torch.randn(10, 16))
        assert torch.allclose(out.norm(dim=1), torch.ones(10), atol=1e-5)

    def test_shape_preserved(self):
        X = torch.randn(7, 5)
        assert l2_normalize_rows(X).shape == X.shape

    def test_zero_row_does_not_nan(self):
        assert not torch.isnan(l2_normalize_rows(torch.zeros(3, 4))).any()

    def test_already_unit_rows_unchanged(self):
        torch.manual_seed(4)
        raw = torch.randn(5, 8)
        unit = raw / raw.norm(dim=1, keepdim=True)
        assert torch.allclose(l2_normalize_rows(unit), unit, atol=1e-5)


class TestFitTransform:
    def _frames(self, n=50, d=16, seed=0):
        torch.manual_seed(seed)
        return torch.randn(n, d)

    def test_center_only_returns_none_for_W(self):
        mu, W = fit_transform(self._frames(), center=True, whiten=False)
        assert W is None
        assert mu.shape == (1, 16)

    def test_whiten_returns_matrix(self):
        mu, W = fit_transform(self._frames(), center=True, whiten=True)
        assert W is not None and W.shape == (16, 16)

    def test_no_center_no_whiten_mu_is_zeros(self):
        mu, W = fit_transform(self._frames(), center=False, whiten=False)
        assert torch.allclose(mu, torch.zeros(1, 16))
        assert W is None

    def test_whiten_without_center_raises(self):
        with pytest.raises(ValueError, match="center"):
            fit_transform(self._frames(), center=False, whiten=True)

    def test_mu_equals_frame_mean(self):
        X = self._frames()
        mu, _ = fit_transform(X, center=True, whiten=False)
        assert torch.allclose(mu, X.mean(dim=0, keepdim=True), atol=1e-6)

    def test_pipeline_produces_unit_norm_frames(self):
        X = self._frames()
        mu, W = fit_transform(X, center=True, whiten=True)
        F = l2_normalize_rows(apply_transform(X, mu, W))
        assert torch.allclose(F.norm(dim=1), torch.ones(50), atol=1e-5)
