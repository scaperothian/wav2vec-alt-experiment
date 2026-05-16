"""Tests for wav2vec2-alt inference layer — model is mocked, no download required."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from src.config import TARGET_SR
from src.embed import embed_full_song


def _make_mock_model(n_hidden: int = 25, n_frames: int = 150, hidden_dim: int = 1024):
    """Fake Wav2Vec2Model whose output mimics the real one's hidden_states structure."""
    hidden_states = tuple(
        torch.randn(1, n_frames, hidden_dim) for _ in range(n_hidden)
    )
    out = SimpleNamespace(hidden_states=hidden_states)
    model = MagicMock()
    model.return_value = out
    return model, n_hidden, n_frames, hidden_dim


class TestEmbedFullSong:
    def _run(self, n_hidden=25, n_frames=150, hidden_dim=1024, duration_sec=2.0,
             output_layer_norm=None):
        model, n_hidden, n_frames, hidden_dim = _make_mock_model(n_hidden, n_frames, hidden_dim)
        waveform = torch.zeros(int(TARGET_SR * duration_sec))
        hidden = embed_full_song(waveform, model, output_layer_norm, device="cpu")
        return hidden, n_hidden, n_frames, hidden_dim

    def test_output_shape(self):
        hidden, n_hidden, n_frames, hidden_dim = self._run()
        assert hidden.shape == (n_hidden, n_frames, hidden_dim)

    def test_output_is_cpu_tensor(self):
        hidden, *_ = self._run()
        assert hidden.device.type == "cpu"

    def test_output_is_float32(self):
        hidden, *_ = self._run()
        assert hidden.dtype == torch.float32

    def test_hidden_state_count_preserved(self):
        for n in [1, 13, 25]:
            hidden, *_ = self._run(n_hidden=n)
            assert hidden.shape[0] == n

    def test_model_called_with_output_hidden_states_true(self):
        model, *_ = _make_mock_model()
        waveform = torch.zeros(TARGET_SR)
        embed_full_song(waveform, model, None, device="cpu")
        _, kwargs = model.call_args
        assert kwargs.get("output_hidden_states") is True

    def test_output_layer_norm_applied_to_last_layer(self):
        n_hidden, n_frames, hidden_dim = 25, 150, 1024
        model, *_ = _make_mock_model(n_hidden, n_frames, hidden_dim)
        norm = torch.nn.LayerNorm(hidden_dim)
        waveform = torch.zeros(int(TARGET_SR * 2.0))

        hidden_with = embed_full_song(waveform, model, norm, device="cpu")

        model2, *_ = _make_mock_model(n_hidden, n_frames, hidden_dim)
        model2.return_value = model.return_value  # same hidden states
        hidden_without = embed_full_song(waveform, model2, None, device="cpu")

        # All layers except the last should be identical
        assert torch.allclose(hidden_with[:-1], hidden_without[:-1], atol=1e-5)
        # Last layer should differ (LayerNorm was applied)
        assert not torch.allclose(hidden_with[-1], hidden_without[-1], atol=1e-5)

    def test_waveform_is_normalised_before_inference(self):
        """Verify the model receives a zero-mean unit-variance waveform."""
        model, *_ = _make_mock_model()
        torch.manual_seed(7)
        # Waveform with non-zero mean and variance
        waveform = torch.randn(TARGET_SR) * 3.0 + 5.0

        embed_full_song(waveform, model, None, device="cpu")

        passed_chunk = model.call_args[0][0]  # shape [1, samples]
        chunk_1d = passed_chunk.squeeze(0)
        assert abs(chunk_1d.mean().item()) < 0.01
        assert abs(chunk_1d.std().item() - 1.0) < 0.1

    def test_chunking_concatenates_correctly(self):
        # Use a very short chunk so the 3-second audio is split across chunks
        n_hidden, n_frames, hidden_dim = 5, 50, 32
        model, *_ = _make_mock_model(n_hidden, n_frames, hidden_dim)
        waveform = torch.zeros(int(TARGET_SR * 3.0))

        hidden = embed_full_song(waveform, model, None, device="cpu",
                                 chunk_sec=1.0)

        # 3 chunks of n_frames each → total_frames = 3 * n_frames
        assert hidden.shape == (n_hidden, n_frames * 3, hidden_dim)
