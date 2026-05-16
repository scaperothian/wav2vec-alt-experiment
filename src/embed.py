"""wav2vec2-alt model loading and full-song inference."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import Wav2Vec2Model

from .config import BASE_MODEL, EXPECTED_HIDDEN, TARGET_SR
from .download import resolve_checkpoint

EMBED_CHUNK_SEC = 30.0


def _strip_speechbrain_prefix(state_dict: dict) -> tuple[dict, int]:
    new_sd, n = {}, 0
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
            n += 1
        else:
            new_sd[k] = v
    return new_sd, n


def load_model(
    ckpt: Path | None,
    apply_output_norm: bool = True,
    device: str = "cpu",
) -> tuple[Wav2Vec2Model, torch.nn.LayerNorm | None]:
    """
    Load the SpeechBrain wav2vec2 checkpoint into a HuggingFace Wav2Vec2Model.

    Returns (model, output_layer_norm | None). The output_layer_norm mirrors
    SpeechBrain's output_norm=true wrapper and is applied only to the final
    transformer layer output. Intermediate layers are returned raw.
    """
    print(f"Loading base architecture: {BASE_MODEL}")
    model = Wav2Vec2Model.from_pretrained(BASE_MODEL)
    assert model.config.hidden_size == EXPECTED_HIDDEN

    ckpt_path = resolve_checkpoint(ckpt)
    print(f"Loading fine-tuned weights: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd, n_stripped = _strip_speechbrain_prefix(raw)
    print(f"  checkpoint keys: {len(raw)}  (stripped 'model.' from {n_stripped})")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    benign = {"masked_spec_embed"}
    real_missing = [k for k in missing if k not in benign]
    if real_missing:
        print(f"  WARNING: {len(real_missing)} unexpected missing keys")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected checkpoint keys")
    if not real_missing and not unexpected:
        print("  state_dict loaded cleanly")

    output_layer_norm = torch.nn.LayerNorm(EXPECTED_HIDDEN) if apply_output_norm else None

    model.eval().to(device)
    if output_layer_norm is not None:
        output_layer_norm.eval().to(device)

    return model, output_layer_norm


def embed_full_song(
    waveform: torch.Tensor,
    model: Wav2Vec2Model,
    output_layer_norm: torch.nn.LayerNorm | None,
    device: str,
    chunk_sec: float = EMBED_CHUNK_SEC,
) -> torch.Tensor:
    """
    Run wav2vec2 over the full waveform in fixed-size chunks, collecting all
    hidden states (CNN output + each transformer block).

    The waveform is zero-mean/unit-variance normalised once before chunking,
    matching the SpeechBrain feature-extractor path. output_layer_norm is
    applied only to the last hidden state (index -1) when provided.

    Args:
        waveform:          1-D float tensor at TARGET_SR
        model:             Wav2Vec2Model (on device, eval mode)
        output_layer_norm: LayerNorm for the final layer, or None
        device:            torch device string
        chunk_sec:         seconds of audio per forward pass (default 30 s)

    Returns:
        [num_hidden_states, total_frames, hidden_dim]
        For wav2vec2-large: num_hidden_states == 25 (1 CNN + 24 transformer)
    """
    from tqdm import tqdm

    # Per-utterance normalisation (matches SpeechBrain's do_normalize=True path)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)

    chunk_samples = int(chunk_sec * TARGET_SR)
    total_samples = waveform.shape[0]
    n_chunks = math.ceil(total_samples / chunk_samples)
    duration_sec = total_samples / TARGET_SR

    all_hidden: list[torch.Tensor] = []
    bar = tqdm(
        total=duration_sec,
        unit="s",
        unit_scale=False,
        desc="Embedding",
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
    )
    with bar:
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[start:end].unsqueeze(0).to(device)

            with torch.inference_mode():
                out = model(chunk, output_hidden_states=True)

            # hidden_states: tuple of tensors, each [1, T, D]
            hidden = torch.stack(out.hidden_states, dim=0).squeeze(1).cpu()  # [L+1, T, D]

            if output_layer_norm is not None:
                hidden[-1] = output_layer_norm(hidden[-1].to(device)).detach().cpu()

            all_hidden.append(hidden)
            bar.update((end - start) / TARGET_SR)

    return torch.cat(all_hidden, dim=1)  # [L+1, total_T, D]
