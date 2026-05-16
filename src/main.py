#!/usr/bin/env python3
"""
hello_alt.py — minimal "does it work" script for the ALT_SpeechBrain
DALI/ALT fine-tuned wav2vec2 encoder.

Loads the singing-fine-tuned wav2vec2 weights into a vanilla HuggingFace
Wav2Vec2Model (bypassing SpeechBrain entirely), runs one audio file
through it, and prints the resulting embedding shape and basic stats.

What this matches in the original training code:
    Their forward (compute_forward in train_wav2vec2_tb.py):
        feats = self.modules.wav2vec2(wavs)
    Their wav2vec2 wrapper (hyperparams.yaml):
        speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2
        source: facebook/wav2vec2-large-960h-lv60-self
        output_norm: true   <-- we apply LayerNorm to match
    So `feats` is the last hidden state of the HF Wav2Vec2Model,
    followed by a LayerNorm over the hidden dim.

Tested against PyTorch 2.7 + transformers >=4.30 + torchaudio >=2.2.
The original repo pins torch 1.9.1, but we only need inference here, so
the version drift is safe.

Usage:
    python hello_alt.py --ckpt path/to/wav2vec2.ckpt --audio path/to/file.wav
    python hello_alt.py --ckpt path/to/wav2vec2.ckpt          # generates a 3s sine
    python hello_alt.py --ckpt path/to/wav2vec2.ckpt --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import Wav2Vec2Model

BASE_MODEL = "facebook/wav2vec2-large-960h-lv60-self"
TARGET_SR = 16000
EXPECTED_HIDDEN = 1024


def strip_speechbrain_prefix(state_dict: dict) -> dict:
    """
    SpeechBrain's HuggingFaceWav2Vec2 wraps the HF model as
    `self.model = Wav2Vec2Model(...)`, so the saved checkpoint keys are
    prefixed `model.` (e.g. `model.encoder.layers.0.attention.k_proj.weight`).
    HF's Wav2Vec2Model expects keys without that prefix. Strip it.
    """
    new_sd = {}
    n_stripped = 0
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
            n_stripped += 1
        else:
            new_sd[k] = v
    return new_sd, n_stripped


def load_audio_or_sine(audio_path: Path | None) -> torch.Tensor:
    """
    Returns a 1-D mono float32 waveform at TARGET_SR.
    If audio_path is None, synthesizes a 3-second 440 Hz sine so this
    script can be run for a sanity check without any audio file.
    """
    if audio_path is None:
        duration_s = 3.0
        t = torch.arange(int(duration_s * TARGET_SR), dtype=torch.float32) / TARGET_SR
        wav = 0.1 * torch.sin(2 * torch.pi * 440.0 * t)
        print(f"  (no --audio given; using synthetic 3 s 440 Hz sine)")
        return wav

    # Lazy import: torchaudio isn't strictly needed for the sine path.
    import torchaudio

    wav, sr_in = torchaudio.load(str(audio_path))  # (channels, samples), float32
    print(f"  loaded: {audio_path}  shape={tuple(wav.shape)}  sr={sr_in}")

    # Downmix to mono.
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample to 16 kHz if needed.
    if sr_in != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr_in, TARGET_SR)
        wav = resampler(wav)
        print(f"  resampled {sr_in} -> {TARGET_SR}")

    return wav.squeeze(0)  # (samples,)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Path to wav2vec2.ckpt from the kept CKPT folder.",
    )
    ap.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Path to an audio file (any format torchaudio supports). "
        "Omit to use a synthetic sine wave.",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available).",
    )
    ap.add_argument(
        "--no-output-norm",
        action="store_true",
        help="Skip the final LayerNorm. The DALI yaml sets output_norm=true, "
        "so by default we apply it to match.",
    )
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1
    if args.audio is not None and not args.audio.exists():
        print(f"ERROR: audio not found: {args.audio}", file=sys.stderr)
        return 1

    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device:         {args.device}")
    print()

    # --- 1. Build the architecture shell from the HF hub ---
    # This downloads ~1.2 GB of the BASE model on first run. We only
    # need it for the architecture + tensor shapes; we'll overwrite
    # the weights immediately with the fine-tuned checkpoint.
    print(f"[1/4] Loading base architecture: {BASE_MODEL}")
    t0 = time.time()
    model = Wav2Vec2Model.from_pretrained(BASE_MODEL)
    print(f"      hidden size: {model.config.hidden_size}")
    assert model.config.hidden_size == EXPECTED_HIDDEN, (
        f"Unexpected hidden size {model.config.hidden_size}; this script "
        f"assumes wav2vec2-large with hidden={EXPECTED_HIDDEN}."
    )
    print(f"      done in {time.time()-t0:.1f}s")

    # --- 2. Load the singing-tuned weights and strip SpeechBrain prefix ---
    print(f"[2/4] Loading fine-tuned weights: {args.ckpt}")
    t0 = time.time()
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd, n_stripped = strip_speechbrain_prefix(raw)
    print(f"      checkpoint keys: {len(raw)}  (stripped 'model.' prefix from {n_stripped})")

    missing, unexpected = model.load_state_dict(sd, strict=False)

    # The HF Wav2Vec2Model has a `masked_spec_embed` parameter that is
    # used only during pretraining; the fine-tuned ckpt won't have it.
    # That's the only "missing" key we expect — anything else is suspect.
    benign_missing = {"masked_spec_embed"}
    real_missing = [k for k in missing if k not in benign_missing]
    if real_missing:
        print(f"      WARNING: {len(real_missing)} unexpected missing keys:")
        for k in real_missing[:5]:
            print(f"        - {k}")
        if len(real_missing) > 5:
            print(f"        ... and {len(real_missing)-5} more")
    if unexpected:
        print(f"      WARNING: {len(unexpected)} unexpected keys in checkpoint:")
        for k in unexpected[:5]:
            print(f"        - {k}")
    if not real_missing and not unexpected:
        print(f"      state_dict loaded cleanly")
    print(f"      done in {time.time()-t0:.1f}s")

    # The DALI yaml has output_norm: true. SpeechBrain applies a
    # LayerNorm over the hidden dim to the wav2vec2 output. We mirror
    # that here. (HF's own model does NOT do this — output_norm is a
    # SpeechBrain wrapper concept.)
    apply_output_norm = not args.no_output_norm
    output_layer_norm = torch.nn.LayerNorm(EXPECTED_HIDDEN) if apply_output_norm else None
    if apply_output_norm:
        # SpeechBrain's LayerNorm is freshly initialized (gamma=1, beta=0)
        # at module construction. The checkpoint doesn't save it under a
        # named key we can map to easily, but identity init is what
        # gets used at inference for output_norm anyway in their wrapper
        # path. If you find this is wrong, --no-output-norm disables it.
        # See: speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2
        pass

    model.eval().to(args.device)
    if output_layer_norm is not None:
        output_layer_norm.eval().to(args.device)
    print()

    # --- 3. Load and prepare audio ---
    print(f"[3/4] Preparing audio")
    wav = load_audio_or_sine(args.audio)
    # wav: (samples,) float32 in [-1, 1] at 16 kHz
    print(f"      waveform: {wav.shape[0]} samples ({wav.shape[0]/TARGET_SR:.2f} s)  "
          f"min={wav.min().item():.3f}  max={wav.max().item():.3f}")

    # wav2vec2-large-960h-lv60-self uses do_normalize=True (zero-mean,
    # unit-variance per-utterance). The SpeechBrain wrapper handles this
    # via the HF feature extractor; we replicate the same normalization
    # directly to avoid pulling in the feature extractor for one line.
    wav = (wav - wav.mean()) / (wav.std() + 1e-7)

    inputs = wav.unsqueeze(0).to(args.device)  # (1, samples)
    print()

    # --- 4. Forward pass ---
    print(f"[4/4] Forward pass")
    t0 = time.time()
    with torch.inference_mode():
        out = model(inputs)
        hidden = out.last_hidden_state  # (1, T, 1024)
        if output_layer_norm is not None:
            hidden = output_layer_norm(hidden)
    elapsed = time.time() - t0

    print(f"      embedding shape:  {tuple(hidden.shape)}")
    print(f"      dtype:            {hidden.dtype}")
    print(f"      frames:           {hidden.shape[1]}  "
          f"(~{hidden.shape[1] / (wav.shape[-1]/TARGET_SR):.2f} Hz frame rate)")
    print(f"      mean / std:       {hidden.mean().item():+.4f} / {hidden.std().item():.4f}")
    print(f"      forward time:     {elapsed*1000:.1f} ms on {args.device}")
    print()
    print("HELLO WORLD OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
