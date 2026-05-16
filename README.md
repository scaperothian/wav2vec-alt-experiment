# wav2vec-alt-experiment

Minimal inference harness for the ALT/DALI singing-fine-tuned wav2vec2 encoder.

Loads a SpeechBrain-format checkpoint (`wav2vec2.ckpt`) into a vanilla HuggingFace
`Wav2Vec2Model`, runs an audio file (or a synthetic sine wave) through it, and prints
the resulting embedding shape and basic stats.

## Setup

```bash
poetry install
```

## Usage

```bash
# With a real audio file
poetry run wav2vec-alt --ckpt model/save/CKPT+2022-05-13+09-25-17+00/wav2vec2.ckpt --audio /path/to/file.wav

# Sanity check with synthetic sine (no audio file needed)
poetry run wav2vec-alt --ckpt model/save/CKPT+2022-05-13+09-25-17+00/wav2vec2.ckpt
```

## Notes

- The `model/` directory is excluded from git (large checkpoints). Keep it locally.
- Base model (`facebook/wav2vec2-large-960h-lv60-self`) is downloaded from HuggingFace on first run (~1.2 GB).
- Tested with PyTorch 2.7 + torchaudio 2.7 on macOS (MPS) and CPU.
