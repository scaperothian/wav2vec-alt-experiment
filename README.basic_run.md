# wav2vec-alt-experiment

Minimal inference harness for the ALT/DALI singing-fine-tuned wav2vec2 encoder.

Loads a SpeechBrain-format checkpoint (`wav2vec2.ckpt`) into a vanilla HuggingFace
`Wav2Vec2Model`, runs an audio file (or a synthetic sine wave) through it, and prints
the resulting embedding shape and basic stats.

## Setup

```bash
poetry install
```

## Model checkpoint

The checkpoint is not tracked in git. It is resolved automatically in this order:

1. `--ckpt <path>` — explicit path, validated before loading
2. `model/save/CKPT+*/wav2vec2.ckpt` — auto-detected if the `model/` directory is present
3. `wav2vec-alt-model.zip` in the project root — drop the zip here and it will be extracted automatically
4. Google Drive — downloaded automatically if `GDRIVE_MODEL_URL` is set in `src/download.py`

To use the manual zip option, just place `wav2vec-alt-model.zip` in the project root and run normally — no flags needed.

## Usage

```bash
# Auto-resolve checkpoint (local model dir, local zip, or download)
poetry run python basic_run.py

# With a specific audio file
poetry run python basic_run.py --audio /path/to/file.wav

# With an explicit checkpoint path
poetry run python basic_run.py --ckpt model/save/CKPT+2022-05-13+09-25-17+00/wav2vec2.ckpt

# Force CPU
poetry run python basic_run.py --device cpu
```

## Notes

- The `model/` directory is excluded from git (large checkpoints). Keep it locally.
- Base model (`facebook/wav2vec2-large-960h-lv60-self`) is downloaded from HuggingFace on first run (~1.2 GB).
- Tested with PyTorch 2.7 + torchaudio 2.7 on macOS (MPS) and CPU.

## Tests

```bash
poetry run pytest
```
