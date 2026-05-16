# CLAUDE.md

## Project

Inference-only harness for a singing-fine-tuned wav2vec2 model (ALT/DALI, SpeechBrain checkpoint).
Entry point: `src/main.py`. No training code here — the `model/` directory holds the pre-trained checkpoint and is not tracked in git.

## Environment

- Python 3.12 via Poetry
- PyTorch 2.7 + torchaudio 2.7
- transformers for `Wav2Vec2Model`

Run everything through `poetry run` or activate the venv with `poetry shell`.

## Key facts

- The checkpoint was saved by SpeechBrain, so weight keys are prefixed `model.` — `strip_speechbrain_prefix()` in `src/main.py` handles that.
- `output_norm: true` in `hyperparams.yaml` means a `LayerNorm` is applied after the encoder; `src/main.py` mirrors this by default. Pass `--no-output-norm` to skip.
- Base model is downloaded from HuggingFace on first run (~1.2 GB); subsequent runs use the local cache.
- `model/` is gitignored — do not commit checkpoint files.

## Checkpoint resolution (src/download.py)

Checkpoints are resolved in this priority order:

1. `--ckpt <path>` — explicit, validated to exist before use
2. `model/save/CKPT+*/wav2vec2.ckpt` — auto-detected (newest CKPT+ dir wins)
3. `wav2vec-alt-model.zip` in the project root — extracted into `model/save/downloaded/`, zip is left intact
4. Google Drive download — `GDRIVE_MODEL_URL` in `src/download.py`, downloaded zip is deleted after extraction

## Commands

```bash
poetry install                  # set up env
poetry run wav2vec-alt          # run with auto-resolved checkpoint
poetry run wav2vec-alt --audio /path/to/file.wav
poetry run pytest               # run tests
```
