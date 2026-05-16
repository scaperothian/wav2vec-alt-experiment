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

- The checkpoint was saved by SpeechBrain, so weight keys are prefixed `model.` — `strip_speechbrain_prefix()` in `main.py` handles that.
- `output_norm: true` in `hyperparams.yaml` means a `LayerNorm` is applied after the encoder; `main.py` mirrors this by default. Pass `--no-output-norm` to skip.
- Base model is downloaded from HuggingFace on first run (~1.2 GB); subsequent runs use the local cache.
- `model/` is gitignored — do not commit checkpoint files.

## Commands

```bash
poetry install                        # set up env
poetry run wav2vec-alt --ckpt model/save/CKPT+2022-05-13+09-25-17+00/wav2vec2.ckpt
```
