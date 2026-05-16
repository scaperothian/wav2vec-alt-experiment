# wav2vec-alt-experiment

Section prototype vs. all-frame similarity analysis using the ALT/DALI
singing-fine-tuned wav2vec2 encoder (SpeechBrain checkpoint).

Loads the fine-tuned checkpoint into a vanilla HuggingFace `Wav2Vec2Model`,
embeds an audio file in 30-second chunks across all transformer layers, then
computes centering + ZCA-whitened cosine similarity between section prototypes
and every frame in the song. Produces the same style of plots as `mert-experiment`.

## Setup

```bash
poetry install
```

## Model checkpoint

The checkpoint is not tracked in git. It is resolved automatically in this order:

1. `--ckpt <path>` — explicit path, validated before loading
2. `model/save/CKPT+*/wav2vec2.ckpt` — auto-detected if the `model/` directory is present
3. `wav2vec-alt-model.zip` in the project root — extracted automatically
4. Google Drive — downloaded automatically if `GDRIVE_MODEL_URL` is set in `src/download.py`

## Usage

```bash
# Audio file — treated as one section spanning the full duration
poetry run wav2vec-alt song.wav

# ProPresenter JSON — sections defined by slide timestamps
poetry run wav2vec-alt song.json

# Explicit checkpoint path
poetry run wav2vec-alt song.wav --ckpt model/save/CKPT+.../wav2vec2.ckpt

# Save plots to files instead of displaying interactively
poetry run wav2vec-alt song.json --plot-output out.png

# Also compute section×section pairwise grid
poetry run wav2vec-alt song.json --also-pairwise

# Save all matrices to a .npz file
poetry run wav2vec-alt song.json --save-npz results.npz

# Disable transforms (baseline cosine without centering or whitening)
poetry run wav2vec-alt song.wav --no-center --no-whiten

# Skip the final LayerNorm (mirrors the original --no-output-norm behaviour)
poetry run wav2vec-alt song.wav --no-output-norm
```

## Plots produced

| Plot | Description |
|------|-------------|
| `out.png` | Per-layer prototype similarity over time — one subplot per probed layer |
| `out_layermean.png` | Layer-mean summary (raw + smoothed) |
| `out_pairwise.png` | Section×section cosine grid (requires `--also-pairwise`) |

## Model details

- Base: `facebook/wav2vec2-large-960h-lv60-self` (24 transformer layers, hidden=1024)
- Fine-tuned on DALI/ALT singing data via SpeechBrain
- Frame rate: 50 Hz (CNN stride 320 at 16 kHz)
- Layers probed by default: 6, 12, 18, 24
- Base model downloaded from HuggingFace on first run (~1.2 GB)
- `model/` directory is excluded from git (large checkpoints)

## Tests

```bash
poetry run pytest
```
