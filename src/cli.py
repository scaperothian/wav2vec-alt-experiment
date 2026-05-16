"""
wav2vec-sim: section prototype vs. all-frame similarity via wav2vec-alt embeddings.

For each probed transformer layer of the ALT/DALI singing-fine-tuned wav2vec2:
  1. Compute raw window means across the full song.
  2. Fit centering / ZCA-whitening from those means (both on by default).
  3. Apply the transform, then L2-normalize, to get frame embeddings F.
  4. Build section prototypes P by transforming each section's windows first,
     averaging, then normalizing.  (Transform-before-average is critical:
     averaging un-transformed windows re-introduces the song-mean we removed.)
  5. Compute A = cosine_block(P, F) — [N_sections × N_frames].
  6. Report argmax accuracy and mean off-diagonal prototype similarity.
  7. Plot similarity scores as a line chart over time (enabled by default).

Probed layers default to [6, 12, 18, 24] (wav2vec2-large has 24 transformer
layers; layer 0 is the CNN feature extractor output).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import HOP_SEC, LAYERS_TO_PROBE, WAV2VEC_FRAME_RATE, WINDOW_SEC
from .embed import embed_full_song, load_model
from .io import load_audio, load_song
from .plotting import (
    _layer_mean_path,
    _pairwise_path,
    plot_layer_mean_similarity,
    plot_prototype_similarity,
    plot_section_grids,
)
from .similarity import cosine_block, cosine_matrix
from .transform import apply_transform, fit_transform, l2_normalize_rows
from .windows import (
    frame_to_section_assignment,
    section_windows,
    whole_song_window_spans,
    window_means,
)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

def analyze_layer(
    frames: torch.Tensor,
    sections: list[dict],
    all_spans: list[tuple[int, int]],
    section_of: list[int],
    layer_idx: int,
    center: bool,
    whiten: bool,
    window: float = WINDOW_SEC,
    hop: float = HOP_SEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build prototypes, frame embeddings, and prototype-vs-frame similarity
    for one transformer layer, with optional centering and ZCA whitening.

    Returns:
        A: [N_sec, N_frames]  prototype-vs-frame cosine similarity
        P: [N_sec, D]         section prototype embeddings (numpy)
        F: [N_frames, D]      all-frame embeddings (numpy)
    """
    F_raw = window_means(frames, all_spans)
    section_raw = [
        window_means(frames, list(section_windows(s["start"], s["stop"], window=window, hop=hop)))
        for s in sections
    ]

    mu, W = fit_transform(F_raw, center=center, whiten=whiten)

    F_norm = l2_normalize_rows(apply_transform(F_raw, mu, W))  # [N_frames, D]

    prototypes = []
    for raw_wins in section_raw:
        if raw_wins.shape[0] == 0:
            raise ValueError(
                "A section produced zero windows — is WINDOW_SEC larger than "
                "the section duration?"
            )
        transformed = apply_transform(raw_wins, mu, W)
        proto = transformed.mean(dim=0, keepdim=True)
        proto = l2_normalize_rows(proto).squeeze(0)
        prototypes.append(proto)
    P = torch.stack(prototypes)  # [N_sec, D]

    A = cosine_block(P, F_norm)  # [N_sec, N_frames]

    argmax = A.argmax(axis=0)
    correct = sum(1 for k, gt in enumerate(section_of) if gt >= 0 and argmax[k] == gt)
    total   = sum(1 for gt in section_of if gt >= 0)
    acc     = f"{correct}/{total} = {correct / total:.1%}" if total else "n/a"

    P_sim    = cosine_block(P, P)
    n        = P_sim.shape[0]
    mask     = ~np.eye(n, dtype=bool)
    mean_off = float(P_sim[mask].mean()) if mask.any() else float("nan")

    print(
        f"  Layer {layer_idx}: argmax accuracy {acc}  |  "
        f"mean off-diagonal prototype sim: {mean_off:.3f}"
        + (" (lower = more separated)" if n > 1 else "")
    )

    return A, P.numpy(), F_norm.numpy()


# ---------------------------------------------------------------------------
# Shared analysis core
# ---------------------------------------------------------------------------

def _transform_tag(center: bool, whiten: bool) -> str:
    if center and whiten:
        return "centered+whitened"
    if center:
        return "centered"
    return "baseline"


def _run(
    audio_path: Path,
    sections: list[dict],
    ckpt: Path | None,
    apply_output_norm: bool,
    device: str,
    center: bool,
    whiten: bool,
    also_pairwise: bool,
    plot: bool,
    plot_output: Path | None,
    save_npz: Path | None,
    audio_name: str,
    window: float = WINDOW_SEC,
    hop: float = HOP_SEC,
    smooth_k: int = 3,
) -> None:
    tag = _transform_tag(center, whiten)
    print(f"Transform: {tag}  |  window={window}s  hop={hop}s")

    model, output_layer_norm = load_model(ckpt, apply_output_norm, device)
    wav = load_audio(audio_path)
    duration = wav.shape[0] / 16_000
    print(f"  Duration : {duration:.2f}s  ({wav.shape[0]:,} samples @ 16000 Hz)")

    print("Running wav2vec2-alt inference...")
    t0 = time.perf_counter()
    hidden = embed_full_song(wav, model, output_layer_norm, device)
    embed_time = time.perf_counter() - t0

    n_layers, total_frames, hidden_dim = hidden.shape
    print(
        f"  Layers   : {n_layers}  |  Frames: {total_frames:,}  |  Dim: {hidden_dim}"
    )
    print(
        f"  Time     : {embed_time:.1f}s  "
        f"({duration / embed_time:.1f}× real-time)"
    )

    all_spans  = whole_song_window_spans(total_frames, window=window, hop=hop)
    section_of = [frame_to_section_assignment(sp, sections) for sp in all_spans]
    timestamps = np.array([(s + e) / 2 / WAV2VEC_FRAME_RATE for s, e in all_spans])

    print("\nAnalysing layers:")
    layer_A: dict[int, np.ndarray] = {}
    layer_P: dict[int, np.ndarray] = {}
    layer_F: dict[int, np.ndarray] = {}

    for layer_idx in LAYERS_TO_PROBE:
        if layer_idx >= hidden.shape[0]:
            continue
        A, P, F = analyze_layer(
            hidden[layer_idx], sections, all_spans, section_of,
            layer_idx, center, whiten, window=window, hop=hop,
        )
        layer_A[layer_idx] = A
        layer_P[layer_idx] = P
        layer_F[layer_idx] = F

    # --- Build all plots, then show once ------------------------------------
    if plot:
        layer_results = {k: (layer_A[k], timestamps) for k in layer_A}
        plot_prototype_similarity(
            layer_results, sections, audio_name,
            output_path=plot_output,
            transform_tag=tag,
        )
        lm_output = _layer_mean_path(plot_output) if plot_output else None
        plot_layer_mean_similarity(
            layer_A, timestamps, sections, audio_name,
            output_path=lm_output,
            transform_tag=tag,
            smooth_k=smooth_k,
        )

    # --- Section×section grid -----------------------------------------------
    if also_pairwise:
        section_labels = [s["label"][:20] for s in sections]
        layer_grids = {
            k: cosine_matrix(torch.from_numpy(layer_P[k]))
            for k in layer_P
        }
        if plot:
            grid_output = _pairwise_path(plot_output) if plot_output else None
            plot_section_grids(layer_grids, section_labels, audio_name, grid_output)
        if save_npz:
            pairwise_npz = _pairwise_path(save_npz)
            np.savez(str(pairwise_npz), **{f"layer{k}": v for k, v in layer_grids.items()})
            print(f"Pairwise matrix saved to {pairwise_npz}")

    # --- Show all interactive figures at once --------------------------------
    if plot and not plot_output:
        import matplotlib.pyplot as plt
        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close("all")

    # --- Save npz ------------------------------------------------------------
    if save_npz:
        flat: dict = {
            "section_starts": np.array([s["start"] for s in sections]),
            "section_stops":  np.array([s["stop"]  for s in sections]),
            "timestamps":     timestamps,
            "transform":      np.bytes_(tag),
        }
        for k in layer_A:
            flat[f"layer{k}_A"] = layer_A[k]
            flat[f"layer{k}_P"] = layer_P[k]
            flat[f"layer{k}_F"] = layer_F[k]
        np.savez(str(save_npz), **flat)
        print(f"Results saved to {save_npz}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wav2vec-sim",
        description=(
            "Section prototype vs. all-frame similarity via wav2vec-alt embeddings. "
            "Mean-centering and ZCA whitening are applied by default to combat "
            "embedding anisotropy. Plots similarity scores over time by default."
        ),
    )
    p.add_argument(
        "input",
        help="Path to a ProPresenter JSON manifest OR an audio file (.wav, .mp3, …).",
    )

    # Checkpoint / model flags (wav2vec-alt specific)
    p.add_argument(
        "--ckpt",
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Path to wav2vec2.ckpt. If omitted, auto-detected from model/save/, "
            "extracted from wav2vec-alt-model.zip, or downloaded from Google Drive."
        ),
    )
    p.add_argument(
        "--no-output-norm",
        action="store_true",
        help=(
            "Skip the final LayerNorm on the last transformer layer output. "
            "The DALI yaml sets output_norm=true, so this is applied by default."
        ),
    )

    # Transform flags (both on by default)
    p.add_argument(
        "--no-center",
        action="store_true",
        help="Disable song-mean centering (centering is on by default).",
    )
    p.add_argument(
        "--no-whiten",
        action="store_true",
        help="Disable ZCA whitening (whitening is on by default).",
    )

    # Windowing parameters
    p.add_argument(
        "--window",
        type=float,
        default=WINDOW_SEC,
        metavar="SEC",
        help=(
            f"Sliding window length in seconds (default: {WINDOW_SEC}). "
            "Each embedding averages this many seconds of wav2vec2 frames."
        ),
    )
    p.add_argument(
        "--hop",
        type=float,
        default=HOP_SEC,
        metavar="SEC",
        help=(
            f"Hop size between windows in seconds (default: {HOP_SEC}). "
            "Controls how often a new embedding is computed."
        ),
    )

    # Plot flags
    p.add_argument(
        "--smooth-k",
        type=int,
        default=3,
        metavar="K",
        help=(
            "Smoothing window length for the layer-mean summary plot (default: 3). "
            "Each point is averaged with the K-1 preceding points. "
            "K=1 disables smoothing."
        ),
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress all matplotlib output.",
    )
    p.add_argument(
        "--plot-output",
        default=None,
        metavar="FILE",
        help=(
            "Save plots to FILE instead of displaying them. "
            "The layer-mean summary is saved with a '_layermean' suffix. "
            "When --also-pairwise is set, the section grid uses '_pairwise'."
        ),
    )
    p.add_argument(
        "--also-pairwise",
        action="store_true",
        help=(
            "Compute the pooled section×section similarity grid and plot it. "
            "If --save-npz is set, also saves the grid as a separate _pairwise.npz."
        ),
    )
    p.add_argument(
        "--save-npz",
        default=None,
        metavar="FILE",
        help="Save similarity matrices and embeddings to a .npz file.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    center = not args.no_center
    whiten = not args.no_whiten
    if whiten and not center:
        parser.error("--no-center cannot be combined with whitening; "
                     "add --no-whiten or remove --no-center.")

    input_path = Path(args.input)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if input_path.suffix.lower() == ".json":
        audio_path, sections = load_song(input_path)
        audio_name = input_path.stem

    elif input_path.suffix.lower() in AUDIO_SUFFIXES:
        import torchaudio
        info = torchaudio.info(str(input_path))
        duration = info.num_frames / info.sample_rate
        sections = [{"label": "full", "start": 0.0, "stop": duration}]
        audio_path = input_path
        audio_name = input_path.stem

    else:
        parser.error(
            f"Unrecognised input '{input_path}'. "
            f"Expected a .json or an audio file ({', '.join(sorted(AUDIO_SUFFIXES))})."
        )
        return

    print(f"\nLoading wav2vec2-alt on {device}...")
    print(f"Audio: {audio_path}")
    print(f"Sections ({len(sections)}):")
    for i, s in enumerate(sections):
        print(f"  [{i}] {s['start']:6.2f}-{s['stop']:6.2f}s :: {s['label']}")
    print()

    try:
        _run(
            audio_path=audio_path,
            sections=sections,
            ckpt=args.ckpt,
            apply_output_norm=not args.no_output_norm,
            device=device,
            center=center,
            whiten=whiten,
            also_pairwise=args.also_pairwise,
            plot=not args.no_plot,
            plot_output=Path(args.plot_output) if args.plot_output else None,
            save_npz=Path(args.save_npz) if args.save_npz else None,
            audio_name=audio_name,
            window=args.window,
            hop=args.hop,
            smooth_k=args.smooth_k,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
