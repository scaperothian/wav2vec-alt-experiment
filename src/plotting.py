"""Matplotlib visualisations for wav2vec-alt similarity analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_prototype_similarity(
    layer_results: dict[int, tuple[np.ndarray, np.ndarray]],
    sections: list[dict],
    audio_name: str = "",
    output_path: str | Path | None = None,
    transform_tag: str = "",
) -> None:
    """
    Line plot of section-prototype similarity scores over time.

    One subplot per layer, stacked vertically with a shared time axis.
    One line per section prototype; vertical dashed lines at section boundaries.

    Args:
        layer_results:  layer_idx -> (A [N_sec, N_frames], timestamps [N_frames])
        sections:       list of dicts with 'label', 'start', 'stop'
        audio_name:     used in the figure title
        output_path:    save to file if given; leave open for plt.show() otherwise
        transform_tag:  e.g. "centered+whitened" — appended to the title
    """
    import matplotlib.pyplot as plt

    layers = sorted(layer_results)
    n_layers = len(layers)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(
        n_layers, 1,
        figsize=(14, 3 * n_layers),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, layer_idx in zip(axes, layers):
        A, timestamps = layer_results[layer_idx]
        for i in range(A.shape[0]):
            label = sections[i]["label"][:20] if i < len(sections) else f"sec{i}"
            ax.plot(
                timestamps, A[i],
                marker="o", markersize=3, linewidth=1,
                color=colors[i % len(colors)],
                label=label,
            )
        boundaries = {s["start"] for s in sections} | {sections[-1]["stop"]}
        for t in sorted(boundaries):
            ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("Cosine similarity", fontsize=8)
        ax.margins(y=0.1)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    parts = ["Section prototype similarity (wav2vec-alt)"]
    if transform_tag:
        parts.append(f"({transform_tag})")
    if audio_name:
        parts.append(f"— {audio_name}")
    fig.suptitle("  ".join(parts), fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_section_grids(
    layer_matrices: dict[int, np.ndarray],
    section_labels: list[str],
    audio_name: str = "",
    output_path: str | Path | None = None,
) -> None:
    """
    Grid heatmap of pooled section×section cosine similarity, one panel per layer.

    Cells show numeric scores on a RdYlGn background (−1 to 1). Panels in one row.
    """
    import matplotlib.pyplot as plt

    layers = sorted(layer_matrices)
    n_layers = len(layers)
    n_sec = len(section_labels)
    cell_size = max(1.5, n_sec * 0.7)

    fig, axes = plt.subplots(
        1, n_layers,
        figsize=(cell_size * n_layers + 1, cell_size + 1.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, layer_idx in zip(axes, layers):
        M = layer_matrices[layer_idx]
        im = ax.imshow(M, vmin=-1.0, vmax=1.0, cmap="RdYlGn", aspect="equal")
        for i in range(n_sec):
            for j in range(n_sec):
                val = float(M[i, j])
                text_color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")
        short = [lb[:14] for lb in section_labels]
        ax.set_xticks(range(n_sec))
        ax.set_yticks(range(n_sec))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        plt.colorbar(im, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)

    title = (
        f"Section×section similarity (wav2vec-alt) — {audio_name}"
        if audio_name else "Section×section similarity (wav2vec-alt)"
    )
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_layer_mean_similarity(
    layer_A: dict[int, np.ndarray],
    timestamps: np.ndarray,
    sections: list[dict],
    audio_name: str = "",
    output_path: str | Path | None = None,
    transform_tag: str = "",
    smooth_k: int = 3,
) -> None:
    """
    Two-panel summary plot averaging similarity across all probed layers.

    Top panel:    mean across layers, one line per section.
    Bottom panel: causal rolling mean of smooth_k samples over the top panel.
    """
    import matplotlib.pyplot as plt

    if not layer_A:
        return

    stack = np.stack(list(layer_A.values()), axis=0)
    mean_A = stack.mean(axis=0)
    smooth_A = causal_rolling_mean(mean_A, k=smooth_k)

    n_sec = mean_A.shape[0]
    colors = plt.cm.tab10.colors
    boundaries = {s["start"] for s in sections} | {sections[-1]["stop"]}

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    subtitles = ["Layer mean", f"Layer mean — {smooth_k}-sample smoothed"]

    for ax, data, subtitle in zip(axes, [mean_A, smooth_A], subtitles):
        for i in range(n_sec):
            label = sections[i]["label"][:20] if i < len(sections) else f"sec{i}"
            ax.plot(timestamps, data[i],
                    marker="o", markersize=3, linewidth=1,
                    color=colors[i % len(colors)], label=label)
        for t in sorted(boundaries):
            ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("Cosine similarity", fontsize=8)
        ax.margins(y=0.1)
        ax.set_title(subtitle, fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    layer_list = ", ".join(str(k) for k in sorted(layer_A))
    parts = [f"Layer-mean similarity (wav2vec-alt, layers {layer_list})"]
    if transform_tag:
        parts.append(f"({transform_tag})")
    if audio_name:
        parts.append(f"— {audio_name}")
    fig.suptitle("  ".join(parts), fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, output_path)


def causal_rolling_mean(A: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Causal rolling mean of length k along the last axis.
    output[..., i] = mean(A[..., max(0, i-k+1) : i+1])
    """
    if k <= 1:
        return A.copy()
    out = np.empty_like(A, dtype=float)
    n = A.shape[-1]
    cumsum = np.cumsum(A, axis=-1)
    for i in range(min(k - 1, n)):
        out[..., i] = cumsum[..., i] / (i + 1)
    if n >= k:
        shifted = np.concatenate(
            [np.zeros((*A.shape[:-1], 1)), cumsum[..., : n - k]], axis=-1
        )
        out[..., k - 1:] = (cumsum[..., k - 1:] - shifted) / k
    return out


def _pairwise_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_stem(p.stem + "_pairwise")


def _layer_mean_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_stem(p.stem + "_layermean")


def _save_or_show(fig, output_path: str | Path | None) -> None:
    import matplotlib.pyplot as plt
    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
        plt.close(fig)
