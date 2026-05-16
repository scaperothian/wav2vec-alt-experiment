"""Sliding-window aggregation over wav2vec2 frame embeddings."""

from typing import Iterator

import torch

from .config import WAV2VEC_FRAME_RATE, WINDOW_SEC, HOP_SEC


def section_windows(
    start_sec: float,
    stop_sec: float,
    window: float = WINDOW_SEC,
    hop: float = HOP_SEC,
    fps: int = WAV2VEC_FRAME_RATE,
) -> Iterator[tuple[int, int]]:
    """
    Yield (start_frame, end_frame) pairs that tile [start_sec, stop_sec).

    A trailing window is added when the section is long enough but the last
    hop would otherwise miss the final frames.
    """
    win_frames = int(round(window * fps))
    hop_frames = int(round(hop * fps))
    start_f = int(round(start_sec * fps))
    stop_f  = int(round(stop_sec  * fps))

    f = start_f
    while f + win_frames <= stop_f:
        yield f, f + win_frames
        f += hop_frames

    if (
        (stop_f - start_f) >= win_frames
        and (stop_f - win_frames) > (f - hop_frames)
    ):
        yield stop_f - win_frames, stop_f


def whole_song_window_spans(
    total_frames: int,
    window: float = WINDOW_SEC,
    hop: float = HOP_SEC,
    fps: int = WAV2VEC_FRAME_RATE,
) -> list[tuple[int, int]]:
    """
    Sliding windows covering [0, total_frames). Final partial hop is dropped
    so every span is exactly win_frames wide (simulates live stream of frames).
    """
    win_f = int(round(window * fps))
    hop_f = int(round(hop * fps))
    spans: list[tuple[int, int]] = []
    f = 0
    while f + win_f <= total_frames:
        spans.append((f, f + win_f))
        f += hop_f
    return spans


def window_means(
    layer_frames: torch.Tensor,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Mean-pool each span WITHOUT L2 normalisation (normalization happens after
    centering/whitening). Returns [W, D] or empty [0, D] if no valid spans.
    """
    rows: list[torch.Tensor] = []
    T = layer_frames.shape[0]
    for s, e in spans:
        e = min(e, T)
        if e - s < 2:
            continue
        rows.append(layer_frames[s:e].mean(dim=0))
    if rows:
        return torch.stack(rows, dim=0)
    return torch.empty(0, layer_frames.shape[-1])


def frame_to_section_assignment(
    frame_span: tuple[int, int],
    sections: list[dict],
    fps: int = WAV2VEC_FRAME_RATE,
) -> int:
    """
    Return the index of the section whose bounds contain the frame's centre.
    Returns -1 if the centre falls outside all sections.
    """
    center_sec = (frame_span[0] + frame_span[1]) / 2 / fps
    for i, sec in enumerate(sections):
        if sec["start"] <= center_sec < sec["stop"]:
            return i
    return -1
