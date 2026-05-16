"""JSON song manifest loading and audio ingestion."""

import json
from pathlib import Path

import torch
import torchaudio

from .config import TARGET_SR


def load_song(json_path: Path) -> tuple[Path, list[dict]]:
    """
    Parse a ProPresenter-style export JSON.

    Returns:
        audio_path: resolved path to the audio file
        sections:   list of dicts with keys label, start, stop (seconds)
    """
    with open(json_path) as f:
        data = json.load(f)
    pres = data["presentation"]
    audio_path = Path(pres["id"]["audio"])

    sections = []
    for group in pres["groups"]:
        for slide in group["slides"]:
            sections.append({
                "label": (next(iter(slide.get("text", "").splitlines()), "")[:40] or "(unlabeled)"),
                "start": float(slide["start time"]),
                "stop":  float(slide["stop time"]),
            })
    return audio_path, sections


def load_audio(audio_path: Path) -> torch.Tensor:
    """
    Load an audio file, downmix to mono, and resample to TARGET_SR (16 kHz).

    Returns:
        1-D float tensor of shape [num_samples]
    """
    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)
