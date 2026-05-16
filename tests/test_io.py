"""Tests for JSON loading and audio ingestion (no real model required)."""

import json
import wave
from pathlib import Path

import pytest
import torch

from src.config import TARGET_SR
from src.io import load_audio, load_song


def _write_song_json(path: Path, audio_path: str, groups: list) -> None:
    path.write_text(json.dumps({
        "presentation": {"id": {"audio": audio_path}, "groups": groups}
    }))


def _minimal_groups():
    return [
        {
            "slides": [
                {"text": "Verse 1\nSecond line", "start time": "0.0",  "stop time": "10.0"},
                {"text": "Chorus\nLine 2",       "start time": "10.0", "stop time": "20.0"},
            ]
        },
        {
            "slides": [
                {"text": "", "start time": "20.0", "stop time": "30.0"},
            ]
        },
    ]


def _write_wav(path: Path, sr: int = TARGET_SR, duration_sec: float = 0.5, channels: int = 1):
    n_samples = int(sr * duration_sec)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n_samples * channels)


# --- load_song ---------------------------------------------------------------

class TestLoadSong:
    def test_returns_audio_path_and_sections(self, tmp_path):
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, str(tmp_path / "song.wav"), _minimal_groups())
        audio_path, sections = load_song(json_file)
        assert isinstance(audio_path, Path)
        assert len(sections) == 3

    def test_section_fields_present(self, tmp_path):
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", _minimal_groups())
        _, sections = load_song(json_file)
        for sec in sections:
            assert {"label", "start", "stop"} <= sec.keys()

    def test_label_is_first_line_only(self, tmp_path):
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", _minimal_groups())
        _, sections = load_song(json_file)
        assert sections[0]["label"] == "Verse 1"

    def test_label_truncated_to_40_chars(self, tmp_path):
        groups = [{"slides": [{"text": "A" * 60, "start time": "0.0", "stop time": "5.0"}]}]
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", groups)
        _, sections = load_song(json_file)
        assert len(sections[0]["label"]) <= 40

    def test_empty_text_becomes_unlabeled(self, tmp_path):
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", _minimal_groups())
        _, sections = load_song(json_file)
        assert sections[2]["label"] == "(unlabeled)"

    def test_timestamps_are_floats(self, tmp_path):
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", _minimal_groups())
        _, sections = load_song(json_file)
        for sec in sections:
            assert isinstance(sec["start"], float)
            assert isinstance(sec["stop"], float)

    def test_multiple_groups_flattened(self, tmp_path):
        groups = [
            {"slides": [{"text": "A", "start time": "0",  "stop time": "5"}]},
            {"slides": [{"text": "B", "start time": "5",  "stop time": "10"},
                        {"text": "C", "start time": "10", "stop time": "15"}]},
        ]
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", groups)
        _, sections = load_song(json_file)
        assert len(sections) == 3

    def test_missing_audio_key_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"presentation": {"id": {}}}))
        with pytest.raises(KeyError):
            load_song(bad)


# --- load_audio --------------------------------------------------------------

class TestLoadAudio:
    def test_mono_wav_returns_1d_tensor(self, tmp_path):
        p = tmp_path / "mono.wav"
        _write_wav(p, sr=TARGET_SR, channels=1)
        assert load_audio(p).dim() == 1

    def test_stereo_wav_downmixed_to_mono(self, tmp_path):
        p = tmp_path / "stereo.wav"
        _write_wav(p, sr=TARGET_SR, channels=2)
        assert load_audio(p).dim() == 1

    def test_sample_count_matches_duration(self, tmp_path):
        duration = 0.5
        p = tmp_path / "half.wav"
        _write_wav(p, sr=TARGET_SR, duration_sec=duration)
        result = load_audio(p)
        assert abs(result.shape[0] - int(TARGET_SR * duration)) <= 1

    def test_resampling_from_8khz(self, tmp_path):
        src_sr = 8_000
        duration = 0.25
        p = tmp_path / "8k.wav"
        _write_wav(p, sr=src_sr, duration_sec=duration)
        result = load_audio(p)
        expected = int(TARGET_SR * duration)
        assert abs(result.shape[0] - expected) < expected * 0.05

    def test_output_is_float32(self, tmp_path):
        p = tmp_path / "f.wav"
        _write_wav(p, sr=TARGET_SR)
        assert load_audio(p).dtype == torch.float32

    def test_target_sr_is_16khz(self):
        assert TARGET_SR == 16_000
