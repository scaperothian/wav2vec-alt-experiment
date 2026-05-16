"""Tests for sliding-window frame aggregation (wav2vec2 @ 50 fps)."""

import pytest
import torch

from src.windows import (
    WAV2VEC_FRAME_RATE,
    frame_to_section_assignment,
    section_windows,
    whole_song_window_spans,
    window_means,
)

FPS = WAV2VEC_FRAME_RATE  # 50


class TestSectionWindows:
    def test_basic_count(self):
        # 2s section, 1s window, 0.5s hop at 50 fps
        # win=50f, hop=25f, start=0, stop=100
        # f=0: yield(0,50) f=25; f=25: yield(25,75) f=50; f=50: yield(50,100) f=75
        # f=75: 75+50>100, exit. tail: stop-win=50, f-hop=50, 50>50 False -> no tail
        windows = list(section_windows(0.0, 2.0, window=1.0, hop=0.5, fps=FPS))
        assert len(windows) == 3
        for s, e in windows:
            assert e - s == 50
            assert s >= 0 and e <= 100

    def test_section_shorter_than_window_yields_nothing(self):
        assert list(section_windows(0.0, 0.5, window=1.0, hop=0.5, fps=FPS)) == []

    def test_section_exactly_one_window(self):
        windows = list(section_windows(0.0, 1.0, window=1.0, hop=0.5, fps=FPS))
        assert len(windows) >= 1
        assert windows[0] == (0, 50)

    def test_non_zero_start(self):
        windows = list(section_windows(10.0, 12.0, window=1.0, hop=0.5, fps=FPS))
        assert len(windows) > 0
        for s, e in windows:
            assert s >= 500   # 10s * 50fps
            assert e <= 600   # 12s * 50fps

    def test_frames_within_section_bounds(self):
        start, stop = 5.0, 8.0
        windows = list(section_windows(start, stop, window=1.0, hop=0.5, fps=FPS))
        start_f = int(round(start * FPS))
        stop_f  = int(round(stop  * FPS))
        for s, e in windows:
            assert s >= start_f and e <= stop_f

    def test_hop_larger_than_window_no_overlap(self):
        windows = list(section_windows(0.0, 4.0, window=1.0, hop=2.0, fps=FPS))
        assert len(windows) >= 2
        assert windows[0][1] <= windows[1][0]


class TestWholeSongWindowSpans:
    def test_all_spans_same_width(self):
        spans = whole_song_window_spans(200, window=1.0, hop=0.5, fps=FPS)
        for s, e in spans:
            assert e - s == 50  # 1s * 50fps

    def test_starts_at_zero(self):
        spans = whole_song_window_spans(200, window=1.0, hop=0.5, fps=FPS)
        assert spans[0][0] == 0

    def test_no_span_exceeds_total_frames(self):
        total = 200
        for s, e in whole_song_window_spans(total, window=1.0, hop=0.5, fps=FPS):
            assert e <= total

    def test_returns_at_least_one_span(self):
        assert len(whole_song_window_spans(200, window=1.0, hop=0.5, fps=FPS)) > 0

    def test_empty_when_total_frames_less_than_window(self):
        assert whole_song_window_spans(10, window=1.0, hop=0.5, fps=FPS) == []

    def test_non_overlapping_with_large_hop(self):
        spans = whole_song_window_spans(400, window=1.0, hop=2.0, fps=FPS)
        for i in range(len(spans) - 1):
            assert spans[i][1] <= spans[i + 1][0]


class TestWindowMeans:
    def _frames(self, n: int, dim: int = 16) -> torch.Tensor:
        torch.manual_seed(55)
        return torch.randn(n, dim)

    def test_output_shape(self):
        frames = self._frames(200)
        out = window_means(frames, [(0, 50), (25, 75), (50, 100)])
        assert out.shape == (3, 16)

    def test_rows_are_not_unit_norm(self):
        frames = self._frames(200) * 10
        out = window_means(frames, [(0, 50)])
        assert out.norm(dim=1).item() > 1.5

    def test_values_equal_slice_mean(self):
        frames = self._frames(200)
        out = window_means(frames, [(0, 50)])
        assert torch.allclose(out[0], frames[0:50].mean(dim=0), atol=1e-5)

    def test_clips_to_tensor_length(self):
        frames = self._frames(80)
        out = window_means(frames, [(0, 50), (50, 200)])
        assert out.shape[0] == 2

    def test_skips_spans_shorter_than_two(self):
        frames = self._frames(200)
        out = window_means(frames, [(0, 50), (100, 101)])
        assert out.shape[0] == 1

    def test_empty_when_no_valid_spans(self):
        frames = self._frames(200)
        out = window_means(frames, [(0, 1)])
        assert out.shape == (0, 16)

    def test_dim_matches_frames(self):
        frames = self._frames(200, dim=32)
        assert window_means(frames, [(0, 50)]).shape[1] == 32


class TestFrameToSectionAssignment:
    def _sections(self):
        return [
            {"label": "Intro",  "start":  0.0, "stop": 10.0},
            {"label": "Verse",  "start": 10.0, "stop": 30.0},
            {"label": "Chorus", "start": 30.0, "stop": 50.0},
        ]

    def test_centre_inside_first_section(self):
        # centre = (0+50)/2 / 50 = 0.5s -> Intro
        assert frame_to_section_assignment((0, 50), self._sections(), fps=FPS) == 0

    def test_centre_inside_second_section(self):
        # centre = (500+550)/2 / 50 = 10.5s -> Verse
        assert frame_to_section_assignment((500, 550), self._sections(), fps=FPS) == 1

    def test_centre_inside_third_section(self):
        # centre = (1500+1550)/2 / 50 = 30.5s -> Chorus
        assert frame_to_section_assignment((1500, 1550), self._sections(), fps=FPS) == 2

    def test_centre_before_all_sections_returns_minus_one(self):
        sections = [{"label": "A", "start": 5.0, "stop": 10.0}]
        assert frame_to_section_assignment((0, 50), sections, fps=FPS) == -1

    def test_centre_after_all_sections_returns_minus_one(self):
        sections = [{"label": "A", "start": 0.0, "stop": 1.0}]
        # centre = (200+250)/2 / 50 = 4.5s, after stop=1.0
        assert frame_to_section_assignment((200, 250), sections, fps=FPS) == -1

    def test_centre_exactly_at_boundary_goes_to_next_section(self):
        # centre exactly at 10.0s (exclusive stop of Intro, inclusive start of Verse)
        # s+e = 10.0 * 2 * 50 = 1000 e.g. (475, 525)
        assert frame_to_section_assignment((475, 525), self._sections(), fps=FPS) == 1

    def test_custom_fps(self):
        sections = [{"label": "A", "start": 0.0, "stop": 10.0}]
        assert frame_to_section_assignment((0, 10), sections, fps=10) == 0
