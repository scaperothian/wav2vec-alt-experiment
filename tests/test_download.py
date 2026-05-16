from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.download import (
    GDRIVE_MODEL_URL,
    _CKPT_FILENAME,
    _ZIP_FILENAME,
    _find_local_checkpoint,
    resolve_checkpoint,
)


def _make_ckpt(root: Path, name: str) -> Path:
    ckpt_dir = root / name
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / _CKPT_FILENAME
    ckpt.write_bytes(b"fake")
    return ckpt


def _make_zip(dest: Path, inner_path: str = _CKPT_FILENAME) -> None:
    with zipfile.ZipFile(dest, "w") as zf:
        zf.writestr(inner_path, b"fake model")


def _fake_download(url, output, quiet, progress=None):
    _make_zip(Path(output))


# --- _find_local_checkpoint ---

def test_find_local_checkpoint_returns_none_when_empty(tmp_path):
    assert _find_local_checkpoint(tmp_path) is None


def test_find_local_checkpoint_finds_single(tmp_path):
    expected = _make_ckpt(tmp_path, "CKPT+2022-05-13+09-25-17+00")
    assert _find_local_checkpoint(tmp_path) == expected


def test_find_local_checkpoint_returns_newest(tmp_path):
    _make_ckpt(tmp_path, "CKPT+2022-01-01+00-00-00+00")
    newest = _make_ckpt(tmp_path, "CKPT+2023-06-15+12-00-00+00")
    assert _find_local_checkpoint(tmp_path) == newest


def test_find_local_checkpoint_ignores_dirs_without_ckpt(tmp_path):
    (tmp_path / "CKPT+2022-01-01+00-00-00+00").mkdir()
    assert _find_local_checkpoint(tmp_path) is None


# --- resolve_checkpoint: explicit path ---

def test_resolve_checkpoint_explicit_path(tmp_path):
    ckpt = tmp_path / "my.ckpt"
    ckpt.write_bytes(b"fake")
    assert resolve_checkpoint(ckpt) == ckpt


def test_resolve_checkpoint_explicit_missing_raises(tmp_path):
    missing = tmp_path / "does_not_exist.ckpt"
    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        resolve_checkpoint(missing)


# --- resolve_checkpoint: auto-detect extracted checkpoint ---

def test_resolve_checkpoint_auto_detects_local(tmp_path):
    save_root = tmp_path / "model" / "save"
    expected = _make_ckpt(save_root, "CKPT+2022-05-13+09-25-17+00")
    assert resolve_checkpoint(None, cwd=tmp_path) == expected


# --- resolve_checkpoint: local zip ---

def test_resolve_checkpoint_uses_local_zip(tmp_path):
    _make_zip(tmp_path / _ZIP_FILENAME)
    result = resolve_checkpoint(None, cwd=tmp_path)
    assert result.name == _CKPT_FILENAME
    assert result.exists()


def test_resolve_checkpoint_local_zip_not_deleted(tmp_path):
    zip_path = tmp_path / _ZIP_FILENAME
    _make_zip(zip_path)
    resolve_checkpoint(None, cwd=tmp_path)
    assert zip_path.exists()


def test_resolve_checkpoint_local_zip_nested_ckpt(tmp_path):
    with zipfile.ZipFile(tmp_path / _ZIP_FILENAME, "w") as zf:
        zf.writestr(f"subdir/{_CKPT_FILENAME}", b"fake model")
    result = resolve_checkpoint(None, cwd=tmp_path)
    assert result.name == _CKPT_FILENAME
    assert result.exists()


def test_resolve_checkpoint_prefers_extracted_over_local_zip(tmp_path):
    save_root = tmp_path / "model" / "save"
    extracted = _make_ckpt(save_root, "CKPT+2022-05-13+09-25-17+00")
    _make_zip(tmp_path / _ZIP_FILENAME)
    result = resolve_checkpoint(None, cwd=tmp_path)
    assert result == extracted


# --- resolve_checkpoint: download ---

def test_resolve_checkpoint_no_url_raises(tmp_path):
    with patch("src.download.GDRIVE_MODEL_URL", None):
        with pytest.raises(RuntimeError, match="GDRIVE_MODEL_URL is not set"):
            resolve_checkpoint(None, cwd=tmp_path)


def test_resolve_checkpoint_downloads_and_extracts_zip(tmp_path):
    with patch("src.download.GDRIVE_MODEL_URL", "https://drive.google.com/fake"):
        with patch("gdown.download", side_effect=_fake_download) as mock_dl:
            result = resolve_checkpoint(None, cwd=tmp_path)
            mock_dl.assert_called_once()
            assert result.name == _CKPT_FILENAME
            assert result.exists()


def test_resolve_checkpoint_downloaded_zip_kept_in_project_root(tmp_path):
    with patch("src.download.GDRIVE_MODEL_URL", "https://drive.google.com/fake"):
        with patch("gdown.download", side_effect=_fake_download):
            resolve_checkpoint(None, cwd=tmp_path)
            assert (tmp_path / _ZIP_FILENAME).exists()


def test_resolve_checkpoint_raises_if_ckpt_missing_from_zip(tmp_path):
    def fake_download(url, output, quiet, progress=None):
        with zipfile.ZipFile(Path(output), "w") as zf:
            zf.writestr("unrelated_file.txt", "nothing useful")

    with patch("src.download.GDRIVE_MODEL_URL", "https://drive.google.com/fake"):
        with patch("gdown.download", side_effect=fake_download):
            with pytest.raises(RuntimeError, match=_CKPT_FILENAME):
                resolve_checkpoint(None, cwd=tmp_path)


def test_resolve_checkpoint_gdown_missing_raises(tmp_path):
    with patch("src.download.GDRIVE_MODEL_URL", "https://drive.google.com/fake"):
        with patch.dict("sys.modules", {"gdown": None}):
            with pytest.raises((RuntimeError, ImportError)):
                resolve_checkpoint(None, cwd=tmp_path)


def test_gdrive_url_is_set():
    assert GDRIVE_MODEL_URL is not None, "GDRIVE_MODEL_URL should be set in download.py"
