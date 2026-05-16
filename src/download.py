from __future__ import annotations

import zipfile
from pathlib import Path

import tqdm

# Accepted formats:
#   https://drive.google.com/file/d/<FILE_ID>/view
#   https://drive.google.com/uc?id=<FILE_ID>
GDRIVE_MODEL_URL: str | None = "https://drive.google.com/file/d/1NH5MI2HL5Wj3B4oo478C6Rl4lfMd_M4w/view?usp=sharing"

_SAVE_SUBDIR = Path("model") / "save"
_CKPT_FILENAME = "wav2vec2.ckpt"
_ZIP_FILENAME = "wav2vec-alt-model.zip"
_DOWNLOAD_DIRNAME = "downloaded"


def _find_local_checkpoint(save_root: Path) -> Path | None:
    """Return the newest wav2vec2.ckpt under save_root/CKPT+*, or None."""
    candidates = sorted(save_root.glob(f"CKPT+*/{_CKPT_FILENAME}"))
    return candidates[-1] if candidates else None


def _extract_zip(zip_path: Path, dest_dir: Path, delete_after: bool = False) -> Path:
    """Extract zip_path into dest_dir and return the path to wav2vec2.ckpt."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm.tqdm(members, desc="  Extracting", unit="file"):
            zf.extract(member, dest_dir)
    if delete_after:
        zip_path.unlink()
    matches = list(dest_dir.rglob(_CKPT_FILENAME))
    if not matches:
        raise RuntimeError(f"Extracted archive but could not find {_CKPT_FILENAME} inside.")
    return sorted(matches)[-1]


def _download_zip(zip_path: Path) -> None:
    """Download GDRIVE_MODEL_URL to zip_path (project root)."""
    if GDRIVE_MODEL_URL is None:
        raise RuntimeError(
            "Checkpoint not found locally and GDRIVE_MODEL_URL is not set.\n"
            "Either pass --ckpt explicitly, drop wav2vec-alt-model.zip in the project "
            "directory, or set GDRIVE_MODEL_URL in src/download.py."
        )
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required for automatic model download. "
            "Run: pip install gdown"
        )

    print(f"  Downloading model archive -> {zip_path}")

    def _progress(current: int, total: int | None) -> None:
        bar.total = total
        bar.update(current - bar.n)

    with tqdm.tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc="  Downloading") as bar:
        gdown.download(GDRIVE_MODEL_URL, str(zip_path), quiet=True, progress=_progress)

    if not zip_path.exists():
        raise RuntimeError(f"Download appeared to succeed but {zip_path} was not created.")


def resolve_checkpoint(explicit: Path | None, cwd: Path | None = None) -> Path:
    """
    Resolve which checkpoint to load, in priority order:
      1. --ckpt argument (explicit path) — validated to exist
      2. Auto-detected from model/save/CKPT+*/wav2vec2.ckpt relative to cwd
      3. wav2vec-alt-model.zip found in the project directory — extracted in place
      4. Downloaded from GDRIVE_MODEL_URL into model/save/downloaded/
    """
    if explicit is not None:
        if not explicit.exists():
            raise RuntimeError(f"Checkpoint not found: {explicit}")
        return explicit

    root = (cwd or Path.cwd())
    save_root = root / _SAVE_SUBDIR

    if save_root.exists():
        found = _find_local_checkpoint(save_root)
        if found:
            print(f"  Auto-detected checkpoint: {found}")
            return found

    local_zip = root / _ZIP_FILENAME
    if not local_zip.exists():
        print("  No local checkpoint found — attempting download from Google Drive...")
        _download_zip(local_zip)

    print(f"  Found archive {local_zip} — extracting...")
    return _extract_zip(local_zip, save_root / _DOWNLOAD_DIRNAME, delete_after=False)
