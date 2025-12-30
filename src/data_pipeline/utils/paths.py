from __future__ import annotations

from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_SIGNATE_DIR = RAW_DIR / "signate"
INTERIM_DIR = DATA_DIR / "interim"


def raw_signate_path(filename: str) -> Path:
    """Return the absolute path to a file inside data/raw/signate."""
    return RAW_SIGNATE_DIR / filename


PathLike = Union[str, Path]


def interim_subdir(*parts: PathLike) -> Path:
    """Create (if needed) and return a subdirectory within data/interim."""
    target = INTERIM_DIR.joinpath(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_parent(path: Path) -> Path:
    """Ensure parents exist before writing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

