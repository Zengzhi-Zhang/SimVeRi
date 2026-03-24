"""Portable path helpers for the SimVeRi validation codebase."""

from __future__ import annotations

import os
from pathlib import Path


_VALIDATION_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _VALIDATION_ROOT.parent
_RELEASE_ROOT = _REPO_ROOT.parent


def _env_or_default(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name, "").strip()
    if value:
        return Path(value).expanduser()
    return default


def get_validation_root() -> str:
    return str(_VALIDATION_ROOT)


def get_repo_root() -> str:
    return str(_REPO_ROOT)


def get_release_root() -> str:
    return str(_RELEASE_ROOT)


def get_default_simveri_root() -> str:
    """Return the default released SimVeRi dataset location."""
    default = _RELEASE_ROOT / "SimVeRi-dataset-v2.0"
    return str(_env_or_default("SIMVERI_DATASET_ROOT", default))


def get_default_veri776_root() -> str:
    """Return the default VeRi-776 location expected by validation scripts."""
    default = _REPO_ROOT / "external_datasets" / "VeRi-776" / "VeRi"
    return str(_env_or_default("VERI776_ROOT", default))


def get_validation_output_dir(*parts: str) -> str:
    return str((_VALIDATION_ROOT / "outputs").joinpath(*parts))


def get_validation_pretrained_path(filename: str = "veri_sbs_R50-ibn.pth") -> str:
    return str(_VALIDATION_ROOT / "pretrained" / filename)
