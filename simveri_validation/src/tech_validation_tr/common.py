"""
Common utilities for the Technical Validation (TR) scripts.

Keep dependencies minimal (json/csv/math/numpy) so CPU-only scripts can run
in the evaluation environment without torch.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any, *, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_csv_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
    return bool(default)


def angdiff_deg(a: float, b: float) -> float:
    """
    Minimal absolute angular difference in degrees in [0, 180].
    """
    # Handle NaN gracefully.
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v) + eps)
    return v / n


def uniform_subsample(items: List[Any], k: int) -> List[Any]:
    if k <= 0 or len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k, dtype=int)
    return [items[i] for i in idx]


def load_tracklets(path: str) -> Tuple[List[dict], dict]:
    """
    Load a tracklets json file.

    Supported formats:
      1) {"meta": {...}, "tracklets": [ {...}, ... ]}
      2) [ {...}, ... ]
      3) {"tracklets": [ ... ]}  (meta optional)
    """
    obj = load_json(path)
    if isinstance(obj, list):
        return obj, {"generated_at": None}
    if isinstance(obj, dict):
        if isinstance(obj.get("tracklets"), list):
            return obj["tracklets"], obj.get("meta", {})
        # Back-compat: treat dict as mapping id->rec
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values()), {"generated_at": None}
    raise ValueError(f"Unsupported tracklets format: {path}")


def index_by(tracklets: Iterable[dict], key: str = "track_id") -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for t in tracklets:
        k = t.get(key)
        if isinstance(k, str) and k:
            out[k] = t
    return out


def load_coview_pairs(path: str) -> Dict[Tuple[str, str], dict]:
    """
    Return mapping (cam_from, cam_to) -> pair_record, where pair_record has:
      - median_dt_s
      - n
      - coview
    """
    obj = load_json(path)
    pairs = obj.get("pairs") if isinstance(obj, dict) else None
    if not isinstance(pairs, dict):
        raise ValueError(f"Invalid coview json: {path}")
    out: Dict[Tuple[str, str], dict] = {}
    for _, rec in pairs.items():
        if not isinstance(rec, dict):
            continue
        a = rec.get("from")
        b = rec.get("to")
        if isinstance(a, str) and isinstance(b, str):
            out[(a, b)] = rec
    return out


@dataclass(frozen=True)
class SpeedPrior:
    v_mean_kmh: float
    v_std_kmh: float
    v_max_kmh: float


def load_speed_prior(path: str) -> SpeedPrior:
    obj = load_json(path)
    return SpeedPrior(
        v_mean_kmh=float(obj["v_mean_kmh"]),
        v_std_kmh=float(obj["v_std_kmh"]),
        v_max_kmh=float(obj.get("v_max_kmh", obj.get("v_max", 120.0))),
    )


def euclidean2d(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    return math.hypot(dx, dy)

