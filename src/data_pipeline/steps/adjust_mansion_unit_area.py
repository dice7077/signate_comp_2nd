from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

from ..utils.paths import INTERIM_DIR, PROJECT_ROOT, ensure_parent, interim_subdir
from .layout import step_output_dir

SOURCE_DIR = INTERIM_DIR / step_output_dir("split_signate_by_type")
OUTPUT_DIR_NAME = step_output_dir("adjust_mansion_unit_area")
RANGE_MARGIN = 2.0
MIN_VALID_SCORE = 1.0
ABSOLUTE_MIN_AREA = 6.0
HARD_MAX_AREA = 1200.0


def adjust_mansion_unit_area(force: bool = True) -> Dict[str, object]:
    """
    Combine unit_area / house_area into unit_house_area_adjusted for mansion rows.
    """

    output_dir = interim_subdir(OUTPUT_DIR_NAME)
    outputs: List[dict] = []

    for dataset_name in ("train_kodate", "train_mansion", "test_kodate", "test_mansion"):
        source_path = SOURCE_DIR / f"{dataset_name}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_path} が見つかりません。split_signate_by_type を完了させてください。"
            )
        df = pd.read_parquet(source_path)
        stats: Dict[str, int] | None = None

        if dataset_name.endswith("_mansion"):
            df = df.copy()
            adjusted, stats = _build_adjusted_series(df)
            df["unit_house_area_adjusted"] = adjusted
        else:
            stats = {
                "rows": int(len(df)),
                "from_unit_area": 0,
                "from_house_area": 0,
                "clipped_to_bounds": 0,
                "forced_nulls": 0,
                "both_missing_inputs": int(
                    (df["unit_area"].isna() & df["house_area"].isna()).sum()
                    if {"unit_area", "house_area"} <= set(df.columns)
                    else 0
                ),
            }

        output_path = output_dir / f"{dataset_name}.parquet"
        if output_path.exists() and not force:
            raise FileExistsError(
                f"{output_path} が既に存在します。--force で上書きしてください。"
            )
        ensure_parent(output_path)
        df.to_parquet(output_path, index=False)

        outputs.append(
            {
                "name": dataset_name,
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "path": str(output_path.relative_to(PROJECT_ROOT)),
                "adjustment": stats,
            }
        )

    manifest = {
        "step": "adjust_mansion_unit_area",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(SOURCE_DIR.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "outputs": outputs,
    }

    manifest_path = output_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    return manifest


def _build_adjusted_series(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, int]]:
    stats = {
        "rows": int(len(df)),
        "from_unit_area": 0,
        "from_house_area": 0,
        "clipped_to_bounds": 0,
        "forced_nulls": 0,
        "both_missing_inputs": int(
            (df["unit_area"].isna() & df["house_area"].isna()).sum()
        ),
    }

    results: List[float | object] = []

    for row in df.itertuples(index=False):
        value, reason = _adjust_single(row)
        results.append(value)
        if reason == "unit":
            stats["from_unit_area"] += 1
        elif reason == "house":
            stats["from_house_area"] += 1
        elif reason == "clipped":
            stats["clipped_to_bounds"] += 1
        elif reason == "null":
            stats["forced_nulls"] += 1

    series = pd.Series(results, dtype="Float64", index=df.index)
    return series, stats


def _adjust_single(row) -> Tuple[float | object, str]:
    unit_area = _coerce_float(getattr(row, "unit_area", None))
    house_area = _coerce_float(getattr(row, "house_area", None))
    room_count = _coerce_float(getattr(row, "room_count", None))
    expected_low, expected_high = _expected_bounds(row)

    value, source, score = _select_best_value(
        unit_area, house_area, expected_low, expected_high, room_count
    )
    if value is not None and score >= MIN_VALID_SCORE:
        return value, source

    fallback = _fallback_to_bounds(value, expected_low, expected_high)
    if fallback is not None:
        return fallback, "clipped"

    return pd.NA, "null"


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _expected_bounds(row) -> Tuple[float | None, float | None]:
    hints = []
    for attr in ("unit_area_min", "unit_area_max"):
        value = _coerce_float(getattr(row, attr, None))
        if value is not None and value > 0:
            hints.append(value)
    if not hints:
        return None, None
    low = min(hints)
    high = max(hints)
    return low, high


def _select_best_value(
    unit_area: float | None,
    house_area: float | None,
    expected_low: float | None,
    expected_high: float | None,
    room_count: float | None,
) -> Tuple[float | None, str | None, float]:
    candidates: List[Tuple[float, str, float]] = []
    if unit_area is not None:
        candidates.append(
            (
                _score_value(
                    unit_area, house_area, expected_low, expected_high, room_count
                ),
                "unit",
                unit_area,
            )
        )
    if house_area is not None:
        candidates.append(
            (
                _score_value(
                    house_area, unit_area, expected_low, expected_high, room_count
                ),
                "house",
                house_area,
            )
        )
    if not candidates:
        return None, None, -math.inf

    best_score, best_source, best_value = max(
        candidates, key=lambda item: (item[0], 1 if item[1] == "unit" else 0)
    )
    return best_value, best_source, best_score


def _score_value(
    value: float,
    other: float | None,
    expected_low: float | None,
    expected_high: float | None,
    room_count: float | None,
) -> float:
    score = 2.0

    if other is not None:
        diff = abs(value - other)
        rel = diff / max(abs(value), abs(other), 1.0)
        if diff <= 1.0 or rel <= 0.01:
            score += 3.0
        elif diff <= 5.0 or rel <= 0.05:
            score += 1.5
        else:
            score -= min(2.0, rel * 2.0)

    score += _range_score(value, expected_low, expected_high)
    score += _magnitude_bonus(value, expected_low, expected_high)
    score += _room_penalty(value, room_count)

    if value < ABSOLUTE_MIN_AREA:
        score -= 6.0
    if value > HARD_MAX_AREA:
        score -= 6.0

    return score


def _range_score(value: float, expected_low: float | None, expected_high: float | None) -> float:
    if expected_low is not None and expected_high is not None:
        if (expected_low - RANGE_MARGIN) <= value <= (expected_high + RANGE_MARGIN):
            return 4.0
        distance = (expected_low - value) if value < expected_low else (value - expected_high)
        return -min(6.0, distance / 4.0)

    if expected_low is not None:
        if value >= expected_low - RANGE_MARGIN:
            return 1.5
        return -min(4.0, (expected_low - value) / 4.0)

    if expected_high is not None:
        if value <= expected_high + RANGE_MARGIN:
            return 1.5
        return -min(4.0, (value - expected_high) / 4.0)

    return 0.0


def _magnitude_bonus(
    value: float, expected_low: float | None, expected_high: float | None
) -> float:
    if expected_low is not None or expected_high is not None:
        penalty = 0.0
        if value < 10.0:
            penalty -= 0.5
        if value > 400.0:
            penalty -= 0.5
        return penalty

    if 15.0 <= value <= 150.0:
        return 2.5
    if 8.0 <= value < 15.0:
        return 0.5
    if 150.0 < value <= 500.0:
        return 1.0
    if 500.0 < value <= 750.0:
        return 0.5
    return -1.0


def _room_penalty(value: float, room_count: float | None) -> float:
    if room_count is None or math.isnan(room_count):
        return 0.0
    if room_count >= 2 and value < 15.0:
        return -3.0
    if room_count <= 1 and value < 12.0:
        return 0.5
    return 0.0


def _fallback_to_bounds(
    preferred_value: float | None,
    expected_low: float | None,
    expected_high: float | None,
) -> float | None:
    if expected_low is None and expected_high is None:
        return None

    if expected_low is None:
        reference = expected_high
        candidate = preferred_value if preferred_value is not None else reference
        return min(candidate, reference)

    if expected_high is None:
        reference = expected_low
        candidate = preferred_value if preferred_value is not None else reference
        return max(candidate, reference)

    candidate = preferred_value if preferred_value is not None else (
        (expected_low + expected_high) / 2.0
    )
    if candidate < expected_low:
        return expected_low
    if candidate > expected_high:
        return expected_high
    return candidate


__all__ = ["adjust_mansion_unit_area"]

