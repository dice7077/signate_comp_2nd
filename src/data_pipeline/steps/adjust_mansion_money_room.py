from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.paths import INTERIM_DIR, PROJECT_ROOT, ensure_parent, interim_subdir
from .layout import step_output_dir

SOURCE_DIR = INTERIM_DIR / step_output_dir("adjust_mansion_unit_area")
OUTPUT_DIR_NAME = step_output_dir("adjust_mansion_money_room")

GLOBAL_SIGMA_MULT = 3.0
PREF_HIGH_RATIO = 8.0
PREF_LOW_RATIO = 0.2
LOCAL_RATIO_LOW = 0.5
LOCAL_RATIO_HIGH = 2.0
BUILDING_MIN_RATIO_COUNT = 5
BUILDING_RATIO_HIGH = 4.0
BUILDING_RATIO_LOW = 0.25
UNIT_MIN_RATIO_COUNT = 2
UNIT_RATIO_HIGH = 3.0
UNIT_RATIO_LOW = 1.0 / 3.0

PREF_PATTERN = re.compile(r"([^0-9]+?[都道府県])")


def adjust_mansion_money_room(force: bool = True) -> Dict[str, object]:
    """
    Detect obvious typos in mansion money_room and create money_room_adjusted.
    """

    output_dir = interim_subdir(OUTPUT_DIR_NAME)
    outputs: List[dict] = []

    for dataset_name in ("train_kodate", "train_mansion", "test_kodate", "test_mansion"):
        source_path = SOURCE_DIR / f"{dataset_name}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_path} が見つかりません。adjust_mansion_unit_area を完了させてください。"
            )
        df = pd.read_parquet(source_path)
        stats: Dict[str, object] | None

        has_money_room = "money_room" in df.columns

        if dataset_name.endswith("_mansion") and has_money_room:
            df = df.copy()
            stats = _adjust_mansion_rows(df)
        elif dataset_name.endswith("_mansion"):
            df = df.copy()
            df["money_room_adjusted"] = pd.Series(pd.NA, dtype="Int64", index=df.index)
            stats = {
                "rows": int(len(df)),
                "adjusted_rows": 0,
                "reason": "money_room_missing",
            }
        else:
            df = df.copy()
            if has_money_room:
                df["money_room_adjusted"] = df["money_room"]
            else:
                df["money_room_adjusted"] = pd.Series(
                    pd.NA, dtype="Int64", index=df.index
                )
            stats = {
                "rows": int(len(df)),
                "adjusted_rows": 0,
                "reason": "non_mansion_passthrough" if has_money_room else "money_room_missing",
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
        "step": "adjust_mansion_money_room",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(SOURCE_DIR.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "outputs": outputs,
    }

    manifest_path = output_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    return manifest


def _adjust_mansion_rows(df: pd.DataFrame) -> Dict[str, object]:
    if "money_room" not in df.columns:
        raise KeyError("money_room 列が存在しません。")
    if "unit_house_area_adjusted" not in df.columns:
        raise KeyError("unit_house_area_adjusted 列が存在しません。")
    if "building_id" not in df.columns:
        raise KeyError("building_id 列が存在しません。")
    if "unit_id" not in df.columns:
        raise KeyError("unit_id 列が存在しません。")

    money = df["money_room"]
    area = df["unit_house_area_adjusted"]
    positive_mask = money.notna() & area.notna() & (area > 0) & (money > 0)

    ppsm = pd.Series(np.nan, dtype="Float64", index=df.index)
    ppsm.loc[positive_mask] = (money[positive_mask] / area[positive_mask]).astype(float)

    ppsm_positive = ppsm.dropna()
    if ppsm_positive.empty:
        df["money_room_adjusted"] = df["money_room"]
        return {
            "rows": int(len(df)),
            "valid_ppsm_rows": 0,
            "corrected_high": 0,
            "corrected_low": 0,
            "note": "ppsm unavailable",
        }

    log_ppsm = np.log(ppsm_positive)
    mean = float(log_ppsm.mean())
    sigma = float(log_ppsm.std(ddof=0))
    if not math.isfinite(sigma):
        sigma = 0.0
    if sigma == 0.0:
        global_low = float(ppsm_positive.quantile(0.01))
        global_high = float(ppsm_positive.quantile(0.99))
    else:
        global_low = float(np.exp(mean - GLOBAL_SIGMA_MULT * sigma))
        global_high = float(np.exp(mean + GLOBAL_SIGMA_MULT * sigma))

    pref_series = df.get("full_address", pd.Series("", index=df.index)).fillna("")
    prefectures = pref_series.map(_extract_prefecture)
    pref_stats = _group_stats(ppsm, prefectures)
    if pref_stats.empty:
        pref_median_series = pd.Series(np.nan, index=df.index)
    else:
        pref_median_series = prefectures.map(pref_stats["median"])

    building_ids = df["building_id"]
    unit_ids = df["unit_id"]

    building_stats = _group_stats(ppsm, building_ids)
    unit_stats = _group_stats(ppsm, unit_ids)

    building_ratio_series, building_ratio_counts = _ratio_series(
        building_stats, building_ids, BUILDING_MIN_RATIO_COUNT
    )
    unit_ratio_series, unit_ratio_counts = _ratio_series(
        unit_stats, unit_ids, UNIT_MIN_RATIO_COUNT
    )

    if building_stats.empty:
        building_median_series = pd.Series(np.nan, index=df.index)
        building_count_series = pd.Series(np.nan, index=df.index)
    else:
        building_median_series = building_ids.map(building_stats["median"])
        building_count_series = building_ids.map(building_stats["count"])

    if unit_stats.empty:
        unit_median_series = pd.Series(np.nan, index=df.index)
        unit_count_series = pd.Series(np.nan, index=df.index)
    else:
        unit_median_series = unit_ids.map(unit_stats["median"])
        unit_count_series = unit_ids.map(unit_stats["count"])

    pref_median_clean = pref_median_series.where(pref_median_series > 0)
    pref_ratio = ppsm / pref_median_clean
    building_ratio = building_ratio_series
    unit_ratio = unit_ratio_series

    global_high_mask = ppsm > global_high
    global_low_mask = ppsm < global_low
    pref_high_mask = pref_ratio > PREF_HIGH_RATIO
    pref_low_mask = pref_ratio < PREF_LOW_RATIO
    building_high_mask = (building_ratio > BUILDING_RATIO_HIGH) & building_ratio_counts.notna()
    building_low_mask = (building_ratio < BUILDING_RATIO_LOW) & building_ratio_counts.notna()
    unit_high_mask = (unit_ratio > UNIT_RATIO_HIGH) & unit_ratio_counts.notna()
    unit_low_mask = (unit_ratio < UNIT_RATIO_LOW) & unit_ratio_counts.notna()

    high_candidates = positive_mask & (
        global_high_mask | pref_high_mask | building_high_mask | unit_high_mask
    )
    low_candidates = positive_mask & (
        global_low_mask | pref_low_mask | building_low_mask | unit_low_mask
    )
    low_candidates &= ~high_candidates

    divisible_by_10 = (money % 10 == 0)
    high_candidates &= divisible_by_10

    ppsm_after_div = ppsm / 10.0
    ppsm_after_mul = ppsm * 10.0

    high_fit_mask = _within_local_bounds(
        ppsm_after_div,
        pref_median_series,
        building_median_series,
        unit_median_series,
        building_count_series,
        unit_count_series,
        global_low,
        global_high,
    )
    low_fit_mask = _within_local_bounds(
        ppsm_after_mul,
        pref_median_series,
        building_median_series,
        unit_median_series,
        building_count_series,
        unit_count_series,
        global_low,
        global_high,
    )

    high_apply = high_candidates & high_fit_mask
    low_apply = low_candidates & low_fit_mask

    adjusted = money.astype("int64").copy()
    adjusted.loc[high_apply] = (adjusted.loc[high_apply] // 10).astype("int64")
    adjusted.loc[low_apply] = (adjusted.loc[low_apply] * 10).astype("int64")
    df["money_room_adjusted"] = adjusted

    return {
        "rows": int(len(df)),
        "valid_ppsm_rows": int(positive_mask.sum()),
        "global_low_ppsm": global_low,
        "global_high_ppsm": global_high,
        "high_candidates": int(high_candidates.sum()),
        "low_candidates": int(low_candidates.sum()),
        "corrected_high": int(high_apply.sum()),
        "corrected_low": int(low_apply.sum()),
    }


def _extract_prefecture(address: str) -> str | None:
    if not address:
        return None
    match = PREF_PATTERN.search(address)
    if not match:
        return None
    return match.group(1)


def _group_stats(values: pd.Series, key: pd.Series) -> pd.DataFrame:
    valid = values.notna() & key.notna()
    if not valid.any():
        return pd.DataFrame(columns=["count", "min", "max", "median"])
    stats = (
        pd.DataFrame({"key": key[valid], "value": values[valid]})
        .groupby("key")["value"]
        .agg(["count", "min", "max", "median"])
    )
    return stats


def _ratio_series(
    stats: pd.DataFrame, key: pd.Series, min_count: int
) -> Tuple[pd.Series, pd.Series]:
    if stats.empty:
        nan_series = pd.Series(np.nan, index=key.index)
        return nan_series, nan_series

    eligible = stats[stats["count"] >= min_count].copy()
    if eligible.empty:
        nan_series = pd.Series(np.nan, index=key.index)
        return nan_series, nan_series

    eligible["ratio"] = eligible["max"] / eligible["min"].replace(0, np.nan)
    eligible.loc[~np.isfinite(eligible["ratio"]), "ratio"] = np.nan

    ratio_series = key.map(eligible["ratio"])
    count_series = key.map(eligible["count"])
    return ratio_series, count_series


def _within_local_bounds(
    candidate_ppsm: pd.Series,
    pref_median: pd.Series,
    building_median: pd.Series,
    unit_median: pd.Series,
    building_counts: pd.Series,
    unit_counts: pd.Series,
    global_low: float,
    global_high: float,
) -> pd.Series:
    if candidate_ppsm is None:
        return pd.Series(False, index=pref_median.index)

    global_ok = candidate_ppsm.between(global_low, global_high)

    pref_ok = pd.Series(False, index=candidate_ppsm.index)
    if pref_median is not None:
        pref_ratio = candidate_ppsm / pref_median.where(pref_median > 0)
        pref_ok = pref_ratio.between(LOCAL_RATIO_LOW, LOCAL_RATIO_HIGH)
        pref_ok = pref_ok.fillna(False)

    building_ok = pd.Series(False, index=candidate_ppsm.index)
    if building_median is not None and building_counts is not None:
        building_ratio = candidate_ppsm / building_median
        building_ok = (
            building_ratio.between(LOCAL_RATIO_LOW, LOCAL_RATIO_HIGH)
            & (building_counts >= 2)
        )
        building_ok = building_ok.fillna(False)

    unit_ok = pd.Series(False, index=candidate_ppsm.index)
    if unit_median is not None and unit_counts is not None:
        unit_ratio = candidate_ppsm / unit_median
        unit_ok = (
            unit_ratio.between(LOCAL_RATIO_LOW, LOCAL_RATIO_HIGH)
            & (unit_counts >= 2)
        )
        unit_ok = unit_ok.fillna(False)

    local_ok = pref_ok | building_ok | unit_ok
    return global_ok & local_ok


__all__ = ["adjust_mansion_money_room"]


