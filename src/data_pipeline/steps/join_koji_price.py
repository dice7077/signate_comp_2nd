from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ..utils.paths import (
    INTERIM_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    ensure_parent,
    interim_subdir,
)
from ..utils.signate_types import TYPE_NAME_MAP
from .layout import step_output_dir

EARTH_RADIUS_KM = 6371.0088
SEARCH_RADIUS_KM = 1.5

KOJI_PRICE_FIELD_MAP: Tuple[Tuple[str, str], ...] = (
    ("2023_koji_price", "L01_101"),
    ("2022_koji_price", "L01_100"),
    ("2021_koji_price", "L01_099"),
    ("2020_koji_price", "L01_098"),
    ("2019_koji_price", "L01_097"),
    ("2018_koji_price", "L01_096"),
)
KOJI_GROWTH_COLUMN = ("koji_price_growth_2023_vs_2022", "2023_koji_price", "2022_koji_price")
KOJI_USAGE_FIELD = "L01_050"
KOJI_TEXT_FIELD_MAP: Tuple[Tuple[str, str], ...] = (
    ("koji_usage_status", "L01_027"),  # 利用現況
    ("koji_building_structure", "L01_030"),  # 建物構造
)

SOURCE_DIR = INTERIM_DIR / step_output_dir("assign_data_id")
OUTPUT_DIR_NAME = step_output_dir("join_koji_price")
KOJI_PRICE_PATH = RAW_DIR / "koji_price" / "L01-23.geojson"


def join_koji_price(force: bool = True) -> Dict[str, object]:
    """公示価格GeoJSONから1.5km以内の最近傍価格を data_id 単位でまとめる。"""

    koji_points = _load_koji_price_points(KOJI_PRICE_PATH)
    tree = _build_spatial_index(koji_points)
    output_dir = interim_subdir(OUTPUT_DIR_NAME)

    manifests: List[dict] = []
    for dataset_name in ("train", "test"):
        source_path = SOURCE_DIR / f"{dataset_name}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_path} not found. Run assign_data_id before join_koji_price."
            )

        signate_df = _load_signate_subset(source_path)
        joined_df, stats = _attach_nearest_koji(signate_df, koji_points, tree)

        output_path = output_dir / f"{dataset_name}.parquet"
        if output_path.exists() and not force:
            raise FileExistsError(
                f"{output_path} already exists. Pass force=True to overwrite."
            )
        ensure_parent(output_path)
        joined_df.to_parquet(output_path)

        manifests.append(
            {
                "dataset": dataset_name,
                "rows": int(len(joined_df)),
                "within_radius": stats["within_radius"],
                "outside_radius": stats["outside_radius"],
                "missing_coordinates": stats["missing_coordinates"],
                "koji_points": stats["koji_points"],
                "search_radius_km": SEARCH_RADIUS_KM,
                "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    manifest = {
        "step": "join_koji_price",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "koji_price_source": str(KOJI_PRICE_PATH.relative_to(PROJECT_ROOT)),
        "outputs": manifests,
    }
    manifest_path = output_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _load_signate_subset(path: Path) -> pd.DataFrame:
    required = ["data_id", "lon", "lat", "bukken_type"]
    df = pd.read_parquet(path, columns=required)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing} in {path}")
    return df


def _attach_nearest_koji(
    signate_df: pd.DataFrame, koji_df: pd.DataFrame, tree: BallTree
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    result = pd.DataFrame(
        {
            "data_id": signate_df["data_id"].copy(),
            "bukken_type": pd.to_numeric(signate_df["bukken_type"], errors="coerce").astype(
                "Int64"
            ),
        }
    )
    type_labels = result["bukken_type"].map(TYPE_NAME_MAP)
    result["bukken_type_label"] = type_labels.astype("string[python]")
    for column_name, _ in KOJI_PRICE_FIELD_MAP:
        result[column_name] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    result["koji_usage_code"] = pd.Series(pd.NA, index=result.index, dtype="string[python]")
    result[KOJI_GROWTH_COLUMN[0]] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    result["koji_distance_km"] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    for column_name, _ in KOJI_TEXT_FIELD_MAP:
        result[column_name] = pd.Series(pd.NA, index=result.index, dtype="string[python]")

    lat = pd.to_numeric(signate_df["lat"], errors="coerce")
    lon = pd.to_numeric(signate_df["lon"], errors="coerce")
    valid_mask = lat.notna() & lon.notna()
    valid_indices = signate_df.index[valid_mask].to_numpy()

    stats = {
        "rows": int(len(signate_df)),
        "valid_coordinates": int(valid_mask.sum()),
        "missing_coordinates": int(len(signate_df) - valid_mask.sum()),
        "within_radius": 0,
        "outside_radius": int(len(signate_df)),
        "koji_points": int(len(koji_df)),
    }

    if valid_mask.any():
        query_coords = np.column_stack([lat.loc[valid_mask], lon.loc[valid_mask]])
        query_rad = np.radians(query_coords.astype(float))
        distances_rad, indices = tree.query(query_rad, k=1)
        distances_km = distances_rad[:, 0] * EARTH_RADIUS_KM
        within_radius = distances_km <= SEARCH_RADIUS_KM

        stats["within_radius"] = int(within_radius.sum())
        stats["outside_radius"] = stats["rows"] - stats["within_radius"]

        if within_radius.any():
            matched_signate_idx = valid_indices[within_radius]
            matched_koji_idx = indices[within_radius, 0]
            matched_points = koji_df.iloc[matched_koji_idx]
            matched_distances = distances_km[within_radius]

            result.loc[matched_signate_idx, "koji_usage_code"] = matched_points[
                "koji_usage_code"
            ].to_numpy()
            for column_name, _ in KOJI_PRICE_FIELD_MAP:
                result.loc[matched_signate_idx, column_name] = matched_points[
                    column_name
                ].to_numpy()
            growth_values = _compute_growth(
                matched_points[KOJI_GROWTH_COLUMN[1]].to_numpy(dtype=float, na_value=np.nan),
                matched_points[KOJI_GROWTH_COLUMN[2]].to_numpy(dtype=float, na_value=np.nan),
            )
            result.loc[matched_signate_idx, KOJI_GROWTH_COLUMN[0]] = growth_values
            result.loc[matched_signate_idx, "koji_distance_km"] = matched_distances
            for column_name, _ in KOJI_TEXT_FIELD_MAP:
                result.loc[matched_signate_idx, column_name] = matched_points[
                    column_name
                ].to_numpy()

    return result, stats


def _build_spatial_index(koji_df: pd.DataFrame) -> BallTree:
    coords = koji_df[["lat", "lon"]].to_numpy(dtype=float)
    coords_rad = np.radians(coords)
    return BallTree(coords_rad, metric="haversine")


def _load_koji_price_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Koji price source not found: {path}")

    with path.open(encoding="utf-8") as fp:
        geojson = json.load(fp)

    records: List[dict] = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates")
        if not coordinates or len(coordinates) < 2:
            continue
        lon = _coerce_float(coordinates[0])
        lat = _coerce_float(coordinates[1])
        if lon is None or lat is None:
            continue
        props = feature.get("properties") or {}
        text_values = {
            column_name: _coerce_str(props.get(raw_field))
            for column_name, raw_field in KOJI_TEXT_FIELD_MAP
        }
        usage_status = text_values.get("koji_usage_status")
        if usage_status is None or "住宅" not in usage_status:
            continue

        record = {
            "lon": lon,
            "lat": lat,
            "koji_usage_code": _coerce_str(props.get(KOJI_USAGE_FIELD)),
        }
        for column_name, raw_field in KOJI_PRICE_FIELD_MAP:
            record[column_name] = _coerce_float(props.get(raw_field))
        for column_name, value in text_values.items():
            record[column_name] = value
        records.append(record)

    if not records:
        raise ValueError("No valid Koji price points found in GeoJSON.")

    df = pd.DataFrame(records).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    df["koji_usage_code"] = df["koji_usage_code"].astype("string[python]")
    for column_name, _ in KOJI_PRICE_FIELD_MAP:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("Float64")
    for column_name, _ in KOJI_TEXT_FIELD_MAP:
        df[column_name] = df[column_name].astype("string[python]")

    return df


def _coerce_float(value) -> float | None:
    try:
        if value in (None, "", "_"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value) -> str | None:
    if value in (None, "", "_"):
        return None
    return str(value)


def _compute_growth(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    curr = current.astype(float)
    prev = previous.astype(float)
    growth = np.full(curr.shape, np.nan, dtype=float)
    valid = np.isfinite(curr) & np.isfinite(prev) & (prev != 0)
    if valid.any():
        growth[valid] = (curr[valid] - prev[valid]) / prev[valid]
    return growth


__all__ = ["join_koji_price"]

