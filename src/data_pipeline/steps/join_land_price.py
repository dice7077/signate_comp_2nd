from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
SECONDARY_SEARCH_RADIUS_KM = 3.0
COORD_ROUND_DIGITS = 6

SOURCE_DIR = INTERIM_DIR / step_output_dir("assign_data_id")
OUTPUT_DIR_NAME = step_output_dir("join_land_price")
RAW_LAND_DIR = RAW_DIR / "land_price"

LAND_GEOJSON_SOURCES: Tuple[dict, ...] = (
    {
        "year": 2023,
        "price_column": "2023_land_price",
        "distance_column": "2023_land_distance_km",
        "usage_column": "2023_land_usage_code",
        "filename": "L02-23.geojson",
        "price_field": "L02_006",
        "usage_field": "L02_046",
    },
    {
        "year": 2022,
        "price_column": "2022_land_price",
        "distance_column": "2022_land_distance_km",
        "usage_column": "2022_land_usage_code",
        "filename": "L02-22.geojson",
        "price_field": "L02_006",
        "usage_field": "L02_046",
    },
    {
        "year": 2021,
        "price_column": "2021_land_price",
        "distance_column": "2021_land_distance_km",
        "usage_column": "2021_land_usage_code",
        "filename": "L02_21.geojson",
        "price_field": "L02_006",
        "usage_field": "L02_046",
    },
    {
        "year": 2020,
        "price_column": "2020_land_price",
        "distance_column": "2020_land_distance_km",
        "usage_column": "2020_land_usage_code",
        "filename": "L02-20.geojson",
        "price_field": "L02_006",
        "usage_field": "L02_045",
        "usage_candidates": ("L02_045", "L02_046"),
    },
    {
        "year": 2019,
        "price_column": "2019_land_price",
        "distance_column": "2019_land_distance_km",
        "usage_column": "2019_land_usage_code",
        "filename": "L02-19.geojson",
        "price_field": None,
        "usage_field": None,
        "price_candidates": ("L02_006", "L02_044", "L02_005"),
        "usage_candidates": ("L02_045", "L02_046"),
        "price_keywords": ("価格", "地価"),
        "usage_keywords": ("用途", "地域"),
    },
)

LAND_PRICE_COLUMNS: Tuple[str, ...] = tuple(cfg["price_column"] for cfg in LAND_GEOJSON_SOURCES)
LAND_DISTANCE_COLUMNS: Tuple[str, ...] = tuple(
    cfg["distance_column"] for cfg in LAND_GEOJSON_SOURCES
)
LAND_USAGE_COLUMNS: Tuple[str, ...] = tuple(cfg["usage_column"] for cfg in LAND_GEOJSON_SOURCES)
LAND_PRICE_SOURCE_PATHS: Tuple[Path, ...] = tuple(
    RAW_LAND_DIR / cfg["filename"] for cfg in LAND_GEOJSON_SOURCES
)


def join_land_price(force: bool = True) -> Dict[str, object]:
    """地価公示GeoJSONから1.5km以内の最近傍地点を data_id 単位で集計する。"""

    land_points_by_year = _load_land_price_points()
    tree_by_year = {
        price_column: _build_spatial_index(df)
        for price_column, df in land_points_by_year.items()
        if not df.empty
    }
    output_dir = interim_subdir(OUTPUT_DIR_NAME)

    manifests: List[dict] = []
    for dataset_name in ("train", "test"):
        source_path = SOURCE_DIR / f"{dataset_name}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_path} not found. Run assign_data_id before join_land_price."
            )

        signate_df = _load_signate_subset(source_path)
        joined_df, stats = _attach_nearest_land(signate_df, land_points_by_year, tree_by_year)

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
                "within_radius_primary": stats["within_radius_primary"],
                "outside_radius": stats["outside_radius"],
                "missing_coordinates": stats["missing_coordinates"],
                "land_points": stats["land_points"],
                "search_radius_km": SEARCH_RADIUS_KM,
                "secondary_search_radius_km": SECONDARY_SEARCH_RADIUS_KM,
                "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    manifest = {
        "step": "join_land_price",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "land_price_sources": [
            str(path.relative_to(PROJECT_ROOT)) for path in LAND_PRICE_SOURCE_PATHS
        ],
        "coordinate_round_digits": COORD_ROUND_DIGITS,
        "search_radius_km": SEARCH_RADIUS_KM,
        "secondary_search_radius_km": SECONDARY_SEARCH_RADIUS_KM,
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


def _attach_nearest_land(
    signate_df: pd.DataFrame,
    land_points_by_year: Dict[str, pd.DataFrame],
    trees_by_year: Dict[str, BallTree],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
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

    for column_name in LAND_PRICE_COLUMNS:
        result[column_name] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    for column_name in LAND_USAGE_COLUMNS:
        result[column_name] = pd.Series(pd.NA, index=result.index, dtype="string[python]")
    for column_name in LAND_DISTANCE_COLUMNS:
        result[column_name] = pd.Series(pd.NA, index=result.index, dtype="Float64")

    lat = pd.to_numeric(signate_df["lat"], errors="coerce")
    lon = pd.to_numeric(signate_df["lon"], errors="coerce")
    valid_mask = lat.notna() & lon.notna()
    valid_indices = signate_df.index[valid_mask].to_numpy()

    stats = {
        "rows": int(len(signate_df)),
        "valid_coordinates": int(valid_mask.sum()),
        "missing_coordinates": int(len(signate_df) - valid_mask.sum()),
        "within_radius_primary": {cfg["price_column"]: 0 for cfg in LAND_GEOJSON_SOURCES},
        "within_radius": {cfg["price_column"]: 0 for cfg in LAND_GEOJSON_SOURCES},
        "outside_radius": {cfg["price_column"]: int(len(signate_df)) for cfg in LAND_GEOJSON_SOURCES},
        "land_points": {
            cfg["price_column"]: int(len(land_points_by_year.get(cfg["price_column"], ())))
            for cfg in LAND_GEOJSON_SOURCES
        },
    }

    if valid_mask.any():
        query_coords = np.column_stack([lat.loc[valid_mask], lon.loc[valid_mask]])
        query_rad = np.radians(query_coords.astype(float))

        for config in LAND_GEOJSON_SOURCES:
            price_col = config["price_column"]
            usage_col = config["usage_column"]
            distance_col = config["distance_column"]
            points_df = land_points_by_year.get(price_col)
            tree = trees_by_year.get(price_col)

            if points_df is None or tree is None or points_df.empty:
                continue

            distances_rad, indices = tree.query(query_rad, k=1)
            distances_km = distances_rad[:, 0] * EARTH_RADIUS_KM
            primary_within = distances_km <= SEARCH_RADIUS_KM
            secondary_within = distances_km <= SECONDARY_SEARCH_RADIUS_KM

            stats["within_radius_primary"][price_col] = int(primary_within.sum())
            stats["within_radius"][price_col] = int(secondary_within.sum())
            stats["outside_radius"][price_col] = stats["rows"] - stats["within_radius"][price_col]

            if secondary_within.any():
                matched_signate_idx = valid_indices[secondary_within]
                matched_land_idx = indices[secondary_within, 0]
                matched_points = points_df.iloc[matched_land_idx]
                matched_distances = distances_km[secondary_within]

                result.loc[matched_signate_idx, price_col] = matched_points[price_col].to_numpy()
                result.loc[matched_signate_idx, usage_col] = matched_points[usage_col].to_numpy()
                result.loc[matched_signate_idx, distance_col] = matched_distances

    return result, stats


def _build_spatial_index(land_df: pd.DataFrame) -> BallTree:
    coords = land_df[["lat", "lon"]].to_numpy(dtype=float)
    coords_rad = np.radians(coords)
    return BallTree(coords_rad, metric="haversine")


def _load_land_price_points() -> Dict[str, pd.DataFrame]:
    points_by_year: Dict[str, pd.DataFrame] = {}
    for config in LAND_GEOJSON_SOURCES:
        df = _load_geojson_for_year(config)
        points_by_year[config["price_column"]] = df
    if not points_by_year:
        raise ValueError("No land price sources configured.")
    return points_by_year


def _load_geojson_for_year(config: dict) -> pd.DataFrame:
    path = RAW_LAND_DIR / config["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Land price source not found: {path}")

    with path.open(encoding="utf-8") as fp:
        geojson = json.load(fp)

    records: List[dict] = []
    price_field_name: str | None = None
    usage_field_name: str | None = None

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
        keys = tuple(props.keys())
        if price_field_name is None:
            price_field_name = _resolve_property_field_name(
                keys,
                explicit=config.get("price_field"),
                candidates=config.get("price_candidates"),
                keywords=config.get("price_keywords"),
                column_label=config["price_column"],
                field_type="price",
            )
        if usage_field_name is None:
            usage_field_name = _resolve_property_field_name(
                keys,
                explicit=config.get("usage_field"),
                candidates=config.get("usage_candidates"),
                keywords=config.get("usage_keywords"),
                column_label=config["usage_column"],
                field_type="usage",
                allow_missing=False,
            )

        record = {
            "_coord_key": _format_coord_key(lat, lon),
            "lon": lon,
            "lat": lat,
            config["price_column"]: _coerce_float(props.get(price_field_name)),
            config["usage_column"]: _coerce_str(props.get(usage_field_name)),
        }
        records.append(record)

    if not records:
        raise ValueError(f"No valid land price points found in {path}.")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="_coord_key", keep="first")
    df = df.dropna(subset=["lon", "lat"])
    df = df.reset_index(drop=True)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce").astype(float)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce").astype(float)
    df[config["price_column"]] = pd.to_numeric(
        df[config["price_column"]], errors="coerce"
    ).astype("Float64")
    df[config["usage_column"]] = df[config["usage_column"]].astype("string[python]")
    return df


def _resolve_property_field_name(
    available_keys: Sequence[str],
    *,
    explicit: str | None,
    candidates: Sequence[str] | None,
    keywords: Sequence[str] | None,
    column_label: str,
    field_type: str,
    allow_missing: bool = False,
) -> str | None:
    if explicit and explicit in available_keys:
        return explicit
    if candidates:
        for candidate in candidates:
            if candidate in available_keys:
                return candidate
    if keywords:
        for keyword in keywords:
            for key in available_keys:
                if keyword and keyword in key:
                    return key
    if allow_missing:
        return None
    raise KeyError(
        f"Unable to locate {field_type} field for {column_label}. "
        f"explicit={explicit}, candidates={candidates}, keywords={keywords}"
    )


def _format_coord_key(lat: float, lon: float) -> str:
    return f"{lat:.{COORD_ROUND_DIGITS}f}_{lon:.{COORD_ROUND_DIGITS}f}"


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


__all__ = ["join_land_price"]





