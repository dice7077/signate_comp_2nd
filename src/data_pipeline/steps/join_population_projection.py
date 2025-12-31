from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..utils.paths import (
    INTERIM_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    ensure_parent,
    interim_subdir,
)
from .layout import step_output_dir


# Years to expose from the population projection GeoJSON.
POPULATION_YEARS: Tuple[int, ...] = (2025, 2035, 2045, 2055)
POPULATION_COLUMNS: Tuple[str, ...] = tuple(
    f"mesh_population_{year}" for year in POPULATION_YEARS
)

POPULATION_RAW_ROOT = RAW_DIR / "population" / "mesh1km_2024"
LOOKUP_DIR = INTERIM_DIR / "lookup_population_mesh"
LOOKUP_FILENAME = "mesh1km_population.parquet"
LOOKUP_PATH = LOOKUP_DIR / LOOKUP_FILENAME

SOURCE_DIR = INTERIM_DIR / step_output_dir("drop_sparse_columns")
OUTPUT_DIR_NAME = step_output_dir("join_population_projection")
MESH_ID_COLUMN = "mesh_id_1km"


def join_population_projection(force: bool = True) -> Dict[str, object]:
    """
    Enrich the cleaned Signate tables with 1km-mesh population projections.

    The population lookup is built from 47 prefecture-level GeoJSON files.
    Future projections for 2025, 2035, 2045, and 2055 are averaged per mesh.
    """

    population_lookup, lookup_stats = _load_population_lookup()
    output_dir = interim_subdir(OUTPUT_DIR_NAME)
    manifests: List[dict] = []

    for dataset_name in ("train", "test"):
        source_path = SOURCE_DIR / f"{dataset_name}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_path} not found. Run drop_sparse_columns first."
            )

        df = pd.read_parquet(source_path)
        enriched_df, join_stats = _attach_population(df, population_lookup)

        output_path = output_dir / f"{dataset_name}.parquet"
        if output_path.exists() and not force:
            raise FileExistsError(
                f"{output_path} already exists. Pass force=True to overwrite."
            )
        ensure_parent(output_path)
        enriched_df.to_parquet(output_path)

        manifests.append(
            {
                "dataset": dataset_name,
                "rows": int(len(enriched_df)),
                "population_columns": POPULATION_COLUMNS,
                "mesh_ids_missing": join_stats["mesh_ids_missing"],
                "population_attached_rows": join_stats["population_attached"],
                "deduplicated_rows": join_stats["deduplicated_rows"],
                "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    manifest = {
        "step": "join_population_projection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lookup_stats": lookup_stats,
        "outputs": manifests,
    }
    manifest_path = output_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _attach_population(
    df: pd.DataFrame, population_lookup: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if "lon" not in df.columns or "lat" not in df.columns:
        missing = {"lon", "lat"} - set(df.columns)
        raise KeyError(f"Missing required coordinate columns: {sorted(missing)}")
    if "data_id" not in df.columns:
        raise KeyError("'data_id' column missing from input dataset.")

    enriched = df.copy()
    enriched[MESH_ID_COLUMN] = _compute_mesh_ids(enriched["lat"], enriched["lon"])

    population_lookup = population_lookup.rename(columns={"mesh_id": MESH_ID_COLUMN})
    merged = enriched.merge(population_lookup, how="left", on=MESH_ID_COLUMN)
    population_cols = list(POPULATION_COLUMNS)

    duplicated_mask = merged["data_id"].duplicated(keep=False)
    deduplicated_rows = int(duplicated_mask.sum())
    if deduplicated_rows:
        population_means = (
            merged.loc[:, ["data_id", *population_cols]]
            .groupby("data_id", as_index=False)
            .mean()
        )
        base = (
            merged.drop(columns=population_cols)
            .drop_duplicates(subset="data_id", keep="first")
        )
        merged = base.merge(population_means, on="data_id", how="left")

    stats = {
        "mesh_ids_missing": int(merged[MESH_ID_COLUMN].isna().sum()),
        "population_attached": int(
            merged[population_cols].notna().any(axis=1).sum()
        ),
        "deduplicated_rows": deduplicated_rows,
    }
    return merged, stats


def _compute_mesh_ids(lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:
    lat = pd.to_numeric(lat_series, errors="coerce")
    lon = pd.to_numeric(lon_series, errors="coerce")
    mesh = pd.Series(pd.NA, index=lat.index, dtype="string[python]")
    mask = lat.notna() & lon.notna()
    if mask.any():
        mesh.loc[mask] = _vectorized_mesh(lat[mask].to_numpy(), lon[mask].to_numpy())
    return mesh


def _vectorized_mesh(lat: np.ndarray, lon: np.ndarray) -> List[str]:
    lat = lat.astype(float)
    lon = lon.astype(float)
    lat_minutes = lat * 60.0
    lon_diff = lon - 100.0

    block_lat = np.floor_divide(lat_minutes, 40).astype(int)
    block_lon = np.floor(lon_diff).astype(int)

    lat_rem = lat_minutes - block_lat * 40.0
    lon_rem = lon_diff - block_lon

    sub_lat = np.floor_divide(lat_rem, 5).astype(int)
    sub_lon = np.floor(lon_rem / 0.125).astype(int)

    lat_rem2 = lat_rem - sub_lat * 5.0
    lon_rem2 = lon_rem - sub_lon * 0.125

    fine_lat = np.floor(lat_rem2 / 0.5 + 1e-9).astype(int)
    fine_lon = np.floor(lon_rem2 / 0.0125 + 1e-9).astype(int)

    fine_lat = np.clip(fine_lat, 0, 9)
    fine_lon = np.clip(fine_lon, 0, 9)
    sub_lon = np.clip(sub_lon, 0, 7)

    mesh_numeric = (
        (block_lat * 1_000_000)
        + (block_lon * 10_000)
        + (sub_lat * 1_000)
        + (sub_lon * 100)
        + (fine_lat * 10)
        + fine_lon
    )
    return [f"{value:08d}" for value in mesh_numeric.tolist()]


def _load_population_lookup() -> Tuple[pd.DataFrame, Dict[str, int]]:
    if LOOKUP_PATH.exists():
        lookup_df = pd.read_parquet(LOOKUP_PATH)
        lookup_df["mesh_id"] = lookup_df["mesh_id"].astype("string[python]")
        return lookup_df, {"lookup_rows": int(len(lookup_df)), "rebuilt": False}

    lookup_df, build_stats = _build_population_lookup()
    ensure_parent(LOOKUP_PATH)
    lookup_df.to_parquet(LOOKUP_PATH, index=False)
    return lookup_df, {**build_stats, "rebuilt": True}


def _build_population_lookup() -> Tuple[pd.DataFrame, Dict[str, int]]:
    if not POPULATION_RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Population source directory not found: {POPULATION_RAW_ROOT}"
        )

    aggregates: Dict[str, Dict[str, List[float]]] = {}
    total_features = 0

    for pref_code, geojson_path in _iter_geojson_files():
        with geojson_path.open() as f:
            data = json.load(f)
        for feature in data.get("features", []):
            total_features += 1
            props = feature.get("properties") or {}
            mesh_id = props.get("MESH_ID")
            if not mesh_id:
                continue
            mesh_id = str(mesh_id)
            entry = aggregates.get(mesh_id)
            if entry is None:
                entry = {
                    "sums": [0.0 for _ in POPULATION_YEARS],
                    "counts": [0 for _ in POPULATION_YEARS],
                }
                aggregates[mesh_id] = entry
            for idx, year in enumerate(POPULATION_YEARS):
                value = _coerce_float(props.get(f"PTN_{year}"))
                if value is None:
                    continue
                entry["sums"][idx] += value
                entry["counts"][idx] += 1

    records: List[dict] = []
    for mesh_id, entry in aggregates.items():
        record = {"mesh_id": mesh_id}
        for idx, column in enumerate(POPULATION_COLUMNS):
            count = entry["counts"][idx]
            record[column] = (
                entry["sums"][idx] / count if count else None
            )
        records.append(record)

    lookup_df = pd.DataFrame(records)
    lookup_df["mesh_id"] = lookup_df["mesh_id"].astype("string[python]")
    lookup_df = lookup_df.sort_values("mesh_id").reset_index(drop=True)
    stats = {"lookup_rows": int(len(lookup_df)), "features_processed": total_features}
    return lookup_df, stats


def _iter_geojson_files() -> Iterable[Tuple[str, Path]]:
    for subdir in sorted(POPULATION_RAW_ROOT.glob("1km_mesh_2024_*_GEOJSON")):
        parts = subdir.name.split("_")
        if len(parts) < 4:
            continue
        pref_code = parts[3]
        geojson_name = f"1km_mesh_2024_{pref_code}.geojson"
        geojson_path = subdir / geojson_name
        if geojson_path.exists():
            yield pref_code, geojson_path


def _coerce_float(value) -> float | None:
    if value in (None, "", "@", "*"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["join_population_projection"]

