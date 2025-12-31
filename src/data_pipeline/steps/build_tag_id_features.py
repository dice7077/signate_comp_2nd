from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Sequence

import pandas as pd

from ..utils.paths import (
    INTERIM_DIR,
    PROJECT_ROOT,
    ensure_parent,
    interim_subdir,
)
from ..utils.signate_types import TYPE_NAME_MAP
from .layout import step_output_dir

UNKNOWN_TYPE_LABEL = "unknown"

SOURCE_DIR = INTERIM_DIR / step_output_dir("assign_data_id")
OUTPUT_DIR_NAME = step_output_dir("build_tag_id_features")

# Keep the order stable so train/test share identical column layouts.
TAG_FEATURES = OrderedDict(
    [
        ("unit_tag_id", {"column_prefix": "unit_tag"}),
        ("building_tag_id", {"column_prefix": "building_tag"}),
        ("statuses", {"column_prefix": "status_tag"}),
    ]
)


def build_tag_id_features(force: bool = True) -> Dict[str, object]:
    """
    Construct lookup + wide matrices for slash-delimited tag columns.

    Generates three Parquet files inside data/interim/02_01_build_tag_id_features/:
    - tag_ids.parquet: feature_name/tag_id lookup (union of train+test).
    - train_tag_ids.parquet: one-hot matrix keyed by data_id.
    - test_tag_ids.parquet: same columns/order as train.
    """

    dataset_frames = _load_source_tables()
    tag_catalog = _collect_tag_catalog(dataset_frames)

    output_dir = interim_subdir(OUTPUT_DIR_NAME)
    outputs: List[dict] = []

    tag_ids_df = _build_catalog_frame(tag_catalog)
    tag_ids_path = output_dir / "tag_ids.parquet"
    _write_parquet(tag_ids_df, tag_ids_path, force=force)
    outputs.append(
        {
            "name": "tag_ids",
            "rows": int(len(tag_ids_df)),
            "columns": int(tag_ids_df.shape[1]),
            "path": str(tag_ids_path.relative_to(PROJECT_ROOT)),
        }
    )

    encoded_tables: Dict[str, pd.DataFrame] = {}
    for dataset_name, df in dataset_frames.items():
        encoded = _encode_dataset(df, tag_catalog)
        encoded_tables[dataset_name] = encoded
        dataset_path = output_dir / f"{dataset_name}_tag_ids.parquet"
        _write_parquet(encoded, dataset_path, force=force)
        outputs.append(
            {
                "name": f"{dataset_name}_tag_ids",
                "rows": int(len(encoded)),
                "columns": int(encoded.shape[1]),
                "path": str(dataset_path.relative_to(PROJECT_ROOT)),
            }
        )

    manifest = {
        "step": "build_tag_id_features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(SOURCE_DIR.relative_to(PROJECT_ROOT)),
        "tag_feature_counts": {
            feature: len(tags) for feature, tags in tag_catalog.items()
        },
        "outputs": outputs,
    }

    manifest_path = output_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _load_source_tables() -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    required_columns = ["data_id", "bukken_type", *TAG_FEATURES.keys()]
    for dataset_name in ("train", "test"):
        source_path = SOURCE_DIR / f"{dataset_name}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_path} not found. Run assign_data_id before this step."
            )
        df = pd.read_parquet(source_path, columns=required_columns)
        df["bukken_type"] = (
            pd.to_numeric(df["bukken_type"], errors="coerce").astype("Int64")
        )
        for feature in TAG_FEATURES.keys():
            df[feature] = df[feature].astype("string[python]")
        frames[dataset_name] = df
    return frames


def _collect_tag_catalog(dataset_frames: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    catalog: Dict[str, List[str]] = OrderedDict()
    for feature in TAG_FEATURES.keys():
        combined_series = pd.concat(
            [df[feature] for df in dataset_frames.values()],
            ignore_index=True,
        )
        catalog[feature] = _extract_unique_tags(combined_series)
    return catalog


def _extract_unique_tags(series: pd.Series) -> List[str]:
    unique_tags: set[str] = set()
    for value in series:
        unique_tags.update(_split_tag_string(value))
    return sorted(unique_tags)


def _build_catalog_frame(tag_catalog: Dict[str, List[str]]) -> pd.DataFrame:
    records: List[dict] = []
    for feature, tag_ids in tag_catalog.items():
        for tag_id in tag_ids:
            records.append({"feature_name": feature, "tag_id": tag_id})
    df = pd.DataFrame(records)
    if not df.empty:
        df["feature_name"] = df["feature_name"].astype("string[python]")
        df["tag_id"] = df["tag_id"].astype("string[python]")
    return df


def _encode_dataset(
    df: pd.DataFrame, tag_catalog: Dict[str, List[str]]
) -> pd.DataFrame:
    encoded = df[["data_id", "bukken_type"]].reset_index(drop=True)
    encoded["bukken_type_label"] = (
        _map_bukken_type_labels(df["bukken_type"]).reset_index(drop=True)
    )
    for feature, meta in TAG_FEATURES.items():
        tags = tag_catalog.get(feature, [])
        block = _encode_feature_block(df[feature], tags, meta["column_prefix"])
        if not block.empty:
            encoded = pd.concat([encoded, block.reset_index(drop=True)], axis=1)
    return encoded


def _encode_feature_block(
    series: pd.Series,
    tag_ids: Sequence[str],
    column_prefix: str,
) -> pd.DataFrame:
    if not tag_ids:
        return pd.DataFrame(index=series.index)

    normalized = (
        series.astype("string[python]")
        .fillna("")
        .str.replace(" ", "", regex=False)
    )
    dummies = normalized.str.get_dummies(sep="/")
    if "" in dummies.columns:
        dummies = dummies.drop(columns="", errors="ignore")

    dummies = dummies.reindex(columns=list(tag_ids), fill_value=0)
    dummies = dummies.astype("uint8", copy=False)
    dummies.columns = [f"{column_prefix}_{tag}" for tag in tag_ids]
    return dummies


def _split_tag_string(value: object) -> List[str]:
    if value is None or pd.isna(value):
        return []
    tokens = str(value).split("/")
    cleaned = [token.strip() for token in tokens if token and token.strip()]
    return cleaned


def _map_bukken_type_labels(series: pd.Series) -> pd.Series:
    labels = series.map(TYPE_NAME_MAP)
    labels = labels.fillna(UNKNOWN_TYPE_LABEL)
    return labels.astype("string[python]")


def _write_parquet(df: pd.DataFrame, path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. Pass force=True to overwrite the file."
        )
    ensure_parent(path)
    df.to_parquet(path, index=False)


__all__ = ["build_tag_id_features"]

