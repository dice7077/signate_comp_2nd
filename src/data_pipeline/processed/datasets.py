from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from ..utils.paths import DATA_DIR, INTERIM_DIR, PROJECT_ROOT


PROCESSED_ROOT = DATA_DIR / "processed"
SPLIT_BASE_DIR = INTERIM_DIR / "01_02_split_by_type"

TYPE_DIRECTORIES: Dict[str, str] = {
    "kodate": "0001_kodate",
    "mansion": "0002_mansion",
}

FEATURE_PLAN: Dict[str, List[str]] = {
    "kodate": [
        "target_ym",
        "lon",
        "lat",
        "unit_area_max",
        "land_area_all",
        "unit_count",
        "year_built",
        "2023_land_price",
        "2023_koji_price",
        "mesh_population_2025",
    ],
    "mansion": [
        "target_ym",
        "lon",
        "lat",
        "unit_area_min",
        "unit_area_max",
        "room_floor",
        "balcony_area",
        "room_count",
        "2023_koji_price",
        "mesh_population_2035",
    ],
}

COLUMN_SOURCES: Dict[str, str] = {
    "2023_land_price": "land",
    "2023_land_usage_code": "land",
    "2023_land_distance_km": "land",
    "2023_koji_price": "koji",
    "2023_koji_usage_code": "koji",
    "2023_koji_distance_km": "koji",
    "mesh_population_2025": "population",
    "mesh_population_2035": "population",
}

SUPPLEMENTARY_SOURCES: Dict[str, Dict[str, str]] = {
    "land": {
        "dir": "05_01_join_land_price",
        "train": "train.parquet",
        "test": "test.parquet",
    },
    "koji": {
        "dir": "03_01_join_koji_price",
        "train": "train.parquet",
        "test": "test.parquet",
    },
    "population": {
        "dir": "04_01_join_population_projection",
        "train": "train_population_features.parquet",
        "test": "test_population_features.parquet",
    },
}


class ProcessedDatasetError(RuntimeError):
    """build_processed_datasets 向けの例外。"""


def build_processed_datasets(
    version: str,
    *,
    types: Sequence[str] | None = None,
    overwrite: bool = False,
) -> List[Path]:
    """
    data/processed 配下に学習用データセットを構築する。
    """

    selected_types = _normalize_types(types)
    outputs: List[Path] = []

    for type_label in selected_types:
        features = FEATURE_PLAN[type_label]
        version_dir = PROCESSED_ROOT / TYPE_DIRECTORIES[type_label] / version
        version_dir.mkdir(parents=True, exist_ok=True)
        split_meta: Dict[str, dict] = {}

        for split in ("train", "test"):
            output_path = version_dir / f"{split}.parquet"
            if output_path.exists() and not overwrite:
                raise ProcessedDatasetError(
                    f"{output_path} が既に存在します。--overwrite を指定してください。"
                )

            df = _load_base_dataframe(split, type_label, features)
            df = _merge_supplementary_features(df, split, features)

            columns = _output_columns(split, features)
            missing = [col for col in columns if col not in df.columns]
            if missing:
                raise ProcessedDatasetError(
                    f"{type_label}/{split} で列が不足しています: {', '.join(missing)}"
                )

            result = df[columns].copy()
            result.to_parquet(output_path, index=False)
            outputs.append(output_path)

            split_meta[split] = {
                "path": str(output_path.relative_to(PROJECT_ROOT)),
                "rows": int(len(result)),
            }

        _write_manifest(
            version_dir=version_dir,
            type_label=type_label,
            version=version,
            features=features,
            split_meta=split_meta,
        )

    return outputs


def _normalize_types(types: Sequence[str] | None) -> List[str]:
    if not types:
        return list(FEATURE_PLAN.keys())
    normalized = []
    for type_label in types:
        if type_label not in FEATURE_PLAN:
            raise ProcessedDatasetError(f"未知のtypeが指定されました: {type_label}")
        if type_label not in normalized:
            normalized.append(type_label)
    return normalized


def _load_base_dataframe(split: str, type_label: str, features: Sequence[str]) -> pd.DataFrame:
    path = SPLIT_BASE_DIR / f"{split}_{type_label}.parquet"
    base_columns = _base_feature_columns(features)
    columns = ["data_id", *base_columns]
    if split == "train":
        columns.insert(1, "money_room")
    return pd.read_parquet(path, columns=columns)


def _base_feature_columns(features: Sequence[str]) -> List[str]:
    cols = [col for col in features if COLUMN_SOURCES.get(col, "base") == "base"]
    # 順序維持のため features の登場順で重複排除
    ordered: List[str] = []
    for col in features:
        if COLUMN_SOURCES.get(col, "base") != "base":
            continue
        if col not in ordered:
            ordered.append(col)
    return ordered


def _merge_supplementary_features(
    df: pd.DataFrame, split: str, features: Sequence[str]
) -> pd.DataFrame:
    required = _supplementary_feature_map(features)
    for source, columns in required.items():
        data = _load_supplementary(source, split, columns)
        df = df.merge(
            data,
            on="data_id",
            how="left",
            validate="one_to_one",
        )
    return df


def _supplementary_feature_map(features: Sequence[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for col in features:
        source = COLUMN_SOURCES.get(col)
        if not source:
            continue
        mapping.setdefault(source, [])
        if col not in mapping[source]:
            mapping[source].append(col)
    return mapping


def _load_supplementary(source: str, split: str, columns: Sequence[str]) -> pd.DataFrame:
    config = SUPPLEMENTARY_SOURCES[source]
    path = INTERIM_DIR / config["dir"] / config[split]
    read_columns = ["data_id", *columns]
    return pd.read_parquet(path, columns=read_columns)


def _output_columns(split: str, features: Sequence[str]) -> List[str]:
    cols = ["data_id"]
    if split == "train":
        cols.append("money_room")
    cols.extend(features)
    return cols


def _write_manifest(
    *,
    version_dir: Path,
    type_label: str,
    version: str,
    features: Sequence[str],
    split_meta: Dict[str, dict],
) -> None:
    manifest = {
        "type_label": type_label,
        "type_directory": TYPE_DIRECTORIES[type_label],
        "version": version,
        "feature_count": len(features),
        "features": list(features),
        "sources": _summarize_sources(features),
        "splits": split_meta,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = version_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)


def _summarize_sources(features: Sequence[str]) -> Dict[str, List[str]]:
    summary: Dict[str, List[str]] = {}
    for col in features:
        source = COLUMN_SOURCES.get(col, "base")
        summary.setdefault(source, [])
        summary[source].append(col)
    return summary


__all__ = ["build_processed_datasets", "ProcessedDatasetError"]


