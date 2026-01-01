from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, cast

import pandas as pd

from ..utils.paths import DATA_DIR, INTERIM_DIR, PROJECT_ROOT


PROCESSED_ROOT = DATA_DIR / "processed"
SPLIT_BASE_DIR = INTERIM_DIR / "01_02_split_by_type"

TYPE_DIRECTORIES: Dict[str, str] = {
    "kodate": "0001_kodate",
    "mansion": "0002_mansion",
}

AUXILIARY_BASE_COLUMNS: Tuple[str, ...] = ("target_ym",)
KOJI_YEARS: Tuple[int, ...] = (2023, 2022, 2021, 2020, 2019, 2018)
LAND_YEARS: Tuple[int, ...] = (2023, 2022, 2021, 2020, 2019)

TARGET_YEAR_FEATURE_SPECS: Dict[str, Dict[str, object]] = {
    "koji_price": {
        "source": "koji",
        "dtype": "Float64",
        "year_columns": {year: f"{year}_koji_price" for year in KOJI_YEARS},
    },
    "koji_usage_code": {
        "source": "koji",
        "dtype": "string[python]",
        "year_columns": {year: f"{year}_koji_usage_code" for year in KOJI_YEARS},
    },
    "koji_distance_km": {
        "source": "koji",
        "dtype": "Float64",
        "year_columns": {year: f"{year}_koji_distance_km" for year in KOJI_YEARS},
    },
    "land_price": {
        "source": "land",
        "dtype": "Float64",
        "year_columns": {year: f"{year}_land_price" for year in LAND_YEARS},
    },
    "land_usage_code": {
        "source": "land",
        "dtype": "string[python]",
        "year_columns": {year: f"{year}_land_usage_code" for year in LAND_YEARS},
    },
    "land_distance_km": {
        "source": "land",
        "dtype": "Float64",
        "year_columns": {year: f"{year}_land_distance_km" for year in LAND_YEARS},
    },
}

MANSION_UNIT_TAG_COLUMNS: Tuple[str, ...] = (
    "unit_tag_230401",
    "unit_tag_220301",
    "unit_tag_230203",
    "unit_tag_223101",
    "unit_tag_230501",
    "unit_tag_340401",
    "unit_tag_230601",
    "unit_tag_250301",
    "unit_tag_240201",
    "unit_tag_340201",
    "unit_tag_331001",
    "unit_tag_340101",
    "unit_tag_290201",
)

MANSION_BUILDING_TAG_COLUMNS: Tuple[str, ...] = (
    "building_tag_310101",
    "building_tag_321101",
    "building_tag_334101",
    "building_tag_330101",
    "building_tag_320401",
    "building_tag_334201",
    "building_tag_433301",
    "building_tag_310401",
)

MANSION_TAG_FEATURES: Tuple[str, ...] = (
    MANSION_UNIT_TAG_COLUMNS + MANSION_BUILDING_TAG_COLUMNS
)

KODATE_UNIT_TAG_COLUMNS: Tuple[str, ...] = (
    "unit_tag_230401",
    "unit_tag_223101",
    "unit_tag_230203",
    "unit_tag_290101",
    "unit_tag_230501",
    "unit_tag_250301",
    "unit_tag_230601",
    "unit_tag_230103",
    "unit_tag_290601",
    "unit_tag_240201",
    "unit_tag_220901",
    "unit_tag_290501",
    "unit_tag_223201",
    "unit_tag_230202",
    "unit_tag_340401",
)

KODATE_BUILDING_TAG_COLUMNS: Tuple[str, ...] = (
    "building_tag_340301",
    "building_tag_210401",
    "building_tag_294201",
    "building_tag_334101",
    "building_tag_334201",
    "building_tag_334001",
)

KODATE_TAG_FEATURES: Tuple[str, ...] = (
    KODATE_UNIT_TAG_COLUMNS + KODATE_BUILDING_TAG_COLUMNS
)

ALL_TAG_FEATURES: Tuple[str, ...] = MANSION_TAG_FEATURES + KODATE_TAG_FEATURES

FEATURE_PLAN: Dict[str, List[str]] = {
    "kodate": [
        "building_structure",
        "total_floor_area",
        "floor_count",
        "year_built",
        "years_old",
        "building_land_area",
        "land_area_all",
        "building_land_chimoku",
        "land_youto",
        "land_toshi",
        "land_chisei",
        "land_kenpei",
        "land_youseki",
        "land_road_cond",
        "balcony_area",
        "dwelling_unit_window_angle",
        "room_count",
        "unit_area",
        "floor_plan_code",
        "flg_investment",
        "post1",
        "post_all",
        "addr1_1",
        "addr_all",
        "rosen_name1",
        "eki_name1",
        "walk_distance1",
        "rosen_name2",
        "eki_name2",
        "walk_distance2",
        "house_area",
        "madori_number_all",
        "madori_kind_all",
        "genkyo_code",
        "usable_status",
        "school_ele_name",
        "school_ele_distance",
        "convenience_distance",
        "super_distance",
        "koji_price",
        "koji_usage_code",
        "koji_distance_km",
        "land_price",
        "land_usage_code",
        "land_distance_km",
        "mesh_population_2025",
        "mesh_population_2035",
        "mesh_population_2045",
        "mesh_population_2055",
    ],
    "mansion": [
        "building_structure",
        "floor_count",
        "year_built",
        "building_land_chimoku",
        "land_youto",
        "land_toshi",
        "land_chisei",
        "management_form",
        "room_floor",
        "balcony_area",
        "dwelling_unit_window_angle",
        "room_count",
        "unit_area",
        "floor_plan_code",
        "flg_investment",
        "post1",
        "post_all",
        "addr1_1",
        "addr_all",
        "walk_distance1",
        "walk_distance2",
        "house_area",
        "room_kaisuu",
        "snapshot_window_angle",
        "madori_number_all",
        "madori_kind_all",
        "money_kyoueki",
        "parking_money",
        "genkyo_code",
        "usable_status",
        "target_ym",
        "lon",
        "lat",
        "unit_area_min",
        "unit_area_max",
        "koji_price",
        "koji_usage_code",
        "koji_distance_km",
        "land_price",
        "land_usage_code",
        "land_distance_km",
        "mesh_population_2025",
        "mesh_population_2035",
        "mesh_population_2045",
        "mesh_population_2055",
    ],
}

FEATURE_PLAN_OVERRIDES: Dict[Tuple[str, str], List[str]] = {
    ("mansion", "0004_add_tags"): FEATURE_PLAN["mansion"] + list(MANSION_TAG_FEATURES),
    ("kodate", "0006_add_tags"): FEATURE_PLAN["kodate"] + list(KODATE_TAG_FEATURES),
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
    "mesh_population_2045": "population",
    "mesh_population_2055": "population",
    "koji_price": "koji",
    "koji_usage_code": "koji",
    "koji_distance_km": "koji",
    "land_price": "land",
    "land_usage_code": "land",
    "land_distance_km": "land",
    **{column: "tags" for column in ALL_TAG_FEATURES},
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
    "tags": {
        "dir": "02_01_build_tag_id_features",
        "train": "train_tag_ids.parquet",
        "test": "test_tag_ids.parquet",
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
        features = _resolve_feature_plan(type_label, version)
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
            df = _apply_target_year_alignment(df)

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


def _resolve_feature_plan(type_label: str, version: str) -> List[str]:
    override = FEATURE_PLAN_OVERRIDES.get((type_label, version))
    if override is not None:
        return list(override)
    return list(FEATURE_PLAN[type_label])


def _load_base_dataframe(split: str, type_label: str, features: Sequence[str]) -> pd.DataFrame:
    path = SPLIT_BASE_DIR / f"{split}_{type_label}.parquet"
    base_columns = _base_feature_columns(features)
    columns = ["data_id"]
    if split == "train":
        columns.append("money_room")
    columns.extend(AUXILIARY_BASE_COLUMNS)
    columns.extend(base_columns)
    ordered_columns = list(dict.fromkeys(columns))
    return pd.read_parquet(path, columns=ordered_columns)


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
    read_columns = ["data_id", *_expand_supplementary_columns(source, columns)]
    return pd.read_parquet(path, columns=read_columns)


def _expand_supplementary_columns(source: str, columns: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for column in columns:
        spec = TARGET_YEAR_FEATURE_SPECS.get(column)
        if spec and spec["source"] == source:
            expanded.extend(spec["year_columns"].values())
        else:
            expanded.append(column)
    # 順序を維持したまま重複排除
    ordered: List[str] = []
    seen = set()
    for column in expanded:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


def _apply_target_year_alignment(df: pd.DataFrame) -> pd.DataFrame:
    if "target_ym" not in df.columns:
        return df

    target_year = _extract_target_year(df["target_ym"])
    if target_year.isna().all():
        return df

    for feature, spec in TARGET_YEAR_FEATURE_SPECS.items():
        year_columns = cast(Dict[int, str], spec["year_columns"])
        dtype = cast(str, spec["dtype"])
        available = {year: col for year, col in year_columns.items() if col in df.columns}
        if not available:
            continue
        if feature not in df.columns:
            df[feature] = pd.Series(pd.NA, dtype=dtype, index=df.index)
        else:
            df[feature] = df[feature].astype(dtype)
        for year, column_name in available.items():
            mask = target_year == year
            if not mask.any():
                continue
            df.loc[mask, feature] = df.loc[mask, column_name]
    return df


def _extract_target_year(target_ym: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(target_ym, errors="coerce")
    years = pd.Series(pd.NA, dtype="Int64", index=target_ym.index)
    valid = pd.notna(numeric)
    if not valid.any():
        return years
    numeric_valid = numeric.loc[valid].astype("int64")
    years.loc[valid] = (numeric_valid // 100).astype("Int64")
    return years


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


