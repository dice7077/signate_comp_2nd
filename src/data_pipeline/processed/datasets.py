from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, cast

import numpy as np

import pandas as pd

from ..utils.overlap import (
    ALL_OVERLAP_FEATURE_COLUMNS,
    OVERLAP_FEATURE_COLUMNS,
    UNIT_ONLY_OVERLAP_FEATURE_COLUMNS,
    compute_overlap_features,
    get_category_specs,
    value_counts,
)
from ..utils.paths import DATA_DIR, INTERIM_DIR, PROJECT_ROOT


PROCESSED_ROOT = DATA_DIR / "processed"
SPLIT_BASE_DIR = INTERIM_DIR / "01_03_adjust_mansion_unit_area"

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

SAME_UNIT_HISTORY_FEATURES: Tuple[str, ...] = (
    "money_room_past_6m",
    "money_room_past_12m",
    "money_room_past_18m",
    "money_room_past_24m",
    "money_room_past_over_30m",
)
OVERLAP_SYNTHETIC_FEATURES: Tuple[str, ...] = tuple(
    dict.fromkeys(OVERLAP_FEATURE_COLUMNS + UNIT_ONLY_OVERLAP_FEATURE_COLUMNS)
)
SYNTHETIC_FEATURES: Tuple[str, ...] = SAME_UNIT_HISTORY_FEATURES + OVERLAP_SYNTHETIC_FEATURES
SYNTHETIC_FEATURE_SET = frozenset(SYNTHETIC_FEATURES)

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
        "unit_house_area_adjusted",
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
    (
        "mansion",
        "0007_cv_building_id",
    ): FEATURE_PLAN["mansion"]
    + list(MANSION_TAG_FEATURES)
    + ["building_id", "unit_id"],
    (
        "mansion",
        "0006_same_unit_id",
    ): FEATURE_PLAN["mansion"]
    + list(MANSION_TAG_FEATURES)
    + ["unit_id"]
    + list(SAME_UNIT_HISTORY_FEATURES),
    (
        "mansion",
        "0008_unit_building_overlap",
    ): FEATURE_PLAN["mansion"]
    + list(MANSION_TAG_FEATURES)
    + ["building_id", "unit_id", *OVERLAP_FEATURE_COLUMNS],
    ("kodate", "0006_add_tags"): FEATURE_PLAN["kodate"] + list(KODATE_TAG_FEATURES),
    (
        "kodate",
        "0007_same_unit_id",
    ): FEATURE_PLAN["kodate"]
    + list(KODATE_TAG_FEATURES)
    + ["unit_id"]
    + list(SAME_UNIT_HISTORY_FEATURES),
    (
        "kodate",
        "0008_test_202207only",
    ): FEATURE_PLAN["kodate"]
    + list(KODATE_TAG_FEATURES)
    + ["unit_id"]
    + list(SAME_UNIT_HISTORY_FEATURES),
    (
        "kodate",
        "0009_same_unit_features_all",
    ): FEATURE_PLAN["kodate"]
    + list(KODATE_TAG_FEATURES)
    + ["unit_id"]
    + list(SAME_UNIT_HISTORY_FEATURES),
    (
        "kodate",
        "0010_unit_overlap",
    ): FEATURE_PLAN["kodate"]
    + list(KODATE_TAG_FEATURES)
    + ["unit_id", "unit_overlap_count", "unit_overlap_bucket", "overlap_category"],
}

SAME_UNIT_ID_DATASET_CONFIGS: Dict[Tuple[str, str], Dict[str, object]] = {
    ("mansion", "0006_same_unit_id"): {},
    ("kodate", "0007_same_unit_id"): {},
    ("kodate", "0008_test_202207only"): {"test_known_target_ym": 202207},
    (
        "kodate",
        "0009_same_unit_features_all",
    ): {
        "drop_rows_without_history": False,
        "test_only_known_units": False,
    },
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
        customizer = None
        customizer_config = SAME_UNIT_ID_DATASET_CONFIGS.get((type_label, version))
        if customizer_config is not None:
            customizer = _SameUnitIdCustomizer(**customizer_config)

        needs_overlap = _requires_overlap_features(features)
        use_building_overlap = _uses_building_overlap_features(features)
        overlap_unit_counts = None
        overlap_building_counts = None

        for split in ("train", "test"):
            output_path = version_dir / f"{split}.parquet"
            if output_path.exists() and not overwrite:
                raise ProcessedDatasetError(
                    f"{output_path} が既に存在します。--overwrite を指定してください。"
                )

            df = _load_base_dataframe(split, type_label, features)
            df = _merge_supplementary_features(df, split, features)
            df = _apply_target_year_alignment(df)
            if customizer is not None:
                df = customizer.transform(df, split)

            if needs_overlap:
                if "unit_id" not in df.columns:
                    raise ProcessedDatasetError(
                        f"{type_label}/{version} は overlap 特徴量に unit_id を必要とします。"
                    )
                if use_building_overlap and "building_id" not in df.columns:
                    raise ProcessedDatasetError(
                        f"{type_label}/{version} は building_id 列が必要ですが見つかりません。"
                    )
                if split == "train":
                    overlap_unit_counts = value_counts(df["unit_id"])
                    overlap_building_counts = (
                        value_counts(df["building_id"]) if use_building_overlap else None
                    )
                    subtract_self = True
                else:
                    if overlap_unit_counts is None:
                        raise ProcessedDatasetError(
                            "train split より前に test split へ overlap 特徴量を適用できません。"
                        )
                    subtract_self = False

                df = _attach_overlap_features(
                    df=df,
                    id_column="data_id",
                    unit_column="unit_id",
                    building_column="building_id" if use_building_overlap else None,
                    unit_counts=overlap_unit_counts,
                    building_counts=overlap_building_counts,
                    subtract_self=subtract_self,
                    use_building=use_building_overlap,
                )

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
    # 順序維持のため features の登場順で重複排除
    ordered: List[str] = []
    for col in features:
        if col in SYNTHETIC_FEATURE_SET:
            continue
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
        if not source or source == "overlap":
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


def _requires_overlap_features(features: Sequence[str]) -> bool:
    return any(feature in ALL_OVERLAP_FEATURE_COLUMNS for feature in features)


def _uses_building_overlap_features(features: Sequence[str]) -> bool:
    return any(
        feature in {"building_overlap_count", "building_overlap_bucket"} for feature in features
    )


def _attach_overlap_features(
    *,
    df: pd.DataFrame,
    id_column: str,
    unit_column: str,
    building_column: str | None,
    unit_counts: pd.Series,
    building_counts: pd.Series | None,
    subtract_self: bool,
    use_building: bool,
) -> pd.DataFrame:
    category_specs = get_category_specs(use_building)
    features = compute_overlap_features(
        df=df,
        id_column=id_column,
        unit_column=unit_column,
        building_column=building_column,
        unit_reference_counts=unit_counts,
        building_reference_counts=building_counts,
        subtract_self=subtract_self,
        include_label=False,
        use_building=use_building,
        category_specs=category_specs,
    )
    return df.merge(features, on=id_column, how="left", validate="one_to_one")


class _SameUnitIdCustomizer:
    """
    target_ym を利用した unit_id ごとの過去 money_room 特徴量を構築し、
    設定に応じて行の除外や保持を切り替えるカスタマイザ。
    """

    BUCKET_SPECS: Tuple[Tuple[str, int, int | None], ...] = (
        ("money_room_past_6m", 1, 6),
        ("money_room_past_12m", 7, 12),
        ("money_room_past_18m", 13, 18),
        ("money_room_past_24m", 19, 24),
        ("money_room_past_over_30m", 30, None),
    )
    FEATURE_NAMES: Tuple[str, ...] = SAME_UNIT_HISTORY_FEATURES

    def __init__(
        self,
        *,
        test_known_target_ym: int | None = None,
        drop_rows_without_history: bool = True,
        test_only_known_units: bool = True,
    ) -> None:
        self._unit_histories: Dict[int, Tuple[np.ndarray, np.ndarray]] | None = None
        self._known_units: set[int] = set()
        self._test_known_units: set[int] | None = None
        self._test_known_target_ym = test_known_target_ym
        self._drop_rows_without_history = drop_rows_without_history
        self._test_only_known_units = test_only_known_units

    def transform(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        if "unit_id" not in df.columns or "target_ym" not in df.columns:
            raise ProcessedDatasetError(
                "unit_id/target_ym 列が不足しているため 0006_same_unit_id を構築できません。"
            )
        if split == "train":
            return self._transform_train(df)
        if split == "test":
            if self._unit_histories is None:
                raise ProcessedDatasetError(
                    "train split を先に処理してから test split を処理してください。"
                )
            return self._transform_test(df)
        raise ProcessedDatasetError(f"未対応の split が指定されました: {split}")

    def _transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        if "money_room" not in df.columns:
            raise ProcessedDatasetError("train split には money_room 列が必要です。")
        self._fit_history(df)
        if not self._drop_rows_without_history:
            df_with_features = df.copy()
            return self._append_history_features(df_with_features)
        mask = self._rows_with_history(df)
        df_filtered = df.loc[mask].copy()
        if df_filtered.empty:
            raise ProcessedDatasetError(
                "unit_id が過去に登場した train 行が存在しません。"
            )
        return self._append_history_features(df_filtered)

    def _transform_test(self, df: pd.DataFrame) -> pd.DataFrame:
        allowed_units = self._test_known_units if self._test_known_units is not None else self._known_units
        if self._test_only_known_units:
            df_filtered = df.loc[df["unit_id"].isin(allowed_units)].copy()
        else:
            df_filtered = df.copy()
        if df_filtered.empty:
            return self._ensure_feature_columns(df_filtered)
        return self._append_history_features(df_filtered)

    def _fit_history(self, df: pd.DataFrame) -> None:
        unit_series = pd.to_numeric(df["unit_id"], errors="coerce")
        target_series = pd.to_numeric(df["target_ym"], errors="coerce")
        money_series = pd.to_numeric(df["money_room"], errors="coerce")
        base = pd.DataFrame(
            {
                "unit_id": unit_series,
                "target_ym": target_series,
                "money_room": money_series,
            }
        ).dropna()
        if base.empty:
            raise ProcessedDatasetError("履歴構築に利用できる train 行がありません。")
        base["unit_id"] = base["unit_id"].astype("int64")
        base["target_ym"] = base["target_ym"].astype("int64")
        base["money_room"] = base["money_room"].astype(float)
        histories: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for unit_id, group in base.groupby("unit_id", sort=False):
            months = self._to_month_index_from_numeric(
                group["target_ym"].to_numpy(dtype=float, copy=True)
            )
            valid = np.isfinite(months)
            if not np.any(valid):
                continue
            months_valid = months[valid].astype(np.int64, copy=True)
            values_valid = group["money_room"].to_numpy(dtype=float, copy=True)[valid]
            order = np.argsort(months_valid, kind="mergesort")
            histories[int(unit_id)] = (
                months_valid[order],
                values_valid[order],
            )
        if not histories:
            raise ProcessedDatasetError("unit_id の履歴を構築できませんでした。")
        self._unit_histories = histories
        self._known_units = set(histories.keys())
        self._test_known_units = self._determine_test_known_units(base)

    def _determine_test_known_units(self, base: pd.DataFrame) -> set[int]:
        if self._test_known_target_ym is None:
            return set(self._known_units)
        target_mask = base["target_ym"] == self._test_known_target_ym
        if not target_mask.any():
            return set()
        units = base.loc[target_mask, "unit_id"].astype("int64", copy=True)
        return set(units.tolist())

    def _rows_with_history(self, df: pd.DataFrame) -> np.ndarray:
        unit_values = pd.to_numeric(df["unit_id"], errors="coerce").to_numpy(dtype=float)
        target_values = pd.to_numeric(df["target_ym"], errors="coerce").to_numpy(
            dtype=float
        )
        mask = np.zeros(len(df), dtype=bool)
        valid = np.isfinite(unit_values) & np.isfinite(target_values)
        if not np.any(valid):
            return mask
        min_target: Dict[int, int] = {}
        valid_units = unit_values[valid].astype(np.int64, copy=False)
        valid_targets = target_values[valid].astype(np.int64, copy=False)
        for uid, target in zip(valid_units, valid_targets):
            prev = min_target.get(uid)
            if prev is None or target < prev:
                min_target[uid] = target
        valid_indices = np.flatnonzero(valid)
        for idx in valid_indices:
            uid = int(unit_values[idx])
            target = int(target_values[idx])
            if target > min_target.get(uid, target):
                mask[idx] = True
        return mask

    def _append_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return self._ensure_feature_columns(df)
        arrays = self._compute_history_feature_arrays(df)
        for name, values in arrays.items():
            df[name] = pd.Series(values, index=df.index, dtype="Float64")
        return df

    def _ensure_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for name in self.FEATURE_NAMES:
            if name not in df.columns:
                df[name] = pd.Series(dtype="Float64")
        return df

    def _compute_history_feature_arrays(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self._unit_histories is None:
            raise ProcessedDatasetError("unit_id 履歴が未構築です。")
        unit_values = pd.to_numeric(df["unit_id"], errors="coerce").to_numpy(dtype=float)
        month_indices = self._to_month_index_from_numeric(
            pd.to_numeric(df["target_ym"], errors="coerce").to_numpy(dtype=float)
        )
        feature_matrix = {
            name: np.full(len(df), np.nan, dtype=float) for name in self.FEATURE_NAMES
        }
        for row_idx in range(len(df)):
            unit_val = unit_values[row_idx]
            month_val = month_indices[row_idx]
            if not np.isfinite(unit_val) or not np.isfinite(month_val):
                continue
            history = self._unit_histories.get(int(unit_val))
            if history is None:
                continue
            months_hist, values_hist = history
            diffs = month_val - months_hist
            available = diffs > 0
            if not np.any(available):
                continue
            diffs = diffs[available]
            hist_values = values_hist[available]
            for name, min_offset, max_offset in self.BUCKET_SPECS:
                bucket = diffs >= min_offset
                if max_offset is not None:
                    bucket &= diffs <= max_offset
                if not np.any(bucket):
                    continue
                avg = hist_values[bucket].mean()
                if avg > 0:
                    feature_matrix[name][row_idx] = float(np.log(avg))
        return feature_matrix

    @staticmethod
    def _to_month_index_from_numeric(values: np.ndarray) -> np.ndarray:
        result = np.full(len(values), np.nan, dtype=float)
        if values.size == 0:
            return result
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return result
        indices = np.flatnonzero(finite_mask)
        ints = values[finite_mask].astype(np.int64, copy=False)
        months = ints % 100
        years = ints // 100
        valid_month = (months >= 1) & (months <= 12)
        if not np.any(valid_month):
            return result
        valid_indices = indices[valid_month]
        result[valid_indices] = years[valid_month] * 12 + (months[valid_month] - 1)
        return result


__all__ = ["build_processed_datasets", "ProcessedDatasetError"]


