from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import pandas as pd

BUCKET_ORDER: Tuple[str, ...] = ("0", "1", "2", "3", "4+")

UNIT_BUILDING_CATEGORY_SPECS: Tuple[Tuple[str, str], ...] = (
    ("unit_ge3", "unit_id 被り 3件以上"),
    ("unit_2", "unit_id 被り 2件"),
    ("unit_1", "unit_id 被り 1件"),
    ("building_ge3", "unit_id 被り0 & building_id 被り 3件以上"),
    ("building_2", "unit_id 被り0 & building_id 被り 2件"),
    ("building_1", "unit_id 被り0 & building_id 被り 1件"),
    ("isolated", "unit_id / building_id とも被り0"),
)

UNIT_ONLY_CATEGORY_SPECS: Tuple[Tuple[str, str], ...] = (
    ("unit_ge3", "unit_id 被り 3件以上"),
    ("unit_2", "unit_id 被り 2件"),
    ("unit_1", "unit_id 被り 1件"),
    ("unit_0", "unit_id 被り 0件"),
)

OVERLAP_FEATURE_COLUMNS: Tuple[str, ...] = (
    "unit_overlap_count",
    "unit_overlap_bucket",
    "building_overlap_count",
    "building_overlap_bucket",
    "overlap_category",
)

UNIT_ONLY_OVERLAP_FEATURE_COLUMNS: Tuple[str, ...] = (
    "unit_overlap_count",
    "unit_overlap_bucket",
    "overlap_category",
)

ALL_OVERLAP_FEATURE_COLUMNS = set(OVERLAP_FEATURE_COLUMNS) | set(
    UNIT_ONLY_OVERLAP_FEATURE_COLUMNS
)


def get_category_specs(use_building: bool) -> Tuple[Tuple[str, str], ...]:
    return UNIT_BUILDING_CATEGORY_SPECS if use_building else UNIT_ONLY_CATEGORY_SPECS


def value_counts(series: pd.Series) -> pd.Series:
    """NaN を除外して value_counts を計算し、int 型へ揃える。"""
    return series.dropna().value_counts().astype(int)


def bucketize_count(count: int) -> str:
    if count >= 4:
        return "4+"
    if count <= 0:
        return "0"
    return str(int(count))


def compute_overlap_features(
    *,
    df: pd.DataFrame,
    id_column: str,
    unit_column: str,
    building_column: str | None,
    unit_reference_counts: pd.Series,
    building_reference_counts: pd.Series | None,
    subtract_self: bool,
    include_label: bool = False,
    use_building: bool = True,
    category_specs: Sequence[Tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """unit/building の被り数・バケット・カテゴリを DataFrame として返す。"""
    unit_counts = _lookup_counts(
        df[unit_column], unit_reference_counts, subtract_self=subtract_self
    )
    if use_building and building_column and building_reference_counts is not None:
        building_counts = _lookup_counts(
            df[building_column], building_reference_counts, subtract_self=subtract_self
        )
    else:
        building_counts = pd.Series(0, index=df.index, dtype=int)
        use_building = False

    result = pd.DataFrame(
        {
            id_column: df[id_column],
            "unit_overlap_count": unit_counts,
            "unit_overlap_bucket": unit_counts.map(bucketize_count).astype("string"),
            "building_overlap_count": building_counts,
            "building_overlap_bucket": building_counts.map(bucketize_count).astype("string"),
        }
    )

    specs = tuple(category_specs) if category_specs is not None else get_category_specs(use_building)
    label_map: Dict[str, str] = dict(specs)
    result["overlap_category"] = [
        assign_overlap_category(u, b, use_building) for u, b in zip(unit_counts, building_counts)
    ]
    if include_label:
        result["overlap_category_label"] = result["overlap_category"].map(label_map)
    return result


def assign_overlap_category(unit_count: int, building_count: int, use_building: bool) -> str:
    if unit_count >= 3:
        return "unit_ge3"
    if unit_count == 2:
        return "unit_2"
    if unit_count == 1:
        return "unit_1"
    if use_building and building_count >= 3:
        return "building_ge3"
    if use_building and building_count == 2:
        return "building_2"
    if use_building and building_count == 1:
        return "building_1"
    return "isolated" if use_building else "unit_0"


def _lookup_counts(
    values: pd.Series, reference_counts: pd.Series, *, subtract_self: bool
) -> pd.Series:
    counts = values.map(reference_counts).fillna(0).astype(int)
    if subtract_self:
        mask = values.notna() & (counts > 0)
        counts.loc[mask] = counts.loc[mask] - 1
    counts[counts < 0] = 0
    return counts

