from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from ..utils.paths import INTERIM_DIR, PROJECT_ROOT, ensure_parent, interim_subdir
from .layout import step_output_dir

# Columns whose null rate exceeds 99% in both train/test after assign_data_id.
SPARSE_COLUMNS = [
    "building_name_ruby",
    "free_rent_duration",
    "free_rent_gen_timing",
    "money_hoshou_company",
    "name_ruby",
    "reform_common_area",
    "reform_common_area_date",
    "reform_date",
    "reform_etc",
    "reform_place",
    "reform_place_other",
    "school_ele_code",
    "school_jun_code",
    "traffic_car",
]


ASSIGN_OUTPUT_DIR = step_output_dir("assign_data_id")
OUTPUT_DIR_NAME = step_output_dir("drop_sparse_columns")


def _load_assigned_dataset(dataset_name: str) -> pd.DataFrame:
    source_path = INTERIM_DIR / ASSIGN_OUTPUT_DIR / f"{dataset_name}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found. Run the assign_data_id step first."
        )
    return pd.read_parquet(source_path)


def drop_sparse_columns(force: bool = True) -> Dict[str, List[dict]]:
    """
    Remove columns with more than 99% null rate from train/test tables and
    attach a few deterministic helper features used across experiments.
    """
    output_dir = interim_subdir(OUTPUT_DIR_NAME)
    stats: List[dict] = []

    for dataset_name in ("train", "test"):
        df = _load_assigned_dataset(dataset_name)
        df = _attach_basic_features(df)
        columns_to_drop = [col for col in SPARSE_COLUMNS if col in df.columns]
        if not columns_to_drop:
            # Nothing to do for this split; persist original schema.
            cleaned_df = df
        else:
            cleaned_df = df.drop(columns=columns_to_drop)
        output_path = output_dir / f"{dataset_name}.parquet"
        if output_path.exists() and not force:
            raise FileExistsError(
                f"{output_path} already exists. Pass force=True to overwrite."
            )
        ensure_parent(output_path)
        cleaned_df.to_parquet(output_path)
        stats.append(
            {
                "dataset": dataset_name,
                "rows": int(len(cleaned_df)),
                "columns_removed": columns_to_drop,
                "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    dropped_columns = sorted(
        {col for entry in stats for col in entry["columns_removed"]}
    )
    manifest = {
        "step": "drop_sparse_columns",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": stats,
        "dropped_columns": dropped_columns,
    }
    manifest_path = output_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    return manifest


def _attach_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["years_old"] = pd.Series(dtype="float64", index=df.index)
        df["post_all"] = pd.Series(dtype="string[python]", index=df.index)
        df["addr_all"] = pd.Series(dtype="string[python]", index=df.index)
        return df

    enriched = df.copy()
    enriched["years_old"] = _compute_years_old(enriched)

    post_front = _format_code_series(enriched.get("post1"), width=3, index=enriched.index)
    post_back = _format_code_series(enriched.get("post2"), width=4, index=enriched.index)
    enriched["post_all"] = _compose_codes(post_front, post_back)

    addr_front = _format_code_series(enriched.get("addr1_1"), width=2, index=enriched.index)
    addr_back = _format_code_series(enriched.get("addr1_2"), width=3, index=enriched.index)
    enriched["addr_all"] = _compose_codes(addr_front, addr_back)
    return enriched


def _compute_years_old(df: pd.DataFrame) -> pd.Series:
    target_series = df.get("target_ym")
    built_series = df.get("year_built")
    if target_series is None or built_series is None:
        return pd.Series(float("nan"), index=df.index, dtype="float64")

    target_dates = _coerce_year_month_series(target_series)
    built_dates = _coerce_year_month_series(built_series)
    if target_dates is None or built_dates is None:
        return pd.Series(float("nan"), index=df.index, dtype="float64")

    delta = target_dates - built_dates
    years = delta.dt.days / 365.25
    return years.astype("float64")


def _coerce_year_month_series(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    values = (
        series.astype("string[python]")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )
    values = values.where(values != "", pd.NA)
    digits = values.str.replace(r"[^0-9]", "", regex=True)
    digits = digits.where(digits.str.len() > 0, pd.NA)

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    lengths = digits.str.len()
    mask6 = (lengths == 6).fillna(False)
    if mask6.any():
        result.loc[mask6] = pd.to_datetime(
            digits.loc[mask6],
            format="%Y%m",
            errors="coerce",
        )
    mask4 = (lengths == 4).fillna(False)
    if mask4.any():
        result.loc[mask4] = pd.to_datetime(
            digits.loc[mask4] + "01",
            format="%Y%m",
            errors="coerce",
        )
    return result


def _format_code_series(
    series: pd.Series | None, *, width: int, index: pd.Index
) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="string[python]")

    values = (
        series.astype("string[python]")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )
    values = values.where(values != "", pd.NA)
    digits = values.str.replace(r"[^0-9]", "", regex=True)
    digits = digits.where(digits.str.len() > 0, pd.NA)

    formatted = digits.astype("string[python]")
    mask_short = formatted.notna() & (formatted.str.len() < width)
    if mask_short.any():
        formatted.loc[mask_short] = formatted.loc[mask_short].str.zfill(width)
    return formatted


def _compose_codes(front: pd.Series, back: pd.Series) -> pd.Series:
    if len(front) != len(back):
        raise ValueError("Series length mismatch when composing codes.")
    combined = pd.Series(pd.NA, index=front.index, dtype="string[python]")
    mask = front.notna() & back.notna()
    if mask.any():
        combined.loc[mask] = front.loc[mask] + "-" + back.loc[mask]
    return combined


__all__ = ["drop_sparse_columns", "SPARSE_COLUMNS"]

