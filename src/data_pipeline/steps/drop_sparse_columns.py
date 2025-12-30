from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from ..utils.paths import INTERIM_DIR, PROJECT_ROOT, ensure_parent, interim_subdir

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


def _load_assigned_dataset(dataset_name: str) -> pd.DataFrame:
    source_path = INTERIM_DIR / "00_assign_data_id" / f"{dataset_name}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found. Run the assign_data_id step first."
        )
    return pd.read_parquet(source_path)


def drop_sparse_columns(force: bool = True) -> Dict[str, List[dict]]:
    """Remove columns with more than 99% null rate from train/test tables."""
    output_dir = interim_subdir("01_drop_sparse_columns")
    stats: List[dict] = []

    for dataset_name in ("train", "test"):
        df = _load_assigned_dataset(dataset_name)
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


__all__ = ["drop_sparse_columns", "SPARSE_COLUMNS"]

