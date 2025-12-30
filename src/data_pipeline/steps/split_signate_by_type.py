from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from ..utils.paths import INTERIM_DIR, PROJECT_ROOT, ensure_parent, interim_subdir


TYPE_NAME_MAP = {
    1202: "kodate",
    1302: "mansion",
}


def _load_clean_dataset(dataset_name: str) -> pd.DataFrame:
    source_path = INTERIM_DIR / "01_join_population_projection" / f"{dataset_name}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found. Run the drop_sparse_columns step first."
        )
    df = pd.read_parquet(source_path)
    if "data_id" not in df.columns:
        raise KeyError(f"'data_id' column missing from {source_path}")
    return df


def split_signate_by_type(force: bool = True) -> Dict[str, List[dict]]:
    """Split the cleaned train/test tables (after dropping sparse columns) by bukken_type."""
    output_dir = interim_subdir("01_split_by_type")
    stats: List[dict] = []

    for dataset_name in ("train", "test"):
        df = _load_clean_dataset(dataset_name)
        if "bukken_type" not in df.columns:
            raise KeyError(f"'bukken_type' column not found in {dataset_name} dataset")
        bukken_series = pd.to_numeric(df["bukken_type"], errors="coerce")

        for type_code, type_label in TYPE_NAME_MAP.items():
            mask = bukken_series == type_code
            subset = df.loc[mask].copy()
            subset.reset_index(drop=True, inplace=True)
            output_path = output_dir / f"{dataset_name}_{type_label}.parquet"
            if subset.empty:
                # Still overwrite with an empty table to make downstream steps predictable.
                subset = subset.head(0)
            if output_path.exists() and not force:
                raise FileExistsError(
                    f"{output_path} already exists. Pass force=True to overwrite."
                )
            ensure_parent(output_path)
            subset.to_parquet(output_path)
            stats.append(
                {
                    "dataset": dataset_name,
                    "bukken_type": int(type_code),
                    "type_label": type_label,
                    "rows": int(len(subset)),
                    "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                }
            )

    manifest = {
        "step": "split_signate_by_type",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": stats,
    }
    manifest_path = output_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    return manifest

