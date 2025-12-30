from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..utils.paths import PROJECT_ROOT, ensure_parent, interim_subdir, raw_signate_path


def assign_data_id(force: bool = True) -> Dict[str, List[dict]]:
    """Attach a unique data_id column to the raw Signate train/test tables."""
    output_dir = interim_subdir("00_assign_data_id")
    stats: List[dict] = []

    dataset_files = {
        "train": "train.csv",
        "test": "test.csv",
    }

    for dataset_name, filename in dataset_files.items():
        csv_path = raw_signate_path(filename)
        df = _read_csv(csv_path)
        df = _attach_data_id(df, dataset_name)
        output_path = output_dir / f"{dataset_name}.parquet"
        if output_path.exists() and not force:
            raise FileExistsError(
                f"{output_path} already exists. Pass force=True to overwrite."
            )
        ensure_parent(output_path)
        df.to_parquet(output_path)
        stats.append(
            {
                "dataset": dataset_name,
                "rows": int(len(df)),
                "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    manifest = {
        "step": "assign_data_id",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": stats,
    }
    manifest_path = output_dir / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    return manifest


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    df = pd.read_csv(path, low_memory=False)
    return _normalize_object_columns(df)


def _normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast mixed-type object columns to pandas' dedicated string dtype."""
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) == 0:
        return df
    df = df.copy()
    df[object_cols] = df[object_cols].astype("string[python]")
    return df


def _attach_data_id(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    if dataset_name == "train":
        df.insert(0, "data_id", pd.RangeIndex(start=0, stop=len(df), step=1))
        df["data_id"] = df["data_id"].astype("int64")
    elif dataset_name == "test":
        if "id" not in df.columns:
            raise KeyError("'id' column not found in test dataset.")
        df.insert(0, "data_id", df["id"].astype("string[python]"))
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return df


