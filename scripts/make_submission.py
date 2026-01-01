#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="kodate/mansionの推論結果を結合し、提出CSVを生成する。",
    )
    parser.add_argument(
        "--kodate-pred",
        required=True,
        help="kodate用 test_predictions ファイルへのパス（parquet/csv）。",
    )
    parser.add_argument(
        "--mansion-pred",
        required=True,
        help="mansion用 test_predictions ファイルへのパス（parquet/csv）。",
    )
    parser.add_argument(
        "--sample",
        help="提出テンプレートCSV。未指定時は data/raw/signate/ の既定ファイルを探索。",
    )
    parser.add_argument(
        "--output",
        help="出力先CSV。未指定時は submissions/submission_<timestamp>.csv。",
    )
    parser.add_argument(
        "--no-round",
        action="store_true",
        help="予測値を四捨五入せずそのまま出力する（デフォルトはround）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_path = Path(args.sample) if args.sample else _default_sample_path()
    kodate_df = _load_predictions(Path(args.kodate_pred))
    mansion_df = _load_predictions(Path(args.mansion_pred))

    sample_df = _load_sample(sample_path)
    submit_df, coverage = _merge_predictions(sample_df, kodate_df, mansion_df)

    if not args.no_round:
        submit_df["money_room"] = np.round(submit_df["money_room"]).astype(int)

    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submit_df.to_csv(output_path, index=False, header=False)

    missing = coverage[1]
    print("=== Submission Created ===")
    print(f"Sample path         : {sample_path}")
    print(f"Kodate predictions  : {Path(args.kodate_pred)}")
    print(f"Mansion predictions : {Path(args.mansion_pred)}")
    print(f"Output path         : {output_path}")
    print(f"Total rows          : {len(submit_df):,}")
    print(f"Missing predictions : {missing}")
    if missing > 0:
        print("[WARN] 欠損した data_id が存在します。推論ファイルを確認してください。")


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    expected_id_cols = [col for col in df.columns if col.lower() in {"data_id", "id"}]
    if not expected_id_cols:
        raise ValueError(f"{path} に data_id/id 列が見つかりません。")
    id_col = expected_id_cols[0]
    if "prediction" not in df.columns and "money_room" in df.columns:
        df = df.rename(columns={"money_room": "prediction"})
    if "prediction" not in df.columns:
        raise ValueError(f"{path} に prediction 列が見つかりません。")
    result = df[[id_col, "prediction"]].copy()
    result.rename(columns={id_col: "data_id"}, inplace=True)
    result["data_id"] = result["data_id"].astype(int)
    result["prediction"] = result["prediction"].astype(float)
    return result


def _load_sample(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = pd.read_csv(path, header=None)

    if df.shape[1] < 2:
        df = pd.read_csv(path, header=None, names=["id", "money_room"])
    elif "id" not in df.columns or "money_room" not in df.columns:
        df = df.iloc[:, :2]
        df.columns = ["id", "money_room"]

    df = df[["id", "money_room"]].copy()
    df["id"] = df["id"].astype(str).str.zfill(6)
    return df


def _merge_predictions(
    sample_df: pd.DataFrame, kodate_df: pd.DataFrame, mansion_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Tuple[int, int]]:
    for df in (kodate_df, mansion_df):
        df["id"] = df["data_id"].astype(int).astype(str).str.zfill(6)
    merged_preds = pd.concat(
        [kodate_df[["id", "prediction"]], mansion_df[["id", "prediction"]]],
        axis=0,
        ignore_index=True,
    )
    if merged_preds["id"].duplicated().any():
        duplicates = merged_preds.loc[merged_preds["id"].duplicated(), "id"].unique()
        raise ValueError(f"data_id が重複しています: {duplicates[:5]}")

    merged = sample_df[["id"]].merge(merged_preds, on="id", how="left")
    missing = int(merged["prediction"].isna().sum())
    merged["money_room"] = merged["prediction"].astype(float)
    merged = merged[["id", "money_room"]].copy()
    return merged, (len(sample_df), missing)


def _default_sample_path() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "signate" / "sample_submission.csv",
        PROJECT_ROOT / "data" / "raw" / "signate" / "sample_submit.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("sample submission が見つかりませんでした。--sample を指定してください。")


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "submissions" / f"submission_{timestamp}.csv"


if __name__ == "__main__":
    main()


