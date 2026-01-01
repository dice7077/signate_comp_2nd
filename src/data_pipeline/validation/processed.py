from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from ..utils.paths import DATA_DIR
from .counts import ValidationError, load_expected_counts

PROCESSED_ROOT = DATA_DIR / "processed"
TYPE_DIRECTORIES: Dict[str, str] = {
    "kodate": "0001_kodate",
    "mansion": "0002_mansion",
}


def validate_processed(
    version: str,
    *,
    types: Sequence[str] | None = None,
    raise_on_error: bool = False,
) -> List[dict]:
    """
    data/processed 配下の学習データで data_id ユニーク件数と期待行数を検証する。
    """

    expected_counts = load_expected_counts()
    target_types = _normalize_types(types)

    results: List[dict] = []
    errors: List[dict] = []

    for type_label in target_types:
        base_dir = PROCESSED_ROOT / TYPE_DIRECTORIES[type_label] / version
        for split in ("train", "test"):
            expected = expected_counts[split]["type_counts"][type_label]
            path = base_dir / f"{split}.parquet"
            result = _validate_file(path, split, type_label, expected)
            results.append(result)
            if result["status"] != "ok":
                errors.append(result)

    if raise_on_error and errors:
        raise ValidationError(_summarize_errors(errors))

    return results


def _normalize_types(types: Sequence[str] | None) -> List[str]:
    if not types:
        return list(TYPE_DIRECTORIES.keys())
    unknown = [t for t in types if t not in TYPE_DIRECTORIES]
    if unknown:
        raise ValueError(f"未知のタイプが指定されました: {', '.join(unknown)}")
    return list(dict.fromkeys(types))


def _validate_file(
    path: Path, split: str, type_label: str, expected_rows: int
) -> dict:
    result = {
        "type_label": type_label,
        "split": split,
        "path": str(path),
        "rows": None,
        "unique_data_ids": None,
        "expected_rows": expected_rows,
        "status": "missing",
        "message": "",
    }

    if not path.exists():
        result["message"] = "出力ファイルが存在しません。"
        return result

    df = pd.read_parquet(path, columns=["data_id"])
    rows = int(len(df))
    unique = int(df["data_id"].nunique(dropna=False))

    result["rows"] = rows
    result["unique_data_ids"] = unique

    issues: List[str] = []
    if rows != expected_rows:
        issues.append(f"期待行数 {expected_rows:,} 件に対し {rows:,} 件")
    if rows != unique:
        issues.append(f"data_id ユニーク件数 {unique:,} 件 ≠ 行数 {rows:,} 件")

    if issues:
        result["status"] = "error"
        result["message"] = "; ".join(issues)
    else:
        result["status"] = "ok"
    return result


def _summarize_errors(errors: Sequence[dict]) -> str:
    head = [
        f"{entry['type_label']}::{entry['split']} - {entry.get('message') or 'Unknown error'}"
        for entry in errors
    ]
    preview = "\n".join(head[:5])
    extra = len(head) - 5
    if extra > 0:
        preview += f"\n…他 {extra} 件"
    return preview


__all__ = ["validate_processed"]


