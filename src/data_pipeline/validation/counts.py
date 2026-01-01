from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from ..steps.layout import STEP_LAYOUT, step_output_dir
from ..utils.paths import DATA_DIR, INTERIM_DIR, PROJECT_ROOT
from ..utils.signate_types import TYPE_NAME_MAP


EXPECTED_COUNTS_PATH = DATA_DIR / "metadata" / "signate_expected_counts.json"


class ValidationError(RuntimeError):
    """データ件数検証でNGになった際に送出する例外。"""


@dataclass(frozen=True)
class OutputSpec:
    filename: str
    dataset: str | None = None
    type_label: str | None = None


TYPE_LABELS = tuple(sorted(TYPE_NAME_MAP.values()))

STEP_OUTPUT_SPECS: "OrderedDict[str, Sequence[OutputSpec]]" = OrderedDict(
    [
        (
            "assign_data_id",
            (
                OutputSpec(filename="train.parquet", dataset="train"),
                OutputSpec(filename="test.parquet", dataset="test"),
            ),
        ),
        (
            "join_koji_price",
            (
                OutputSpec(filename="train.parquet", dataset="train"),
                OutputSpec(filename="test.parquet", dataset="test"),
            ),
        ),
        (
            "join_land_price",
            (
                OutputSpec(filename="train.parquet", dataset="train"),
                OutputSpec(filename="test.parquet", dataset="test"),
            ),
        ),
        (
            "build_tag_id_features",
            (
                OutputSpec(filename="train_tag_ids.parquet", dataset="train"),
                OutputSpec(filename="test_tag_ids.parquet", dataset="test"),
            ),
        ),
        (
            "drop_sparse_columns",
            (
                OutputSpec(filename="train.parquet", dataset="train"),
                OutputSpec(filename="test.parquet", dataset="test"),
            ),
        ),
        (
            "join_population_projection",
            (
                OutputSpec(
                    filename="train_population_features.parquet",
                    dataset="train",
                ),
                OutputSpec(
                    filename="test_population_features.parquet",
                    dataset="test",
                ),
            ),
        ),
        (
            "split_signate_by_type",
            tuple(
                OutputSpec(
                    filename=f"{dataset}_{label}.parquet",
                    dataset=dataset,
                    type_label=label,
                )
                for dataset in ("train", "test")
                for label in TYPE_LABELS
            ),
        ),
        (
            "adjust_mansion_unit_area",
            tuple(
                OutputSpec(
                    filename=f"{dataset}_{label}.parquet",
                    dataset=dataset,
                    type_label=label,
                )
                for dataset in ("train", "test")
                for label in TYPE_LABELS
            ),
        ),
    ]
)


def _ordered_step_names(step_names: Sequence[str] | None) -> List[str]:
    canonical_order = list(STEP_LAYOUT.keys())
    if not step_names:
        return [name for name in canonical_order if name in STEP_OUTPUT_SPECS]
    unknown = [name for name in step_names if name not in STEP_OUTPUT_SPECS]
    if unknown:
        raise ValueError(f"未対応のステップが指定されました: {', '.join(unknown)}")
    return [name for name in canonical_order if name in step_names]


def load_expected_counts(path: Path = EXPECTED_COUNTS_PATH) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"期待件数ファイルが見つかりません: {path}. "
            "raw/signate から集計し JSON を用意してください。"
        )
    with path.open(encoding="utf-8") as fp:
        data = json.load(fp)
    for split in ("train", "test"):
        if split not in data:
            raise KeyError(f"{split} の件数が {path} に含まれていません。")
        if "total_rows" not in data[split] or "type_counts" not in data[split]:
            raise KeyError(f"{split} の total_rows/type_counts が欠落しています。")
    return data


def validate_steps(
    step_names: Sequence[str] | None = None,
    *,
    raise_on_error: bool = False,
) -> List[dict]:
    expected_counts = load_expected_counts()
    results: List[dict] = []
    errors: List[dict] = []

    for step_name in _ordered_step_names(step_names):
        specs = STEP_OUTPUT_SPECS.get(step_name, ())
        for spec in specs:
            result = _validate_output(step_name, spec, expected_counts)
            results.append(result)
            if result["status"] != "ok":
                errors.append(result)

    if raise_on_error and errors:
        raise ValidationError(_summarize_errors(errors))

    return results


def _validate_output(step_name: str, spec: OutputSpec, expected_counts: Dict[str, dict]) -> dict:
    output_dir = INTERIM_DIR / step_output_dir(step_name)
    path = output_dir / spec.filename
    expected_rows = _expected_rows(spec, expected_counts)

    result = {
        "step": step_name,
        "output_label": spec.filename,
        "path": _relative_path(path),
        "rows": None,
        "unique_data_ids": None,
        "expected_rows": expected_rows,
        "status": "missing",
        "message": "",
    }

    if not path.exists():
        result["message"] = "出力ファイルが存在しません。"
        return result

    try:
        df = pd.read_parquet(path, columns=["data_id"])
    except (ValueError, KeyError) as exc:
        result["rows"] = _count_rows(path)
        result["message"] = f"data_id列を読み込めません: {exc}"
        result["status"] = "error"
        return result

    rows = int(len(df))
    unique = int(df["data_id"].nunique(dropna=False))

    result["rows"] = rows
    result["unique_data_ids"] = unique

    issues: List[str] = []
    if expected_rows is not None and rows != expected_rows:
        issues.append(f"期待行数 {expected_rows:,} 件に対し {rows:,} 件")
    if unique != rows:
        issues.append(f"data_id ユニーク件数 {unique:,} 件 ≠ 行数 {rows:,} 件")

    if issues:
        result["status"] = "error"
        result["message"] = "; ".join(issues)
    else:
        result["status"] = "ok"
        result["message"] = ""
    return result


def _count_rows(path: Path) -> int:
    # data_id が存在しない場合でもせめて行数だけは把握する。
    df = pd.read_parquet(path)
    return int(len(df))


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _expected_rows(spec: OutputSpec, expected_counts: Dict[str, dict]) -> int | None:
    if spec.dataset is None:
        return None
    dataset_info = expected_counts.get(spec.dataset)
    if dataset_info is None:
        raise KeyError(f"{spec.dataset} の期待件数情報がありません。")
    if spec.type_label:
        type_counts = dataset_info.get("type_counts") or {}
        if spec.type_label not in type_counts:
            raise KeyError(
                f"{spec.dataset} の {spec.type_label} 件数が期待テーブルにありません。"
            )
        return int(type_counts[spec.type_label])
    return int(dataset_info.get("total_rows"))


def _summarize_errors(errors: Iterable[dict]) -> str:
    head = []
    for entry in errors:
        head.append(
            f"{entry['step']}::{entry['output_label']} - {entry.get('message') or 'Unknown error'}"
        )
    preview = "\n".join(head[:5])
    extra = len(head) - 5
    if extra > 0:
        preview += f"\n…他 {extra} 件"
    return preview


__all__ = ["ValidationError", "validate_steps"]


