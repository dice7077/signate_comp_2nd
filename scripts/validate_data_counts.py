#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.validation import ValidationError, validate_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="各ステップのParquet件数とdata_idユニーク件数を検証するユーティリティ。"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        metavar="STEP_NAME",
        help="検証対象ステップを限定する（未指定なら全ステップ）。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="結果をJSON形式で標準出力へ書き出す。",
    )
    parser.add_argument(
        "--only-errors",
        action="store_true",
        help="NG項目だけを表示する（--json時は無効）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = validate_steps(args.steps, raise_on_error=False)
    errors = [entry for entry in results if entry["status"] != "ok"]

    if args.json:
        payload = {"ok": not errors, "results": results, "errors": errors}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_results(results, only_errors=args.only_errors)
        if errors:
            print(f"\n検証NG: {len(errors)}件")
        else:
            print("\n検証OK: すべての出力が期待件数に一致しました。")

    if errors:
        raise SystemExit(1)


def _print_results(results: List[dict], *, only_errors: bool) -> None:
    for entry in results:
        if only_errors and entry["status"] == "ok":
            continue
        status = "OK " if entry["status"] == "ok" else "NG!"
        rows = "-" if entry["rows"] is None else f"{entry['rows']:,}"
        unique = (
            "-"
            if entry["unique_data_ids"] is None
            else f"{entry['unique_data_ids']:,}"
        )
        expected = (
            "-"
            if entry["expected_rows"] is None
            else f"{entry['expected_rows']:,}"
        )
        message = entry.get("message") or ""
        print(
            f"{status} step={entry['step']:<25} output={entry['output_label']:<25} "
            f"rows={rows:>10} unique={unique:>10} expected={expected:>10} {message}"
        )


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"検証で例外が発生しました: {exc}", file=sys.stderr)
        raise


