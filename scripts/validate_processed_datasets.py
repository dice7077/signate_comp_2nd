#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.validation.counts import ValidationError
from src.data_pipeline.validation.processed import validate_processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="data/processed 以下の学習データ件数と data_id ユニーク数を検証する。",
    )
    parser.add_argument(
        "--version",
        default="0001_initial",
        help="検証対象バージョン（例: 0001_initial）。",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["kodate", "mansion"],
        help="検証対象タイプを限定する（未指定なら両方）。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="結果をJSON形式で出力する。",
    )
    parser.add_argument(
        "--only-errors",
        action="store_true",
        help="NG項目のみ表示する（--json指定時は無効）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        results = validate_processed(
            version=args.version,
            types=args.types,
            raise_on_error=False,
        )
    except (ValidationError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    errors = [entry for entry in results if entry["status"] != "ok"]

    if args.json:
        payload = {"ok": not errors, "results": results, "errors": errors}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_results(results, only_errors=args.only_errors)
        if errors:
            print(f"\n検証NG: {len(errors)}件")
        else:
            print("\n検証OK: すべての出力が期待件数に一致しました。")

    if errors:
        raise SystemExit(1)


def _print_results(results: list[dict], *, only_errors: bool) -> None:
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
            f"{status} type={entry['type_label']:<8} split={entry['split']:<5} "
            f"rows={rows:>10} unique={unique:>10} expected={expected:>10} {message}"
        )


if __name__ == "__main__":
    main()


