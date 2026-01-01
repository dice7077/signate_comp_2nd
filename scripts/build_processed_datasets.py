#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.processed import ProcessedDatasetError, build_processed_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="data/processed 以下に戸建て/マンション向け学習データを構築するスクリプト。",
    )
    parser.add_argument(
        "--version",
        default="0001_initial",
        help="出力先のバージョン名（例: 0001_initial）。",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["kodate", "mansion"],
        help="対象タイプを限定する（未指定なら両方）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存出力を上書きする場合に指定。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        outputs = build_processed_datasets(
            version=args.version,
            types=args.types,
            overwrite=args.overwrite,
        )
    except ProcessedDatasetError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    for path in outputs:
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()


