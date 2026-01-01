#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.experiment import _plot_scatter, _plot_scatter_loglog


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def _resolve_summary_paths(exp_dirs: Sequence[str]) -> Iterable[Path]:
    if exp_dirs:
        for raw in exp_dirs:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            summary_path = candidate / "summary.json"
            if summary_path.exists():
                yield summary_path
            else:
                print(f"[WARN] summary.json が見つかりません: {summary_path}")
        return

    for summary_path in sorted(EXPERIMENTS_DIR.glob("*/*/summary.json")):
        yield summary_path


def _update_summary(summary_path: Path, payload: dict) -> None:
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _process_summary(summary_path: Path, overwrite: bool) -> bool:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    type_label = summary.get("type_label", "unknown")
    experiment_name = summary.get("experiment_name", summary_path.parent.name)

    plots_dir = summary_path.parent / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = plots_dir / "oof_scatter.png"
    log_path = plots_dir / "oof_scatter_loglog.png"

    oof_rel = summary.get("artifacts", {}).get("oof_predictions")
    if not oof_rel:
        print(f"[SKIP] OOFファイル情報がありません: {summary_path}")
        return False

    oof_path = PROJECT_ROOT / oof_rel
    if not oof_path.exists():
        print(f"[SKIP] OOFファイルが存在しません: {oof_path}")
        return False

    needs_linear = overwrite or not scatter_path.exists()
    needs_log = overwrite or not log_path.exists()
    if not (needs_linear or needs_log):
        print(f"[SKIP] 既に最新: {summary_path}")
        return False

    oof_df = pd.read_parquet(oof_path)
    required_cols = {"y_true", "y_pred"}
    if not required_cols.issubset(oof_df.columns):
        missing = required_cols.difference(oof_df.columns)
        print(f"[SKIP] 必須列欠損 ({missing}): {oof_path}")
        return False

    if needs_linear:
        _plot_scatter(
            oof_df["y_true"].to_numpy(),
            oof_df["y_pred"].to_numpy(),
            scatter_path,
            type_label,
            experiment_name,
        )
        print(f"[OK] プロット生成: {scatter_path}")

    if needs_log:
        _plot_scatter_loglog(
            oof_df["y_true"].to_numpy(),
            oof_df["y_pred"].to_numpy(),
            log_path,
            type_label,
            experiment_name,
        )
        print(f"[OK] プロット生成: {log_path}")

    plots_entry = summary.setdefault("artifacts", {}).setdefault("plots", {})
    updated = False
    scatter_rel = str(scatter_path.relative_to(PROJECT_ROOT))
    if plots_entry.get("oof_scatter") != scatter_rel:
        plots_entry["oof_scatter"] = scatter_rel
        updated = True
    rel_plot_path = str(log_path.relative_to(PROJECT_ROOT))
    if plots_entry.get("oof_scatter_loglog") != rel_plot_path:
        plots_entry["oof_scatter_loglog"] = rel_plot_path
        updated = True

    if updated:
        _update_summary(summary_path, summary)
        print(f"[OK] summary更新: {summary_path}")
    else:
        print(f"[SKIP] summary更新なし: {summary_path}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="既存実験のOOF散布図（線形・両対数）を再生成します。"
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="*",
        help="対象実験ディレクトリ（PROJECT_ROOTからの相対パス可）。空なら全件対象。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の oof_scatter_loglog.png があっても上書きします。",
    )
    args = parser.parse_args()

    touched = 0
    updated = 0
    for summary_path in _resolve_summary_paths(args.experiment_dirs):
        touched += 1
        if _process_summary(summary_path, args.overwrite):
            updated += 1

    print(f"[DONE] 対象:{touched} 件 / 更新:{updated} 件")


if __name__ == "__main__":
    main()

