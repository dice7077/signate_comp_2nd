#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.processed.datasets import TYPE_DIRECTORIES
from src.training import ExperimentConfig, ExperimentResult, run_experiment
from src.training.experiment import ExperimentError, default_lightgbm_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5fold CVでLightGBMを学習し、結果を experiments/ 以下に保存する。",
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=sorted(TYPE_DIRECTORIES.keys()),
        help="学習対象タイプ（kodate/mansion）。",
    )
    parser.add_argument(
        "--version",
        default="0001_initial",
        help="data/processed 以下のデータバージョン。",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="実験名（例: 0001_initial）。",
    )
    parser.add_argument(
        "--description",
        required=True,
        help="実験の簡潔な説明。",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="KFold の分割数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード。",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=4000,
        help="LightGBM のラウンド数。",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=200,
        help="early stopping patience。",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="LightGBMパラメータを上書きする（YAML形式可）。例: --param learning_rate=0.03",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="同名実験ディレクトリが存在する場合に上書きする。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        param_overrides = _parse_param_overrides(args.param)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    params = default_lightgbm_params()
    params.update(param_overrides)
    params["seed"] = args.seed

    config = ExperimentConfig(
        type_label=args.type,
        dataset_version=args.version,
        experiment_name=args.experiment_name,
        description=args.description,
        folds=args.folds,
        random_state=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        lightgbm_params=params,
    )

    try:
        result = run_experiment(config, overwrite=args.overwrite)
    except ExperimentError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    snapshot_code(result, Path(__file__), vars(args))
    print_summary(result)


def _parse_param_overrides(entries: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--param には KEY=VALUE 形式を指定してください: {entry}")
        key, value = entry.split("=", 1)
        key = key.strip()
        try:
            overrides[key] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ValueError(f"--param の値を解析できません: {entry}") from exc
    return overrides


def snapshot_code(result: ExperimentResult, script_path: Path, cli_args: Dict[str, Any]) -> None:
    target_dir = result.experiment_dir / "code"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_path, target_dir / script_path.name)
    metadata_path = target_dir / "execution.json"
    payload = {
        "script": str(script_path.relative_to(PROJECT_ROOT)),
        "args": cli_args,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def print_summary(result: ExperimentResult) -> None:
    metrics = result.metrics["overall"]
    print("\n=== Experiment Completed ===")
    print(f"Type                : {result.config.type_label}")
    print(f"Experiment          : {result.config.experiment_name}")
    print(f"Description         : {result.config.description}")
    print(f"Output Directory    : {result.experiment_dir}")
    print(f"Validation MAPE     : {metrics['mape']:.6f}")
    print(f"Validation MAE      : {metrics['mae']:.2f}")
    print(f"Validation RMSE     : {metrics['rmse']:.2f}")
    print("Fold metrics:")
    for entry in result.fold_metrics:
        print(
            f"  Fold {entry['fold']}: "
            f"MAPE={entry['mape']:.6f} MAE={entry['mae']:.2f} "
            f"RMSE={entry['rmse']:.2f} best_iter={entry['best_iteration']}"
        )


if __name__ == "__main__":
    main()


