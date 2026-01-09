#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import GroupKFold, KFold
import yaml

from src.data_pipeline.processed.datasets import TYPE_DIRECTORIES
from src.data_pipeline.utils.paths import DATA_DIR

from src.training import ExperimentConfig, run_experiment
from src.training.experiment import (
    ExperimentError,
    _bucket_metrics_dataframe,
    _compute_sample_weight,
    _fold_progress_callback,
    _has_pre_dedup_overlap_columns,
    _prepare_categorical_features,
    _prepare_overlap_analysis,
    _prepare_pre_dedup_overlap_analysis,
    _regression_metrics,
    _resolve_feature_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="既存ExperimentConfigを基にLightGBMパラメータをOptunaで探索するツール。",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="ExperimentConfig(JSON/YAML)ファイルのパス。",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Optunaの試行回数。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="探索の最大秒数。指定なしなら回数優先。",
    )
    parser.add_argument(
        "--metric",
        choices=("mape", "mae", "rmse"),
        default="mape",
        help="最適化対象の検証指標。",
    )
    parser.add_argument(
        "--trial-prefix",
        help="試行ごとの experiment_name 接頭辞。未指定なら 'trial_'。",
    )
    parser.add_argument(
        "--trial-subdir",
        default="optuna_trials",
        help="試行成果物を格納するサブディレクトリ名（base experiment配下）。空文字で無効化。",
    )
    parser.add_argument(
        "--trial-run-name",
        help="optuna_trials 以下に作成するラン専用サブディレクトリ名（例: 0001_optuna）。",
    )
    parser.add_argument(
        "--fast-fold-index",
        type=int,
        default=0,
        help="高速探索用に指定foldのみを評価する（1-based）。0なら通常の全fold。",
    )
    parser.add_argument(
        "--study-name",
        help="Optuna study名。storage指定時のみ有効。",
    )
    parser.add_argument(
        "--storage",
        help="Optuna storage。例: sqlite:///mansion_optuna.db",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="同名studyが存在する場合は再利用する。",
    )
    parser.add_argument(
        "--no-default-storage",
        action="store_true",
        help="デフォルトのSQLite storage作成を無効化する（Optunaがメモリ上で動作）。",
    )
    parser.add_argument(
        "--pruner",
        choices=("none", "median", "percentile"),
        default="median",
        help="Optuna prunerの種類（fast-fold時のみ有効）。",
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=200,
        help="Prunerを有効化する前に確保する学習イテレーション数。",
    )
    parser.add_argument(
        "--pruner-percentile",
        type=float,
        default=25.0,
        help="percentile pruner利用時のパーセンタイル値（0-100）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="TPESamplerの乱数シード。",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        help="探索時に上書きするnum_boost_round（未指定ならconfig値）。",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        help="探索時に上書きするearly_stopping_rounds。",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="各試行のexperiments配下成果物を残す（デフォルトは削除）。",
    )
    parser.add_argument(
        "--summary-output",
        help="探索履歴/ベスト試行をJSONで保存するパス。",
    )
    parser.add_argument(
        "--export-config",
        help="ベストパラメータを埋め込んだExperimentConfig出力先。",
    )
    parser.add_argument(
        "--final-experiment-name",
        help="export-configに書き込むexperiment_name（未指定なら base名 + _optuna_best）。",
    )
    parser.add_argument(
        "--final-description",
        help="export-configに書き込むdescription。未指定なら base説明 + ' (optuna best)'.",
    )
    parser.add_argument(
        "--skip-base-trial",
        action="store_true",
        help="Optuna探索でベースライン設定を強制的に試行しない。",
    )
    return parser.parse_args()


def load_experiment_config(path_str: str) -> ExperimentConfig:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"configファイルが見つかりません: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp)
    else:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError("ExperimentConfigファイルの構造が不正です。")
    return ExperimentConfig(**payload)


@dataclass
class FastFoldContext:
    features: pd.DataFrame
    target_raw: np.ndarray
    target: np.ndarray
    sample_weight: np.ndarray | None
    categorical_cols: list[str]
    feature_cols: list[str]
    group_values: pd.Series | None
    splits: list[tuple[np.ndarray, np.ndarray]]
    bucket_keys: np.ndarray | None
    bucket_category_specs: Tuple[Tuple[str, str], ...] | None


def _clamp_float(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _clamp_int(value: int, low: int, high: int) -> int:
    return int(min(max(value, low), high))


def _snap_to_step(value: int, low: int, high: int, step: int) -> int:
    snapped = low + round((value - low) / step) * step
    return _clamp_int(snapped, low, high)


def build_base_trial_params(config: ExperimentConfig) -> Dict[str, Any]:
    params = config.lightgbm_params
    get = params.get
    base = {
        "learning_rate": _clamp_float(get("learning_rate", 0.1), 0.02, 0.2),
        "num_leaves": _snap_to_step(int(get("num_leaves", 63)), 63, 511, 8),
        "max_depth": get("max_depth", -1),
        "min_data_in_leaf": _clamp_int(int(get("min_data_in_leaf", 40)), 20, 400),
        "feature_fraction": _clamp_float(get("feature_fraction", 0.9), 0.6, 1.0),
        "bagging_fraction": _clamp_float(get("bagging_fraction", 0.9), 0.6, 1.0),
        "bagging_freq": _clamp_int(int(get("bagging_freq", 1)), 1, 7),
        "lambda_l1": _clamp_float(max(get("lambda_l1", 0.0), 1e-5), 1e-5, 10.0),
        "lambda_l2": _clamp_float(max(get("lambda_l2", 0.0), 1e-5), 1e-5, 10.0),
        "min_gain_to_split": _clamp_float(get("min_gain_to_split", 0.0), 0.0, 0.5),
    }
    allowed_depths = [-1, 6, 8, 10, 12, 14, 16]
    if base["max_depth"] not in allowed_depths:
        base["max_depth"] = -1
    return base


def _params_match(params: Dict[str, Any], signature: Dict[str, Any]) -> bool:
    for key, target in signature.items():
        if key not in params:
            return False
        value = params[key]
        if isinstance(target, float):
            if not math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-9):
                return False
        else:
            if value != target:
                return False
    return True


def prepare_fast_fold_context(config: ExperimentConfig) -> FastFoldContext:
    type_dir = TYPE_DIRECTORIES[config.type_label]
    dataset_dir = DATA_DIR / "processed" / type_dir / config.dataset_version
    train_path = dataset_dir / "train.parquet"
    test_path = dataset_dir / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise ExperimentError(f"データセットが存在しません: {dataset_dir}")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    group_values: pd.Series | None = None
    if config.group_column:
        if config.group_column not in train_df.columns:
            raise ExperimentError(f"group_column '{config.group_column}' が train データに存在しません。")
        group_values = train_df[config.group_column]
        if group_values.isna().any():
            raise ExperimentError(f"group_column '{config.group_column}' に欠損が含まれています。")

    feature_cols = _resolve_feature_columns(train_df, config)
    target_raw = train_df[config.target_column].astype(float).to_numpy()
    if not np.all(np.isfinite(target_raw)):
        raise ExperimentError("Target列に無効な値が含まれています。")

    target = target_raw.copy()
    if config.log_target:
        if np.any(target_raw <= 0):
            raise ExperimentError("log_target=True の場合、targetは正の値でなければなりません。")
        target = np.log(target_raw)

    sample_weight = _compute_sample_weight(config, target)
    features = train_df[feature_cols].copy()
    test_features = test_df[feature_cols].copy()

    categorical_cols: list[str] = []
    if config.categorical_features:
        categorical_cols = list(dict.fromkeys(config.categorical_features))
        missing_cats = [col for col in categorical_cols if col not in feature_cols]
        if missing_cats:
            raise ExperimentError(f"Categorical列が存在しません: {', '.join(missing_cats)}")
        _prepare_categorical_features(features, test_features, categorical_cols)

    bucket_keys: np.ndarray | None = None
    bucket_category_specs: Tuple[Tuple[str, str], ...] | None = None
    if config.bucket_analysis:
        use_pre_dedup = (
            not config.bucket_analysis.use_building
            and _has_pre_dedup_overlap_columns(train_df)
        )
        if use_pre_dedup:
            overlap_analysis = _prepare_pre_dedup_overlap_analysis(
                train_df=train_df,
                test_df=test_df,
                id_column=config.id_column,
            )
        else:
            overlap_analysis = _prepare_overlap_analysis(
                train_df=train_df,
                test_df=test_df,
                id_column=config.id_column,
                config=config.bucket_analysis,
            )
        bucket_category_specs = overlap_analysis["category_specs"]  # type: ignore[assignment]
        train_assignments = overlap_analysis["train_assignments"]
        bucket_lookup = (
            train_assignments.set_index(config.id_column)["overlap_category"]  # type: ignore[index]
        )
        ordered_ids = train_df[config.id_column]
        bucket_keys = bucket_lookup.reindex(ordered_ids).to_numpy(copy=False)

    if config.group_column:
        splitter = GroupKFold(n_splits=config.folds)
        split_iter = splitter.split(features, target, groups=group_values.to_numpy(copy=False))
    else:
        splitter = KFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
        split_iter = splitter.split(features, target)

    splits = list(split_iter)
    return FastFoldContext(
        features=features,
        target_raw=target_raw,
        target=target,
        sample_weight=sample_weight,
        categorical_cols=categorical_cols,
        feature_cols=feature_cols,
        group_values=group_values,
        splits=splits,
        bucket_keys=bucket_keys,
        bucket_category_specs=bucket_category_specs,
    )


def _resolve_pruner_metric(lightgbm_params: Dict[str, Any]) -> str:
    metric = lightgbm_params.get("metric", "l2")
    if isinstance(metric, (list, tuple)):
        return metric[0]
    return str(metric)


def run_fast_fold_training(
    trial: optuna.trial.Trial,
    config: ExperimentConfig,
    data: FastFoldContext,
    fold_index: int,
    enable_pruning: bool,
) -> Dict[str, Any]:
    if fold_index < 1 or fold_index > len(data.splits):
        raise ExperimentError(f"fast-fold index {fold_index} がfold数 {len(data.splits)} を超えています。")

    train_idx, valid_idx = data.splits[fold_index - 1]
    train_weight = data.sample_weight[train_idx] if data.sample_weight is not None else None
    valid_weight = data.sample_weight[valid_idx] if data.sample_weight is not None else None

    train_set = lgb.Dataset(
        data.features.iloc[train_idx],
        label=data.target[train_idx],
        feature_name=list(data.feature_cols),
        categorical_feature=data.categorical_cols or None,
        weight=train_weight,
    )
    valid_set = lgb.Dataset(
        data.features.iloc[valid_idx],
        label=data.target[valid_idx],
        reference=train_set,
        categorical_feature=data.categorical_cols or None,
        weight=valid_weight,
    )

    callbacks = [
        lgb.early_stopping(config.early_stopping_rounds, verbose=False),
        _fold_progress_callback(fold_index, config.progress_period),
    ]
    if enable_pruning:
        metric_name = _resolve_pruner_metric(config.lightgbm_params)
        callbacks.append(LightGBMPruningCallback(trial, metric=metric_name, valid_name="valid"))

    start = time.perf_counter()
    booster = lgb.train(
        config.lightgbm_params,
        train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    elapsed = time.perf_counter() - start

    best_iteration = booster.best_iteration or config.num_boost_round
    train_pred = booster.predict(
        data.features.iloc[train_idx],
        num_iteration=best_iteration,
    )
    valid_pred = booster.predict(
        data.features.iloc[valid_idx],
        num_iteration=best_iteration,
    )
    if config.log_target:
        train_pred = np.exp(train_pred)
        valid_pred = np.exp(valid_pred)

    train_target = data.target_raw[train_idx]
    eval_target = data.target_raw[valid_idx]
    train_metrics = _regression_metrics(train_target, train_pred)
    valid_metrics = _regression_metrics(eval_target, valid_pred)

    if data.bucket_keys is not None and data.bucket_category_specs is not None:
        def _log_bucket_metrics(
            stage: str,
            indices: np.ndarray,
            y_true_values: np.ndarray,
            y_pred_values: np.ndarray,
        ) -> None:
            bucket_subset = data.bucket_keys[indices]
            bucket_summary = _bucket_metrics_dataframe(
                bucket_keys=bucket_subset,
                y_true=y_true_values,
                y_pred=y_pred_values,
                category_specs=data.bucket_category_specs,
            )
            print(
                f"[fast-fold {fold_index}] {stage} bucket metrics:",
                flush=True,
            )
            for _, row in bucket_summary.iterrows():
                count = int(row["data_id_count"])
                ratio = row["ratio"]
                mape = row["mape"]
                ratio_pct = ratio * 100 if pd.notna(ratio) else 0.0
                mape_text = f"{mape:.6f}" if pd.notna(mape) else "N/A"
                print(
                    f"    - {row['label']}: {count:,} rows ({ratio_pct:.2f}%) MAPE={mape_text}",
                    flush=True,
                )

        _log_bucket_metrics("train", train_idx, train_target, train_pred)
        _log_bucket_metrics("valid", valid_idx, eval_target, valid_pred)

    fold_entry = {
        "fold": fold_index,
        "rows_train": int(len(train_idx)),
        "rows_valid": int(len(valid_idx)),
        "best_iteration": int(best_iteration),
        "train_mape": train_metrics["mape"],
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "valid_mape": valid_metrics["mape"],
        "valid_mae": valid_metrics["mae"],
        "valid_rmse": valid_metrics["rmse"],
        "mape": valid_metrics["mape"],
        "mae": valid_metrics["mae"],
        "rmse": valid_metrics["rmse"],
        "train_time_sec": float(elapsed),
    }
    return fold_entry


def suggest_lgbm_params(trial: optuna.trial.Trial, base_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(base_params)
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.003, 0.2, log=True)
    params["num_leaves"] = trial.suggest_int("num_leaves", 63, 511, step=4)
    params["max_depth"] = trial.suggest_int("max_depth", 6, 32)
    params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 20, 400)
    params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.4, 1.0)
    params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)
    params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)
    params["lambda_l1"] = trial.suggest_float("lambda_l1", 1e-6, 10.0, log=True)
    params["lambda_l2"] = trial.suggest_float("lambda_l2", 1e-6, 10.0, log=True)
    params["min_gain_to_split"] = trial.suggest_float("min_gain_to_split", 0.0, 0.5)
    params.setdefault("feature_pre_filter", False)
    params.setdefault("verbosity", -1)
    return params


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round_to_step(value: int, lower: int, upper: int, step: int) -> int:
    clipped = int(_clip(value, lower, upper))
    steps = round((clipped - lower) / step)
    candidate = lower + steps * step
    return int(_clip(candidate, lower, upper))


def _resolve_max_depth(value: int) -> int:
    allowed = [-1, 6, 8, 10, 12, 14, 16]
    if value in allowed:
        return value
    positive = [depth for depth in allowed if depth >= 0]
    if not positive:
        return -1
    closest = min(positive, key=lambda depth: abs(depth - value))
    return closest


def build_base_trial_params(config: ExperimentConfig) -> Dict[str, Any]:
    base_params = config.lightgbm_params

    def _get(name: str, default: float) -> float:
        return float(base_params.get(name, default))

    params: Dict[str, Any] = {}
    params["learning_rate"] = float(_clip(_get("learning_rate", 0.1), 0.02, 0.2))
    params["num_leaves"] = _round_to_step(int(base_params.get("num_leaves", 63)), 63, 511, 8)
    params["max_depth"] = _resolve_max_depth(int(base_params.get("max_depth", -1)))
    params["min_data_in_leaf"] = int(_clip(base_params.get("min_data_in_leaf", 40), 20, 400))
    params["feature_fraction"] = float(_clip(_get("feature_fraction", 0.9), 0.6, 1.0))
    params["bagging_fraction"] = float(_clip(_get("bagging_fraction", 0.9), 0.6, 1.0))
    params["bagging_freq"] = int(_clip(base_params.get("bagging_freq", 1), 1, 7))
    params["lambda_l1"] = float(_clip(base_params.get("lambda_l1", 1e-5), 1e-5, 10.0))
    params["lambda_l2"] = float(_clip(base_params.get("lambda_l2", 1e-5), 1e-5, 10.0))
    params["min_gain_to_split"] = float(_clip(base_params.get("min_gain_to_split", 0.0), 0.0, 0.5))
    params.setdefault("feature_pre_filter", False)
    params.setdefault("verbosity", -1)
    return params


def create_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=args.pruner_warmup_steps)
    if args.pruner == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=args.pruner_percentile,
            n_warmup_steps=args.pruner_warmup_steps,
        )
    raise ValueError(f"未知のprunerが指定されました: {args.pruner}")


def export_best_config(
    base_config: ExperimentConfig,
    best_params: Dict[str, Any],
    output_path: str,
    experiment_name: str | None,
    description: str | None,
) -> Path:
    payload = asdict(base_config)
    payload["lightgbm_params"] = dict(base_config.lightgbm_params)
    payload["lightgbm_params"].update(best_params)
    payload["experiment_name"] = experiment_name or f"{base_config.experiment_name}_optuna_best"
    payload["description"] = description or f"{base_config.description} (optuna best)"
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return out_path


def _normalize_run_name(name: str) -> str:
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("trial-run-name は空にできません。")
    if any(sep in trimmed for sep in ("/", "\\")):
        raise ValueError("trial-run-name にパス区切り文字は使用できません。")
    if trimmed in {".", ".."}:
        raise ValueError("trial-run-name に '.' や '..' は使用できません。")
    return trimmed


def resolve_trial_root(base_config: ExperimentConfig, args: argparse.Namespace) -> Path:
    root = Path(base_config.experiment_name)
    if args.trial_subdir:
        root = root / args.trial_subdir
    if args.trial_run_name:
        root = root / _normalize_run_name(args.trial_run_name)
    return root


def resolve_storage_settings(
    args: argparse.Namespace,
    base_config: ExperimentConfig,
    trial_root: Path,
) -> tuple[str | None, str | None, bool]:
    storage_url = args.storage
    auto_storage = False
    if not storage_url and not args.no_default_storage:
        base_dir = (
            PROJECT_ROOT
            / "experiments"
            / base_config.type_label
        )
        base_dir = base_dir / trial_root
        base_dir.mkdir(parents=True, exist_ok=True)
        storage_path = base_dir / "optuna_study.db"
        storage_url = f"sqlite:///{storage_path}"
        auto_storage = True

    if storage_url:
        storage_url = _prepare_sqlite_storage_path(storage_url)

    study_name = args.study_name
    if storage_url and not study_name:
        study_name = base_config.experiment_name

    load_if_exists = args.load_if_exists or auto_storage
    return storage_url, study_name, load_if_exists


def _prepare_sqlite_storage_path(storage_url: str) -> str:
    parsed = urlparse(storage_url)
    if parsed.scheme != "sqlite":
        return storage_url

    raw_path = (parsed.netloc or "") + (parsed.path or "")
    if raw_path.startswith("//"):
        raw_path = raw_path[1:]
    if not raw_path:
        return storage_url

    db_path = Path(raw_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def build_objective(
    base_config: ExperimentConfig,
    metric_key: str,
    trial_prefix: str,
    trial_dir: str,
    cleanup: bool,
    num_boost_round: int | None,
    early_stopping_rounds: int | None,
    base_trial_signature: Dict[str, Any] | None,
):
    trial_root = Path(trial_dir)

    def _objective(trial: optuna.trial.Trial) -> float:
        trial_config = deepcopy(base_config)
        suggested_params = suggest_lgbm_params(trial, base_config.lightgbm_params)
        is_base_trial = (
            base_trial_signature is not None and _params_match(trial.params, base_trial_signature)
        )
        if is_base_trial:
            lgbm_params = deepcopy(base_config.lightgbm_params)
            trial.set_user_attr("is_base_config", True)
        else:
            lgbm_params = suggested_params
            lgbm_params["seed"] = trial_config.random_state + trial.number
        lgbm_params.setdefault("feature_pre_filter", False)
        lgbm_params.setdefault("verbosity", -1)
        trial_config.lightgbm_params = lgbm_params
        trial_path = trial_root / f"{trial_prefix}{trial.number:04d}"
        trial_config.experiment_name = trial_path.as_posix()
        if is_base_trial:
            trial_config.description = f"{base_config.description} (base config trial)"
        else:
            trial_config.description = f"{base_config.description} (optuna trial {trial.number})"
        if num_boost_round is not None:
            trial_config.num_boost_round = num_boost_round
        if early_stopping_rounds is not None:
            trial_config.early_stopping_rounds = early_stopping_rounds

        try:
            result = run_experiment(trial_config, overwrite=True)
        except ExperimentError as exc:
            raise optuna.TrialPruned(f"Experiment failed: {exc}") from exc

        value = result.metrics["overall"][metric_key]
        trial.set_user_attr("experiment_name", trial_config.experiment_name)
        trial.set_user_attr("metric_value", value)
        trial.set_user_attr("fold_metrics", result.fold_metrics)
        trial.set_user_attr("lightgbm_params", trial_config.lightgbm_params)
        if cleanup:
            shutil.rmtree(result.experiment_dir, ignore_errors=True)
        return value

    return _objective


def build_fast_objective(
    base_config: ExperimentConfig,
    metric_key: str,
    fold_index: int,
    data: FastFoldContext,
    num_boost_round: int | None,
    early_stopping_rounds: int | None,
    enable_pruning: bool,
    base_trial_signature: Dict[str, Any] | None,
):
    def _objective(trial: optuna.trial.Trial) -> float:
        trial_config = deepcopy(base_config)
        suggested_params = suggest_lgbm_params(trial, base_config.lightgbm_params)
        is_base_trial = (
            base_trial_signature is not None and _params_match(trial.params, base_trial_signature)
        )
        if is_base_trial:
            lgbm_params = deepcopy(base_config.lightgbm_params)
            trial.set_user_attr("is_base_config", True)
            trial_config.description = f"{base_config.description} (fast fold base trial)"
        else:
            lgbm_params = suggested_params
            lgbm_params.setdefault("seed", trial_config.random_state + trial.number)
            trial_config.description = f"{base_config.description} (fast fold trial {trial.number})"
        lgbm_params.setdefault("feature_pre_filter", False)
        lgbm_params.setdefault("verbosity", -1)
        trial_config.lightgbm_params = lgbm_params

        if num_boost_round is not None:
            trial_config.num_boost_round = num_boost_round
        if early_stopping_rounds is not None:
            trial_config.early_stopping_rounds = early_stopping_rounds

        metrics = run_fast_fold_training(
            trial=trial,
            config=trial_config,
            data=data,
            fold_index=fold_index,
            enable_pruning=enable_pruning,
        )

        value = metrics[metric_key]
        trial.set_user_attr("experiment_name", None)
        trial.set_user_attr("metric_value", value)
        trial.set_user_attr("fold_metrics", [metrics])
        trial.set_user_attr("lightgbm_params", trial_config.lightgbm_params)
        return value

    return _objective


def dump_summary(path_str: str, study: optuna.Study, base_config_path: str, metric: str) -> Path:
    payload = {
        "base_config": str(base_config_path),
        "metric": metric,
        "best_value": study.best_value if study.best_trial else None,
        "best_trial_number": study.best_trial.number if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else {},
        "n_completed_trials": len(study.trials),
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
        ],
    }
    out_path = Path(path_str)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    trial_prefix = args.trial_prefix or "trial_"
    trial_root = resolve_trial_root(base_config, args)
    trial_root_str = trial_root.as_posix()
    fast_mode = args.fast_fold_index > 0
    fast_context: FastFoldContext | None = None
    base_trial_signature: Dict[str, Any] | None = None

    if fast_mode:
        if args.fast_fold_index > base_config.folds:
            raise SystemExit(f"--fast-fold-index は 1~{base_config.folds} の範囲で指定してください。")
        if args.keep_artifacts:
            print("[WARN] fast-foldモードでは成果物を保存しません。--keep-artifacts は無視されます。", file=sys.stderr)
        fast_context = prepare_fast_fold_context(base_config)
        trial_dir_str = ""
        cleanup = True
    else:
        cleanup = not args.keep_artifacts
        trial_dir_str = trial_root_str

    storage_url, study_name, load_if_exists = resolve_storage_settings(args, base_config, trial_root)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    pruner = create_pruner(args)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )
    if storage_url:
        print(
            f"[INFO] Optuna storage: {storage_url} (study={study_name}, load_if_exists={load_if_exists})",
            flush=True,
        )

    if not args.skip_base_trial:
        base_trial_signature = build_base_trial_params(base_config)
        study.enqueue_trial(base_trial_signature)

    if fast_mode:
        assert fast_context is not None
        objective = build_fast_objective(
            base_config=base_config,
            metric_key=args.metric,
            fold_index=args.fast_fold_index,
            data=fast_context,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            enable_pruning=args.pruner != "none",
            base_trial_signature=base_trial_signature,
        )
    else:
        objective = build_objective(
            base_config=base_config,
            metric_key=args.metric,
            trial_prefix=trial_prefix,
            trial_dir=trial_dir_str,
            cleanup=cleanup,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            base_trial_signature=base_trial_signature,
        )

    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    except KeyboardInterrupt:
        print("探索がユーザーにより中断されました。", file=sys.stderr)

    if not study.best_trial:
        print("有効な試行が存在しませんでした。", file=sys.stderr)
        return

    best = study.best_trial
    print("\n=== Optuna Best Trial ===")
    print(f"Trial #{best.number}")
    print(f"{args.metric}: {best.value:.6f}")
    for key, val in sorted(best.params.items()):
        print(f"  {key}: {val}")

    if args.summary_output:
        out_path = dump_summary(args.summary_output, study, args.config, args.metric)
        print(f"探索サマリを保存: {out_path}")

    if args.export_config:
        exported_path = export_best_config(
            base_config=base_config,
            best_params=best.params,
            output_path=args.export_config,
            experiment_name=args.final_experiment_name,
            description=args.final_description,
        )
        print(f"ベスト設定を書き出しました: {exported_path}")
        print(
            "最終実験を実行するには run_experiment.py --config で上記ファイルを指定してください。"
        )


if __name__ == "__main__":
    main()


