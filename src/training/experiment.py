from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold

from ..data_pipeline.processed.datasets import TYPE_DIRECTORIES
from ..data_pipeline.utils.paths import DATA_DIR, PROJECT_ROOT

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


class ExperimentError(RuntimeError):
    """実験の実行に関する例外。"""


def default_lightgbm_params() -> Dict[str, Any]:
    return {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 40,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "seed": 42,
        "verbosity": -1,
    }

@dataclass
class ExperimentConfig:
    type_label: str
    dataset_version: str
    experiment_name: str
    description: str
    folds: int = 5
    random_state: int = 42
    num_boost_round: int = 4000
    early_stopping_rounds: int = 200
    lightgbm_params: Dict[str, Any] = field(default_factory=default_lightgbm_params)
    target_column: str = "money_room"
    id_column: str = "data_id"


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    experiment_dir: Path
    metrics: Dict[str, Any]
    fold_metrics: List[Dict[str, Any]]
    artifacts: Dict[str, Any]


def run_experiment(config: ExperimentConfig, *, overwrite: bool = False) -> ExperimentResult:
    """
    加工済みデータを用いて LightGBM モデルを5-fold CVで学習し、各種成果物を保存する。
    """

    _validate_config(config)
    type_dir = TYPE_DIRECTORIES[config.type_label]
    dataset_dir = DATA_DIR / "processed" / type_dir / config.dataset_version
    train_path = dataset_dir / "train.parquet"
    test_path = dataset_dir / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise ExperimentError(f"データセットが存在しません: {dataset_dir}")

    experiment_dir = PROJECT_ROOT / "experiments" / config.type_label / config.experiment_name
    if experiment_dir.exists():
        if not overwrite:
            raise ExperimentError(f"{experiment_dir} は既に存在します。--overwrite を指定してください。")
        shutil.rmtree(experiment_dir)

    models_dir = experiment_dir / "artifacts" / "models"
    plots_dir = experiment_dir / "artifacts" / "plots"
    predictions_dir = experiment_dir / "artifacts" / "predictions"
    reports_dir = experiment_dir / "artifacts" / "reports"
    for path in (models_dir, plots_dir, predictions_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    feature_cols = _resolve_feature_columns(train_df, config)
    target = train_df[config.target_column].astype(float).to_numpy()
    features = train_df[feature_cols]
    test_features = test_df[feature_cols]

    oof_predictions = np.zeros(len(train_df), dtype=float)
    oof_folds = np.zeros(len(train_df), dtype=int)
    test_predictions_folds: List[np.ndarray] = []
    fold_importances: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []
    model_paths: List[str] = []

    lgbm_params = dict(config.lightgbm_params)
    # seedはfoldごとに固定する
    lgbm_params.setdefault("seed", config.random_state)
    lgbm_params.setdefault("feature_pre_filter", False)
    lgbm_params.setdefault("verbosity", -1)

    kf = KFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.random_state,
    )

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(features, target), start=1):
        fold_start = time.perf_counter()
        train_set = lgb.Dataset(features.iloc[train_idx], label=target[train_idx], feature_name=list(feature_cols))
        valid_set = lgb.Dataset(features.iloc[valid_idx], label=target[valid_idx], reference=train_set)

        booster = lgb.train(
            lgbm_params,
            train_set,
            num_boost_round=config.num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        best_iteration = booster.best_iteration or config.num_boost_round
        valid_pred = booster.predict(features.iloc[valid_idx], num_iteration=best_iteration)
        test_pred = booster.predict(test_features, num_iteration=best_iteration)

        oof_predictions[valid_idx] = valid_pred
        oof_folds[valid_idx] = fold_idx
        test_predictions_folds.append(test_pred)

        model_path = models_dir / f"fold_{fold_idx}.txt"
        booster.save_model(str(model_path), num_iteration=best_iteration)
        model_paths.append(str(model_path.relative_to(PROJECT_ROOT)))

        importance = booster.feature_importance(importance_type="gain")
        fold_importances.append(
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": importance,
                    "fold": fold_idx,
                }
            )
        )

        fold_time = time.perf_counter() - fold_start
        fold_metrics.append(
            {
                "fold": fold_idx,
                "rows_train": int(len(train_idx)),
                "rows_valid": int(len(valid_idx)),
                "best_iteration": int(best_iteration),
                "mape": _safe_mape(target[valid_idx], valid_pred),
                "mae": float(np.mean(np.abs(target[valid_idx] - valid_pred))),
                "rmse": float(math.sqrt(np.mean((target[valid_idx] - valid_pred) ** 2))),
                "train_time_sec": float(fold_time),
            }
        )

    overall_metrics = _summarize_metrics(target, oof_predictions, fold_metrics)

    test_prediction = np.mean(np.vstack(test_predictions_folds), axis=0)

    oof_path = predictions_dir / "oof_predictions.parquet"
    test_path_out = predictions_dir / "test_predictions.parquet"
    feature_importance_path = reports_dir / "feature_importance.csv"
    feature_importance_fold_path = reports_dir / "feature_importance_by_fold.csv"
    metrics_path = reports_dir / "metrics.json"
    summary_path = experiment_dir / "summary.json"
    config_snapshot_path = experiment_dir / "config.json"

    oof_df = pd.DataFrame(
        {
            config.id_column: train_df[config.id_column],
            "fold": oof_folds,
            "y_true": target,
            "y_pred": oof_predictions,
        }
    )
    oof_df.to_parquet(oof_path, index=False)

    test_df_out = pd.DataFrame(
        {
            config.id_column: test_df[config.id_column],
            "prediction": test_prediction,
        }
    )
    test_df_out.to_parquet(test_path_out, index=False)

    fold_importance_df = pd.concat(fold_importances, ignore_index=True)
    fold_importance_df.to_csv(feature_importance_fold_path, index=False)

    aggregated_importance = (
        fold_importance_df.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    aggregated_importance.rename(columns={"mean": "importance_mean", "std": "importance_std"}, inplace=True)
    aggregated_importance.to_csv(feature_importance_path, index=False)

    _write_json(metrics_path, overall_metrics)

    plot_paths = _create_plots(
        y_true=target,
        y_pred=oof_predictions,
        fold_metrics=fold_metrics,
        plots_dir=plots_dir,
        type_label=config.type_label,
        experiment_name=config.experiment_name,
    )

    artifacts = {
        "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
        "oof_predictions": str(oof_path.relative_to(PROJECT_ROOT)),
        "test_predictions": str(test_path_out.relative_to(PROJECT_ROOT)),
        "feature_importance": str(feature_importance_path.relative_to(PROJECT_ROOT)),
        "feature_importance_by_fold": str(feature_importance_fold_path.relative_to(PROJECT_ROOT)),
        "plots": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in plot_paths.items()},
        "models": model_paths,
    }

    summary_payload = {
        "type_label": config.type_label,
        "dataset_version": config.dataset_version,
        "experiment_name": config.experiment_name,
        "description": config.description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "folds": config.folds,
        "random_state": config.random_state,
        "features": list(feature_cols),
        "metrics": overall_metrics,
        "artifacts": artifacts,
        "paths": {
            "experiment_dir": str(experiment_dir.relative_to(PROJECT_ROOT)),
            "dataset_dir": str(dataset_dir.relative_to(PROJECT_ROOT)),
        },
        "config": asdict(config),
    }

    _write_json(summary_path, summary_payload)
    _write_json(config_snapshot_path, asdict(config))

    return ExperimentResult(
        config=config,
        experiment_dir=experiment_dir,
        metrics=overall_metrics,
        fold_metrics=fold_metrics,
        artifacts=artifacts,
    )


def _validate_config(config: ExperimentConfig) -> None:
    if config.type_label not in TYPE_DIRECTORIES:
        raise ExperimentError(
            f"未知のtypeが指定されました: {config.type_label}. choices={list(TYPE_DIRECTORIES)}"
        )
    if not config.experiment_name:
        raise ExperimentError("experiment_name は必須です。")
    if config.folds < 2:
        raise ExperimentError("folds は2以上を指定してください。")
    if not config.description.strip():
        raise ExperimentError("description は1文字以上の説明を入力してください。")


def _resolve_feature_columns(df: pd.DataFrame, config: ExperimentConfig) -> List[str]:
    reserved = {config.id_column, config.target_column}
    return [col for col in df.columns if col not in reserved]


def _safe_mape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def _summarize_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, fold_metrics: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = _safe_mape(y_true, y_pred)
    return {
        "overall": {
            "mape": mape,
            "mae": mae,
            "rmse": rmse,
        },
        "folds": list(fold_metrics),
    }


def _create_plots(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_metrics: Sequence[Dict[str, Any]],
    plots_dir: Path,
    type_label: str,
    experiment_name: str,
) -> Dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = plots_dir / "oof_scatter.png"
    residual_path = plots_dir / "residual_hist.png"
    fold_bar_path = plots_dir / "fold_mape.png"

    _plot_scatter(y_true, y_pred, scatter_path, type_label, experiment_name)
    _plot_residuals(y_true, y_pred, residual_path, type_label, experiment_name)
    _plot_fold_mape(fold_metrics, fold_bar_path, type_label, experiment_name)

    return {
        "oof_scatter": scatter_path,
        "residual_hist": residual_path,
        "fold_mape": fold_bar_path,
    }


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path, type_label: str, exp: str) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=10, alpha=0.3, edgecolor=None)
    max_val = np.percentile(np.concatenate([y_true, y_pred]), 99.5)
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="y=x")
    plt.xlabel("Actual money_room")
    plt.ylabel("Predicted money_room")
    plt.title(f"{type_label}::{exp} OOF actual vs. predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: Path, type_label: str, exp: str) -> None:
    residuals = y_pred - y_true
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Prediction - Actual")
    plt.ylabel("Count")
    plt.title(f"{type_label}::{exp} residual distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_fold_mape(fold_metrics: Sequence[Dict[str, Any]], path: Path, type_label: str, exp: str) -> None:
    plt.figure(figsize=(6, 4))
    fold_ids = [entry["fold"] for entry in fold_metrics]
    values = [entry["mape"] for entry in fold_metrics]
    sns.barplot(x=fold_ids, y=values, color="#4C72B0")
    plt.xlabel("Fold")
    plt.ylabel("MAPE")
    plt.title(f"{type_label}::{exp} fold-wise MAPE")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


__all__ = [
    "ExperimentConfig",
    "ExperimentError",
    "ExperimentResult",
    "default_lightgbm_params",
    "run_experiment",
]


