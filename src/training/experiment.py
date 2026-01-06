from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
from sklearn.model_selection import GroupKFold, KFold

from ..data_pipeline.processed.datasets import TYPE_DIRECTORIES
from ..data_pipeline.utils.overlap import (
    compute_overlap_features,
    get_category_specs,
    value_counts,
)
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
class BucketAnalysisConfig:
    unit_column: str
    building_column: str | None = None
    output_prefix: str = "unit_building_overlap"
    use_building: bool = True


PRE_DEDUP_BUCKET_SPECS: Tuple[Tuple[str, str], ...] = (
    ("0", "unit_id被り0件"),
    ("1", "unit_id被り1件"),
    ("2", "unit_id被り2件"),
    ("3", "unit_id被り3件"),
    ("4+", "unit_id被り4件以上"),
)


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
    feature_columns: Sequence[str] | None = None
    categorical_features: Sequence[str] | None = None
    progress_period: int = 500
    log_target: bool = False
    group_column: str | None = None
    weight_strategy: str | None = None
    bucket_analysis: BucketAnalysisConfig | None = None

    def __post_init__(self) -> None:
        if self.bucket_analysis and isinstance(self.bucket_analysis, dict):
            self.bucket_analysis = BucketAnalysisConfig(**self.bucket_analysis)


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
    overlap_analysis = None
    category_specs: Tuple[Tuple[str, str], ...] | None = None
    if config.bucket_analysis:
        use_pre_dedup_overlap = (
            not config.bucket_analysis.use_building
            and _has_pre_dedup_overlap_columns(train_df)
        )
        if use_pre_dedup_overlap:
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
        category_specs = overlap_analysis["category_specs"]
    train_overlap_lookup: pd.DataFrame | None = None
    if overlap_analysis is not None:
        train_overlap_lookup = overlap_analysis["train_assignments"].set_index(config.id_column)
    group_values: pd.Series | None = None
    if config.group_column:
        if config.group_column not in train_df.columns:
            raise ExperimentError(
                f"group_column '{config.group_column}' が train データに存在しません。"
            )
        if config.group_column not in test_df.columns:
            raise ExperimentError(
                f"group_column '{config.group_column}' が test データに存在しません。"
            )
        group_values = train_df[config.group_column]
        if group_values.isna().any():
            raise ExperimentError(
                f"group_column '{config.group_column}' に欠損が含まれています。"
            )

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

    categorical_cols = []
    if config.categorical_features:
        categorical_cols = list(dict.fromkeys(config.categorical_features))
        missing_cats = [col for col in categorical_cols if col not in feature_cols]
        if missing_cats:
            raise ExperimentError(
                f"Categorical列が存在しません: {', '.join(missing_cats)}"
            )
        _prepare_categorical_features(features, test_features, categorical_cols)

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

    if config.group_column:
        splitter = GroupKFold(n_splits=config.folds)
        split_iter = splitter.split(
            features, target, groups=group_values.to_numpy(copy=False)
        )
    else:
        splitter = KFold(
            n_splits=config.folds,
            shuffle=True,
            random_state=config.random_state,
        )
        split_iter = splitter.split(features, target)

    for fold_idx, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        fold_start = time.perf_counter()
        print(
            f"[fold {fold_idx}] 学習開始: train={len(train_idx)} valid={len(valid_idx)} features={len(feature_cols)}",
            flush=True,
        )
        train_weight = sample_weight[train_idx] if sample_weight is not None else None
        valid_weight = sample_weight[valid_idx] if sample_weight is not None else None
        train_set = lgb.Dataset(
            features.iloc[train_idx],
            label=target[train_idx],
            feature_name=list(feature_cols),
            categorical_feature=categorical_cols or None,
            weight=train_weight,
        )
        valid_set = lgb.Dataset(
            features.iloc[valid_idx],
            label=target[valid_idx],
            reference=train_set,
            categorical_feature=categorical_cols or None,
            weight=valid_weight,
        )

        booster = lgb.train(
            lgbm_params,
            train_set,
            num_boost_round=config.num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                _fold_progress_callback(fold_idx, config.progress_period),
            ],
        )

        best_iteration = booster.best_iteration or config.num_boost_round
        valid_pred = booster.predict(features.iloc[valid_idx], num_iteration=best_iteration)
        test_pred = booster.predict(test_features, num_iteration=best_iteration)
        if config.log_target:
            valid_pred = np.exp(valid_pred)
            test_pred = np.exp(test_pred)

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
        eval_target = target_raw[valid_idx]
        fold_entry = {
            "fold": fold_idx,
            "rows_train": int(len(train_idx)),
            "rows_valid": int(len(valid_idx)),
            "best_iteration": int(best_iteration),
            "mape": _safe_mape(eval_target, valid_pred),
            "mae": float(np.mean(np.abs(eval_target - valid_pred))),
            "rmse": float(math.sqrt(np.mean((eval_target - valid_pred) ** 2))),
            "train_time_sec": float(fold_time),
        }
        fold_metrics.append(fold_entry)
        if train_overlap_lookup is not None and category_specs is not None:
            valid_ids = train_df.iloc[valid_idx][config.id_column]
            overlap_subset = train_overlap_lookup.loc[valid_ids]
            bucket_summary = _bucket_metrics_dataframe(
                bucket_keys=overlap_subset["overlap_category"].to_numpy(),
                y_true=target_raw[valid_idx],
                y_pred=valid_pred,
                category_specs=category_specs,
            )
            print(f"[fold {fold_idx}] bucket metrics:", flush=True)
            for _, row in bucket_summary.iterrows():
                count = int(row["data_id_count"])
                ratio_pct = row["ratio"] * 100 if row["ratio"] is not None else 0.0
                mape_text = f"{row['mape']:.6f}" if row["mape"] is not None else "N/A"
                print(
                    f"    - {row['label']}: {count:,} rows ({ratio_pct:.2f}%) MAPE={mape_text}",
                    flush=True,
                )
        print(
            "[fold {fold}] 完了: best_iter={best_iter} "
            "MAPE={mape:.6f} MAE={mae:.2f} RMSE={rmse:.2f} time={time:.1f}s".format(
                fold=fold_idx,
                best_iter=fold_entry["best_iteration"],
                mape=fold_entry["mape"],
                mae=fold_entry["mae"],
                rmse=fold_entry["rmse"],
                time=fold_entry["train_time_sec"],
            ),
            flush=True,
        )

    overall_metrics = _summarize_metrics(target_raw, oof_predictions, fold_metrics)

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
            "y_true": target_raw,
            "y_pred": oof_predictions,
        }
    )
    pre_dedup_artifacts = None
    if "pre_dedup_unit_overlap_bucket" in train_df.columns:
        aux_columns = [config.id_column, "pre_dedup_unit_overlap_bucket"]
        for candidate in ("pre_dedup_unit_overlap_count", "pre_dedup_unit_size"):
            if candidate in train_df.columns:
                aux_columns.append(candidate)
        dedup_assignments = train_df[aux_columns].copy()
        oof_df = oof_df.merge(
            dedup_assignments,
            on=config.id_column,
            how="left",
            validate="one_to_one",
        )
        pre_dedup_artifacts = _persist_pre_dedup_metrics(
            oof_df=oof_df,
            reports_dir=reports_dir,
        )
    bucket_artifacts = None
    if overlap_analysis is not None:
        oof_df = oof_df.merge(
            overlap_analysis["train_assignments"],
            on=config.id_column,
            how="left",
            validate="one_to_one",
        )
        bucket_artifacts = _persist_bucket_artifacts(
            oof_df=oof_df,
            overlap_analysis=overlap_analysis,
            config=config,
            reports_dir=reports_dir,
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
        y_true=target_raw,
        y_pred=oof_predictions,
        fold_metrics=fold_metrics,
        plots_dir=plots_dir,
        type_label=config.type_label,
        experiment_name=config.experiment_name,
        aggregated_importance=aggregated_importance,
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
    if bucket_artifacts:
        artifacts["bucket_analysis"] = bucket_artifacts
    if pre_dedup_artifacts:
        artifacts["pre_dedup_overlap_metrics"] = pre_dedup_artifacts

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
    if config.feature_columns:
        ordered = list(dict.fromkeys(config.feature_columns))
        missing = [col for col in ordered if col not in df.columns]
        if missing:
            raise ExperimentError(f"指定された特徴量が存在しません: {', '.join(missing)}")
        return ordered
    reserved = {config.id_column, config.target_column}
    if config.group_column:
        reserved.add(config.group_column)
    return [col for col in df.columns if col not in reserved]


def _compute_sample_weight(config: ExperimentConfig, target: np.ndarray) -> np.ndarray | None:
    strategy = config.weight_strategy
    if not strategy:
        return None
    if strategy == "inverse_log_target":
        if not config.log_target:
            raise ExperimentError("weight_strategy='inverse_log_target' を利用するには log_target=True が必要です。")
        if np.any(target <= 0):
            raise ExperimentError("対数変換後のtargetに非正の値が含まれているためweightを計算できません。")
        weights = 1.0 / target
        if not np.all(np.isfinite(weights)):
            raise ExperimentError("weightの計算結果に無効値が含まれています。")
        return weights
    raise ExperimentError(f"未対応のweight_strategyが指定されました: {strategy}")


def _prepare_categorical_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_cols: Sequence[str]
) -> None:
    for col in categorical_cols:
        train_series = train_df[col].astype("string[python]")
        test_series = test_df[col].astype("string[python]")
        combined = pd.concat([train_series, test_series], ignore_index=True)
        categories = pd.unique(combined.dropna())
        dtype = CategoricalDtype(categories=categories, ordered=False)
        train_df[col] = train_series.astype(dtype)
        test_df[col] = test_series.astype(dtype)


def _prepare_overlap_analysis(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
    config: BucketAnalysisConfig,
) -> Dict[str, pd.DataFrame | None]:
    required = [config.unit_column]
    if config.use_building and config.building_column:
        required.append(config.building_column)
    missing = [col for col in required if col not in train_df.columns]
    if missing:
        raise ExperimentError(
            "bucket_analysis で指定された列が train データに存在しません: "
            + ", ".join(missing)
        )

    unit_counts = value_counts(train_df[config.unit_column])
    building_counts = (
        value_counts(train_df[config.building_column])
        if config.use_building and config.building_column
        else None
    )
    category_specs = get_category_specs(config.use_building)

    train_assignments = compute_overlap_features(
        df=train_df,
        id_column=id_column,
        unit_column=config.unit_column,
        building_column=config.building_column,
        unit_reference_counts=unit_counts,
        building_reference_counts=building_counts,
        subtract_self=True,
        include_label=True,
        use_building=config.use_building,
        category_specs=category_specs,
    )

    test_assignments = None
    if all(col in test_df.columns for col in required):
        test_assignments = compute_overlap_features(
            df=test_df,
            id_column=id_column,
            unit_column=config.unit_column,
            building_column=config.building_column,
            unit_reference_counts=unit_counts,
            building_reference_counts=building_counts,
            subtract_self=False,
            include_label=True,
            use_building=config.use_building,
            category_specs=category_specs,
        )

    return {
        "train_assignments": train_assignments,
        "test_assignments": test_assignments,
        "category_specs": category_specs,
    }


def _has_pre_dedup_overlap_columns(df: pd.DataFrame) -> bool:
    required = [
        "pre_dedup_unit_overlap_count",
        "pre_dedup_unit_overlap_bucket",
        "pre_dedup_overlap_category",
    ]
    return all(column in df.columns for column in required)


def _prepare_pre_dedup_overlap_analysis(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
) -> Dict[str, pd.DataFrame | Tuple[Tuple[str, str], ...]]:
    required = [
        "pre_dedup_unit_overlap_count",
        "pre_dedup_unit_overlap_bucket",
        "pre_dedup_overlap_category",
    ]
    missing = [col for col in required if col not in train_df.columns]
    if missing:
        raise ExperimentError(
            "pre_dedup overlap 列が不足しています: " + ", ".join(missing)
        )

    category_specs = get_category_specs(False)
    label_map = dict(category_specs)

    def build(df: pd.DataFrame) -> pd.DataFrame:
        subset = pd.DataFrame(
            {
                id_column: df[id_column],
                "unit_overlap_count": df["pre_dedup_unit_overlap_count"].astype(int),
                "unit_overlap_bucket": df["pre_dedup_unit_overlap_bucket"].astype("string"),
                "overlap_category": df["pre_dedup_overlap_category"].astype("string"),
            }
        )
        label_column = "pre_dedup_overlap_category_label"
        if label_column in df.columns:
            subset["overlap_category_label"] = df[label_column].astype("string")
        else:
            subset["overlap_category_label"] = subset["overlap_category"].map(label_map)
        return subset

    train_assignments = build(train_df)
    test_assignments = None
    if _has_pre_dedup_overlap_columns(test_df):
        test_assignments = build(test_df)

    return {
        "train_assignments": train_assignments,
        "test_assignments": test_assignments,
        "category_specs": category_specs,
    }


def _persist_bucket_artifacts(
    *,
    oof_df: pd.DataFrame,
    overlap_analysis: Dict[str, pd.DataFrame | None],
    config: ExperimentConfig,
    reports_dir: Path,
) -> Dict[str, str]:
    assert config.bucket_analysis is not None  # for mypy/static analyzers
    prefix = config.bucket_analysis.output_prefix

    train_assign_path = reports_dir / f"{prefix}_train_assignments.parquet"
    overlap_analysis["train_assignments"].to_parquet(train_assign_path, index=False)  # type: ignore[arg-type]

    test_assign_path = None
    test_assignments = overlap_analysis.get("test_assignments")
    if isinstance(test_assignments, pd.DataFrame):
        test_assign_path = reports_dir / f"{prefix}_test_assignments.parquet"
        test_assignments.to_parquet(test_assign_path, index=False)

    summary_df, metric_paths = _summarize_bucket_metrics(
        oof_df=oof_df,
        prefix=prefix,
        reports_dir=reports_dir,
        category_specs=overlap_analysis["category_specs"],  # type: ignore[arg-type]
    )
    metric_paths_rel = {
        "metrics_csv": str(metric_paths["csv"].relative_to(PROJECT_ROOT)),
        "metrics_json": str(metric_paths["json"].relative_to(PROJECT_ROOT)),
    }
    artifacts = {
        **metric_paths_rel,
        "train_assignments": str(train_assign_path.relative_to(PROJECT_ROOT)),
    }
    if test_assign_path is not None:
        artifacts["test_assignments"] = str(test_assign_path.relative_to(PROJECT_ROOT))
    artifacts["summary"] = summary_df.to_dict(orient="records")

    # 解析結果を console に軽く表示しておくとログ確認が楽になる
    print(f"[bucket] {prefix} summary:")
    for _, row in summary_df.iterrows():
        label = row["label"]
        count = int(row["data_id_count"])
        ratio = row["ratio"]
        mape = row["mape"]
        ratio_pct = ratio * 100 if pd.notna(ratio) else 0.0
        mape_text = f"{mape:.6f}" if pd.notna(mape) else "N/A"
        print(f"    - {label}: {count:,} rows ({ratio_pct:.2f}%) MAPE={mape_text}")

    return artifacts


def _persist_pre_dedup_metrics(
    *,
    oof_df: pd.DataFrame,
    reports_dir: Path,
) -> Dict[str, Any] | None:
    column = "pre_dedup_unit_overlap_bucket"
    if column not in oof_df.columns:
        return None
    summary_df = _pre_dedup_metrics_dataframe(
        bucket_keys=oof_df[column].to_numpy(),
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        specs=PRE_DEDUP_BUCKET_SPECS,
    )
    csv_path = reports_dir / "pre_dedup_unit_overlap_metrics.csv"
    json_path = reports_dir / "pre_dedup_unit_overlap_metrics.json"
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print("[pre_dedup] unit overlap summary:")
    for _, row in summary_df.iterrows():
        label = row["label"]
        count = int(row["data_id_count"])
        ratio = row["ratio"]
        mape = row["mape"]
        ratio_pct = ratio * 100 if pd.notna(ratio) else 0.0
        mape_text = f"{mape:.6f}" if pd.notna(mape) else "N/A"
        print(f"    - {label}: {count:,} rows ({ratio_pct:.2f}%) MAPE={mape_text}")
    return {
        "metrics_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "metrics_json": str(json_path.relative_to(PROJECT_ROOT)),
        "summary": summary_df.to_dict(orient="records"),
    }


def _summarize_bucket_metrics(
    *,
    oof_df: pd.DataFrame,
    prefix: str,
    reports_dir: Path,
    category_specs: Tuple[Tuple[str, str], ...],
) -> tuple[pd.DataFrame, Dict[str, Path]]:
    summary_df = _bucket_metrics_dataframe(
        bucket_keys=oof_df["overlap_category"].to_numpy(),
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        category_specs=category_specs,
    )
    csv_path = reports_dir / f"{prefix}_bucket_metrics.csv"
    json_path = reports_dir / f"{prefix}_bucket_metrics.json"
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    return summary_df, {"csv": csv_path, "json": json_path}


def _pre_dedup_metrics_dataframe(
    *,
    bucket_keys: np.ndarray,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    specs: Tuple[Tuple[str, str], ...],
) -> pd.DataFrame:
    bucket_series = pd.Series(bucket_keys, dtype="string").fillna("unknown")
    bucket_array = bucket_series.to_numpy(dtype=object)
    total = len(bucket_array)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rows: List[Dict[str, Any]] = []
    known_keys = [key for key, _ in specs]
    for key, label in specs:
        mask = bucket_array == key
        count = int(np.sum(mask))
        ratio = (count / total) if total else 0.0
        mape = _safe_mape(y_true_arr[mask], y_pred_arr[mask]) if count else None
        rows.append(
            {
                "bucket_key": key,
                "label": label,
                "data_id_count": count,
                "ratio": ratio,
                "mape": mape,
            }
        )
    if total:
        residual_mask = ~np.isin(bucket_array, known_keys, assume_unique=False)
        residual_count = int(np.sum(residual_mask))
        if residual_count:
            ratio = residual_count / total
            mape = _safe_mape(y_true_arr[residual_mask], y_pred_arr[residual_mask])
            rows.append(
                {
                    "bucket_key": "other",
                    "label": "その他",
                    "data_id_count": residual_count,
                    "ratio": ratio,
                    "mape": mape,
                }
            )
    return pd.DataFrame(rows)


def _bucket_metrics_dataframe(
    *,
    bucket_keys: np.ndarray,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    category_specs: Tuple[Tuple[str, str], ...],
) -> pd.DataFrame:
    total = len(bucket_keys)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rows: List[Dict[str, Any]] = []
    for key, label in category_specs:
        mask = bucket_keys == key
        count = int(np.sum(mask))
        ratio = (count / total) if total else 0.0
        mape = _safe_mape(y_true_arr[mask], y_pred_arr[mask]) if count else None
        rows.append(
            {
                "bucket_key": key,
                "label": label,
                "data_id_count": count,
                "ratio": ratio,
                "mape": mape,
            }
        )
    return pd.DataFrame(rows)


def _fold_progress_callback(fold_idx: int, period: int):
    if period <= 0:
        def _noop(env):
            return

        return _noop

    period = max(1, period)

    def _callback(env: lgb.callback.CallbackEnv) -> None:
        iteration = env.iteration + 1
        end_iteration = getattr(env, "end_iteration", env.iteration + 1)
        if iteration % period != 0 and iteration != end_iteration:
            return
        valid_metrics = [
            (eval_name, result)
            for data_name, eval_name, result, _ in env.evaluation_result_list
            if data_name == "valid"
        ]
        if not valid_metrics:
            return
        metrics_text = " ".join(f"{name}={value:.6f}" for name, value in valid_metrics)
        print(f"[fold {fold_idx}] iter={iteration}: {metrics_text}", flush=True)

    _callback.order = 10
    return _callback


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
    aggregated_importance: pd.DataFrame | None = None,
) -> Dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = plots_dir / "oof_scatter.png"
    log_scatter_path = plots_dir / "oof_scatter_loglog.png"
    residual_path = plots_dir / "residual_hist.png"
    fold_bar_path = plots_dir / "fold_mape.png"
    fi_path = plots_dir / "feature_importance.png"

    _plot_scatter(y_true, y_pred, scatter_path, type_label, experiment_name)
    _plot_scatter_loglog(y_true, y_pred, log_scatter_path, type_label, experiment_name)
    _plot_residuals(y_true, y_pred, residual_path, type_label, experiment_name)
    _plot_fold_mape(fold_metrics, fold_bar_path, type_label, experiment_name)
    fi_available = aggregated_importance is not None and not aggregated_importance.empty
    if fi_available:
        _plot_feature_importance(aggregated_importance, fi_path, type_label, experiment_name)

    return {
        "oof_scatter": scatter_path,
        "oof_scatter_loglog": log_scatter_path,
        "residual_hist": residual_path,
        "fold_mape": fold_bar_path,
        **({"feature_importance": fi_path} if fi_available else {}),
    }


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path, type_label: str, exp: str) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=10, alpha=0.3, edgecolor=None)
    all_vals = np.concatenate([y_true, y_pred])
    min_val = np.min(all_vals)
    max_val = np.max(all_vals)
    pad = max(1e-6, 0.05 * (max_val - min_val))
    lower = max(0.0, min_val - pad)
    upper = max_val + pad
    plt.plot([lower, upper], [lower, upper], color="red", linestyle="--", label="y=x")
    plt.xlabel("Actual money_room")
    plt.ylabel("Predicted money_room")
    plt.title(f"{type_label}::{exp} OOF actual vs. predicted")
    plt.legend()
    plt.tight_layout()
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_scatter_loglog(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    type_label: str,
    exp: str,
    min_percentile: float = 0.0,
    max_percentile: float = 100.0,
    eps: float = 1e-6,
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = (y_true > 0) & (y_pred > 0)
    if not np.any(mask):
        # 正の値がない場合は通常の散布図を保存する（単調増加の確認を優先）
        _plot_scatter(y_true, y_pred, path, type_label, exp)
        return

    y_true_pos = np.clip(y_true[mask], eps, None)
    y_pred_pos = np.clip(y_pred[mask], eps, None)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true_pos, y=y_pred_pos, s=10, alpha=0.3, edgecolor=None)

    all_vals = np.concatenate([y_true_pos, y_pred_pos])
    min_val = max(np.percentile(all_vals, min_percentile), eps)
    max_val = np.percentile(all_vals, max_percentile)
    if max_val <= min_val:
        max_val = min_val * 10

    pad_ratio = 0.15
    lower = max(min_val / (1 + pad_ratio), eps)
    upper = max_val * (1 + pad_ratio)

    plt.plot([lower, upper], [lower, upper], color="red", linestyle="--", label="y=x")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100))
    ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)
    plt.xlabel("Actual money_room (log scale)")
    plt.ylabel("Predicted money_room (log scale)")
    plt.title(f"{type_label}::{exp} OOF actual vs. predicted (log-log)")
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


def _plot_feature_importance(
    aggregated_importance: pd.DataFrame,
    path: Path,
    type_label: str,
    exp: str,
    top_n: int = 30,
) -> None:
    if aggregated_importance.empty:
        return

    plot_df = (
        aggregated_importance.sort_values("importance_mean", ascending=False)
        .head(top_n)
        .copy()
    )
    plot_df = plot_df.iloc[::-1]  # 水平棒グラフで上位を上に表示するため反転

    heights = max(4, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8, heights))
    stds = plot_df["importance_std"].fillna(0).to_numpy()
    error_kw = {"capsize": 3, "ecolor": "#333333", "elinewidth": 1}
    ax.barh(
        y=np.arange(len(plot_df)),
        width=plot_df["importance_mean"],
        xerr=stds,
        error_kw=error_kw,
        color="#4C72B0",
        alpha=0.9,
    )
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Gain importance (mean)")
    ax.set_ylabel("Feature")
    ax.set_title(f"{type_label}::{exp} feature importance (top {len(plot_df)})")
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


