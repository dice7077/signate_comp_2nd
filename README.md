# signate_comp_2nd

## Git hooks

This repo ships shared git hooks under `scripts/git-hooks`. They enforce basic
workflow rules—currently blocking direct commits to `main`.

Install them once per clone:

```
git config core.hooksPath scripts/git-hooks
```

After that, any `git commit` while on `main` will abort with a friendly message.
Create a feature branch (e.g. `git checkout -b feat/my-change`) and commit
there instead.

## 実験運用ルール

- すべての実験成果物は `experiments/<type>/<experiment_name>/` に保存する。
- 同名ディレクトリが存在する場合は `--overwrite` を明示しない限り上書きしない。
- `README` の「実験一覧」テーブルへ validation MAPE（小数第4位程度）と簡潔な説明を必ず追記し、公開スコアは「提出一覧」テーブルで提出単位に手動入力する。

### 学習・評価

5-fold クロスバリデーションと推論をまとめて実行する:

```
source .venv/bin/activate
python scripts/run_experiment.py \
  --type kodate \
  --version 0001_initial \
  --experiment-name 0002_feature_eng \
  --description "タグ特徴量追加 + LightGBM" \
  --folds 5 \
  --seed 42 \
  --num-boost-round 4000 \
  --early-stopping-rounds 200
```

主な成果物:

- `artifacts/predictions/oof_predictions.parquet` / `test_predictions.parquet`
- `artifacts/reports/metrics.json`（validation MAPE/MAE/RMSE）
- `artifacts/plots/*.png`
- `artifacts/models/fold_*.txt`
- `summary.json`（実験メタ情報）
- `code/` 配下に実行時の `run_experiment.py` と CLI パラメータをスナップショット

### 提出ファイルの生成

任意の kodate/mansion 推論結果を結合して提出 CSV を作る:

```
source .venv/bin/activate
python scripts/make_submission.py \
  --kodate-pred experiments/kodate/0001_initial/artifacts/predictions/test_predictions.parquet \
  --mansion-pred experiments/mansion/0001_initial/artifacts/predictions/test_predictions.parquet \
  --output submissions/submission_0001.csv
```

- サンプル提出は `data/raw/signate/sample_submit.csv`（存在しない場合は `sample_submission.csv`）を自動参照。
- デフォルトでは予測値を四捨五入してヘッダー無し CSV (`id,money_room`) を出力。
- 欠損 data_id がある場合は警告を表示する。

### 実験一覧（戸建て）

| Dataset Version | Experiment   | Description                | Val MAPE |
|-----------------|--------------|----------------------------|----------|
| 0001_initial    | 0001_initial | LightGBM baseline (kodate) | 0.2615   |
| 0003_school_ele_name | 0004_few_features | 少数特徴 + logターゲットLGBM | 0.1696   |

### 実験一覧（マンション）

| Dataset Version | Experiment   | Description                 | Val MAPE |
|-----------------|--------------|-----------------------------|----------|
| 0001_initial    | 0001_initial | LightGBM baseline (mansion) | 0.2371   |

### 提出一覧

| Kodate Experiment | Mansion Experiment | Submission Path                                       | Public Score | Notes                |
|-------------------|--------------------|-------------------------------------------------------|--------------|----------------------|
| 0001_initial      | 0001_initial       | `submissions/submission_kodate0001_mansion0001.csv`   | -            | 初回ベースライン提出 |
