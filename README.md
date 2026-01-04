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

- すべての実験成果物は `experiments/<type>/<experiment_name>/` に保存する。same_unit_id 系列は `experiments/kodate_same_unit_id/<experiment_name>/` および `experiments/mansion_same_unit_id/<experiment_name>/` を利用する。
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

設定ファイル（`ExperimentConfig` JSON/YAML）経由で同じことを行う場合は `--config` を使う。CLI引数よりファイル内容が優先され、`--overwrite` などのフラグだけ任意に追加できる。

```
source .venv/bin/activate
python scripts/run_experiment.py \
  --config experiments/mansion/0004_add_tags/config.json \
  --overwrite
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
| 0004_add_many_feature_baseline | add_many_feature_baseline | 多特徴ベースライン + logターゲットLGBM | 0.1999   |
| 0005_targetyear_geo_features | 0005_targetyear_geo_features | target-year koji/land + mesh人口4期 (lr=0.1, log target) | 0.1475   |
| 0006_add_tags   | 0006_add_tags | target-year geo + mesh人口4期 + 指定unit/buildingタグone-hot (lr=0.1, log target) | 0.1447   |
| 0006_add_tags   | 0010_inverse_weights | 0006構成 + log(money_room)逆数weightでL2学習をL1相当に補正 | 0.1431   |

### 実験一覧（マンション）

| Dataset Version | Experiment   | Description                 | Val MAPE |
|-----------------|--------------|-----------------------------|----------|
| 0001_initial    | 0001_initial | LightGBM baseline (mansion) | 0.2371   |
| 0002_few_features | 0002_few_features | 少数特徴 + logターゲットLGBM (lr=0.1) | 0.1246   |
| 0003_targetyear_geo_features | 0003_targetyear_geo_features | target-year koji/land + mesh人口4期 (lr=0.1, log target) | 0.1127   |
| 0004_add_tags   | 0004_add_tags | target-year geo + mesh人口4期 + 指定unit/buildingタグone-hot (lr=0.1, log target) | 0.1098   |
| 0004_add_tags   | 0006_add_unit_house_area_adjusted | 0004構成 + unit/house areaを維持しつつ unit_house_area_adjusted を追加 | 0.1097   |
| 0004_add_tags   | 0007_inverse_weights | 0006構成 + log(money_room)逆数weightでL2学習をL1相当に補正 | 0.1088   |

### 実験一覧（戸建て: same_unit_id）

| Dataset Version | Experiment | Description | Val MAPE |
|-----------------|------------|-------------|----------|
| 0007_same_unit_id | 0007_same_unit_id | 戸建てタグ + 同一unit履歴ログ特徴 (log target, group by unit_id) | 0.0694 |
| 0008_test_202207only | 0008_test_202207only | 0007ベース + test unit抽出をtrain target_ym=202207重複に限定 | 0.0691 |

### 実験一覧（マンション: same_unit_id）

| Dataset Version | Experiment | Description | Val MAPE |
|-----------------|------------|-------------|----------|
| 0006_same_unit_id | 0006_same_unit_id | マンションタグ + 同一unit履歴ログ特徴 (lr=0.03, log target, group by unit_id) | 0.0754 |

### 提出一覧

| Kodate Experiment | Mansion Experiment | Submission Path                                       | Public Score | Notes                |
|-------------------|--------------------|-------------------------------------------------------|--------------|----------------------|
| 0001_initial      | 0001_initial       | `submissions/submission_kodate0001_mansion0001.csv`   | -            | 初回ベースライン提出 |
| 0004_few_features | 0002_few_features  | `submissions/submission_kodate0004_mansion0002.csv`   | 17.4984      | log target LGBM few features |
| 0005_targetyear_geo_features | 0003_targetyear_geo_features | `submissions/submission_kodate0005_mansion0003.csv`   | 15.4122      | target-year geo + mesh人口4期 |
| 0006_add_tags     | 0004_add_tags      | `submissions/submission_kodate0006_mansion0004.csv`   | 15.0005      | target-year geo + mesh人口4期 + 指定タグone-hot |
| 0006_add_tags     | 0005_adjust_unit_house_area | `submissions/submission_kodate0006_mansion0005.csv`   | 15.1726      | target-year geo + mesh人口4期 + 指定タグone-hot + unit面積調整 |
| 0007_same_unit_id (mix) | 0006_same_unit_id (mix) | `submissions/submission_kodate0007mix_mansion0006mix.csv` | 14.9789      | same_unit_id系mix推論（log target, group by unit_id） |
| 0006_add_tags     | 0005_adjust_unit_house_area + 0006_same_unit_id | `submissions/submission_kodate0006_mansion0005_sameunit0006.csv` | 15.2263      | mansion 0005 + same_unit_id 0006 ミックス（log target, group by unit_id） |
| 0010_inverse_weights | 0007_inverse_weights | `submissions/submission_kodate0010_mansion0007.csv`   | 14.8860      | inverse weight補正組合せ（round有り） |
