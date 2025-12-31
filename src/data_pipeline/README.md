# データパイプライン構成

決定的な変換スクリプトは `src/data_pipeline/steps/` に配置します。各ステップは以下を守ってください。

1. 入力は必ず `data/raw`・`data/external`・`data/interim` のいずれかから宣言済みの Parquet / CSV を読む。
2. 出力は次のステップに渡す前に `data/interim/<step_name>.parquet`（複数出力ならサブフォルダ）へ保存する。
3. ロジックが重複しそうな処理は `src/data_pipeline/utils/` に切り出し、ステップ側は薄く保つ。

ノートブックはアドホックな検証に集中させ、確定したフローは `scripts/` 配下の軽量 CLI エントリから実行します。

## 利用可能なステップ

- `assign_data_id`: Signate の `train.csv` / `test.csv` を読み込み、`data_id` 列を付与（train は `0..N-1`、test は既存の `id` を使用）し、`data/interim/00_01_assign_data_id/{train,test}.parquet` を生成。
- `drop_sparse_columns`: `00_01_assign_data_id` の出力を受け取り、train/test ともに欠損率 99% 超のカラムを削除して `data/interim/01_01_drop_sparse_columns/{train,test}.parquet` を生成。
- `split_signate_by_type`: `01_01_drop_sparse_columns` の出力を読み込み、`bukken_type`（1202=戸建、1302=マンション）ごとに分割し、`data/interim/01_02_split_by_type/` に 4 ファイルを書き出し。
- `build_tag_id_features`: `00_01_assign_data_id` の出力から `unit_tag_id` / `building_tag_id` / `statuses` の `/` 区切りコードをユニーク化し、`tag_ids.parquet`（feature_name と tag_id の対応表）と、`unit_tag_<code>` / `building_tag_<code>` / `status_tag_<code>` を横持ちした `train_tag_ids.parquet`・`test_tag_ids.parquet` を `data/interim/02_01_build_tag_id_features/` に生成。
- `join_koji_price`: `00_01_assign_data_id` の出力から `lon/lat` を使用し、1.5km 以内にある住宅用途の公示地価ポイントを最近傍検索して特徴量化、`data/interim/03_01_join_koji_price/{train,test}.parquet` に保存。
- `join_land_price`: `00_01_assign_data_id` の出力を `lon/lat` ベースで地価調査ポイント（2019-2023年）と突合し、年度別の公示価格・用途地域コード・最短距離 (km) を `data/interim/05_01_join_land_price/{train,test}.parquet` に保存。
- `join_population_projection`: `01_01_drop_sparse_columns` の出力に対して 1km メッシュ人口予測（2025/2035/2045/2055年の `PTN` 値）を `lon` / `lat` から算出したメッシュ ID でマージし、`data/interim/04_01_join_population_projection/{train,test}_population_features.parquet` を生成（data_id ベースの枝データとして保持）。47 都道府県分の GeoJSON を一括して `data/interim/lookup_population_mesh/mesh1km_population.parquet` にキャッシュする。

登録済みステップを順番どおり実行するには:

```
python scripts/run_data_pipeline.py
```

実行せずレジストリ内容だけ確認するには:

```
python scripts/run_data_pipeline.py --list
```

## Interim ディレクトリ命名ルール（エージェント向け）

- 現役ステップは `AA_BB_step_name` 形式。`AA` はブランチ/フェーズ、`BB` はそのフェーズ内の手続き順（いずれも 0 埋め 2 桁）。分岐したら `AA` を変えることで系列を追いやすくする。
- 使わなくなったステップ（または別バージョン）の成果物を残す場合は、フォルダ名を `99_ver<timestamp>_...` へ変更して保管する。`<timestamp>` はローカル時間の `YYYYMMDD_HHMM` 形式（例: `99_ver20251230_1708_split_by_type/`）とし、同日に複数回出しても衝突しないようにする。こうしておくと Deprecated 版が一目で分かり、後続処理でも誤って参照しにくい。
