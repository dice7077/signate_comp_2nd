# Data Directory Structure

- `raw/`: Original competition data (unzipped CSV etc., never modified).
- `external/`: Public/external datasets kept separate from competition drops.
- `interim/`: Step-by-step intermediate tables (store Parquet artifacts here).
- `processed/`: Final feature matrices ready for training/inference.
- `metadata/`: Schema info, data dictionaries, and column mapping docs.
- `submissions/`: Out-of-repo backups of generated submission CSVs.

These folders are ignored by Git—only commit lightweight descriptors (e.g., this README).

## 現行DAG

```
raw/signate/{train,test}.csv
        └─ 00_01_assign_data_id ──┬─▶ 02_01_build_tag_id_features/{train,test}_tag_ids.parquet
                                   ├─▶ 03_01_join_koji_price/{train,test}.parquet
                                   └─▶ 01_01_drop_sparse_columns/{train,test}.parquet
                                             ├─▶ 01_02_split_by_type/{train,test}_{kodate,mansion}.parquet
                                             │        └─▶ 01_03_adjust_mansion_unit_area/{train,test}_{kodate,mansion}.parquet
                                             └─▶ 04_01_join_population_projection/{train,test}_population_features.parquet

raw/population/mesh1km_2024 → lookup_population_mesh/mesh1km_population.parquet
                                         │
                                         └─ (参照) join_population_projection
```

| ステップ | 入力 | 主な処理 | 出力アーティファクト |
| --- | --- | --- | --- |
| `assign_data_id` | `raw/signate/train.csv`, `raw/signate/test.csv` | trainへ連番`data_id`、testへ既存`id`を付与 | `interim/00_01_assign_data_id/{train,test}.parquet` |
| `join_koji_price` | `00_01_assign_data_id` | lon/latから1.5km圏内で最も近い公示価格点を年次ごとに探索し、未ヒットの場合は3kmまで拡張して用途区分/現況/構造・2018-2023年価格・各年の用途コード/距離(km)を付与 | `interim/03_01_join_koji_price/{train,test}.parquet` |
| `build_tag_id_features` | `00_01_assign_data_id` | スラッシュ区切りタグを展開し、タグ辞書と one-hot 行列を作成 | `interim/02_01_build_tag_id_features/tag_ids.parquet`, `train_tag_ids.parquet`, `test_tag_ids.parquet` |
| `drop_sparse_columns` | `00_01_assign_data_id` | 欠損率99%以上の列（13列）を除去 | `interim/01_01_drop_sparse_columns/{train,test}.parquet` |
| `join_population_projection` | `01_01_drop_sparse_columns`, `lookup_population_mesh/mesh1km_population.parquet` | lon/latから1kmメッシュを求め、将来人口(2025-2055)を data_id 枝として出力 | `interim/04_01_join_population_projection/{train,test}_population_features.parquet` |
| `split_signate_by_type` | `01_01_drop_sparse_columns` | `bukken_type`（1202=戸建, 1302=マンション）別に分割 | `interim/01_02_split_by_type/{train,test}_{kodate,mansion}.parquet` |
| `adjust_mansion_unit_area` | `01_02_split_by_type` | マンション行のみ `unit_area` と `house_area` を統合し、外れ値を補正した `unit_house_area_adjusted` を付与 | `interim/01_03_adjust_mansion_unit_area/{train,test}_{kodate,mansion}.parquet` |

補足:

- `lookup_population_mesh/mesh1km_population.parquet` は `join_population_projection` 実行時に GeoJSON 群から自動生成・キャッシュされる。
- 旧版アーティファクトは `interim/99_verYYYYMMDD_HHMM_*` に退避する慣習を守ることで、DAG 表示と実行フローの順番が一致する。
