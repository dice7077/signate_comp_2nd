# データパイプライン構成

決定的な変換スクリプトは `src/data_pipeline/steps/` に配置します。各ステップは以下を守ってください。

1. 入力は必ず `data/raw`・`data/external`・`data/interim` のいずれかから宣言済みの Parquet / CSV を読む。
2. 出力は次のステップに渡す前に `data/interim/<step_name>.parquet`（複数出力ならサブフォルダ）へ保存する。
3. ロジックが重複しそうな処理は `src/data_pipeline/utils/` に切り出し、ステップ側は薄く保つ。

ノートブックはアドホックな検証に集中させ、確定したフローは `scripts/` 配下の軽量 CLI エントリから実行します。

## 利用可能なステップ

- `assign_data_id`: Signate の `train.csv` / `test.csv` を読み込み、`data_id` 列を付与（train は `0..N-1`、test は既存の `id` を使用）し、`data/interim/00_01_assign_data_id/{train,test}.parquet` を生成。
- `drop_sparse_columns`: `00_01_assign_data_id` の出力を受け取り、train/test ともに欠損率 99% 超のカラムを削除して `data/interim/01_01_drop_sparse_columns/{train,test}.parquet` を生成。あわせて `target_ym`・`year_built` から算出する `years_old` と、`post1/post2`・`addr1_1/addr1_2` のゼロ埋め連結 (`post_all`, `addr_all`) も付与する。
- `split_signate_by_type`: `01_01_drop_sparse_columns` の出力を読み込み、`bukken_type`（1202=戸建、1302=マンション）ごとに分割し、`data/interim/01_02_split_by_type/` に 4 ファイルを書き出し。
- `adjust_mansion_unit_area`: `01_02_split_by_type` の成果物を受け取り、マンション行のみ `unit_area` / `house_area` をスコアリングして統合し、外れ値は `unit_area_min/max` をもとにクリップした `unit_house_area_adjusted` 列を追加して `data/interim/01_03_adjust_mansion_unit_area/{train,test}_{kodate,mansion}.parquet` を出力する（戸建ては透過的にコピー）。
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

## `adjust_mansion_unit_area` の変換ルール

マンション物件はデータ出典の都合で `unit_area`（住戸の専有面積）と `house_area`（同じく専有面積と思われる別表記）が並存しており、どちらか一方が桁ズレしたり極端に欠損していたりする。`adjust_mansion_unit_area` では次のルールで `unit_house_area_adjusted` を決定する。

1. **入力列の取得**  
   - `unit_area` / `house_area` / `room_count` / `unit_area_min` / `unit_area_max` を float 化。  
   - `unit_area_min/max` から得られる最小・最大レンジ（建物公式の想定値）を「期待値レンジ」として扱う。

2. **候補スコアリング**  
   - `unit_area` と `house_area` を候補とし、以下の要素で点数化。高得点の方を採用候補にする（引き分け時は `unit_area` 優先）。  
     - 候補同士の一致度（差分≦1㎡ or 1%: +3点、≦5㎡ or 5%: +1.5点、乖離が大きいほど減点）。  
     - 期待値レンジとの整合性（レンジ±2㎡以内: +4点、外れ距離に応じて最大 -6点）。レンジが欠損している場合は常識的な面積帯（15–150㎡で +2.5点 など）を緩やかに評価。  
     - 部屋数との相性（2部屋以上で 15㎡未満なら -3点、ワンルームで 12㎡未満なら +0.5 点など）。  
     - 絶対的な下限/上限チェック（6㎡未満 or 1,200㎡超は -6点の強い減点）。

3. **採用・クリップ・欠損の決定**  
   - 最高スコアが `MIN_VALID_SCORE(=1.0)` 以上なら、その値を採用し、出どころを `from_unit_area` or `from_house_area` として統計に記録。  
   - スコアが基準未満でも、期待値レンジがあれば `unit_area_min/max` の範囲へクリップ（無い方の境界しか無い場合はその値を超えないよう片側クリップ）。  
   - いずれの手段でも妥当な値が得られない場合は `unit_house_area_adjusted = NA` とし、以降の処理で明示的に欠損扱いにできるようにする。

4. **戸建ては透過**  
   - `train_kodate` / `test_kodate` は `split_signate_by_type` の出力をそのままコピーし、新列は生成しない（後続で戸建て専用ロジックを追加予定）。

ステップ完了時は `data/interim/01_03_adjust_mansion_unit_area/*.parquet` に `unit_house_area_adjusted` が追加されたマンションデータが保存され、`build_processed_datasets` でもこの列を参照する。manifest には「どの列から採用したか」「何件クリップしたか」などの統計を出力しているため、パイプライン確認時に異常値の混入を検知しやすい。

### 検証結果メモ

2026/01/01 時点で `01_02_split_by_type` と比較したところ、次のようにデータ品質が改善していることを確認済み。

- 欠損率: train で `unit_house_area_adjusted` の欠損は 0.004%（8 行）、test でも 0.005%（3 行）に抑制。`unit_area` の欠損 17.6% をほぼ完全に補完できている。
- 範囲外値の抑制: 調整後の最大値は 459㎡に収まり、`>500㎡` の行はゼロ。さらに `unit_area_min/max` のレンジ内に入る割合が train 80.4%→86.4%、test 79.7%→85.8% まで改善し、レンジ外超過は 12%台→8%台へ低減。
- 価格単価の健全化（train の `money_room` 利用）:
  - 5万円/㎡未満の極端な安値は 0.014%→0.001% (約 23 行→2 行) まで削減し、1万円/㎡未満は消滅。
  - 上振れ側も 11.3 百万円/㎡ → 8.5 百万円/㎡ に抑えられ、1百万円/㎡超の比率は 4.4%→4.9% へ微増したが、これは面積を正規化した結果に伴う自然な単価上昇（高級物件）で悪化とは判断せず。
- クリップ件数: train 4,403 行 (2.2%) / test 1,233 行 (2.1%) が `unit_area_min/max` にスナップ。既知の桁ズレ行（例: data_id 17795, 58924の 2,642㎡）はレンジ上限へ丸められていることを目視確認。
- 両列とも異常で復元不能だった行は train 8 / test 3 行のみで NA のまま保持（桁ズレでヒントなしのケース）。要件どおり無理に埋めていない。

これらの数値は以下のコマンドで再現可能:

```
.venv/bin/python - <<'PY'
# 01_02 vs 01_03 の列統計・価格単価比較
...
PY
```

（スクリプトは `src/data_pipeline/README.md` 作業コミットに添付されたものを参照。）

## Interim ディレクトリ命名ルール（エージェント向け）

- 現役ステップは `AA_BB_step_name` 形式。`AA` はブランチ/フェーズ、`BB` はそのフェーズ内の手続き順（いずれも 0 埋め 2 桁）。分岐したら `AA` を変えることで系列を追いやすくする。
- 使わなくなったステップ（または別バージョン）の成果物を残す場合は、フォルダ名を `99_ver<timestamp>_...` へ変更して保管する。`<timestamp>` はローカル時間の `YYYYMMDD_HHMM` 形式（例: `99_ver20251230_1708_split_by_type/`）とし、同日に複数回出しても衝突しないようにする。こうしておくと Deprecated 版が一目で分かり、後続処理でも誤って参照しにくい。
