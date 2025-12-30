# Cursor エージェント向けルール

- `data/interim` のステップフォルダは `STEP_REGISTRY` の順番と一致するよう、常に `NN_step_name` という命名パターンを守ること。
- 廃止予定・検証用の成果物を残す場合はフォルダ名を `99_ver<timestamp>_...` に変更し（例: `99_ver20251230_1708_split_by_type/`）、人間や自動処理が容易にスキップできるようにする。
- 決定的なパイプラインステップが `data/interim` に Parquet を出力するたび、同じディレクトリ構成を踏襲した検査ノートブックを `notebooks/data/...` に用意し、形状・スキーマ・先頭データを確認できるようにする。
- 検査ノートブックでは列を行に転置した状態でおおよそ 20 行を表示し、`pd.options.display.max_rows` を少なくとも 400 に設定して縦方向のトリミングを避ける。
- Cursor 上で自動化を実行する際は `.venv` を作成・有効化し（`python3 -m venv .venv` → `source .venv/bin/activate` → `pip install -r requirements.txt` with `required_permissions: ['all']`）、以降のパイプライン実行では常に `.venv/bin/python` を使ってモジュール不足を防ぐ。この際、sandboxを使用すると失敗するので、sandbox外で行うこと。


