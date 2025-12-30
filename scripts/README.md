# Orchestration Scripts

Use this folder for thin CLI entrypoints (e.g. `python scripts/build_features.py`).
Scripts should only parse args/environment, call functions inside
`src/data_pipeline/steps`, and handle logging.
