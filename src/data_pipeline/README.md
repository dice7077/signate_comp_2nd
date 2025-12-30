# Data Pipeline Layout

Place deterministic transformation scripts under `src/data_pipeline/steps/`.  Each
step should:

1. Read its declared Parquet/CSV inputs from `data/raw`, `data/external`, or
   `data/interim`.
2. Persist its Parquet output into `data/interim/<step_name>.parquet` (or a
   nested folder if multiple outputs) before feeding the next step.
3. Keep pure Python logic in `src/data_pipeline/utils/` to avoid duplication.

You can orchestrate the steps from lightweight CLI entrypoints inside
`scripts/`, keeping notebooks focused on ad-hoc EDA.
