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

## Available steps

- `assign_data_id`: loads the raw Signate `train.csv` / `test.csv`, adds a
  `data_id` column (train gets `0..N-1`, test reuses the provided `id`), and
  writes `data/interim/00_assign_data_id/{train,test}.parquet`.
- `split_signate_by_type`: reads the `00_assign_data_id` outputs, splits each by
  `bukken_type` (1202 = kodate, 1302 = mansion), and persists the four tables to
  `data/interim/01_split_by_type/`.

Run all registered steps in order:

```
python scripts/run_data_pipeline.py
```

See the registry without executing anything:

```
python scripts/run_data_pipeline.py --list
```
