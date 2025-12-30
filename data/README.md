# Data Directory Structure

- `raw/`: Original competition data (unzipped CSV etc., never modified).
- `external/`: Public/external datasets kept separate from competition drops.
- `interim/`: Step-by-step intermediate tables (store Parquet artifacts here).
- `processed/`: Final feature matrices ready for training/inference.
- `metadata/`: Schema info, data dictionaries, and column mapping docs.
- `submissions/`: Out-of-repo backups of generated submission CSVs.

These folders are ignored by Git—only commit lightweight descriptors (e.g., this README).
