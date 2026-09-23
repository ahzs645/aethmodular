# Data Space recipe: unified filter Parquet

`unified_filter_parquet.manifest.json` declares the source, output, version, and validation contract. `unified_filter_parquet.py` performs the trusted local conversion. The recipe does not remove rows or apply scientific exclusions; those remain flagged through the existing AETH analysis helpers.

From the repository root:

```sh
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/data_space_recipes/unified_filter_parquet.py
```

The output and its separate provenance JSON are written under `research/ftir_hips_chem/output/tables/` and are gitignored. Use `--data-root` to select another AETH data root and `--output` for an alternate output file. The script reads the new Parquet back and checks exact DataFrame equality before replacing an existing output.

The Parquet can be uploaded through Zoer's Datasets view, rebuilt there, and added to a Data Space. The manifest alone does not upload data or authorize a remote server to execute Python.

`check_zoer_dataset.py` uses Python to read the local Parquet baseline and list the datasets on the hosted Zoer instance:

```sh
uv run python research/ftir_hips_chem/workflows/data_space_recipes/check_zoer_dataset.py --list
```

After uploading and rebuilding the Parquet in Zoer, pass its dataset ID to compare row count, ETAD count, and distinct site count through Zoer's read-only SQL API:

```sh
uv run python research/ftir_hips_chem/workflows/data_space_recipes/check_zoer_dataset.py --dataset-id DATASET_ID
```

`hosted_lab_check.ipynb` is a small companion for the separately hosted Zoer
JupyterLab. Upload only the notebook to the lab: the Hub mounts Zoer's hosted
Datasets collection read-only, and the notebook resolves the AETH Parquet from
that mount. It can also run locally against the recipe output. It reads one
Parquet column with pandas and runs an aggregate directly against the file
with DuckDB, checking the version 1 baseline of 44,493 rows, 11,374 ETAD
rows and four sites. Rerun it in the hosted lab to confirm that lab's mount.
