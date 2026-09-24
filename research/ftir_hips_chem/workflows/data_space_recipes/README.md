# Data Space recipe: unified filter Parquet

`unified_filter_parquet.manifest.json` declares the source, output, version, and validation contract. `unified_filter_parquet.py` performs the trusted local conversion. The recipe does not remove rows or apply scientific exclusions; those remain flagged through the existing AETH analysis helpers.

Recipes that talk to Zoer read the server address from `ZOER_URL` in the repo-root `.env` (copy `.env.example`). From the repository root:

```sh
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/data_space_recipes/unified_filter_parquet.py
```

The output and its separate provenance JSON are written under `research/ftir_hips_chem/output/tables/` and are gitignored. Use `--data-root` to select another AETH data root and `--output` for an alternate output file. The script reads the new Parquet back and checks exact DataFrame equality before replacing an existing output.

The Parquet can be uploaded through Zoer's Datasets view, rebuilt there, and added to a Data Space. The manifest alone does not upload data or authorize a remote server to execute Python.

## Complete IMPROVE portal exports

`improve_raw_parquet.manifest.json` pins the SHA-256 hashes of the two 2026-04-22 portal workbooks. `improve_raw_parquet.py` streams both Data sheets to typed, compressed Parquet, retains the source `Date` strings, and adds `sample_date` for filtering. It verifies seven identical metadata sheets across the workbooks and stores them once; the two distinct Parameters sheets are separate. Empty numeric cells become null. No scientific exclusions or joins are applied.

```sh
uv run python research/ftir_hips_chem/workflows/data_space_recipes/improve_raw_parquet.py --source-dir "<path to Davis Data/Improve>"
```

The ignored `output/tables/zoer_sources/improve_raw/` directory contains 11 Parquet tables and a provenance JSON. Their contents are hosted in Zoer (Zoer dataset `768e3bb0-d2c8-45ea-8b7f-a50fad328cf9`). The original Excel binaries remain in Drive. The hosted cleaned optics table (Zoer dataset `9c6f0a21-c0c6-4955-9bc1-b40077ca9622`) is a filtered and joined derivative; do not add its rows to source counts as independent observations. The AETH Data Space catalog's `catalog_lineage` table records this relationship and the SPARTAN HIPS/site lookup links.

## IMPROVE Aerosol TOR archive and hosted join audit

`ingest_improve_tor_archive.py` verifies the 14-file `Davis Data/improve_tor_fractions` inventory against the matching generated repository copy and source manifest, hosts nine raw reports as checksum-verified downloadable archives (one split into ordered parts), and transforms two derived CSVs to typed Zstd Parquet. The hosted TOR dataset (Zoer dataset `bf0bc7b0-6a0b-4948-b6cc-9428f4572e87`) has 143,948 source sample rows, 13,031 derived FTIR-pool matches, and a 14-row source manifest. `improve_tor_hosted.receipt.json` records file IDs and hashes. The Drive provider did not complete full-byte reads during the initial ingest. A later independent check matched **all nine mounted raw files** byte-for-byte with the repository copy using `verify_improve_tor_drive.py`; its [report](improve_tor_drive_verification.json) records five top-level mount read timeouts. Those five files were separately fetched through the authenticated Google Drive connector and all five hashes matched; see [connector report](improve_tor_connector_verification.json). The connector listed no children under the Drive `raw/` folder at the time, so remote cloud sync of those nine files remains unverified. The Zoer-hosted archive bytes passed their own readback hashes. `link_improve_tor_space.py` adds the ready dataset to the existing Space with revision-checked readback.

`audit_hosted_joins.js` runs read-only SQL against temporary copies of **hosted** DuckDB artifacts from a Zoer backend pod. Its checked aggregate output is `hosted_join_audit_2026-09-23.jsonl`, interpreted in the join and overlap guide kept with the Zoer project notes on Drive (`My Drive/Projects/Zoer/output/aethmodular-data-audits-2026-09-23/`). The `catalog_lineage` table and Space description expose the verified shared full FTIR/HIPS filter IDs, shortened ChemSpec base IDs, method/media distinction, sample windows and release overlaps. `clarify_shared_filters.py` saves the concise Space-level caveat with a revision check and readback.

After a hosted dataset changes, run `uv run python research/ftir_hips_chem/workflows/data_space_recipes/build_zoer_hosted_catalog.py` and `uv run python research/ftir_hips_chem/workflows/data_space_recipes/publish_zoer_catalog.py`. The generator verifies local Parquet hashes, downloads the three hosted SPARTAN source files to verify their hashes, and extends the 19-dataset baseline with the Davis ingest receipt. It does not place local absolute Drive paths in the published catalog.

## Davis Data Drive source families

The 2026-09-23 inventory, kept with the Zoer project notes on Drive (`My Drive/Projects/Zoer/output/aethmodular-data-audits-2026-09-23/`), maps all 136 audited data-like files in the named Davis Data folders. Three `Spartan` originals and two transformed `Improve` workbooks were already hosted. `ingest_davis_drive.py` copies 113 smaller originals into family datasets; its `source_manifest` records original relative paths and hashes. The EC/HIPS working SQLite is preserved as a downloadable ZIP member, and its four tables are separately converted to Parquet because the SQLite columns mix storage classes. App ZIPs, the Han PDF and PurpleAir request are retained as opaque ZIP members. The Drive originals remain untouched.

`ingest_davis_large.py` converts five oversized FTIR CSVs into 17 ZSTD Parquet files, checking parsed row counts, and preserves 12 RDS models and the offline IMPROVE SQLite mirror as 70 numbered ZIP parts. The split scan and spectral tables include `source_row` for the original row order. To reconstruct an RDS or SQLite on another device, download the ZIP files listed for its source path in the [receipt](davis_drive_hosted.receipt.json), extract each `partNNN.bin`, concatenate in ascending part order, and compare the result with the receipt's original SHA-256. `verify_davis_ingest.py` performs this check against the hosted downloads; it also verifies all 113 smaller originals and 17 Parquet part hashes. The large CSV originals themselves remain on Drive; the hosted Parquet is a typed, row-count-checked analytical representation.

`ingest_davis_notes.py` indexes 11 markdown archive notes and the text of the Google Doc pointer `Notes about Data.gdoc`. `link_davis_space.py` adds the 11 hosted datasets to the existing AETH Data Space with a revision check and readback. The Data Space's manifest URL is the stable entry point for users and agents, and the `catalog_*` tables provide the source map.

## Kyan corrected hourly aethalometry

`kyan_hourly_parquet.manifest.json` and `kyan_hourly_parquet.py` preserve the four Kyan hourly CSVs in one typed, compressed, provisional source view. Pass the mounted `Aethalometry Data/Kyan Data` directory explicitly:

```sh
uv run python research/ftir_hips_chem/workflows/data_space_recipes/kyan_hourly_parquet.py --source-dir "<path to Aethalometry Data/Kyan Data>"
```

The output and SHA-256 provenance receipt are under the ignored `research/ftir_hips_chem/output/tables/kyan_aethalometry/`. The recipe retains the ten Central hours with both `og` and `api_filled` rows, and validates an exact Parquet round trip. It does not choose a preferred source or validate filter matching. The provisional output is hosted in Zoer (Zoer dataset `d76ecec7-6caf-44e2-a551-92586b68223a`); the Kyan audit in the Zoer project notes on Drive (`My Drive/Projects/Zoer/output/aethmodular-data-audits-2026-09-23/`) covers overlap and clock caveats.

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
