# Charcoal FTIR Reference Data

Workspace for the charcoal / biochar FTIR continuation of the SPARTAN EC
investigation. The goal is to collect open reference spectra that can help test
how charcoal, biochar, and highly carbonized material appear under FTIR methods.

## What is tracked

- `sources.json` records the open datasets to pull.
- `scripts/pull_reference_spectra.py` downloads the raw files and verifies
  hashes when the source repository exposes them.
- `data/raw/`, `data/interim/`, `data/processed/`, and `output/` are local-only
  working directories. The raw data should be regenerated with the downloader,
  not committed.

## Data sources

| Source id | Repository | Contents |
|---|---|---|
| `minatre_dryad_charcoal_temperature` | Dryad, DOI `10.5061/dryad.cnp5hqcbj` | Charcoal transmission FTIR spectra, 950-3500 cm^-1; reference chars at 200-800 deg C plus combustion-facility chars. |
| `gosling_figshare_charcoal_temperature` | Figshare article `5979544` | Modern reference charcoal and ancient charcoal FTIR spectra. |
| `maezumi_zenodo_charcoal_analog` | Zenodo record `5156747` | Charcoal FTIR reference data plus R analogue-matching code. |
| `mccall_acs_figshare_biochar_stability` | ACS/Figshare article `28891269` | ATR-FTIR biochar spectra and H:C / O:C stability-ratio data across feedstocks and HTTs. |
| `barbosa_figshare_sewage_sludge_biochar` | Figshare article `24167544` | Sewage-sludge biochar raw data; downloader pulls `FTIR.xlsx`. |
| `tigalana_mendeley_rice_straw_biochar_pb` | Mendeley dataset `zcshgj7kyy` | FTIR CSV spectra for pristine, KOH-modified, and Pb-loaded rice-straw biochar. |
| `zeba_dataone_pine_pyom_ageing` | KNB/ESS-DIVE package `ess-dive-6205497946b0d9d-20230407T153732819220` | Pine-wood PyOM data, including FTIR relative peak heights and elemental data. |
| `epa_direct_soil_ftir` | EPA/Data.gov direct workbook | Soil FTIR workbook; useful background fingerprints, not charcoal-only. |
| `mnhn_zenodo_atr_ftir_database` | Zenodo record `10658337` | ATR-FTIR reference library with Bruker OPUS, JCAMP-DX, and sample metadata. |
| `pereira_ufscar_binchotan_thesis` | UFSCar DSpace item `20.500.14289/20327` | Binchotan thesis PDF plus extracted text for material characterization tables and FTIR/XRD discussion. |
| `kwon_woodj_white_charcoal_quality` | Journal of Korean Wood Science & Technology article `10.5658/WOOD.2018.46.5.527` | Downloadable white-charcoal quality tables: proximate analysis, density/pH/EMC, calorific value, hardness, and refinement degree. |

See `leads.md` for KBr-pellet papers, figure/supplement-driven targets, and
reference-library comparisons that are not yet automated pulls.

## Pull the data

From this folder:

```bash
/opt/anaconda3/bin/python3.13 scripts/pull_reference_spectra.py --dry-run
/opt/anaconda3/bin/python3.13 scripts/pull_reference_spectra.py
```

To pull only one source:

```bash
/opt/anaconda3/bin/python3.13 scripts/pull_reference_spectra.py \
  --source mccall_acs_figshare_biochar_stability
```

Downloaded files land under `data/raw/<source_id>/`. The script also writes
`data/raw/_download_manifest.json` plus per-source `_source_metadata.json`
files; all of these are intentionally ignored by git.

## Notes for analysis

- Treat ATR and transmission/ZnSe spectra as separate domains until an explicit
  correction or normalization strategy is chosen.
- Keep KBr-pellet charcoal experiments in a separate local subfolder under
  `data/raw/` or `data/interim/`; do not commit instrument exports.
- If a source blocks command-line downloads, use `--dry-run` to print the
  source file URLs, download manually from the repository page, and keep the
  files in the same `data/raw/<source_id>/` location.
- Dryad may block command-line file requests with HTTP 403. In that case, open
  the Dryad repository page in a browser, use **Download full dataset**, then
  move and extract the ZIP into `data/raw/minatre_dryad_charcoal_temperature/`.
