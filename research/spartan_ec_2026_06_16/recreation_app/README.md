# FTIR calibration — local recreation

A local, offline recreation of the UC Davis FTIR calibration Shiny tool, so you can hook up
your own spectra and build/refine/apply OC & EC calibrations **without** the finicky online tool.

Backed by our validated PLS pipeline (`sklearn PLSRegression(scale=False)` == the tool's R
`kernelpls` to ~1e-10), so calibrations match the tool's math.

## Run
```
python research/spartan_ec_2026_06_16/recreation_app/app.py
# → http://127.0.0.1:5057
```
(`flask`, `pandas`, `numpy`, `scikit-learn` — all in the anaconda env.)

## The local workflow (recursive: build → refine → apply)
1. **Pick a dataset** (top-left dropdown):
   - **EC / OC tool training set** — the tool's *exact* X/Y (RDS-extracted, n=906). Coefficients
     built here are bit-identical to the tool.
   - **Your spectra** — see "Hook up your spectra" below.
2. **Choose components** — type a number, or **click the RMSEP line**.
3. **Refine Sample List** — opens the table sorted by |residual|; click rows to mark, then
   **Rerun as filtered** → the backend drops them and refits live (R²/RMSE update).
4. **Inspect** — click any cross-plot point → filter detail in the sidebar + **View spectrum**.
5. **Export Coefficients** — downloads the tool-format CSV (Wavenumber=0 intercept + per-wavenumber `b`).
6. **Apply to →** — pick another spectra set (target) and **Apply** → predicts the current
   calibration on it → table + **Download predictions** CSV. This is the recursive step:
   build a biomass calibration, then apply it to Addis/Adama/other spectra.

## Hook up your spectra
- **As a calibration dataset** (you want to *build* a calibration from it): drop an offline
  **"Download Spectra"** CSV (metadata + wavenumber columns) into **`datasets/`**. It appears as
  `<name> · EC` and `<name> · OC` — measured loading is joined by **Site + date** from the IMPROVE
  reference. *Note:* that join is the approximate `improve_valid_cleaned` Y (not the tool's exact
  matched Y), so coefficients won't be bit-identical to the tool — fine for local exploration.
- **As an apply target only** (no measured Y, e.g. **Addis/SPARTAN**): drop the spectra CSV into
  **`targets/`**. It shows up in the "Apply to" dropdown; apply any calibration to get predictions.

## API (for scripting)
`POST /api/calibrate {dataset,ncomp,removed,rmsep}` · `POST /api/spectrum {dataset,id}` ·
`POST /api/export {dataset,ncomp,removed}` · `POST /api/apply {dataset,ncomp,removed,target}` ·
`POST /api/apply_export {...}` · `GET /api/datasets` · `GET /api/targets`

## Limits
- Dev server, localhost only. RMSEP sweep is downsampled (every 6th wavenumber) for responsiveness;
  coefficients/predictions use full resolution.
- Filter **Status / QV comments** are DB-only (not in the offline export), so the inspect panel
  shows what's in the data (Site, date, ids, measured/predicted/resid) + the spectrum.
- `datasets/*.csv`, `targets/*.csv`, and large RDS/X files are gitignored.
