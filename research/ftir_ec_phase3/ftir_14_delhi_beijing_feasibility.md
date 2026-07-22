# Delhi/Beijing in PLS score space — blocked on a data pull (2026-07-17)

Satoshi's suggestion from the July 2026 meeting: instead of comparing sites in the 2-D
EC/Fabs plane, project Delhi (INDH) and Beijing (CHTS) samples into the ~20-component
score space of the EC calibration model and measure how close they actually sit to
Addis (ETAD) — samples can share an EC range while being compositionally far apart.

## Why it can't run yet

All the machinery exists (`pls_transfer.project_scores`, `mahalanobis_distance_squared`,
`pairwise_score_distance_squared` — already exercised in ftir_09), but **no SPARTAN FTIR
spectra other than ETAD exist locally**:

- `FTIR/local_db/tables/ftir_catalog.csv` (169,566 analyses, 182 sites) contains only
  IMPROVE sites — checked 2026-07-17: zero rows for ETAD, INDH, CHTS, or USPA.
- The ETAD spectra came from a dedicated pull
  (`…/DAVIS/ETAD FTIR/etad_lots_query.sql` + the `pull_*.ps1` scripts in
  `FTIR/local_db/`); the equivalent pull was never run for the other SPARTAN sites.

## What to request / run

A repeat of the ETAD pull with the site filter widened to INDH, CHTS, **ETBI (Bishoftu —
now has 26 HIPS filters, see `output/tables/context/etad_etbi_hips_summary.csv`)**, and
USPA as a low-concentration control:

1. Adapt `etad_lots_query.sql` → site codes INDH/CHTS/ETBI/USPA, same columns.
   Exact target shapes, so the local pipeline runs unmodified:
   - `<SITE>_FTIR_spectra.csv`: `SampleAnalysisId, MediaId, <2722 wavenumber columns>`
     (3998.42 → ~500 cm⁻¹ descending, identical grid to `ETAD_FTIR_spectra.csv`);
   - `<SITE>_metadata.csv`: `SiteCode, Description, Latitude, Longitude, MediaId,
     ExternalFilterId, ExternalFilterType, SamplingStartDate, SamplingEndDate,
     MassCollectedOnFilter_ug, SampleVolume_m3`.
2. Export as `<SITE>_FTIR_spectra.csv` + `<SITE>_metadata.csv` next to the ETAD files.
3. Then the analysis is ~30 lines with existing helpers: project all sites into the
   deployed EC model's score space (and the ftir_11 lowest-OCEC model's), report
   Mahalanobis distance and Q residual distributions per site vs the Addis cloud.

Per the meeting's sequencing note (Satoshi/Ann): Delhi and Beijing should be used as
**test sets** once the Addis approach settles — not folded into calibration cohorts —
unless charcoal influence there is established independently.
