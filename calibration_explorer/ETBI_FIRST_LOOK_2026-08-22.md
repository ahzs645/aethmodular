# ETBI (Bishoftu) — first out-of-country evaluation, 2026-08-22

> **Superseded (same week):** this doc's headline — that the intercept is
> "Addis-specific" — was revised twice. The five-site table
> (`CROSS_SITE_EVALUATION_2026-08-22.md`) made it a two-city ordering, and
> the York/EIV re-fit (`research/ftir_ec_phase3/OFFSET_ADJUDICATION_2026-08-23.md`)
> is the current statement: Addis certain, Delhi suggestive, slope anomalies
> at Delhi/Pasadena. The ETBI transfer result itself (0.93x, near-zero
> intercept) stands.

The pre-registered test (quartz_tor_campaign one-pager): do the Addis
calibrations transfer to a second Ethiopian SPARTAN site they have never seen?

## Provenance (fully reproducible)

- Spectra pulled directly from AQRC-SQL `Networks_1_0` (`spartan.Filters` ⋈
  `spartan.Sites` SiteCode='ETBI'; `ftir.SampleAnalysis` → `ftir.Scan` blobs),
  via `get_etbi_spectra.ps1` on the VPN'd Windows machine.
- 72 filters in the DB: **63 PM2.5 samples all on lot 251** + 9 field blanks
  (lot 253). Dated **2025-10-20 → 2026-05-19** (the meeting knew only of the
  Oct–Dec batch; five more months exist). 49 FTIR-scanned so far.
- Blobs decoded as float32 single-beam (fmt 1, YScale 1, grids 2519/2513 pts,
  ~4003→416 cm⁻¹); absorbance = −log10(sample/background) per analysis with
  its own background scan; interpolated to the app's 2722-column raw grid.
- Reference: HIPS `Fabs` from SPARTAN_HIPS_Batch1-51.v2 (32 ETBI rows, 26 with
  Fabs — all currently dry-season). AIRSpec df1=6 correction computed with the
  validated port → `targets/etbi/spectra_corrected.csv` (loader extension
  added same day).
- Target: `calibration_explorer/targets/etbi/` — **n=26 usable filters**
  (Fabs 19.0–42.1 Mm⁻¹, all Dry (Oct–Feb)). "Evaluate on → etbi (custom)"
  in the app.

## Results (Deming, MAC 10, all pairs, n=26)

| calibration | at Addis (239) | at ETBI (26) |
|---|---|---|
| **ocec-450 × AIRSpec k=9** (dense-sweep winner) | 0.92x−1.36 | **0.93x−0.56**, R² 0.51 |
| ocec-800 × AIRSpec k=5 (locked) | 0.87x−1.71 | 0.73x−0.59, R² 0.38 |
| ocec-450 × deriv2 k=20 | 0.97x−1.05 | 0.51x+0.07, R² 0.13 |
| ocec-800 × raw k=6 | 1.67x−3.76 | 0.78x−0.65, **R² 0.86** |

## Reading

1. **The Addis offset largely vanishes at Bishoftu.** Every calibration lands
   at intercept −0.6 ± 0.1 µg/m³ (≈6 Mm⁻¹ absorption-equivalent) vs −1.6 to
   −4.3 at Addis (≈21.5 Mm⁻¹ invariant). Same HIPS instrument, same lot 251,
   same SPARTAN protocol, same country — different city, offset mostly gone.
   This is the strongest evidence yet that the intercept is **Addis-specific
   (site/aerosol), not HIPS-generic** — it moves weight from the
   loading-artifact/curve-geometry branches toward real Addis-specific non-EC
   absorption (or an Addis-specific sampling condition).
2. **The dense-sweep winner transfers best on slope**: 0.93x out-of-country,
   vs 0.73x for the locked 800 cohort. The cutoff-450 basin holds up where it
   was never fitted.
3. **Raw spectra work at Bishoftu (R² 0.86) while failing at Addis** — the
   background-leakage mechanism that breaks raw-Addis apparently isn't present
   at ETBI (consistent with "Addis rides a higher baseline background").
   Note raw's slope changes wildly by site (1.67 at Addis, 0.78 at ETBI) —
   corrected models are far more slope-stable across sites.
4. **The deriv2 k=20 basin member does NOT transfer** (0.51x, R² 0.13) — that
   manual-k row was overfit; the AIRSpec k=9 member is the robust one. Honest
   asterisk for the dense-sweep doc.

## Caveats

n=26, dry-season only, moderate Fabs range, no ETBI TOR (Fabs is the only
reference, so MAC ambiguity applies here too), HIPS uncertainties not yet
propagated. The remaining ~35 scanned-but-unreferenced filters gain Fabs as
HIPS catches up — rerun then (rebuild is one script + drop-in).
