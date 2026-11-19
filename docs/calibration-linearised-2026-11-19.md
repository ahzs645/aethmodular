# Which calibration, once HIPS is linearised (19 November 2026)

The question (19 Nov): linearising HIPS took the Addis intercept to about zero and left a slope near 0.55. Is there a
calibration, with or without a baseline, that brings the slope to 1 as well? No calibration choice had been re-read
against linearised HIPS since [hips-loading-protocol.md](hips-loading-protocol.md).

- Gallery: Meeting tab, Nov 19, **"calibration, HIPS linearised"** (`#tab=meeting&meeting=calibration-linearised-20261119`).
  The Calibration tab's nine saved configurations also follow Plot, HIPS, Linearised now (`gallery/app/src/lib/hipsRun.ts`).
- Sweep: `research/ftir_hips_chem/workflows/run_linearised_calibration_sweep_20261119.py` (about 20 minutes; needs the
  22 Oct decision bundle, which is only on Drive, and the 16 Nov run's cached VIBES spectra).
- Page data: `gallery/data/export_meeting_20261119_calibration_linearised.py`; figures in
  `gallery/app/src/meetings/calibration-linearised-20261119.meeting.ts`; the titles are held to the data by
  `gallery/app/src/lib/calibrationLinearised.test.mjs`.

## What was run

33 calibrations at 1 to 30 components: raw spectra, the spline baseline and the VIBES baseline (rank 70, tau 0.1,
converged), each on eleven training sets (every QC-passed IMPROVE filter; thermal EC below 150, 50 or 25 µg; the six
urban sites; the locked lowest-OC/EC cohort; the smokiest tenth by non-soil potassium; filters of 15 µg EC or more; lot
251 only; the 500 spectra nearest the Addis median spectrum; every filter, weighted toward heavy ones). Each model is
applied to the 233 counted Addis filters and fitted against HIPS BC (Deming, the closeout's error ratio), as measured
and linearised at the central IMPROVE ceiling (2.26) and both ends of its interval. The same model is scored against
thermal EC on IMPROVE sites held out of its training (five whole-site folds; the urban level is the median FTIR ÷ TOR
at the six urban sites). No model is fitted to an Addis reference value.

## What it found

Against linearised HIPS, central ceiling, 957 fits at 2 to 30 components:

| Question | Answer |
|---|---|
| Does any calibration reach a slope of 1 at MAC 10? | **No.** The largest slope is 0.72, and no fit reaches 0.9. |
| Which is closest to slope 1 and intercept 0 at MAC 10? | Raw spectra trained only on the 745 filters with 15 µg EC or more, k 19: 0.72x − 0.05, R² 0.79. Its median error on held-out IMPROVE sites is 5.33 µg, against 1.20 µg for raw spectra with every filter. |
| Closest among models that keep the IMPROVE error under 2 µg? | Every filter, weighted toward heavy ones: VIBES at k 30 gives 0.70x − 1.36, raw spectra at k 29 give 0.66x + 0.17. Only the raw one keeps the intercept near zero. |
| With or without a baseline? | **Without.** Raw spectra with every filter keep the intercept between −0.20 and +0.26 µg/m³ and R² between 0.75 and 0.79 at every k. The VIBES intercept falls to −1.69 µg/m³ at k 18. The spline slope never passes 0.22. |
| Where do models that are right on IMPROVE put Addis? | Raw-spectra fits within 5 % of thermal EC at IMPROVE's urban sites give Addis slopes of 0.39 to 0.71, median 0.56. |
| And at a MAC measured against thermal EC (16.4 m²/g, urban IMPROVE)? | Raw spectra reach 1.00x + 0.02 (smokiest tenth, k 28), 1.01x + 0.03 (six urban sites, k 21), 1.00x − 0.16 (every filter, weighted, k 22), 0.98x + 0.20 (lot 251, k 30) and 0.93x + 0.20 (every filter, k 30). Seven of the eight closest are raw spectra. |

The plain models, every IMPROVE filter in training:

| Model | Slope at MAC 10 | at MAC 16.4 | Intercept (µg/m³) | R² | IMPROVE urban FTIR ÷ TOR | IMPROVE median error (µg) |
|---|---|---|---|---|---|---|
| Raw spectra, k 18 (as SPARTAN's published EC is built) | 0.49 | 0.81 | +0.11 | 0.78 | 0.98 | 1.20 |
| VIBES baseline, k 4 (as the frozen model) | 0.11 | 0.18 | +0.18 | 0.71 | 0.60 | 1.43 |
| Spline baseline, k 5 | 0.11 | 0.18 | +0.33 | 0.57 | 0.75 | 1.28 |

The nine configurations saved on the Calibration tab reverse order. The default (lowest OC/EC 800, spline baseline)
goes from 0.87x − 1.71 as measured to 0.23x + 0.37 linearised; every IMPROVE filter with raw spectra goes from
1.80x − 4.28 to 0.47x + 0.19.

## How to read it

After linearising, the slope is the FTIR level × 10 ÷ MAC. Every MAC measured against thermal EC is above 10 (13.1 on
light IMPROVE filters, 16.4 at IMPROVE's urban sites, 18.9 at Adama), so with FTIR EC exactly right the slope at MAC 10
would be 0.53 to 0.76. A slope of 1 at MAC 10 would mean FTIR EC equals Fabs ÷ 10, which reads above thermal EC
wherever both exist. So the slope is not a calibration fault to tune away. What a calibration can be judged on against
linearised HIPS is the intercept and R², and there raw spectra do best.

## Limits

- HIPS is optical. A slope against it compares two instruments; it is not a score against EC.
- The ceiling is IMPROVE's. Across its interval (1.82 to 2.70) the intercept of one model moves by about ±1 µg/m³; the
  order of the models does not.
- "Closest" is chosen on Addis, so the top rows are fitted to the target they are scored on. The IMPROVE columns are
  the independent check.
- The slope keeps rising slowly with k, so many models are closest at 25 to 30 components, where IMPROVE scores have
  stopped improving.
- With every filter in training the urban level is 0.98 at k 18; with the six filters above 150 µg left out it is 0.91,
  the 10 Nov figure. Six training filters move the level by 7 %, as the 16 Nov check found for single spectra.
- IMPROVE's urban sites have a thermal OC/EC near 3 and Addis is near 1. No IMPROVE filter tests a calibration at Addis
  composition, and no thermal EC has been measured on an Addis filter.
- The Calibration tab's grid (specification curve, cutoff and k sweeps, heatmap) still reads HIPS as measured: it keeps
  one fitted line per configuration. Rescoring it needs a rerun that saves per-filter predictions.

This bears on the 16 Nov decision (training set and k) and the 10 Nov decision (what the Addis offset finding changes)
in [open-items.md](open-items.md).
