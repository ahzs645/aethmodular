# July 07 — Does the smoke-only calibration actually help?

One question: on the real Ethiopia (ETAD/Addis) filters, does a **biomass-burning-only** calibration
bring FTIR-EC **closer to the independent HIPS** ground truth than the **regular all-data**
calibration? This is the "old/original data (regular all-data cal) EC vs HIPS" crossplot rendered in
the same format as `../ftir_ec_calibration_2026_06_25/figures/fig08_etad_newec_vs_hips.png`, put next
to the smoke calibration for a like-for-like read.

## Notebooks

| File | What it does |
|---|---|
| **`regular_before_cal0_vs_smoke_hips.ipynb`** ⭐ | **Primary.** "Regular" = the **original reported EC we used *before* CAL-0** (the April group-talk `ftir_ec`, n=189), *not* a refit. Reproduces the group-talk Addis crossplot **style** and puts smoke (biomass) next to it. Baseline joins by date via the ftir_hips_chem pipeline; smoke EC joins by date too (189/189). |
| `regular_vs_smoke_ec_vs_hips.ipynb` | First pass. "Regular" = **CAL-0** (PLS k=20 refit on the 906-sample set) — superseded as the baseline by the notebook above, but kept: applies CAL-0 + biomass to the 319 ETAD spectra, joins SPARTAN HIPS (n=259), fig08-style axes. |

Build either with its `_build_*.py` then `nbconvert --to notebook --execute --inplace`.

### Why two "regular" baselines
CAL-0 (all-data, no filtering) is a *fresh PLS refit*; the **reported EC** is the calibration SPARTAN
actually deploys — "the data we used before CAL-0." The primary notebook uses the reported EC, which is
the honest baseline and (usefully) the one closest to HIPS.

Rebuild:
```bash
cd research/July07
/opt/anaconda3/bin/python3.13 _build_regular_vs_smoke_hips.py
/opt/anaconda3/bin/python3.13 -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 regular_vs_smoke_ec_vs_hips.ipynb
```

## Plot standard (2026-07-07)
- **HIPS (ground truth) on x, FTIR-EC on y.**
- Equal **0–20 µg/m³** range both axes (fit to ETAD; the 0–400 tool standard is for the
  predicted-vs-measured EC plots, not this HIPS crossplot).
- HIPS → EC-equivalent via `EC = Fabs / 10` (MAC = 10; 6-vs-10 still open).

## Result — the smoke calibration does **not** help

Primary notebook (n=189, EC on x / HIPS/MAC on y, group-talk style):

| calibration | slope | intercept | R² | |slope−1| |
|---|---|---|---|---|
| **Regular — reported EC (before CAL-0)** | **0.40** | +2.84 | **0.784** | **0.60** |
| Smoke-only (biomass) | 0.26 | +3.32 | 0.704 | 0.74 |
| CAL-0 refit (reference) | 0.29 | +3.18 | 0.736 | 0.71 |

The smoke calibration pushes the slope **further from 1** (0.40 → 0.26) and **lowers R²** (0.78 → 0.70)
— on the Addis filters it does **not** bring FTIR-EC closer to HIPS. The reported EC we already deploy
is the closest to the 1:1 line. (The first-pass CAL-0-baseline notebook reaches the same conclusion in
its own n=259 units.)

**Caveat:** "regular" = IMPROVE-based 906-sample PLS; "smoke" = SPARTAN lot-251 biomass — they differ
in *training population* as well as *sample selection*, so read the **direction**, not the absolute
level. The confound-free test is **general lot-251 vs biomass lot-251** (same lot); drop those coeffs
into the same harness when they land. Also worth a `MAC = 6` re-run to bound the 6-vs-10 question.

## Outputs
- `figures/regular_vs_smoke_ec_vs_hips.png`
- `tables/regular_vs_smoke_verdict.csv`, `tables/etad_regular_vs_smoke_ec.csv`
