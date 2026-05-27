# Addis fAbs vs FTIR-EC — errors-in-variables fit & intercept shift

Week of 2026-05-26. Quantifies whether the stubborn ~28–29 Mm⁻¹ intercept in the Addis Ababa
fAbs-vs-FTIR-EC scatter is consistent with the hypothesis that **FTIR underestimates EC in
charcoal-heavy air** (fully-charred carbon has no IR-active bonds, so true EC sits further right).

## Notebooks
- `addis_fabs_ec_deming.ipynb` — the errors-in-variables fit + EC-shift exercise (below).
- `addis_fabs_ec_offset_correction.ipynb` — isolates the offset, states the fix values, and shows
  three side-by-side regressions: **raw**, **additive (EC + Δ)**, **multiplicative (EC × k)**, each
  re-fit with OLS. Demonstrates that **only the additive correction zeros the offset** (slope kept at
  ~4); a multiplier leaves the intercept at ~28 and just flattens the slope. Notebook-only (no CSV).

### `addis_fabs_ec_deming.ipynb` — three steps:

1. **Errors-in-variables fit.** OLS assumes all error is in Y (fAbs); the intercept is sensitive to
   X-error. Re-fit with orthogonal / Deming regression (the method Boris et al. 2019 used).
   `scipy.odr` cross-checks the closed-form Deming.
2. **Assumed measurement uncertainty.** σ_EC = 0.2 µg/m³, σ_fAbs = 1.0 Mm⁻¹ (ordinary noise, *not*
   the missing-char effect). A sensitivity sweep shows how slope/intercept move with the error ratio.
3. **EC shift that zeros the intercept.** Holding the slope, solve for the additive Δ and the
   multiplicative k that bring the fixed-slope line to the origin — and judge constant vs. multiplier.

## Key results (n = 189 paired filters)
| fit | slope (Mm⁻¹/µgm⁻³) | intercept (Mm⁻¹) | additive Δ | Δ / mean EC | multiplier k |
|---|---|---|---|---|---|
| OLS | 3.99 | 28.4 (R²=0.78) | 7.1 µg/m³ | 1.40 | 2.40 |
| Deming (λ=25) | 4.45 | 26.1 | 5.9 µg/m³ | 1.15 | 2.15 |

- The intercept **shrinks but does not vanish** under the errors-in-variables fit → not just an
  artifact of ignoring X-error.
- The additive shift **exceeds the mean EC** (Δ/mean EC > 1) → a constant offset is not physical
  (filters don't all carry the same char).
- The multiplicative factor is **~2.2–2.4×**, matching the meeting's 2–3× estimate → the missing-char
  hypothesis is at least plausible, though it requires FTIR-EC to be low by roughly a factor of two.

## Scope
**Addis only — not Delhi.** Delhi's scatter has no points near the origin; shifting it would be the
over-interpretation Warren cautioned against.

## Notes
- Data/loaders come from the sibling `../ftir_hips_chem/scripts` project (added to `sys.path`).
- `_build_notebook.py` regenerates the notebook (`python3.13 _build_notebook.py` then
  `jupyter nbconvert --execute`). fAbs is recovered in Mm⁻¹ as `hips_fabs × MAC_VALUE`.
