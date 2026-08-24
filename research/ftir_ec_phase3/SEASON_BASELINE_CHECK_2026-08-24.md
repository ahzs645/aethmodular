# Does season interact with the calibration and the baseline? First direct table

Prompted by the user 2026-08-24; the systematic season-stratified readout that
ftir_24 (Ann's dry/wet ask) was reserved for. Prior partial answers: ftir_15
(corrected residuals are a season-stable constant), ftir_17 (spectral shape is
season-stable, loading is not), ftir_22 (dry season separates downward in
residuals), char_06 (the dry-season spectral anomaly). What was never run: the
same cohort refit under each baseline, read out PER SEASON.

## The table (ocec-450, site-held-out, Addis; per-season Deming λ*=2.96, MAC 10)

| baseline | all-year | Dry (n=105) | Belg (n=61) | Kiremt (n=73) |
|---|---|---|---|---|
| raw (k=6) | 2.35x−6.08 | **1.27x−1.80** | 2.21x−5.15 | 2.25x−5.06 |
| AIRSpec (k=9) | 0.93x−1.42 | **0.60x−0.22** | 0.84x−0.91 | 0.86x−0.73 |
| deriv2 (k=9) | 1.61x−3.45 | 1.44x−2.86 | 1.36x−2.04 | 1.71x−3.93 |

## Readings

1. **The raw calibration's badness is a wet-season phenomenon.** Its all-year
   2.35x−6.08 decomposes into Belg/Kiremt at ~2.2x−5.1 and Dry at 1.27x−1.80 —
   the background-leakage error concentrates in the wet seasons.
2. **Baselining does not remove seasonality; it relocates it.** Under AIRSpec
   the wet seasons become nearly clean (0.84–0.86x, intercepts −0.7…−0.9) but
   **Dry drops to 0.60x−0.22**: a dry-season slope deficit with almost no
   intercept. The all-year −1.42 intercept is carried by the wet seasons; the
   dry season contributes a slope-form deviation instead.
3. **Direction vs Ann's hypothesis**: she proposed dry (diesel) predicts right
   and wet (charcoal) breaks. Under the corrected winner it is closer to the
   reverse — wet seasons sit near 1:1 and DRY is the anomalous regime
   (consistent with char_06's dry-season spectral anomaly and ftir_22's
   dry-separates-downward residuals). Under raw, her direction holds. So the
   answer to "does the baseline interact with season" is emphatically yes —
   the seasonal failure mode flips with the baseline choice.
4. deriv2 stays season-unstable in the intercept (−2.0…−3.9) — another mark
   against the derivative family at Addis.

## Caveats and next

Per-season Deming on restricted x-ranges (λ held at 2.96) — treat slopes as
indicative; a season-interaction fit (shared slope + season offsets, or
York per season) is the rigorous version. Group schemes differ across sites
(Ethiopian seasons at ETAD/ETBI, quarters at CHTS/INDH/USPA), so any
cross-site seasonal pooling needs the mapping decision first. This table is
the seed of ftir_24 — building it properly (all six setup-matrix cohorts ×
both protocols × per-season, with the interaction test) is the owed
deliverable.
