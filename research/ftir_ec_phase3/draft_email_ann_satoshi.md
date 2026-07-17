# Draft — update email to Ann and Satoshi

Subject: FTIR next steps — results from the to-do list

Hey Ann and Satoshi,

I worked through the to-do list from our last meeting. Short version: the HIPS-transfer test
and the VIP diagnostics both point the same way — the current calibration predicts EC through
OC-linked features that don't hold at Addis — and Ann's low-OC/EC cohort idea is the best
performer so far. Details:

1. **Current calibration leans on OC.** The EC and OC VIP profiles share most of their
   structure (Spearman r = 0.93; 179 of the top-200 wavenumbers overlap).

2. **HIPS transfer (Satoshi's suggestion).** The IMPROVE-trained HIPS model does not transfer
   to Addis (median predicted Fabs 15 vs 47 observed). An Addis-only HIPS model works well
   (nested-CV R² = 0.88), and its top-200 VIP wavenumbers have zero overlap with the
   IMPROVE-trained model's — the priority features really do change.

3. **Analog selection (Mahalanobis + Q + VIP-weighted mismatch).** Promising at first, but the
   improvement did not survive a locked disjoint-site TOR test, so I'm not treating it as real.

4. **Ann's low-OC/EC cohort — best locked result so far.** Training on the 800 lowest-OC/EC
   IMPROVE filters (OC/EC ≤ 2.3 vs pool median 5.5) gives, on a fixed 190-filter Addis cohort
   at MAC = 10: slope 1.59, intercept −3.22, R² 0.77, RMSE 1.16 µg/m³ (deployed: 1.90 / −4.17 /
   0.76 / 1.49). It also beats ten size-matched random cohorts decisively on the held-out TOR
   test (RMSE 3.4 vs 4.4–7.3 µg/filter). Most interesting: its VIP profile correlates r = 0.74
   with the Addis-only HIPS model, while the current smoke calibration correlates r = 0.12 —
   two independent routes converging on the same spectral features. The intercept is still
   about −3, so it's progress on the hypothesis, not a fix.

5. **The ~1600 cm⁻¹ band is not amine.** Its Addis peak center is 1617–1619 cm⁻¹ (tight IQR,
   and unchanged after AIRSpec baselining; every IMPROVE cohort peaks at ≥1633), and it
   co-varies with CH (ρ = 0.88) and carbonyl (ρ = 0.93)
   but not with the 3100–3400 N–H/O–H window (ρ = 0.17, vs 0.6–0.7 in every IMPROVE group).
   That leaves carboxylate and/or aromatic C=C, which we can't separate on Teflon spectra
   because the ~1400 symmetric partner is below the trusted region — consistent with what
   Satoshi suggested, and something the sub-1500 model could resolve.

6. **AIRSpec EDF 6–8.** R disappeared from my machine, so I ported the APRLssb smoothing-spline
   baseline to Python and validated it against the R output we still had for the Ethiopia
   spectra (agreement to ~10⁻⁷ absorbance), then corrected all 13,634 IMPROVE lot-248/251
   spectra plus Addis at EDF 6 and 8. Three findings: (a) EDF 6 vs 8 makes no practical
   difference, so the exact choice in your suggested range isn't sensitive; (b) on corrected
   spectra the low-OC/EC calibration's Addis intercept halves to −1.6 with slope 0.86 at
   MAC = 10 and a held-out TOR slope of 1.01 — the best locked result yet, though noisier than
   the raw version; (c) the 906-smoke calibration collapses on corrected spectra (slope 0.37),
   which suggests a lot of its raw-spectrum response to Addis was carried by the broad
   baseline — i.e., the sloping absorption component. The remaining fork is MAC: the raw
   low-OC/EC model is self-consistent at MAC = 6 (slope 0.95, intercept −3.2) and the
   corrected one at MAC = 10 (slope 0.86, intercept −1.6). An independent Addis TOR/EC value
   or an agreed HIPS MAC bridge would decide between them.

7. **Delhi/Beijing score-space check.** Blocked locally: our DB pull has IMPROVE spectra only.
   If someone can export INDH and CHTS spectra the same way the ETAD file was pulled, the
   analysis itself is ready to run against the fitted models.

Notebooks and tables are in the repo under `research/ftir_ec_phase3/` (ftir_11–ftir_13) if you
want the details.

Thanks,
Ahmad
