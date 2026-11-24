# Where the HIPS work stands after the 24 November meeting (24 November 2026)

One page for picking the work back up. Numbers come from the gallery exports and notebooks named in each line.
Unpublished UC Davis results (the Phoenix drain-disc pairs) are analysed locally and kept out of this repository and
the public gallery until Warren White and Ann agree.

## What Ann decided on 24 Nov

- HIPS reads low on heavily loaded filters, and that, not FTIR, is the main problem at Addis ("I believed Warren, I did
  not think it was HIPS, and that was incorrect").
- SPARTAN should report FTIR EC, a measurement, not corrected HIPS. Corrected HIPS is for QC and guidance on the
  heavily loaded sites; filters past the cut have *compromised Fabs*, not lost filters.
- Write a paper parallel to White et al. (2025) on what this means for SPARTAN, and talk to Warren again first.
- After that: FTIR EC and the good (lightly loaded) HIPS against the aethalometers at all aethalometer sites.
- Ann will bring an IMPROVE screen and a drain disc to the next meeting. Next contact is by email (she is off for
  Thanksgiving, 25–28 Nov).

## What is established

| Finding | Where |
|---|---|
| HIPS optical depth bends on every IMPROVE filter (348k), not only on group medians; τ∞ 2.26 [1.82, 2.70] from the 17 group medians, 1.7 to 2.9 from single-filter fits | [hips-bend-checks-2026-11-20.md](hips-bend-checks-2026-11-20.md); Meeting tab, Nov 24 |
| White et al. (2025) already describe the drop at high loading (screen pattern, α 10 for the network, pixelation limit τ 1.016) | [hips-warren-model-2026-11-24.md](hips-warren-model-2026-11-24.md); Meeting page `hips-warren` |
| The 2008–16 against 2017–25 ceiling gap (2.97 against 1.73) is the July to October months; November to June is 3.2 in both | same |
| White's own IMPROVE-versus-CSN test, rerun on 2,372 daily pairs at four cities: our curve's shape fits (τ∞ 2.29 [2.09, 2.50]) if IMPROVE reads 10 % above CSN; α 10 fits the network means with no offset | `gallery/data/export_hips_csn.py`; Experimental, SPARTAN's own HIPS curve |
| The Addis intercept and MAC follow the size of the correction: MAC for slope 1 is 18.3 at τ∞ 2.26, 13.3 at 2.7, 10.4 at 3.26, about 30 at 1.9 | Experimental, SPARTAN's own HIPS curve (`#sc-mac`) |
| Addis sampled volume is constant (7.03 to 7.36 m³), so at Addis a dark filter is a dirty day: loading and composition cannot be separated there | [spartan_own_curve_design_20261124.ipynb](../research/improve_hips_offset/spartan_own_curve_design_20261124.ipynb) |
| Against the self-calibrated aethalometer, Addis HIPS rises with log-log slope 0.46 (τ∞ about 1.9, a lower bound because errors in x steepen the bend). The bend is in HIPS | same; `#sc-addis-aeth` |
| Delhi sampled 24, then 12, then 6 h inside the same 9am to 9am days at 5 L/min. In the same month, HIPS Fabs per unit of PM2.5 mass falls with volume (b_V −0.28 ± 0.05; −0.44 against FTIR EC; −0.24 expected from the IMPROVE curve). A loading effect in SPARTAN's own data | [delhi_run_length_test_20261124.ipynb](../research/improve_hips_offset/delhi_run_length_test_20261124.ipynb); `#sc-delhi-test` |
| The MA350's loading constant can be set without FTIR: the tape-advance step and Kyan's BC–ATN criterion agree at Addis (k 0.0115 IR, 0.0102 Red, 0.0089 Blue), so the aethalometer is not tuned to FTIR | [aeth-loading-k-2026-11-24.md](aeth-loading-k-2026-11-24.md); Experimental, Aethalometer loading k |

## Corrections to what was said on 24 Nov

- τ∞ 2.26 is the IMPROVE 17-group fit, not "the Addis 90th percentile".
- Delhi's correction is not weaker "because the Addis range was used": Delhi's 41 filters below τ 0.45 fit 0.90x + 0.14
  with no correction.
- The six urban IMPROVE sites (11,020 filters): 65 % are *past* τ 0.45 and 18 % past 1.016. Rural IMPROVE: 7 % past
  0.45 and 0.6 % past 1.016.
- Delhi does not use a lower flow (every site runs 5 L/min); it samples fewer hours.
- Addis is not "less bent, more uniform": its exponent b is 0.46 [0.43, 0.50] against 0.57 expected from its darkness on
  the IMPROVE curve. Addis sits above the IMPROVE curve because it absorbs more per µg of EC, which is a different thing.
- Ann does not like "saturation" for this: use *loading effect*, the term the aethalometer literature uses.

## Open questions

1. **How big the correction is.** Candidates on the median Addis filter run from ×1.1 to ×1.75 (IMPROVE November to June
   curve, White's α 10, the published curve). This decides the Addis MAC (10 to 18) and whether the intercept reaches 0.
2. **The Delhi placebo.** FTIR EC ÷ PM2.5 rises with volume (+0.24 ± 0.08): either FTIR reads relatively high on heavier
   filters or PM2.5 mass is inflated on light ones. Check Delhi field blanks and weighing offsets, and FTIR against the Delhi
   aethalometer (2024–25).
3. **One ceiling or one per site.** Fit τ∞ per site against the aethalometers (Addis, Delhi, Pasadena, Beijing) and against
   FTIR EC everywhere; a per-site ceiling is needed only if they differ by more than their errors.
4. **SPARTAN's mesh.** Open fraction and deposit unevenness from photographs of loaded SPARTAN filters, the one number that
   compares directly with White's α and our τ∞.
5. **Are IMPROVE's dark filters smoke?** Share past τ 0.45 and past 1.016 in July to October against the rest of the year.

## The experiment that settles it

Two SPARTAN filters on the same air, one run 1/4 of the time of the other, at the same flow (vary run time, not flow;
spread the short run through the same days, as Delhi's 6 h runs already are). Simulated with 4 % reading error per filter,
20 pairs give the Addis MAC to about ±1 to ±2; 10 pairs separate MAC 10 from 18. Needs no TOR, no FTIR and no transfer
from IMPROVE's screen. Design and power: Experimental, SPARTAN's own HIPS curve (`#sc-power`).

## Next steps

- Email Ann: the corrections above, the CSN and Delhi results, the experiment proposal, and the questions for Warren.
- Questions for Warren: whether the Phoenix drain-disc deposits are uniform and whether we may use those filters; what
  closes the CSN gap if α is smaller than 10; whether SPARTAN's holder would pattern at 1.4 L/min/cm²; whether the corrected
  IMPROVE product is out and which α it uses.
- Ask SPARTAN whether the SS5 can run two slots on the same days at different duty cycles.
