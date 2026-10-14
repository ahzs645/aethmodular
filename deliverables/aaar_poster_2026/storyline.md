# AAAR poster: storyline draft (practice run Thursday)

Agreed with Ann on 9 Oct: tell the "it is not only Ethiopia" story. Leave the PLS calibration
search (specification curve, cohorts, VIBES) off the poster; keep it for discussion. Write for an
audience that has never thought about FTIR EC. Ann suggested talking it through as slides first, then
laying the slides out as the poster; either is fine.

Numbers below come from the completed rank-70 meeting run
(`research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/meeting/panel_stats.csv`),
spline (AIRSpec) baseline, Deming at MAC 10. The gallery views to screenshot are named for each panel.

## Working title

FTIR elemental carbon calibrated on US filters transfers poorly to heavily loaded, Ethiopia-like samples,
in Addis Ababa and in the US network itself

## One-sentence message

A well-established FTIR method that predicts EC well across the IMPROVE network does much worse on
the IMPROVE filters that look most like Addis Ababa, so the Addis problem is a calibration-coverage
problem that also exists in the US network, not a quirk of one site.

## Panels

1. **The problem (Addis Ababa).** The current SPARTAN FTIR EC against HIPS absorption (Fabs / MAC 10)
   at Addis: slope 2.30, intercept −6.09 µg/m³, R² 0.78 (n 189). Strong correlation, wrong scale.
   *Gallery: Meeting › Starting point.*

2. **The method works for typical US filters.** Same spline-baseline approach, all-IMPROVE calibration
   (lot 251), read out on 2,078 IMPROVE filters from sites held out of the calibration:
   FTIR vs TOR EC slope 0.93, intercept 0.02, R² 0.73.
   *Gallery: Meeting › Three crossplots, Calibration pool · lot 251, IMPROVE filters: All held-out IMPROVE.*

3. **It does not work as well for US filters that look like Addis.** The 97 held-out IMPROVE filters whose
   spectra correlate best with Addis: their reference methods agree (TOR vs HIPS slope 0.91, R² 0.82),
   but FTIR under-reads TOR (slope 0.48, R² 0.46). Worst for Dry-season analogs (FTIR vs TOR 0.44,
   R² 0.33). Same calibration, same baseline; only the samples changed.
   *Gallery: same section, IMPROVE filters: Analogs, Season: All Addis and Dry.*

4. **Why: Addis-like samples are rare in the calibration.** Per filter, the Addis median HIPS-implied
   EC loading is about 9 times the IMPROVE median and above IMPROVE's 99th percentile (in concentration it
   looks like 41 times, because IMPROVE samples ~33 m³ per filter and SPARTAN ~7 m³). Composition differs too:
   the Addis spectra carry a strong silicate (dust) band near 1035 cm⁻¹, 1.4 to 2.6 times IMPROVE by season.
   *Notebook: `research/ftir_hips_chem/ann_followup_20261009.ipynb`, sections 3 and 4
   (`output/plots/ann_followup_20261009/`).*

5. **Context: Addis chemistry by season.** Stacked composition (medians, Bega / Belg / Kiremt), Kiremt with the
   highest concentrations (low boundary layer). One small panel; it orients a general audience.
   *Gallery: Meeting › Seasonal composition (or Part of a whole).*

6. **Takeaway.** Calibrations built on networks dominated by clean, low-loading US filters transfer poorly to
   polluted, dust- and smoke-rich samples typical of the Global South, and the same weakness is visible inside
   the US network for its most Addis-like filters. Validating on the whole network hides it.

## Things to decide before Thursday

- Season names: Ann is fine with "Bega" for Dry; the literature mostly says Dry / Belg / Kiremt.
- Whether panel 3 uses FTIR vs TOR (cleanest, an independent thermal reference) or FTIR vs HIPS (matches panel 1).
  Suggest TOR in panel 3 and HIPS in panel 1, with one sentence explaining why Addis has no TOR.
- Units: show panels 2 and 3 in µg per filter as well as µg/m³ (gallery: Units switch), so the loading gap in
  panel 4 is visible on the same axes.
- Keep for discussion only: VIBES baseline, the specification curve, the IMPROVE-predicts-itself check.
