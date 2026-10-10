# Slide of the week, 11 Sep 2026

Rebuild: `~/anaconda3/bin/python deliverables/slide_of_the_week_2026-09-11/build.py`
(reads the 10 Sep weekly run in `research/ftir_hips_chem/output/tables/ann_weekly_20260910/` and the
AIRSpec df 6 spectra caches; nothing refitted; the builder checks that the Deming fits reproduce).

## One-breath summary

We gave each Addis season its own calibration. The dry season loses the offset but goes flat.
Kiremt gets the slope but still reads low. No season fixes both. The training spectra match Addis
in shape but span a much wider loading range, so the next step is to keep only the analogs whose
absorbance looks like Addis.

## The slide is light on numbers; they are in the notes

| Season | Deming slope [95% CI] | Intercept [95% CI] | Test set TOR R² |
|---|---|---|---|
| Dry | 0.31 [0.24, 0.36] | −0.06 [−0.26, 0.18] | 0.43 |
| Belg | 0.53 [0.32, 0.74] | −0.36 [−1.40, 0.68] | 0.64 |
| Kiremt | 0.90 [0.81, 1.09] | −0.97 [−2.24, −0.37] | 0.70 |

All three fail the 0.85 TOR screen, so none is adopted. Analogs shared between seasons (of 500):
Dry/Belg 6, Dry/Kiremt 1, Belg/Kiremt 160.

## What Ann asked for in the meeting

- Report slopes and intercepts as value [low, high].
- Drop above 3500 cm⁻¹ from the PLS fit as well, not only from matching.
- After correlation, keep the 50 to 100 analogs closest in absorbance: plot variance against rank and cut just before the jump.
- PMF: also try wood + charcoal and sea salt + polluted marine combined.
- Put the spectra at the start of the talk, then the results. Lead with seasons for Satoshi next Thursday.
- Adama: add HIPS on a second y axis next to the quartz plot; plot FTIR vs Fabs coloured by sampled volume; look for PurpleAir or other PM data at Adama and Bishoftu vs Addis.
- AAAR poster: start the introduction and methods from the old poster; results come about a week before practice.

## Don't get caught

- The Deming λ here is 3.34 (HIPS uncertainty vs historical TOR RMSE), not the 2.96 used elsewhere.
- Spectra are every individual filter, not averages. Replicate scans of one Addis filter are averaged.
- In the dry season Addis overlaps more of the training range than in the rainy seasons; the "wider range" point is strongest for Belg and Kiremt.
- Raising MAC would steepen the dry slope, but Addis concentrations would then be implausibly low.

## Adama (separate thread, not on the slide)

Five quartz TOR filters from July 2024. Adama's OC/EC sits mid-IMPROVE, and its FTIR agrees with TOR.
Adama and Bishoftu don't stand in for Addis, so Ann will ask JPL to fund quartz sampling in Addis and
will pull Christian's co-located quartz/FTIR filters from Delhi.
