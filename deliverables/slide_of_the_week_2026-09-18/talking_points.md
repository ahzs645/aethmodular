# Slide of the week, 18 Sep 2026 (Ann + Satoshi call)

Rebuild: `~/anaconda3/bin/python deliverables/slide_of_the_week_2026-09-18/build.py`
(reads `gallery/app/public/data/calibration.json`; it checks the entire-network numbers quoted in the call).

## One-breath summary

Of everything we vary, the baseline moves the Addis intercept the most. Raw spectra sit far below
zero, second derivative falls in between, and the spline baseline lands closest. So from here on we
use the spline baseline only. The catch is that the slope then drops. Satoshi's next test asks
whether the IMPROVE analogs we pick show the same bias as Addis.

## Behind the slide (notes have the detail)

- Entire IMPROVE network calibration, Option A, MAC 10: raw 1.94x − 4.82 → spline baseline 0.47x − 0.78.
- Median intercept over 637 configurations per row: raw −5.1, second derivative −2.3, spline baseline −1.4.
- About 31% of spline-baseline configurations land within ±1 µg/m³, against 2% for raw.

## Asks from the call

- **Satoshi:** for each season's analog set, plot TOR EC vs HIPS Fabs/10, FTIR EC vs TOR EC on held-out analogs, and FTIR EC vs Fabs/10.
- **Satoshi:** predict the analogs with the same model used for Addis and see whether they land in the same quadrant.
- **Satoshi:** per-filter nearest analogs instead of the season median. Lower priority, since spectra are homogeneous within a season.
- **Send Satoshi:** raw Addis spectra, so Francois Kamper can baseline them; runnable code hopefully next week.
- **Ann:** pull the TOR fractions (OPR, OPT, EC1 to EC3) and colour the crossplots by OP.
- **Ann:** settle the cross-validation choice so it stops being a lever.
- **Settled:** spline baseline only. Exclude the CO₂ band and the top end (3500 or 3600 cm⁻¹, nine filters apart, pick one) from matching, and keep the same exclusions for the new baseline.

## Don't get caught

- The biomass-smoke calibration export showed its baseline incorrectly; it is left off the slide.
- The ±1 µg/m³ share depends on which cohorts and cutoffs are in the grid; treat it as descriptive.
- In Kiremt, Addis spectra sit below most of the training range. Extrapolating down is usually fine; extrapolating up is not.
