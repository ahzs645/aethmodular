# Slide of the week, 25 Sep 2026

Rebuild: `~/anaconda3/bin/python deliverables/slide_of_the_week_2026-09-25/build.py`
(reads the gallery exports `gallery/app/public/data/filters.json` and `calibration.json`).

## One-breath summary

Addis FTIR EC sits about 6 µg/m³ below HIPS at zero (Deming 2.24x − 5.86, λ* 2.96, MAC 10).
The bias moves with the season: FTIR reads high in Kiremt and low in the dry season. None of the
1,911 all-season calibrations reaches slope 1, intercept 0. The best guardrail passer is still
at −1.82, and the one in-box point near zero has a test set R² of 0.06. Next step: one calibration
per season, plotted on the same slope vs intercept plane.

## What Ann asked for in the meeting

| Ask | Status on this slide |
|---|---|
| Drop OLS, show R² + Deming equation | Done: Deming only, per-season equations and R² |
| Poster focuses on Addis only, no other sites | Done: other-site intercepts are in NOTES only |
| Say "test set", not "held-out TOR" | Done on axes and notes |
| Rename protocol to "cross-validation protocol"; "raw spectra" / "spline baseline" | Used in notes; gallery labels still to update |
| Put seasonal calibrations on the slope vs intercept plot | Open: this is the next step the slide sets up |
| Show stacked bars in concentration and in percent, by season | Open (gallery option) |
| Spec curve: readable labels grouped by category, heat map underneath | Open |
| Split calibration-set metrics (RMSE, R²) from test-set metrics in the cutoff sweep | Open |
| VIBES: plot down to 400 cm⁻¹, cut the overlay at 1500, rerun correlations full range | Open, so VIBES is left off the slide |

## Don't get caught

- The meeting's gallery panel used λ = 1 (Addis −6.52; Beijing −0.63, Delhi −1.37, JPL +0.10).
  The slide uses λ* 2.96 (−5.86). Both round to "about −6".
- Each seasonal intercept is less negative than the all-season one because each fit spans a
  narrower range. The point is that the seasons sit on different lines, not the individual intercepts.
- Right panel: 53 extreme configurations are pinned to the plot edge. Second derivative
  configurations are still in the cloud and will be dropped.
- Analog cutoffs resolve below nominal (top 500 gives 477).
- The grid export is dated 2026-10-01, after the meeting. The closest-to-zero point quoted in the meeting
  (IMPROVE cross-validation R² 0.030, test set R² 0.062) reproduces exactly in it.

## Blocked

- IMPROVE download from the FED site has failed for four days. Support has been emailed; if it
  stays down, Ann will chase a contact.
