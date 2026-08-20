# Variant-matrix results — 2026-08-19 meeting follow-ups

22 runs through the explorer API (site-held-out protocol, rule k unless noted;
Deming at MAC 10; "fixed" = the 190-filter deployed cohort, "all" = all pairs).
Every configuration is cached, so each row is one click to reproduce in the UI.

## A. Select × calibrate 2×2 (Ann: "do baseline for both")

| cohort | select on | calibrate on | k | fixed Deming | all Deming | held-out TOR R² |
|---|---|---|---|---|---|---|
| Eth-shaped 300 | raw | raw | 10 | 1.85x−4.96 | 1.69x−4.29 | 0.00 |
| Eth-shaped 300 | raw | AIRSpec | 6 | 0.30x−0.60 | 0.25x−0.37 | 0.24 |
| Eth-shaped 300 | AIRSpec | raw | 20 | 1.86x−4.50 | 1.67x−3.60 | 0.63 |
| Eth-shaped 300 | AIRSpec | AIRSpec | 4 | 0.30x−0.68 | 0.28x−0.54 | 0.40 |
| Analogs 500 | raw | raw | 4 | 3.09x−9.32 | 2.93x−8.75 | 0.37 |
| Analogs 500 | raw | AIRSpec | 5 | 0.70x−1.42 | 0.61x−0.99 | 0.30 |
| Analogs 500 | AIRSpec | raw | 15 | 1.94x−4.68 | 1.79x−4.08 | 0.70 |
| Analogs 500 | AIRSpec | AIRSpec | 23 | 0.64x−1.34 | 0.58x−1.20 | 0.69 |

**Reading.** The two knobs do different jobs and neither substitutes for the
other. *Calibrating* on corrected spectra is what moves the intercept
(−4.3…−8.7 → −0.4…−1.2, regardless of selection space). *Selecting* on
corrected spectra is what buys TOR skill (eth held-out R² 0.00 → 0.63, analogs
0.37 → 0.70) — a more coherent cohort — but leaves a raw calibration's offset
intact. The double-baseline combination gets the small intercepts **at the cost
of the slope** (0.28 / 0.58): the model under-responds, compressing predictions
toward the mean — a near-zero intercept earned that way is not a fixed
calibration. The best analog cell (sel+cal AIRSpec: 0.58x−1.20, ho 0.69, k=23 —
Satoshi's "more components" instinct shows up here) still doesn't beat
lowest-OC/EC-800 + AIRSpec (0.87x−1.71, ho 0.90) on the slope–intercept balance.

## B. Lot hypothesis (train lot × eval lot, ocec-800)

| train lot | eval lot | calibrate | k | fixed Deming | all Deming | held-out TOR R² |
|---|---|---|---|---|---|---|
| all (800) | all | raw | 6 | 1.81x−4.34 | 1.67x−3.76 | 0.91 |
| all (800) | 251 | raw | 6 | 1.82x−4.32 | 1.68x−3.80 | 0.91 |
| 251 (619) | all | raw | 7 | 1.95x−4.74 | 1.81x−4.21 | 0.95 |
| 251 (619) | 251 | raw | 7 | 1.96x−4.73 | 1.82x−4.24 | 0.95 |
| all (800) | all | AIRSpec | 5 | 0.95x−2.09 | 0.87x−1.71 | 0.90 |
| all (800) | 251 | AIRSpec | 5 | 0.95x−2.01 | 0.86x−1.63 | 0.90 |
| 251 (619) | all | AIRSpec | 4 | 0.92x−2.08 | 0.83x−1.72 | 0.93 |
| 251 (619) | 251 | AIRSpec | 4 | 0.92x−2.01 | 0.83x−1.62 | 0.93 |

**Reading: the lot hypothesis fails this test.** If baselining worked by
erasing 248-vs-251 media differences, a lot-251-only raw calibration evaluated
on lot-251 filters should have recovered part of the baselining benefit. It
does the opposite — the raw intercept *worsens* (−3.76 → −4.24) while held-out
TOR improves (0.91 → 0.95, a cleaner training set), and the corrected results
barely move (−1.71 → −1.62). Whatever baselining removes, it is not the
248/251 split — evidence for "it's the aerosol/background, not the lot."
The residual lot-side signal is on the ETAD readout: eval-lot 248 alone (n=34)
reads 0.54x−0.55 under AIRSpec — very different, but confounded with sampling
period/season, so treat with care.

## C. Cutoff scan — Ethiopia-shaped, select-corrected × calibrate-corrected

| cutoff | k | fixed Deming | all Deming | all R² | held-out TOR R² |
|---|---|---|---|---|---|
| 150 | 4 | 0.33x−0.71 | 0.32x−0.63 | 0.62 | 0.39 |
| 200 | 6 | 0.33x−0.60 | 0.28x−0.33 | 0.48 | 0.20 |
| 250 | 8 | 0.22x−0.19 | 0.20x−0.07 | 0.47 | 0.22 |
| 300 | 4 | 0.30x−0.68 | 0.28x−0.54 | 0.59 | 0.40 |
| 400 | 15 | 0.61x−1.12 | 0.54x−0.73 | 0.66 | 0.35 |
| 500 | 21 | 0.79x−1.93 | 0.72x−1.59 | 0.69 | 0.57 |

**Reading.** A clean trade with no sweet spot: tighter cohorts drive the
intercept toward zero but only by collapsing the slope (the 250-cutoff
0.20x−0.07 row is the reductio — near-perfect intercept, useless calibration);
loosening to 500 climbs back toward the ocec+AIRSpec operating point
(0.72x−1.59). Shape-based selection at any cutoff converges toward — but never
beats — the composition-based cohort.

## Bottom line

- Intercept alone is a misleading objective: the smallest intercepts in this
  matrix belong to the most under-responsive models. Score candidates on
  distance to (slope 1, intercept 0) with a TOR-skill floor — which is exactly
  the explorer Optimize tab's objective — or read the Pareto front.
- Best operating point so far remains **lowest-OC/EC-800 + AIRSpec,
  site-held-out** (rule k=5: 0.95x−2.09 fixed; optimizer's k=9 on eval-lot 251:
  0.99x−1.69, held-out 0.87).
- The baselining benefit is **not** a lot artifact (Set B), and corrected-space
  *selection* independently earns real TOR skill for shape-based cohorts
  (Set A) — those two are presentable conclusions for the group talk.
