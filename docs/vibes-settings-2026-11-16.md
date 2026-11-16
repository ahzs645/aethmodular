# VIBES settings: the solver, the rank and tau, and what moves EC instead

Date: 16 November 2026. Follows [vibes-newton-adoption-tests-2026-11-03.md](vibes-newton-adoption-tests-2026-11-03.md)
and [vibes-rank-and-wasm-2026-11-10.md](vibes-rank-and-wasm-2026-11-10.md).

- Notebook, executed: [`research/ftir_hips_chem/vibes_rank_tau_ec_20261116.ipynb`](../research/ftir_hips_chem/vibes_rank_tau_ec_20261116.ipynb)
- Gallery page: Meeting tab, Nov 16, "Which VIBES setting" (`#tab=meeting&meeting=vibes-settings-20261116`)
- Decisions it leaves open: [open-items.md](open-items.md#decide-the-training-set-and-k-of-the-improve-ec-calibration-together-16-nov-2026)

The question was which VIBES settings to use for new work, and whether the browser tool needs presets. EC is in µg per
filter throughout. Unless a section says the calibration was rebuilt, EC is read with the frozen k 4 IMPROVE model
(`pls_full_pool_uncapped_full.npz`), which was trained on spectra corrected at rank 70 and tau 0.1.

## Summary

| Question | Answer (measured) |
|---|---|
| Which settings for new work? | **Converged solver, rank 70, tau 0.1.** No other setting has evidence behind it. |
| Does the browser tool need presets? | **No.** It opens on rank 70, tau 0.1, converged, with a "Reset rank and tau" button. |
| Does the converged solver change EC? | **Only on the October run's early stops.** 3,529 PM2.5 scans move by at most 0.09 µg. The 97 early stops move by a median 5 %; the 10 at Addis by a median 11 %, all down. |
| Does the rank matter? | **Not from 40 up.** Rank 40 gives the same EC as rank 70 (median change 0.01 %, largest 0.03 µg). Rank 20 moves EC 0.5 %, rank 10 moves it 3 %. |
| Does tau matter with the frozen calibration? | **A little.** tau 0.05 moves EC a median 1.2 %, tau 0.2 moves it 2.0 %. That is the cost of a mismatch, not a ranking. |
| Is another tau better once the calibration is rebuilt for it? | **No.** No tau is better at every k. tau 0.05 is never better than 0.1. tau 0.2 is better at k 4 and 8 on median error and worse at k 11 and 18 on RMSE. |
| Is the rule that picks k stable? | **No.** It picks k 4, 11 and 4 at tau 0.1, 0.05 and 0.2, and 4, 8 and 3 with one QC-failed spectrum left out. |
| Does one training spectrum matter? | **Yes, at low k.** With or without IMPROVE filter 2008808, held-out RMSE is 3.85 or 3.87 µg, but SPARTAN PM2.5 EC from the k 4 model moves by a median 17 %. |
| Does trimming the heaviest training filters pin EC down? | **No.** It removes the pull of single spectra (6 % to 0.4 %), but EC still spreads 8 % when IMPROVE sites are resampled, and the cut-off itself moves EC by 3 to 30 %. |
| Does the Addis slope against HIPS still follow k with the training set held fixed? | **Yes, under all six sets.** And the set moves it at every k: trained below 25 µg, the slope is about half and never reaches 1. |

The solver, the rank and tau are settled. What moves EC is the calibration: its number of components and its training
spectra. Those are open, and they are one decision.

## 1. The solver

VIBES fits a noise level for each scan, then the baseline. In the 19 Oct run the optimiser stopped within 30 steps on
155 scans, leaving the noise level too high. The converged solver fits the same objective until it cannot improve.

| PM2.5 scans (87-blank background) | Count | Median change | Largest | Moved down / up |
|---|---|---|---|---|
| Not an early stop | 3,529 | 0.002 % | 0.09 µg | 2,130 / 1,399 |
| Early stop in October | 97 | 5.1 % | 2.45 µg | 76 / 21 |
| Addis early stops | 10 | 11.4 % | 2.45 µg | 10 / 0 |

CHTS-0830, analysis 1975, is an early stop: 4.64 µg in October, 4.79 µg converged (+3.3 %). The filter's other scan,
analysis 1929, reads 13.17 µg under either solver. Which scan of a filter is read can matter more than the solver.

The Addis early stops are flatter in a featureless window after converging (3 Nov tests). That their lower EC is more
correct is not shown.

## 2. The rank

Every one of the 4,376 scans saved under the 87-blank background was refitted to convergence at ranks 70, 40, 20 and
10 and tau 0.05, 0.1 and 0.2. All were certified at all 12 settings. The rank was lowered by keeping the first
components of the rank-70 background, not by building a new background at each rank.

| Rank, at tau 0.1 | Median EC change | Scans moved over 10 % | Largest | Field-blank spread (IQR) | Repeat-scan gap |
|---|---|---|---|---|---|
| 70 | 0 | 0 | 0 | 0.002 µg | 0.052 µg |
| 40 | 0.01 % | 0 % | 0.03 µg | 0.006 µg | 0.052 µg |
| 20 | 0.49 % | 3.5 % | 0.52 µg | 0.022 µg | 0.055 µg |
| 10 | 2.96 % | 17.5 % | 1.80 µg | 0.080 µg | 0.056 µg |

Nothing is gained by lowering the rank, and field blanks are tightest at 70.

## 3. tau with the frozen calibration

| tau, at rank 70 | Median EC change | Scans moved over 10 % | Largest | Median noise level (×10⁻⁴) |
|---|---|---|---|---|
| 0.05 | 1.17 % | 3.3 % | 14.4 µg | 3.1 |
| 0.1 | 0 | 0 | 0 | 6.1 |
| 0.2 | 2.03 % | 6.0 % | 3.9 µg | 11.8 |

These are the cost of correcting a spectrum one way and calibrating another. The fitted noise level follows tau, so a
noise level only compares between fits made at one tau.

## 4. tau with the calibration rebuilt

IMPROVE (12,393 spectra) and Addis (233) were corrected at each tau and each tau was scored with its own model. At tau
0.1 the rebuild returns the 3 Nov calibration. Scores are compared at a fixed k.

| Held-out RMSE (µg), 2,327 filters | tau 0.05 | tau 0.1 | tau 0.2 |
|---|---|---|---|
| k 4 | 3.88 | 3.87 | 3.85 |
| k 8 | 3.28 | 3.23 | 3.24 |
| k 11 | 3.24 | 3.17 | 3.31 |
| k 18 | 3.00 | 2.90 | 2.99 |

Against tau 0.1 on the same filters, with sites resampled 2,000 times (90 % interval):

- tau 0.05 is never better. It is worse at k 11 on median error (+0.13 µg), at k 18 on RMSE (+0.10 µg) and at k 8
  out of site (+0.02 µg).
- tau 0.2 is better on median error at k 4 (−0.05 µg) and k 8 (−0.12 µg), and out of site at k 8 and 12. It is worse
  on RMSE at k 11 (+0.15 µg) and k 18 (+0.09 µg).
- The gap between k 4 and k 11 is several times the largest gap between tau values at one k (0.15 µg).

## 5. The rule that picks k

The frozen procedure draws a site-grouped cross-validation curve on the train split and takes the first major minimum.

| Train split | k at tau 0.05 | k at tau 0.1 | k at tau 0.2 |
|---|---|---|---|
| Whole (the frozen procedure) | 11 | 4 | 4 |
| Without the QC-failed spectrum | 8 | 4 | 3 |

An early dip near k 4 and a later one near k 8 to 11 are close in depth, so small changes swap them. **Compare settings
at a fixed k.** A comparison at the k the rule picks compares k.

## 6. One QC-failed training spectrum

IMPROVE filter 2008808 (TONT1) fails raw QC in the 22 Oct decision bundle and is in the train split of the frozen k 4
model. Its corrected spectrum is 364 times farther from the mean than a typical train spectrum. The k 4 model was
fitted with and without it and both were applied to the converged network run.

| Scans | Count | Median EC with it | without it | Median change | Scans moved over 10 % |
|---|---|---|---|---|---|
| PM2.5 | 3,626 | 3.76 µg | 4.33 µg | 17 % | 64 % |
| Addis PM2.5 | 274 | 7.88 µg | 8.06 µg | 5 % | 26 % |
| PM10 | 215 | 2.33 µg | 1.68 µg | 30 % | 89 % |
| Field blanks | 535 | 1.27 µg | 0.30 µg | | |

- Site medians move from −64 % (PRFJ) to +51 % (ILHA), so the order of sites changes with one training spectrum.
- IMPROVE cannot tell the two models apart: held-out RMSE 3.850 against 3.873 µg.
- No check picks a winner. Blanks read closer to zero without it; repeat scans agree less well (0.051 against
  0.074 µg); the Addis slope against HIPS is 0.41 either way.
- The pull shrinks with k: 17 % at k 4, 8 % at k 8, 3 % at k 12.
- The next most extreme train spectra passed QC and are heavy filters (thermal EC 78 to 393 µg). Dropping one of them
  alone moves network EC by up to 6 %. Dropping 100 random spectra moves it 0.25 %.

The 3 Nov transfer table was made with this spectrum in training. The scoring harness
(`run_transfer_diagnosis_20261102`) now leaves it out, so that table's low-k Addis slopes are stale: at k 5 the slope
is 0.61, not 0.41. From k 12 up the two agree.

### A correction

An earlier reply on 16 Nov said the k 4 model "barely moves" without this spectrum. That was read from the held-out
RMSE alone and was wrong. Judge a model change by applying both models to the network scans, not by IMPROVE scores.

## 7. Training without the heaviest filters

No network PM2.5 scan reads above 50 µg and 1.5 % read above 25 µg, so the heaviest training filters lie outside
SPARTAN's range. Six training sets were tried. Each model was refitted with one of its six most extreme spectra
dropped, and on IMPROVE sites drawn with replacement 30 times.

| Training set, k 4 | Spectra | One spectrum dropped | Scan spread, sites resampled | Site-median spread | Worst site | Held-out RMSE below 25 µg | EC change from "all" |
|---|---|---|---|---|---|---|---|
| All QC-passed | 10,065 | 6.1 % | 13.9 % | 10.8 % | 61 % (PRFJ) | 3.04 µg | |
| EC below 150 µg | 10,059 | 2.9 % | 14.3 % | 10.8 % | 33 % (PRFJ) | 2.94 | 7.9 % |
| EC below 100 µg | 10,053 | 2.9 % | 11.7 % | 8.4 % | 33 % (AEAZ) | 2.88 | 5.7 % |
| EC below 50 µg | 10,005 | 2.0 % | 9.3 % | 7.5 % | 19 % (AEAZ) | 2.71 | 11.6 % |
| EC below 25 µg | 9,824 | 0.4 % | 7.6 % | 5.9 % | 13 % (ETAD) | 2.66 | 17.7 % |
| Nearest 99 % of spectra | 9,964 | 0.3 % | 12.7 % | 10.4 % | 83 % (PRFJ) | 3.25 | 9.1 % |

- One spectrum stops mattering.
- The training sites still matter: 8 % per scan and 6 % per site median at the tightest cut-off.
- Each cut-off gives a different EC. The six k 4 models are a median 3 to 30 % apart on network scans. Addis's median
  goes from 8.1 µg to 5.9 µg.
- A higher k does not remove the spread: below 25 µg it is 7.6, 7.0 and 6.6 % at k 4, 8 and 12.
- Trimming by spectral distance is worse than trimming by EC. It keeps filters up to 256 µg, and PRFJ's spread grows.

## 8. The Addis slope with the training set held fixed

Earlier work read the Addis slope of FTIR EC on HIPS BC as following the number of PLS components (the 14 Oct k
sweep, and step 4 of the gallery page). Section 7 showed the slope also moves with
the training set, so the reading was repeated with the training set held fixed: one model per k 1 to 25 for each of
the six sets, applied to the 233 Addis spectra. The range in brackets is the 5th to 95th percentile over 30 refits on
resampled IMPROVE sites.

| Training set | Slope at k 4 | k 8 | k 12 | k 18 | First k with slope ≥ 1 |
|---|---|---|---|---|---|
| All QC-passed | 0.41 (0.29 to 0.61) | 0.94 (0.64 to 1.15) | 1.45 (1.16 to 1.72) | 2.11 (1.56 to 2.50) | 9 |
| EC below 150 µg | 0.45 (0.27 to 0.69) | 0.98 (0.65 to 1.17) | 1.42 (1.05 to 1.63) | 1.84 (1.38 to 2.12) | 9 |
| EC below 100 µg | 0.42 (0.30 to 0.56) | 0.96 (0.77 to 1.13) | 1.34 (1.12 to 1.51) | 1.68 (1.33 to 1.87) | 9 |
| EC below 50 µg | 0.30 (0.25 to 0.39) | 0.71 (0.60 to 0.89) | 1.05 (0.87 to 1.28) | 1.20 (0.99 to 1.40) | 11 |
| EC below 25 µg | 0.23 (0.18 to 0.31) | 0.50 (0.36 to 0.58) | 0.72 (0.60 to 0.85) | 0.81 (0.61 to 0.91) | never |
| Nearest 99 % of spectra | 0.55 (0.37 to 0.71) | 1.10 (0.78 to 1.25) | 1.79 (1.46 to 1.99) | 2.05 (1.70 to 2.40) | 8 |

- **The slope follows k under every training set.** The earlier finding holds.
- **The training set moves it too, at every k.** Trained below 25 µg, the slope is 0.38 to 0.56 of the slope with every
  spectrum in, and it never reaches 1.
- **Cut-offs of 150 and 100 µg are not distinguishable from "all".** From k 2 up they stay inside the band that
  resampling IMPROVE sites draws around it.
- **The training sites alone move the slope by 20 to 50 %.** That is the width of the band.
- **R² does not choose.** It stays between 0.58 and 0.78 from k 3 up under every set. k and the training set change
  the scale of Addis EC, not how closely it follows HIPS. The median Addis EC is 8.1 µg at k 4 and 18.0 µg at k 12
  with every spectrum in; 5.9 and 13.0 µg below 25 µg.

So "the k that matches HIPS" is not a property of k alone. A result of the form "the slope follows k" holds only for a
stated training set. The slopes in step 4 of the gallery page (0.42 at k 4, 1.78 at k 18) come from a third set, every
eligible IMPROVE filter below 150 µg rather than the train split, which is why they differ slightly from this table.

## 9. What follows from it

1. **The frozen model stays as it is.** A trimmed training set is steadier and more accurate on light IMPROVE filters.
   It also lowers Addis EC by about a quarter and halves the Addis slope, and nothing measured here says the lower
   reading is more correct. That needs a reference SPARTAN does not have: thermal EC on SPARTAN filters.
2. **Quote EC from this calibration with a spread.** The table below is the spread for a site's median EC from the k 4
   model with every QC-passed spectrum in. It is the standard deviation over 30 refits on resampled IMPROVE sites. It
   does not include the choice of k or of training set, which move EC by more.
3. **The training set and k are one decision, for Ann and Satoshi.** See
   [open-items.md](open-items.md#decide-the-training-set-and-k-of-the-improve-ec-calibration-together-16-nov-2026).

| Site | Median EC (µg) | Spread | | Site | Median EC (µg) | Spread |
|---|---|---|---|---|---|---|
| PRFJ | 0.54 | 61 % | | AUMN | 1.90 | 11 % |
| AEAZ | 2.66 | 40 % | | CASH | 2.63 | 11 % |
| ILHA | 2.93 | 22 % | | CAHA | 2.00 | 10 % |
| ILNZ | 3.72 | 20 % | | MXMC | 5.76 | 10 % |
| NGIL | 3.75 | 19 % | | INJA | 6.69 | 10 % |
| ETAD | 8.06 | 18 % | | TWTA | 2.61 | 9 % |
| KRSE | 6.15 | 17 % | | ZAJB | 5.79 | 9 % |
| BDDU | 6.95 | 16 % | | CLST | 5.56 | 9 % |
| CLTA | 2.03 | 15 % | | ZAPR | 5.47 | 8 % |
| IDBD | 9.94 | 14 % | | TWKA | 6.16 | 8 % |
| BIBU | 12.53 | 13 % | | CHTS | 7.07 | 8 % |
| ETBI | 4.04 | 12 % | | KRUL | 3.51 | 7 % |
| INDH | 7.09 | 12 % | | USPA | 2.67 | 5 % |
| | | | | USSL | 3.13 | 4 % |
| | | | | USNO | 3.08 | 3 % |

Twelve sites are under 10 %, thirteen between 10 and 20 %, and PRFJ, AEAZ and ILHA above 20 %. A site-to-site
comparison of EC from the k 4 model is not safe for those three.

## Limits

- Neither model in section 6, and none in section 7, has been checked against thermal EC on SPARTAN filters.
- HIPS is optical and reads low on dark filters, so the Addis slope is a comparison, not a score against truth.
- Ranks below 70 are truncations of the rank-70 background.
- The resampling uses 30 draws, enough for a spread, not for a tail.
- The background was not tested here. A feature about 1,500 cm⁻¹ wide is removed at every rank (10 Nov check).
- Season composition and fingerprint grouping have still not been rerun on the converged run.

## Reproduce

The run tables are under `research/ftir_hips_chem/output/tables/` (git-ignored): `vibes_rank_tau_ec_20261116/` and
`vibes_tau_calibration_20261116/`. The calibration refit needs the 22 Oct decision bundle, which is only on Drive
(`My Drive/Aethmodular VIBES/decision_workbook_20261022/bundle`).

```bash
uv run python research/ftir_hips_chem/workflows/run_vibes_rank_tau_ec_20261116.py --area research/ftir_hips_chem --out research/ftir_hips_chem/output/tables/vibes_rank_tau_ec_20261116
uv run python research/ftir_hips_chem/workflows/run_vibes_tau_calibration_20261116.py --bundle BUNDLE --area research/ftir_hips_chem --out research/ftir_hips_chem/output/tables/vibes_tau_calibration_20261116
uv run python research/ftir_hips_chem/workflows/build_vibes_rank_tau_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace research/ftir_hips_chem/vibes_rank_tau_ec_20261116.ipynb
uv run python gallery/data/export_meeting_20261116_vibes_settings.py
```

| Piece | Where |
|---|---|
| Every scan at 12 rank and tau settings | `workflows/run_vibes_rank_tau_ec_20261116.py` |
| Calibration at each tau, the QC-failed spectrum, the training sets, the Addis slope by k | `workflows/run_vibes_tau_calibration_20261116.py` (`main`, `bootstrap`, `qc_spectrum_effect`, `training_domain_stability`, `addis_by_training_set`) |
| Notebook builder | `workflows/build_vibes_rank_tau_notebook.py` |
| Gallery page | `gallery/data/export_meeting_20261116_vibes_settings.py`, `gallery/app/src/meetings/vibes-settings-20261116.meeting.ts`, test `gallery/app/src/lib/vibesSettings.test.mjs` |
