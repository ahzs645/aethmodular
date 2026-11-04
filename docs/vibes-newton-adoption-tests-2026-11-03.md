# VIBES with the ELBO solved to convergence: adoption tests

Date: 3 November 2026. Follows [vibes-reference-webgpu-decision-2026-11-03.md](vibes-reference-webgpu-decision-2026-11-03.md),
which proposed this sequence: port L-BFGS-B, refit the IMPROVE calibration, rerun the analyses that read corrected
spectra, then publish a versioned run. The figures and tables are in
[`vibes_newton_refit_20261103.ipynb`](../research/ftir_hips_chem/vibes_newton_refit_20261103.ipynb).

## Summary

| Test | Result |
|---|---|
| L-BFGS-B port (JAX) | Reaches the noise floor: 144 of 155 premature stops reproduced, against 15 for the earlier JAX code |
| Calibration rebuilt on Newton baselines | Same k (4), same test RMSE (3.850), coefficients r = 0.999998 |
| EC scoring at k 5-18 | No conclusion changes; IMPROVE MAE within 0.006 µg |
| Functional-group band areas | Unchanged on ~97 % of spectra; the rest move, Addis carbonyl and 1600 cm⁻¹ most (r ~0.95) |
| Lots and spikes | Blank residue same or better, 8 failed fits become 0, spike recovery unchanged; lot 241a spike error worse at 2 amplitudes |
| Blank banks | Same conclusions; held-out residue lower |
| Continuum | Unchanged: production VIBES still returns ~0.7 % of an injected continuum |
| The 10 Addis early stops | All 10 flatter in a featureless window (p = 0.002). No clear gain against the site line: 6 of 10 closer (p = 0.75) |
| Continuum-protected background | Has no optimiser problem: SciPy and Newton agree to better than 0.03 % |
| v2 run | Built next to the 19 Oct run: 4,452 of 4,452 fit and certified, 155 flagged |

The changes hold up. Nothing that was checked gets worse in a way that matters, except the small lot 241a spike result
below. EC-level conclusions do not depend on the optimiser. Individual spectra do.

## 1. L-BFGS-B port

`vibes_jax.fit_elbo_jax(engine="lbfgsb")` implements SciPy's `__lbfgsb.c` (v1.16.3): generalised Cauchy point with
breakpoints, direct primal subspace step, Morales-Nocedal projection, memory refreshes, and the block factorisations of
`bmv` and `formk`. It also carries the gradient of the fixed tau, which L-BFGS-B keeps in `y` and which therefore sets
`theta`. The earlier batched code missed that term as well as the Cauchy step.

Whole network, A100, against the saved run:

| Run | sigma > 10 % | > 2× | Baselines moved > 5 % | > 20 % | Stops < 30 its | Shared with saved (155) | ELBO time |
|---|---|---|---|---|---|---|---|
| Earlier JAX | 13.3 % | 9.8 % | 467 | 277 | 281 | 15 | 144 s |
| **L-BFGS-B port** | **6.8 %** | **1.8 %** | **131** | **34** | **150** | **144** | 185 s |
| Fused SciPy (noise floor, CPU) | 5.6 % | 1.1 % | 100 | 23 | 156 | 154 | ~420 s, 44 cores |

Vendored SciPy on the Mac differs from the saved run more than the port does (sigma > 10 % on 15 % of 440 spectra). A
separate one-off compile takes ~95 s.

It tracks SciPy iterate by iterate: within 1e-13 for the early iterations on typical spectra, with the same evaluation
counts, then it separates chaotically. Real SciPy does the same against itself: on row 3802 the fused and vendored runs
jump from 5e-12 to 1e-5 at the same iteration. The port stays within about an order of magnitude of that self-divergence
until both are fully decorrelated.

Limit: the ELBO's `-log d` keeps iterates off the bound, so the case of a `d` resting at its bound across iterations was
never exercised. The unit tests pin the port to SciPy on Cauchy breakpoints, not on that path.

The port is the fast way to run a procedure close to upstream's. It does not change the recommendation: its fits stop at
the same early points SciPy's do.

**A tighter `ftol` alone is not a cure.** On 60 premature-stop spectra with `maxiter` 10,000:

| `ftol` | Median iterations | Median ELBO gap to optimum | Spectra > 1 from optimum |
|---|---|---|---|
| 2.2e-9 (default) | 11 | 5,963 | 60 |
| 1e-12 | 4,506 | 935 | 36 |
| 1e-15 | 8,300 | 41 | 33 |
| 0 | 10,000 | 3.6 | 30 |

The optimiser is slow on these problems, not only stopping early. The default's spectra cost almost no iterations; the
tighter settings run about 100× longer and half still fall short.

## 2. Calibration

All 16,143 spectra were refit with Newton in ~7 minutes on the Mac, all converged and certified. The IMPROVE training set
has the same problem as the network: 297 of 12,393 stopped within 30 iterations (2.4 %), 306 baselines change by > 5 %,
114 by > 20 %, and 5.8 % of fits are more than 0.1 % below the optimum.

Rebuilt with the frozen procedure (site-grouped 5-fold curve, first major minimum, fit on the train split). On the saved
baselines, the rebuild returns the frozen coefficients to 6e-15, so it is the same procedure.

| Baselines | Selected k | Test R² | RMSE | MAE | Bias |
|---|---|---|---|---|---|
| Saved | 4 | 0.5416 | 3.8501 | 2.2612 | 0.2106 |
| Newton | 4 | 0.5417 | 3.8499 | 2.2611 | 0.2117 |

Coefficients correlate at 0.999998. The calibration does not care which optimiser produced its training baselines.

Transfer scoring at higher k, same code as the 2 Nov diagnosis:

| k | IMPROVE out-of-site MAE, saved / Newton (µg) | Addis OLS slope vs HIPS, saved / Newton |
|---|---|---|
| 5 | 1.335 / 1.339 | 0.426 / 0.414 |
| 8 | 1.252 / 1.248 | 1.025 / 1.029 |
| 12 | 1.145 / 1.141 | 1.391 / 1.410 |
| 18 | 1.154 / 1.148 | 1.724 / 1.744 |

Addis against the TOR yardstick at k 18: 0.667 saved, 0.655 Newton. Higher k is more sensitive on single filters: at
k 18 one Addis prediction moves by 10.6 µg (mean 0.28 µg), against a maximum of about 2.4 µg at the frozen k 4.

## 3. Analyses that read corrected spectra

**Functional-group band areas.** Windows from `run_meeting_20261028`.

| Group | Band | Moved > 5 % | > 20 % | r (saved, Newton) |
|---|---|---|---|---|
| Addis (233) | Carbonyl 1650-1800 | 8 | 7 | 0.953 |
| | 1600 band 1500-1650 | 8 | 7 | 0.955 |
| | Fingerprint 1250-1500 | 8 | 2 | 0.962 |
| | C-H 2800-3000 | 7 | 2 | 0.996 |
| Network (3,517) | Carbonyl | 61 | 18 | 0.999 |
| | C-H | 85 | 45 | 1.000 |
| IMPROVE (12,393) | C-H | 298 | 183 | 1.000 |

This is the main place the switch matters. Any filter-level reading of carbonyl or the 1600 band on a premature-stop
spectrum was reading a residual baseline. Every Addis outlier in those two panels falls below the 1:1 line.

**Lots** (`vibes_lot241_20261026`, 6,321 jobs, its own scoring). Blank residue is the same or lower in every lot and
family, for example lot 241a under the 87-blank bank: 6.2e-4 to 5.0e-4. Held-out lot 241 blanks keep their gain. All 8
fits that failed in the `lot241_spartan_r70` family now succeed. Spike recovery is unchanged to about 0.01. One result
goes the other way: lot 241a `rel_rmse` under the 87-blank bank rises at amplitudes 0.05 and 0.15 (0.22 to 0.31, 0.10 to
0.13), and improves at 0.01 (0.61 to 0.46). That group has 5 blanks.

**Blank banks** (`run_oct1_remaining_20261007`, its own scoring). Same selected k and same conclusion: more blanks do not
fix VIBES. IMPROVE test scores match to 3 decimals; Addis standard-model slopes move by about 0.02 at most. Held-out blank
residue falls on the rank-80 bank (PTFE p90 8.9e-5 to 3.7e-5). Season-group analog medians for `allblanks-x1` move more
(Belg slope 2.03 to 1.66); those are small repeated subsets.

**Continuum.** Injection recovery is unchanged: 0.66 % for the production background, 100 % for the deflated ones. The
continuum loss belongs to the background, not the optimiser. On the continuum-protected `anchorhp` background, Newton and
SciPy agree to 2.5e-4 of corrected RMS at worst and SciPy takes about 19 iterations. The scores match to 3 decimals. So
the premature stops belong to the production background's conditioning. A switch to `anchorhp` would remove the optimiser
question altogether.

## 4. The Addis scans, and a correction

The previous note read Newton as slightly further from HIPS. That test was biased, because HIPS sits far above FTIR EC at
Addis, so any downward move looks worse. Two fairer checks:

- **Within-site line.** Fit EC on HIPS using only each site's unchanged filters, then measure the changed filters'
  residual. Network-wide this is a tie (40 for Newton, 35 for saved). All 10 Addis early-stop PM2.5 filters move down, by
  6-23 %, and 6 of 10 land closer (sign test p = 0.75). Mean |residual| falls from 0.20 to 0.15 µg/m³ (Wilcoxon p = 0.19),
  and the mean signed residual goes from +0.10 to -0.06. This check does not separate the two runs.
- **Featureless window, 2000-2500 cm⁻¹.** Mean |corrected absorbance| should be near zero there. The 10 Addis early stops
  fall from a median of 0.0011 to 0.00015, 10 of 10 flatter (sign test p = 0.002). Addis stands out: the saved value is
  about ten times other sites', which show no pattern (USPA 4 of 9, TWTA 4 of 8, CASH 5 of 7, PRFJ 0 of 5). Network-wide it
  is mixed: 71 to 61 for changed PM spectra, 63 to 71 for blanks.

So at Addis the premature stops left a real broad hump, and Newton removes it. That is shown by the spectra, and it is
why EC falls on all 10. Whether the lower EC is more correct is not shown: the site line can't separate the runs. ETAD-0053
is the clearest single case. Elsewhere the effect is neutral on these measures.

An earlier version of this note reported "6 of 7 closer, mean residual +0.21 to +0.02". Those seven were the Addis filters
whose EC moved by more than 0.1 µg/m³, a cut set before the test but one that falls between ETAD-0056 (-0.100, closer) and
ETAD-0150 (-0.099, further). All three of the left-out filters moved away from the line, so that figure overstated the
result. Per-filter values are in `vibes_newton_20261103/addis_premature_checks.csv`.

## 5. The v2 run

`output/tables/network_baselines_v2_20261103/` sits next to the 19 Oct run, which is untouched.

- Same inputs, background and tau; lot 241/241a analyses use the gallery's override banks (76 analyses).
- 4,452 of 4,452 fit and certified, including the 3 field blanks that failed on 19 Oct.
- No fit has a lower ELBO than the 19 Oct run.
- `vibes_fit_diagnostics.csv` carries `premature_stop` (155 analyses), `elbo_gain`, `baseline_change_frac_rms`, and EC
  under the frozen and rebuilt models. `changed_or_premature.csv` is the short list.
- For the 76 lot-241 analyses, `baseline_change_frac_rms` mostly reflects the bank, not the optimiser; compare on
  `vibes_model == "b87"` (266 moved > 5 %, 106 > 20 %).

The gallery's SPARTAN pages still read the 19 Oct run; the Experimental tab's "VIBES solved to convergence" page shows
v2 against it. Making v2 the default means pointing `export_spartan_network.py` and `export_baseline_ftir_ec.py` at the v2
folder and cutting a data release.

## Limits

- EC comparisons use the frozen model family, at k 4 and in the transfer harness at k 5-18. The published SPARTAN EC is a
  separate model that we do not have.
- "Holds up" covers the analyses tested. Season-composition and fingerprint-grouping results were not rerun.
- Timings are from single runs, some sharing the machine.
- Only 10 Addis filters and 75 network filters carry the within-site test, and it has little power at that size.

## Reproduce

```bash
W=research/ftir_hips_chem/workflows/vibes_newton_refit_20261103.py
python $W fit --bundle DECISION_BUNDLE --background background_model.npz --out OUT
python $W calibration --bundle DECISION_BUNDLE --out OUT --area AREA
python $W anchorhp --continuum AREA/output/tables/vibes_continuum_20261102 --out OUT
python $W transfer|inject --bundle DECISION_BUNDLE --out OUT --area AREA
python $W bands --bundle DECISION_BUNDLE --out OUT
python $W lots --out OUT --area AREA
python $W banks --out OUT --area AREA --blankbank "DRIVE/results/blankbank_20261007"
python $W publish --out OUT_V2 --area AREA --pls OUT/pls_full_pool_newton_k4.npz
python research/ftir_hips_chem/workflows/build_vibes_newton_notebook.py
```
