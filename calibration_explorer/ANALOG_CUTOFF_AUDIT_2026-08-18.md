# Spectral-analog cutoff audit — 2026-08-18

## Question

Does “Spectral analogs 500” mean the first 500 rows of the full ranking followed
by TOR-eligibility filtering, or the first 500 TOR-eligible filters in the ranking?

The locked phase-2 cohort uses the latter definition. Under the former definition,
the raw and AIRSpec-corrected selections resolve to 477 and 486 fitted filters,
respectively.

## Test

Both definitions were reconstructed from the same raw and corrected rankings. Each
cohort was calibrated on **raw spectra**, using the locked site-held-out protocol,
automatic component selection, and a component curve through 24. Addis crossplot
statistics below use the fixed deployment-period subset at MAC = 10. Held-out metrics
are against the disjoint TOR test sites.

| Selection space | Cutoff rule | Cohort n | Train n | k | Addis OLS | Addis R² | Held-out TOR R² | Held-out RMSE |
|---|---|---:|---:|---:|---|---:|---:|---:|
| raw | cutoff, then eligibility | 477 | 361 | 4 | 2.5688x − 6.5597 | 0.7442 | 0.6189 | 3.6271 |
| raw | **eligibility, then cutoff** | **500** | **372** | 4 | 2.4814x − 6.3467 | 0.7464 | **0.3666** | 3.8393 |
| AIRSpec corrected | cutoff, then eligibility | 486 | 366 | 17 | 1.5556x − 2.8471 | 0.7245 | 0.7129 | 1.3928 |
| AIRSpec corrected | **eligibility, then cutoff** | **500** | **379** | **15** | 1.6337x − 3.1697 | 0.7319 | **0.7041** | 1.4055 |

The raw and corrected eligibility-first top-500 cohorts overlap by 4 filters.

## Conclusion

The mismatch is real and material. Adding the 23 eligible raw analogs changes the
held-out conclusion substantially; the earlier 0.62 value must not be described as a
top-500 result. Corrected-space selection still performs much better on the held-out
TOR test after the correction, although its selected k and fitted line change.

`calibration_explorer.app._ranking` now restricts rankings to the TOR-eligible pool
before `resolve_cohort` applies cutoff N. A cold-process provenance check reproduces
the locked raw cohort at 500/500; both raw and corrected requests resolve to exactly
500. An already-running Flask process must be restarted to load this behavior.

