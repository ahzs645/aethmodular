# The five-site full grid — 2026-08-23

Every identified combination, evaluated at every site. Launched via
`/api/batch_start` with exactly:

```json
{"cohorts": ["eth_shaped", "analogs", "ocec", "smoke", "pool"],
 "spectra": ["raw", "airspec", "deriv2"],
 "modes": ["site_heldout"],
 "corrsel": true, "sweep_k": true, "cutoff_step": 10,
 "cutoff_ranges": {"eth_shaped": [100, 900], "analogs": [100, 1500],
                   "ocec": [100, 2000]},
 "targets": ["addis", "indh", "chts", "uspa", "etbi"]}
```

9,555 configuration×site jobs (≈1,911 configurations × 5 sites; targets are
the innermost batch loop so each fitted calibration is reused across sites),
k-sweep per configuration, protocol A only (B/B2 have no held-out test and
are excluded from honest ranking anyway). Completed same day — the curve
caches from the 08-20 dense sweeps carried most of the cost. Result:
`cache/batch_results.jsonl` now holds **71,263 rows**
(addis 21,955; indh / chts / uspa / etbi 12,327 each). Zero errors.

## Scoring lesson first (do not skip)

Ranked naively by `|intercept| + 0.5·|slope−1|` with only the held-out
floor, the "winners" at Addis are **slope-0.42–0.44 configurations with
intercepts ≈ 0** — the full-scale slope trap (ftir_17) reappearing at grid
scale: a flat enough line always has a small intercept. Every ranking below
constrains **slope to 0.85–1.18** first. The Optimize tab's score alone is
gameable; use it with the slope box or a larger w.

## Corrected audit: keep target fit separate from calibration held-out fit

The original version of this section mixed estimators and, for Delhi, described
the IMPROVE calibration's held-out TOR R² as if it were Delhi target fit.  The
saved rows have now been re-audited by
`research/ftir_ec_phase3/scripts/audit_five_site_grid.py`.  The table below uses
the **same Deming estimator everywhere** and prints both R² quantities.  `target
R²` is the SPARTAN crossplot; `TOR R²` is the held-out IMPROVE calibration test.

| site | reported family | Deming target readout | target R² | TOR R² | extrapolated |
|---|---|---:|---:|---:|---:|
| **Addis** | **ocec-440 × AIRSpec, k=8** | **0.96x−1.55** | 0.72 | 0.92 | unavailable in saved row |
| **Delhi** | **analogs-530 (AIRSpec-selected) × deriv2, k=20** | **0.87x−0.04** | 0.72 | 0.86 | **96.1%** |
| Beijing | ocec-1850 × deriv2, k=14 | 1.06x−0.00 | 0.60 | 0.91 | 20.8% |
| Pasadena | ocec-120 × raw, k=5 | 0.96x−0.01 | **0.18** | 0.97 | 3.2% |
| Bishoftu | ocec-1010 × AIRSpec, k=9 | 0.86x−0.21 | 0.60 | 0.86 | 7.7% |

Three readings:

1. **The Addis result is stable.** The wide range (100–2000, step 10) found
   nothing better than the known ocec-440/450 × AIRSpec k=8–9 basin — the
   08-20 dense-sweep conclusion survives a 4× wider search.
2. **Delhi's in-sample target line looked calibratable, but was almost wholly
   extrapolated.** The spectral-analog cohort at 530, AIRSpec-selected, in
   second-derivative space, hits Deming 0.87x−0.04 at Delhi, but **96.1% of
   Delhi filters lie beyond the training score-distance p95**.  The 26 newly
   reconstructed, never-screened Delhi filters confirm the concern: the locked
   k=20 model reads OLS 0.63x+1.11 (York 0.66x+0.87), not the screened line.
3. **Each site's optimum is a different cohort size** (120 → 440 → 530 →
   1010 → 1850): cohort choice is doing per-site work that a single global
   calibration cannot.

## The two-city result (the reason the grid was run)

The grid contains **12,327 unique configuration×k rows per target**, generated
from 1,911 base configurations.  Exactly **one** unique row lands in the slope
box at both Addis and Delhi under either consistent estimator:
analogs-440 × AIRSpec-selected × deriv2, k=20.  Under Deming it reads
0.99x−2.22 at Addis and 1.13x−0.69 at Delhi (target R² 0.65 / 0.64); under OLS,
0.88x−1.69 and 0.98x+0.26.  No row reaches target R² ≥ 0.70 at both sites.

> **No combination in the entire grid — any cohort, any size 100–2000, any
> spectral space, any k — gives a well-supported near-1:1 line at both Addis
> and Delhi.** The one slope-box row retains material offsets and moderate
> target R², and it fails on the newly reconstructed Addis/Delhi holdouts.
> Calibration choice alone does not reconcile the two cities.

## Provenance

- Rows: `cache/batch_results.jsonl` (dedup-keyed; the run is re-startable
  with the same JSON and skips everything already scored).
- Reproduction of the adjudication-week analyses:
  `research/ftir_ec_phase3/ftir_34_offset_adjudication_checks.ipynb`.
- Figures: `research/ftir_ec_phase3/output/plots/offset_story/`.
- UI equivalents: Optimize tab → Sites scope "all five", dense step 10,
  wide range; leaderboard "Rank on site" per city.
