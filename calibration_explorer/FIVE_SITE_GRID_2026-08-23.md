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

## Best per site (in-box, held-out R² ≥ 0.85)

| site | best family | readout |
|---|---|---|
| **Addis** | **ocec-440 × AIRSpec, k=8–9** | 0.89x−1.19 … 0.91x−1.25, ho 0.92 |
| **Delhi** | **analogs-530 (AIRSpec-selected) × deriv2, k=17–20** | 0.87x−0.04, ho 0.86–0.88 |
| Beijing | ocec-1840–1850 × deriv2, k=14 | 1.06x−0.00, ho 0.91 |
| Pasadena | ocec-120 × raw, k=5 | 0.96x−0.01, ho 0.97 |
| Bishoftu | ocec-1010–1020 × AIRSpec, k=9 | 0.86x−0.21, ho 0.86 |

Three readings:

1. **The Addis result is stable.** The wide range (100–2000, step 10) found
   nothing better than the known ocec-440/450 × AIRSpec k=8–9 basin — the
   08-20 dense-sweep conclusion survives a 4× wider search.
2. **Delhi is calibratable — by a different family.** The spectral-analog
   cohort at 530, AIRSpec-selected, in second-derivative space, hits
   0.87x−0.04 at Delhi. Treat with care: manual k=17–20, the analog/deriv2
   family is exactly the one that failed the ETBI transfer (0.51x), and
   12,327 Delhi rows were screened (winner's curse — confirm before
   adopting; the honest confirmation would be held-out Delhi seasons or a
   second Indian site).
3. **Each site's optimum is a different cohort size** (120 → 440 → 530 →
   1010 → 1850): cohort choice is doing per-site work that a single global
   calibration cannot.

## The two-city result (the reason the grid was run)

Configurations landing in the slope box at **both Addis and Delhi**: **2 of
~1,900** (analogs-440 × AIRSpec-sel × deriv2 k=20, twice via near-duplicate
rows) — and they still carry **−1.7 at Addis**. The Addis winner evaluated
at Delhi reads 2.36x−4.70.

> **No combination in the entire grid — any cohort, any size 100–2000, any
> spectral space, any k — reconciles Addis and Delhi simultaneously.** The
> Addis–Delhi difference is not a calibration-choice problem; it lives in
> the aerosol/reference side (consistent with the York adjudication: Addis
> is an intercept anomaly, Delhi a slope anomaly, different chemistry axes).

## Provenance

- Rows: `cache/batch_results.jsonl` (dedup-keyed; the run is re-startable
  with the same JSON and skips everything already scored).
- Reproduction of the adjudication-week analyses:
  `research/ftir_ec_phase3/ftir_34_offset_adjudication_checks.ipynb`.
- Figures: `research/ftir_ec_phase3/output/plots/offset_story/`.
- UI equivalents: Optimize tab → Sites scope "all five", dense step 10,
  wide range; leaderboard "Rank on site" per city.
