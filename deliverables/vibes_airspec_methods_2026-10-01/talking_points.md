# VIBES vs AIRSpec baselining: presenter brief

**One-breath summary:** AIRSpec reads the background off each spectrum, assuming it is
smooth and that fixed windows hold all the analyte, and only works from 4000 to 1425 cm⁻¹.
VIBES (pyvibes 1.0.0) learns the background from 87 blank filters and fits each sample
as a weighted mix of 30 blank patterns over the full measured range, 4000 to 500 cm⁻¹.
Full-range VIBES cleans blanks about 28x better, recovers added peaks better at every
height, and gives the best IMPROVE EC model. It also transfers worst to Addis (slope 1.52,
intercept −4.33 µg/m³), and loaded filters push it 4 to 6x beyond its blank library.
AIRSpec stays the EC default.

## Route through the deck (15 slides, ~15 min)

- **1-2:** roadmap; the two pipelines on one page.
- **3-4:** AIRSpec on one real filter (ETAD-0116-4, the median AIRSpec/VIBES difference
  of 253), full x-axis so the "no output below 1425" gap is visible; df sensitivity.
- **5-7:** VIBES: 87 blanks (59 IMPROVE + 28 ETAD) reduced to 30 patterns over 4000 to 500;
  the rank is at the cap, and the leave-one-blank-out error is still falling. Pinball loss
  (τ = 0.1, 9:1 penalty). The per-filter fit as mean blank + weighted patterns, through
  the PTFE bands.
- **8-10:** same filter through both; all 253 filters by region, including the four
  VIBES-only regions below 1425; **new** weight-size diagnostic (blanks 5.4, IMPROVE 19.9,
  Addis 31.4; prior ≈ 5.5).
- **11-13:** held-out blanks (135); injected peaks; EC on IMPROVE (n 2078) and Addis transfer (n 233).
- **14-15:** side-by-side table; takeaways and next steps.

## Key numbers

| | AIRSpec | VIBES full range |
|---|---|---|
| Range | 4000 to 1425 | 3998 to 500 (2722 channels) |
| Held-out blank RMS, 4000 to 1425 | 0.00055 | 0.000020 (28x lower) |
| Added-peak recovery RMSE, heights 0.01 / 0.05 / 0.15 | 0.00043 / 0.0018 / 0.0054 | 0.00034 / 0.00025 / 0.00014 |
| IMPROVE held-out TOR RMSE (µg/filter), R² | 3.31, 0.73 | 3.23, 0.74 |
| Addis vs HIPS Fabs/10, Deming (µg/m³) | 0.56x −1.08 | 1.52x −4.33 |

For context, the runs with VIBES restricted to 1425: VIBES cut gives 3.46 / 0.91x −2.75; fitting
the full range and using only 4000 to 1425 gives 3.41 / 0.73x −1.65.

## Questions likely to come up

- **Why full range?** pyvibes has no window. Cutting it to 1425 was only for like-for-like with
  AIRSpec. Below 1425 is information AIRSpec cannot give at all.
- **Is a cleaner blank proof VIBES is better?** No. Blanks are what VIBES is trained on, and much
  of AIRSpec's blank residual is the CO2 band near 2350 that a spline cannot remove.
- **Why is the full-range blank result 28x, when the cut run gave about 80x?** The same 30
  patterns now also have to cover the region from 1425 to 500.
- **Is rank 30 enough?** Probably not. The cap binds, and the uncapped one-SE rank is about 67 to
  70, but that run is only 3.5% complete.
- **Did you change pyvibes?** No. The vendored copy is byte-identical to pyvibes-main. The adapter
  only adds grid checks, optimizer logging, and one logged retry: 856 of 12,808 fits retried, 0 failed.

## Don't get caught

- IMPROVE EC errors are **µg/filter**; the Addis comparison is **µg/m³** against HIPS Fabs/10 at
  MAC 10 (an optical equivalent, not thermal EC). Deming λ = 3.34, from measurement uncertainties.
- The IMPROVE RMSE differences have no intervals. "Best" is a point estimate on one split.
- The weight-extrapolation result (slide 10) is **new and only in build.py**.
- In the earlier cut run, AIRSpec won at the 0.01 injection height; with the full range, VIBES wins there.
  The margin is small.
- The injection test reuses 9 parent blanks at each height: that is 9 units, not 27.
- The representative filter is typical by selection rule. It is not a mean spectrum.
- The earlier results deck (VIBES cut to 1425) is at `deliverables/airspec_vibes_2026-09-22/`.

## Rebuild

```bash
~/anaconda3/bin/python deliverables/vibes_airspec_methods_2026-10-01/build.py
```
