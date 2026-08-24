# Conversation-to-optimizer audit — 2026-08-24

This maps the Aug-2026 Ann discussion to executable explorer controls. It is an
audit of coverage, not a claim that every grid cell is scientifically equally
credible.

## What the app now searches automatically

| Discussion item | Automatic implementation | Status |
|---|---|---|
| Select on raw vs baseline-corrected spectra, independently calibrate on raw vs corrected | Corrected selection is on by default for Ethiopia-shaped and analog cohorts; raw/AIRSpec calibration remains independent, so all four cells are generated | complete |
| Compare the five cohort families | Ethiopia-shaped, analogs, lowest-OC/EC, biomass smoke, and the entire network | complete |
| Do not fix the analog cutoff at 500 | Ladder, dense cutoff step, wide ranges, and ±10 hill-climb | complete |
| Check Satoshi's `k≈21` idea and “everything in between” | Browser finalist scan defaults to every integer `k=1..30`; server batch offers sparse (always includes 21 and 30) or dense `1..30` | fixed in this pass |
| Focus on lot 251 and compare like with like | Optimizer defaults to both all-lot and train-lot 251; “match Addis evaluation lot” pairs train 251 with eval 251 | fixed in this pass |
| Minimize intercept, with slope as the second objective | `|intercept| + w·|slope−1|` plus an explicit slope box | complete, hardened |
| Reject a visually attractive but invalid winner | Held-out TOR R², score-space extrapolation %, slope range, and negative-prediction % are hard pass/fail guardrails | fixed in this pass |
| Ensure the answer is not city-specific | Server batch evaluates Addis, Delhi, Beijing, Pasadena, and Bishoftu; “robust” ranks each configuration by its worst loaded site | fixed in this pass |
| Check whether season structure survives | Every run returns group medians, their span, negative %, >8 %, and median; the Series tab still shows the dated/grouped record | quantified, not optimized |
| Inspect MAC 6/10 and the suggested MAC 17 | MAC 17 is now a sensitivity readout with the correct Deming-lambda scaling | fixed in this pass; deliberately not searched |

The focused UI smoke test evaluated lowest-OC/EC-800 × AIRSpec on train/eval lot
251. At MAC 10, fixed-set Deming, rule `k=4` read `0.92x−2.01` (held-out TOR
R² 0.93, extrapolation 4%, negative predictions 0%); explicit `k=21` read
`1.17x−1.81` (held-out R² 0.85, extrapolation 9%, negatives 0%). Under the
default objective, `k=21` therefore did **not** beat the rule choice: its small
intercept improvement cost slope and held-out/domain robustness. A bounded
saved-grid test loaded 7,607 rows and formed five-site worst-case rows without
browser warnings or errors.

## What should not become an optimizer knob

- **MAC choice.** MAC changes the x-axis scale and therefore the slope. Selecting
  MAC 17 because it makes a screened slope look closest to one would tune the
  reference definition, not improve the FTIR model. Report MAC 6/10/17 as a
  sensitivity analysis and choose the scientific MAC independently.
- **Locked reconstructed targets.** These remain unavailable to browser, batch,
  and cutoff optimization. They are confirmation data. Reusing them for tuning
  would erase the only useful protection against winner's curse.
- **Season ordering.** The optimizer records group medians but does not reward a
  preconceived seasonal order. That would force the answer being checked.
- **Adama.** The conversation asks for an Adama summary, not another calibration
  target. Its current sample size and data products do not support joining this
  optimization grid.

## What the existing data already answered

1. The selection×calibration `2×2` was run. Calibration on corrected spectra is
   what moves the intercept; corrected-space selection independently improves
   held-out TOR skill for the shape-based cohorts. The double-corrected cells can
   still collapse the slope, which is why the slope box is mandatory.
2. The lot hypothesis was run. Lot-251-only training/evaluation did not reproduce
   the baselining gain; it slightly worsened the raw offset and barely changed the
   corrected result. Baselining is not merely erasing the 248/251 split.
3. Dense cutoffs and the five-site grid were run. The Addis AIRSpec lowest-OC/EC
   basin is stable, while the apparent Delhi winner was 96.1% out of domain and
   failed on locked filters. No configuration reconciled Addis and Delhi.

Sources: `VARIANT_RESULTS_2026-08-19.md`, `FIVE_SITE_GRID_2026-08-23.md`, and
`MEETING_ACTIONS_2026-08-19.md`.

## Literature consequence for the workflow

The local design follows three relevant results:

- Reggente, Dillner, and Takahama (2016) use Mahalanobis distance in PLS score
  space to anticipate error on new sites. That supports the app's extrapolation
  guardrail, but the paper describes the distance as a crude diagnostic rather
  than a guarantee: <https://amt.copernicus.org/articles/9/441/2016/>.
- Varma and Simon (2006) show that reporting the same CV estimate used to tune a
  model is optimistically biased and that the whole tuning procedure must be
  nested or evaluated independently: <https://pmc.ncbi.nlm.nih.gov/articles/PMC1397873/>.
- Cawley and Talbot (2010) show that a model-selection criterion itself can be
  overfit, with effects comparable to differences between algorithms. This is
  why the app reports a candidate/Pareto set and keeps locked targets out of the
  search: <https://www.jmlr.org/papers/v11/cawley10a.html>.

The remaining methods upgrade is therefore not “search even more and quote the
best row.” It is an outer validation layer that reruns the complete cohort,
baseline, cutoff, lot, protocol, and `k` selection inside each outer split, then
reports selection frequency and uncertainty for the finalists.

## Still missing (next, in order)

1. **Nested selection validation / site-cluster bootstrap.** Required before a
   paper can attach an unbiased performance claim to the auto-selected winner.
2. **Model stability output.** Selection frequency by cohort/baseline/cutoff/lot
   and confidence intervals, not only one winning row.
3. **PLS Q-residual domain diagnostics.** Score distance detects leverage but not
   every spectral residual; add Q residual alongside extrapolation %.
4. **A scientifically independent baseline family.** Neutral baseline exists for
   site-spectrum diagnostics, but the full IMPROVE calibration pool has no
   committed neutral-baseline cache. It must not appear as an optimizer option
   until the pool and every target are processed reproducibly.
5. **External reference data.** Additional quartz TOR/HIPS pairs across seasons
   and sites will resolve more than further target-driven hyperparameter search.
