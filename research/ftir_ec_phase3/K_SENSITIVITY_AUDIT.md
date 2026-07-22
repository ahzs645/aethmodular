# Which phase-3 figures depend on the component choice?

Every calibration in this project is PLS, and the number of components `k` comes from a
**protocol** — see `scripts/calibration_modes.py`, which defines the two:

- **Calibration app** — 10-fold interleaved CV, k = first within 5% of the curve minimum,
  final model fitted on all cohort filters.
- **Site-held-out** — site-grouped 5-fold CV, k = first major minimum, fitted on the
  training side of a site-disjoint 80/20 split (seed 20260717).

ftir_21 shows the protocol moves the Addis intercept by up to 1.4 µg/m³ on a fixed cohort.
So the fair question is: **which committed figures would change if the protocol changed?**
This audit answers that figure by figure, so we know what has to be re-run and what does
not. Three categories:

- **k-dependent** — a calibration is fitted and its predictions are plotted. Changes.
- **structurally invariant** — the figure's *conclusion* is a property of the algebra or of
  the data, and holds for any k. The plotted numbers may move; the claim does not.
- **k-free** — no calibration model is involved at all.

---

## k-free — nothing to re-run

| Figure | Why it is independent |
|---|---|
| `ftir12/peak_center_1600_by_group.png` | Band centre/height from spectra; no model. |
| `ftir12/median_spectra_diagnostic_windows.png` | Median spectra only. |
| `ftir16/addis_implied_mac_distribution.png` | Implied MAC = Fabs / TOR-EC; no PLS. |
| `ftir16/improve_implied_mac_curve.png` | Same — a ratio of two measurements. |
| `ftir17/improve_vs_addis_spectra_side_by_side.png` | CH-normalised spectra; no model. |
| `ftir17/addis_spectra_by_season.png` | Spectra by season; no model. |
| `deck/filtering_by_ocec.png` | Cohort composition by TOR OC/EC ratio. |
| `deck/airspec_1_baseline.png`, `_2_corrected.png`, `_3_background_gap.png` | Raw spectra and fitted baselines. |

This is the reassuring half: the **band-identity story (ftir_12), the MAC bridge (ftir_16),
the "Addis is a deficit not an exotic peak" spectra (ftir_17) and the whole AIRSpec
explainer** are untouched by the component argument. They rest on measurements, not on a
calibration.

---

## Structurally invariant — numbers move, conclusions do not

| Figure | The invariant claim |
|---|---|
| `deck/mac_effect_all_calibrations.png` (ftir_19) — both-protocol version: `ftir22/mac_pivot_by_mode.png` | Changing MAC rescales x by a constant, so **every** calibration keeps its intercept and R² exactly and its slope scales by 10/6 — true for any k, any cohort. The plotted intercepts are k-dependent; the pivot-on-the-intercept result is not. |
| `deck/mac_slope_pivot.png` (ftir_19) | Same: slope@MAC6 = 0.6 × slope@MAC10 is algebra. |
| `ftir20/component_selection_all_setups.png` | Sweeps k from 1–30 by construction — it *is* the sensitivity analysis. |
| `ftir20/k_by_rule_ladder.png` | Reports the selected k under each rule; that is the subject. |

ftir_19's headline — "the MAC fork cannot move an intercept" — survives any protocol
change. Worth saying explicitly in the deck, because it is the one conclusion that is
immune to this entire argument.

---

## k-dependent — these would change

| Figure | Status |
|---|---|
| `ftir11/addis_crossplots_ocec_cohorts.png` | **Re-derived** → `ftir22/cohort_size_sweep_by_mode.png` (N sweep) and `ftir21/both_modes_crossplots.png`. |
| `ftir13/addis_crossplots_airspec.png` | Same — the AIRSpec panel of the ftir_21 figure. |
| `deck/intercept_ladder.png` | Superseded by `deck/intercept_slope_by_mode.png` (both protocols). |
| `deck/calibration_setup_matrix.png` | **Rebuilt** with both intercept columns. |
| `ftir11/cohort_cut_and_cv_curves.png` | The CV curve panel is protocol-specific (site-grouped). Superseded by ftir_20's four-curve version. |
| `ftir15/bootstrap_intercepts.png` | **Re-derived** → `ftir22/bootstrap_intercepts_by_mode.png`. |
| `ftir15/residual_vs_D2_by_season.png` | **Re-derived** → `ftir22/residual_vs_D2_by_mode.png`. |
| `ftir15/hybrid_vs_ocec_crossplots.png` | **Not re-run** — cohort membership is itself model-selected (see below). |
| `deck/no_cleaning_fullpool_crossplots.png` (ftir_17) | **Re-derived** → the IMPROVE-network panel of `ftir21/both_modes_crossplots.png`. |
| `deck/transfer_roundup.png` (ftir_18) | **Mixed provenance** — see below. |
| `deck/deployed_alldata_crossplot.png` (ftir_17) | The deployed model is external and fixed; its k is whatever SPARTAN shipped. Not ours to re-choose. |

### The mixed case worth knowing about

`transfer_roundup.png` puts six models on one set of axes, but they do **not** share a
protocol: the HIPS-916 transfer, smoke-906, deployed SPARTAN and the Addis nested-CV panels
come from committed phase-2 predictions with their own component choices, while the
full-pool panel was retrained at the locked k = 6/5. It is a fair picture of *what exists*,
not a controlled comparison. `ftir21/both_modes_crossplots.png` is the controlled version.

---

## The second-order dependency: cohort *membership*

For four cohorts, k affects only the fit. For two, it affects **which filters are in the
cohort at all**, because the selection itself runs through a fitted PLS model:

| Cohort | Selection axis | Membership depends on k? |
|---|---|---|
| Entire IMPROVE network | none | No |
| Biomass-smoke (906) | Katie–George smoke lists | No |
| Ethiopia-shaped smoke (300) | CH / carbonyl / 1600 band features | No — band heights, not a model |
| **Spectral analogs (locked 500)** | score-space D² + VIP-weighted spectral RMSE | **Yes** — both come from a fitted PLS (`improve_model`, `improve_k` score columns, VIP weights) |
| Lowest-OC/EC (800) | TOR OC/EC ratio | No |
| **ftir_15 hybrid cohort** | low-OC/EC ∩ spectral similarity | **Yes** — same score-space machinery |

So for the spectral-analog and hybrid cohorts, ftir_21's comparison holds membership fixed
and varies only the fit; a fully consistent app-protocol version would re-select the cohort
too. Given both already fail the held-out TOR test, this is documented rather than pursued.

---

## Reproducibility: interleaved CV is order-dependent

Found while reconciling ftir_21 against ftir_22, which disagreed on the AIRSpec cohort
(k = 19 / intercept −1.65 vs k = 18 / −1.78) on **identical filters**. Neither was wrong:
interleaved CV assigns folds by row position (`i % 10`), so the answer depends on the order
the cohort happens to be listed in. The same 800 filters give:

| Row order | Calibration app k | Site-held-out k |
|---|---|---|
| ranked by OC/EC | 18 | 5 |
| committed cohort CSV | 19 | 5 |
| randomly shuffled | 15 | 5 |

Site-grouped folds are defined by site *label*, so they cannot depend on row order. This is
a reproducibility property, not a preference: two analysts running the Calibration app on
the same data in different sort orders ship different calibrations. Demonstrated in ftir_22
section 2b (`output/tables/ftir22/row_order_sensitivity.csv`).

---

## Standalone per-protocol figures

The ftir_21/ftir_22 figures overlay both protocols, which is right for *arguing* that the
protocol matters. For building slides you usually want one clean figure per protocol.
`scripts/build_protocol_variants.py` writes each k-dependent figure twice, one folder per
protocol, with **matching file names** so a deck can be built from one folder and swapped
wholesale by pointing at the other:

```
output/plots/deck/by_protocol/
  calibration_app/   <- interleaved CV, within-5%, fitted on all filters
  site_held_out/     <- site-grouped CV, first-major-minimum, site-disjoint fit
```

Both folders contain the same eight file names:

| File | Replaces |
|---|---|
| `calibration_setup_matrix.png` | the deck setup matrix, this protocol's intercept column only |
| `crossplots_all_setups.png` | ftir_11 / ftir_13 Addis crossplots |
| `intercept_slope_ladder.png` | the deck intercept ladder |
| `mac_effect_all_setups.png` | ftir_19 MAC-effect crossplots |
| `mac_slope_pivot.png` | ftir_19 MAC slope dumbbells |
| `bootstrap_intercept_ci.png` | ftir_15 bootstrap |
| `residual_vs_d2.png` | ftir_15 residual structure |
| `cohort_size_sweep.png` | ftir_11 cohort-size sweep |

Colour encodes the **calibration setup**, matching the combined setup matrix and intercept
ladder — not the protocol, which the folder and each figure's own subtitle already carry.
(That palette would not pass a strict categorical CVD check — deck blue and purple are
close — but the encoding is redundant everywhere: every setup is also named by its panel
title or row label, so no chart asks the reader to tell two hues apart.) ⚠ marks the two
setups with no held-out TOR skill. All read from
committed ftir_21/ftir_22 tables — no refitting — so regenerating the set after a change
takes seconds and cannot drift. `index.csv` lists every path with its protocol.

---

## What this means for the deck

1. **The band-identity, MAC-bridge and spectra slides are safe** — no calibration in them.
2. **ftir_19's MAC result is safe** — it is algebra, true for any k.
3. **The intercept column of the setup matrix is protocol-conditional** and should either
   carry both protocols or a footnote naming the one it uses.
4. **ftir_15's uncertainty results have now been re-derived under both protocols**
   (ftir_22) and both survive: the raw-vs-corrected residual distinction holds either way
   (D² r = 0.87/0.71 raw, 0.33/0.21 corrected) and the corrected bootstrap CIs overlap
   heavily ([−2.07, −1.03] app vs [−1.78, −1.17] site-held-out, both excluding zero).
5. **ftir_11's "N = 800 is the sweet spot" is protocol-dependent** and needs its protocol
   stated: under the app protocol intercept and RMSE improve monotonically to N = 1600.
   What actually selects 800 is the held-out TOR test, which only one protocol produces.
6. **The strongest single argument is reproducibility**, not accuracy — see the row-order
   section above.
