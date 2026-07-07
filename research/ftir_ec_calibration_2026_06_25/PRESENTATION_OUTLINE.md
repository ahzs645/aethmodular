# Presentation storyboard — ~2026-07-02

Update deck for the advisors, continuing directly from the 2026-06-25 review. Keep it **step by
step** (the meeting's explicit ask): simple plots first, don't layer everything into one figure.

**Read `notes/whats_new_vs_last_week.md` first.** Last week's deck already showed the 906-sample
calibration, the outlier removal, the jagged-K variance + ensemble recommendation, the OPR char/soot,
and the charcoal library. This outline is built to **advance**, not repeat. Tags:
🆕 = new this week · ♻️ = same data, corrected framing · ⛔ = shown last week, don't re-present.

Each slide: figure/table source + the one thing to say.

---

**1. The reframe — last week was too clean** ♻️🗣️
Last week's headline was the *cleaning win* (61 removed → R²=0.942). The feedback: **too perfect.**
Absolute residual thresholds unfairly kill high-EC smoke samples (big absolute error = small relative
error). New plan: **keep most samples; the below-1:1 samples are the signal**; evaluate against
**Tor, not FTIR-vs-FTIR**. *(This directly reverses last week's slides 3–5 — say so.)*

**2. Adama — the 5 FTIR spectra** 🆕 `figures/fig01_adama_spectra.png`
The five Adama spectra with the **Teflon artifacts** marked (CF double peak; sloping baseline at
4000). Hook: **the highest-EC sample (J1269) is not the biggest-peak spectrum** → "something is
missing." *(Last week only showed removed IMPROVE spectra, not these.)*

**3. Adama — Tor vs. FTIR-EC, against 1:1** ♻️ `figures/fig02_adama_tor_vs_ftir.png`
Tor on x (ground truth), FTIR on y, 1:1 line. **4 of 5 below 1:1** (FTIR under-reports, ratios
0.58–0.81); **J1269 the lone one above.** *(Last week showed side-by-side bars — this is the meeting's
corrected framing.)* ⏳ Overlay biomass FTIR-EC once predictions exist.

**4. Adama — char vs. soot, OPR and OPT** 🆕 `figures/fig03_adama_char_soot.png`
Per-sample char (EC1−OP) vs soot (EC2+EC3). OPR reproduces last week (soot-leaning, c/s 0.02–0.58);
**OPT flips char-EC negative for 4/5 samples** — the meeting's "try OPT" ask, and a concrete new
result on whether the OP convention changes the story. *(OPR alone = last week; OPT = new.)*

**5. Ethiopia (ETAD) — all spectra** 🆕 `figures/fig04_etad_spectra_all.png`
All ~319 Addis/Ethiopia spectra + mean. **No Ethiopia appeared in last week's deck** — this orients
everyone to the dataset that the whole hypothesis is about.

**6. Ethiopia — biomass vs. diesel months** 🆕 `figures/fig04_etad_spectra_by_season.png`
Spectra split by month. ⏳ Replace the placeholder split with **Navid's** high-biomass vs. diesel
months, then contrast the two clouds. Feeds EC-vs-FABS (MAC=10) next.

**7. Component count — the rule, not the ensemble** ♻️ `figures/fig05_ec_rmsecv_curve.png`
Real RMSECV on the 906-sample EC set — the jagged "rose then dipped" curve (shown last week). **New
this week:** don't re-pitch the ensemble; fix a single reproducible rule ("first major minimum",
`rel_tol`), and note k≈13 vs k=21 is a tolerance choice. Then **Weakley**: 2nd-derivative → ~4
components (first components model the Teflon) — the direction to try. *(Curve = last week; rule +
Weakley = new.)*

**8. Calibration variants CAL-0…CAL-5 — the 1:1 grid** 🆕 `figures/fig06_variants_1to1_grid.png`
All six variants, predicted-vs-measured at a common k=20 (`tables/calibration_variants_results.csv`).
Headline: **CAL-1 (cleaned) has the best R² (0.94) but the *worst* RMSECV (21.5 vs CAL-0's 16.6)** —
the "too perfect" trap quantified. Also: ≥70 µg keeps only **22/906** (too aggressive).

**9. The inverse calibrations — which samples they use** 🆕 `fig06_inverse_below11.png`, `fig06_inverse_removed.png`
One plot each: **CAL-3** trains on everything *below 1:1* (under-predicted); **CAL-2** trains on the
3σ-removed samples. This is "the weird samples may be the point," made literal.

**10. Number of components + 2nd-derivative** 🆕 `fig06_rmsecv_by_variant.png`, `fig06_second_derivative_rmsecv.png`
Fix one rule (`first_major_min`) for every variant. **2nd-derivative** spectra reach a good RMSECV at
far fewer components (Weakley) — the direction to adopt. `fig06_second_derivative_spectra.png` shows
why (flattens the PTFE baseline).

**11–13. Supporting papers (Han + Weakley)** 🆕 (`notes/paper_slides_han_weakley.md`)
3 slides: Han char/soot TOR equations (**char-EC = EC1 − POC**, confirmed in the paper); Weakley
source-specific EC calibration; Weakley RMSECV component selection + 2nd-derivative/BMCUVE. Label
everything **TOR-defined** (Han's caveat).

**14. Asks / next steps** 🗣️
Sean: comments export + smoke code. Confirm Alex's Ethiopia spectra = the ETAD file. Then build the
label-dependent `biomass` variant, re-run Adama + Ethiopia against the chosen calibrations, add OPT.

*(Separate, private:* the scope/timeline/funding 1:1 — not part of this deck.)*
*(Skip ⛔: the charcoal reference library / PCA spectral map — the meeting shelved it.)*

---
### Slide → asset map
| Slide | Asset | Tag |
|-------|-------|-----|
| 2 | `figures/fig01_adama_spectra.png` | 🆕 |
| 3 | `figures/fig02_adama_tor_vs_ftir.png`, `tables/adama_tor_vs_ftir.csv` | ♻️ |
| 4 | `figures/fig03_adama_char_soot.png`, `tables/adama_char_soot_opr_opt.csv` | 🆕 (OPT) |
| 5 | `figures/fig04_etad_spectra_all.png` | 🆕 |
| 6 | `figures/fig04_etad_spectra_by_season.png` | 🆕 |
| 7 | `figures/fig05_ec_rmsecv_curve.png` | ♻️ + Weakley |
| 8 | `figures/fig06_variants_1to1_grid.png`, `tables/calibration_variants_results.csv` | 🆕 |
| 9 | `figures/fig06_inverse_below11.png`, `figures/fig06_inverse_removed.png` | 🆕 |
| 10 | `figures/fig06_rmsecv_by_variant.png`, `figures/fig06_second_derivative_rmsecv.png`, `..._spectra.png` | 🆕 |
| 11–13 | `notes/paper_slides_han_weakley.md` | 🆕 |
