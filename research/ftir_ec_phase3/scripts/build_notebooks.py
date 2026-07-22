"""Convert the percent-format run_ftir_*.py scripts into executed notebooks.

Usage: python scripts/build_notebooks.py [11 12 ...]
Run from research/ftir_ec_phase3/. Each script's tl;dr / Takeaways placeholders
are replaced with the finalized text below, then the notebook is executed
top-to-bottom with nbclient so every number in the committed notebook comes
from a real run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PLACEHOLDER = "(filled in by the finalize step after execution)"

TLDR = {
    "11": """\
Ann's lowest-OC/EC strategy is the **best-performing calibration cohort of phases 2–3, but it
does not zero the Addis intercept**. The 800 lowest-OCEC IMPROVE filters (TOR OC/EC ≤ 2.27 —
the pool median is 5.54) give the strongest site-held-out TOR test of any cohort tried
(R² **0.911**, slope **0.87**, RMSE **3.41 µg/filter**) and improve the fixed-190-filter Addis
comparison to intercept **−3.22** and RMSE **1.16 µg/m³** at MAC = 10, versus **−4.17** and
**1.49** for the deployed SPARTAN calibration. Five size-matched random cohorts show intercepts
can drift toward zero by chance (−1.8 to −3.4) — but only with collapsed slopes (0.93–1.49) and
worse RMSE; the OCEC cohort is the only one that improves intercept, RMSE, and the TOR test
simultaneously. N = 400 is too small to support a model (k = 3, held-out R² 0.19); N = 1600
dilutes the effect. The intercept is MAC-invariant: at MAC = 6 the slope becomes **0.95** with
the same −3.22 offset. An independent replication (separate implementation, same protocol and
seed; outputs in `output/tables/ftir_11/`) reproduces the headline metrics to the last digit
and adds a VIP comparison.""",
    "12": """\
The Addis ~1600 cm⁻¹ band **does not behave like amine or liquid water, and is consistent with
carboxylate / oxygenated-organic absorption**. Its peak center sits tightly at
**1616–1620 cm⁻¹** (IQR), below every IMPROVE cohort (medians ≥ **1633**), and its height
covaries strongly with carbonyl (r = **0.94**) and CH (r = **0.88**) but barely with
N–H/O–H 3100–3400 (r = **0.11**) — the amine companion that should be there if the band were
amine. The symmetric-COO⁻ partner test near 1400 cm⁻¹ is **inconclusive by construction**: that
window lies outside the baselined region, so it must be revisited with AIRSpec-corrected
spectra and, ideally, lab standards. The band also tracks deployed FTIR EC (r = **0.80**) and
HIPS τ (r = **0.66**), consistent with the phase-2 finding that the EC calibration leans on
oxygenated-organic covariates. An independent replication with different window conventions
(outputs in `output/tables/ftir_12/`) reaches the same ordering (Addis ≈ 1617 vs IMPROVE
≥ 1635).""",
}

TLDR["13"] = """\
Real AIRSpec baselines (validated Python port, worst deviation **6×10⁻⁷** vs the R run) change
the story in three ways. **(1) They do not fix the HIPS transfer**: an IMPROVE HIPS-τ model
rebuilt on EDF-6/8 baselined spectra still predicts median Addis Fabs **≈12.5 vs 47.1**
observed (bias **−35**) — the ftir_08 transfer failure is compositional, not a baseline
artifact. **(2) They produce the closest-to-zero Addis intercept of any locked-protocol
calibration yet**: the lowest-OCEC-800 cohort on df1 = 6 spectra gives intercept **−1.62** and
slope **0.86** at MAC = 10 with the project's best held-out TOR test (R² **0.904**, slope
**1.01**, k = 5) — but Addis scatter grows (R² 0.66, RMSE 2.41 vs 0.77/1.16 raw) and a
**−2.3 µg/m³** mean bias remains. **(3) The 906-sample smoke calibration collapses** on
corrected spectra (Addis slope **0.37**): its raw-spectra behavior rode on baseline structure.
df1 = 6 vs 8 is immaterial throughout. ftir_12 follow-up: the corrected 1600-band center stays
at **1617–1620 cm⁻¹**, and the apparent 1520–1560 companion (raw-height r = 0.87) vanishes
under CH normalization (Spearman **−0.29**) — a loading artifact, leaving peak position plus
the missing N–H companion as the carboxylate evidence."""

TAKEAWAYS = {
    "11": """\
- **The OCEC hypothesis has real signal.** Selecting calibration filters by low TOR OC/EC —
  with no spectral or Addis information at all — produces the only cohort that improves the
  Addis intercept, Addis RMSE, and the locked TOR test at the same time.
- **It is still ~3.2 µg/m³ from closing the gap.** If composition mismatch in OC/EC were the
  whole story, the intercept should approach zero; it does not. The remaining candidates from
  the meeting are the un-baselined spectra (ftir_13) and genuinely absent charcoal-like
  samples in IMPROVE.
- **Random-cohort intercepts are a trap.** A near-zero intercept with a collapsed slope is
  worse, not better; any cohort comparison must report slope, intercept, RMSE, and a held-out
  TOR test together.
- **Cohort size matters non-monotonically.** 400 is too few sites/filters to support the model;
  1600 pulls the OC/EC cut to 2.8 and starts re-admitting the ordinary IMPROVE mixture.
- **Next decisive test** remains an independent Addis EC reference (TOR on collocated filters
  or an agreed MAC/protocol bridge), since HIPS/MAC still sets the comparison scale.""",
    "12": """\
- **Amine and water-bend assignments are disfavored** for the Addis 1600-band: the center is
  too low and the N–H/O–H companion covariation is absent.
- **Carboxylate (or a related oxygenated-organic feature) is the leading candidate**, matching
  Satoshi's suggestion; charcoal-burning chemistry plausibly produces carboxylate-rich aerosol.
- **The definitive tests are still open**: the symmetric-COO⁻ partner near 1400 cm⁻¹ and the
  nitro symmetric band near 1340 cm⁻¹ both sit below the 1425 cm⁻¹ edge of the baselined
  region. AIRSpec-corrected spectra (ftir_13) recover 1425–1500 but not 1400; resolving this
  fully needs the sub-1500 model Satoshi's group is developing, or lab standards.
- **Calibration relevance**: the band's strong covariation with carbonyl/CH and with deployed
  EC supports the phase-2 conclusion that FTIR EC predictions ride on oxygenated-organic
  covariates — exactly the proportionality that charcoal burning could break.""",
}

TLDR["15"] = """\
Three questions, three clean answers. **(1) The corrected-model intercept is real and so is
its remaining gap**: a site-cluster bootstrap (B = 200) puts the AIRSpec OCEC-800 Addis
intercept at 95% CI **[−1.78, −1.06]** — excluding zero *and* entirely disjoint from the raw
model's **[−4.88, −2.81]**. The AIRSpec improvement is not split luck. **(2) The two models
miss differently**: the raw model's Addis residuals track extrapolation distance (D² r =
**0.71**) and swing with season (+0.49 Kiremt to −1.25 Dry), while the corrected model's
residuals are decoupled from D² (r ≈ 0.2 / Spearman −0.05) and sit at a near-constant
**−2.0 to −2.6 µg/m³ across all three seasons** — a stable offset, with |residual| growing
with loading. That is the signature of a missing constant component or a reference/MAC
mismatch, not composition-driven scatter. **(3) The hybrid cohort fails**: selecting the 800
spectrally-nearest-to-Addis filters inside the lowest-2,000-OC/EC pool collapses the held-out
TOR test (R² **0.19**, slope **0.20**) — cohort engineering has hit its ceiling, and
OCEC-800 + AIRSpec stands as the best candidate."""

TAKEAWAYS["15"] = """\
- **Stop tuning cohorts.** The hybrid negative result plus the bootstrap CIs close out the
  cohort-engineering direction: no selection strategy tested in phases 2–3 moves the intercept
  further without breaking the TOR test.
- **Prefer the corrected model for interpretation even where its RMSE is worse.** The raw
  model's Addis residuals are extrapolation-driven (D² r = 0.71) and season-dependent; the
  corrected model's are a clean, season-stable offset. A constant, explainable miss beats a
  smaller but structured one.
- **The season-independent −2 to −2.6 µg/m³ offset is the sharpest clue yet**: it behaves like
  a missing constant absorption component (Satoshi's EC sloping-absorption discussion) or a
  HIPS MAC/protocol mismatch — both resolvable only with an external anchor, not more IMPROVE
  data.
- **Decisive next data remain**: an independent Addis EC reference (TOR on archived filters or
  Teflon/quartz pairs at Adama/Bishoftu) and the INDH/CHTS/ETBI spectra pull."""

TAKEAWAYS["13"] = """\
- **The baseline caveat is discharged.** Phase-2's ">1500 cm⁻¹ offset correction is not
  AIRSpec" disclaimer no longer shields any hypothesis: with validated EDF-6/8 baselines, the
  transfer gap and most of the intercept persist. What remains points at composition — the
  charcoal-like samples IMPROVE may simply not contain.
- **OCEC-800 + AIRSpec is the best candidate calibration by the meeting's stated objective**
  (intercept toward zero without sacrificing the TOR test), at the price of Addis scatter and
  a residual negative bias. Whether slope 0.86 at MAC = 10 or 0.51 at MAC = 6 is "right"
  still hinges on the unresolved MAC/protocol bridge.
- **Baseline-sensitive skill is not skill**: the smoke model's raw-spectra Addis behavior was
  an artifact. Any future cohort comparison should run on AIRSpec-corrected spectra from the
  start (the port runs at ~15 ms/spectrum, so this is now free).
- **Band identity**: center 1617–1620 cm⁻¹ is rock-stable under real baselining; the
  1520–1560 covariation was loading-driven. Settling carboxylate vs aromatic ring modes needs
  the region below 1425 cm⁻¹ (Satoshi's sub-1500 model) or lab standards.
- **Fixed-df caveat**: this run uses fixed (df1, df2) = (6, 4) and (8, 4) as in the validated
  ETAD R run; AIRSpec's per-spectrum NAF-based EDF selection is not replicated."""

SCRIPTS = {
    "11": ("run_ftir_11.py", "ftir_11_ocec_ratio_cohort.ipynb"),
    "12": ("run_ftir_12.py", "ftir_12_band_1600_identity.ipynb"),
    "13": ("run_ftir_13.py", "ftir_13_airspec_corrected_calibrations.ipynb"),
    "15": ("run_ftir_15.py", "ftir_15_uncertainty_and_hybrid.ipynb"),
    "16": ("run_ftir_16.py", "ftir_16_mac_decision_prep.ipynb"),
    "17": ("run_ftir_17.py", "ftir_17_deck_figures_and_seasonal_spectra.ipynb"),
    "18": ("run_ftir_18.py", "ftir_18_transfer_roundup.ipynb"),
    "19": ("run_ftir_19.py", "ftir_19_mac_effect_on_calibrations.ipynb"),
    "20": ("run_ftir_20.py", "ftir_20_component_selection_across_setups.ipynb"),
    "21": ("run_ftir_21.py", "ftir_21_both_protocols_end_to_end.ipynb"),
    "22": ("run_ftir_22.py", "ftir_22_figures_under_both_protocols.ipynb"),
    "23": ("run_ftir_23.py", "ftir_23_component_selection_by_protocol.ipynb"),
}

TLDR["21"] = """\
Every calibration setup run end-to-end **twice** — once under the **Calibration app**
protocol (10-fold interleaved CV, k = first within 5% of the minimum, model fitted on all
cohort filters) and once under the **site-held-out** protocol (site-grouped 5-fold CV,
first-major-minimum, fitted on the training side of a site-disjoint 80/20 split) — with the same cohorts, the
same spectra and the same PLS math. The two modes are switchable in
`scripts/calibration_modes.py`, named for what they do rather than for who runs them.
**Provenance first**: the `site_heldout` mode reproduces the
committed ftir_11/ftir_13 calibrations exactly (k = 6 / 5, Addis slope and intercept to
< 1e-6), so the comparison is a like-for-like re-run and not a re-derivation.
**The protocol moves the Addis answer, and by more than the MAC fork does.** On the
lowest-OC/EC cohort the Calibration app protocol turns the site-held-out
**1.59x − 3.22 (RMSE 1.16)** into **2.15x − 4.59 (RMSE 2.03)** — the same 800 filters, made to look a full 1.4 µg/m³ worse
in intercept purely by how k was chosen. Biomass-smoke swings hardest (**2.43x − 6.35**
app-protocol, **0.50x − 0.99** site-held-out at k = 4), and the whole IMPROVE network goes
1.95x − 4.05 → 1.65x − 3.44. The one setup the protocol barely touches is
**lowest-OC/EC + AIRSpec**: −1.65 vs −1.62, slope 0.96 vs 0.86 — a calibration whose Addis
answer does not depend on who processed it, which is exactly the robustness the deck
should claim for it. **The held-out TOR test, which only the site-held-out protocol produces,
also re-sorts the field**: lowest-OC/EC + AIRSpec (R² 0.90, slope 1.01) and lowest-OC/EC
(0.91 / 0.87) pass cleanly, the IMPROVE network passes on slope (0.70 / 0.99), and
**Ethiopia-shaped smoke collapses (R² 0.00, slope −2.20)** — a setup still carried in the
matrix at intercept −3.69 turns out to have no held-out skill at all once sites are
disjoint."""

TAKEAWAYS["21"] = """\
- **Report the protocol with the number.** "Intercept −3.22" and "intercept −4.59" are the
  same cohort, the same filters and the same maths; only the component-selection protocol
  differs. Any cross-project comparison of Addis intercepts that doesn't state the protocol
  is comparing two things at once.
- **The protocol is a bigger lever than the MAC fork.** ftir_19 showed MAC cannot move an
  intercept at all (it rescales x, so each calibration pivots on its own intercept). The
  protocol moves the lowest-OC/EC intercept by 1.4 µg/m³ — so of the two open arguments,
  this is the one that changes the headline number.
- **Lowest-OC/EC + AIRSpec is protocol-robust**, landing at −1.65 / −1.62 either way. Taken
  with ftir_20's result that the same cohort is indifferent to the CV fold structure
  (optimism ×1.01), this is the strongest argument the deck has: its answer does not depend
  on analyst choices.
- **Ethiopia-shaped smoke should be retired from the matrix, or asterisked.** Under a
  site-disjoint test it has R² 0.00 and slope −2.20 — no held-out skill. It joins the
  spectral analogs as a setup whose attractive Addis intercept is not backed by a TOR test.
- **Two confounds are inherent to an end-to-end protocol comparison, not bugs**: the app
  mode fits on the whole cohort (e.g. 800 filters) while the site-held-out mode fits on the
  training part (606), so training size travels with the protocol; and the app mode produces no
  site-disjoint test by construction, which is why that column is blank rather than filled
  with an in-sample number.
- **Caveat**: the Calibration app protocol is applied here to *each of these cohorts*,
  which is not literally what the app ships — it trains on 21 hardcoded IMPROVE sites plus the
  BiomassDetected lists. This isolates the protocol while holding the cohort fixed, which
  is the comparison that answers "what does the protocol cost", not "what exactly is
  deployed"."""

TLDR["16"] = """\
Three preparation results for the MAC = 6-vs-10 decision. **(1) SPARTAN's public
`ChemSpec_EC` for ETAD is confirmed to be HIPS Fabs / 10** (median ratio 0.101, r = 0.89,
implied MAC 9.89, n = 175) — it is a unit convention, not an independent EC reference, and
must not be used to break the tie. **(2) The Adama-composition bridge sharpens the fork into a
trichotomy**: if Addis aerosol had Adama's TOR OC/EC (4.6–7.2, median 6.1), the MAC that
reconciles Addis HIPS Fabs with FTIR OC would be **≈47 m²/g (IQR 36–56)** — far outside any
physical EC MAC (~4–13). Therefore at least one of these is true: Addis OC/EC is really ~5–8×
lower than Adama's (massive EC); a large fraction of Addis Fabs is **non-EC absorption**
(BrC, dust/iron, or filter artifact — consistent with the season-stable −2 to −2.6 µg/m³
corrected-model offset in `ftir_15`); or FTIR OC is badly low at Addis. **(3) A modest
co-located quartz TOR campaign decides it**: each sampling day separates the MAC = 6 and
MAC = 10 hypotheses by ~3σ, so **11–13 days per season (≈36 total)** reaches 5σ per season
even if half the signal is lost to protocol systematics. TOR requires quartz, so archived
Addis Teflon filters cannot substitute; the Adama Batch-54 chain proves the logistics exist.
**(4) The IMPROVE implied-MAC bridge (151,843 matched filters, run after the relocated
`FTIR/local_db` was found) favors MAC = 10 at Addis-like composition**: implied MAC =
Fabs/TOR-EC has overall median 11.96 (IQR 9.0–15.7), and in the Addis-like OC/EC ≤ 2.27
subset (n = 6,503) median **10.05** (IQR 6.9–13.1) — MAC = 10 sits at the center, MAC = 6 in
the lower tail. Together with ftir_13/ftir_15 this tilts the fork toward MAC ≈ 10 + the
corrected model + a genuine non-EC absorption component at Addis, pending the quartz
campaign's direct answer."""

TAKEAWAYS["16"] = """\
- **Do not cite ChemSpec EC as independent support for MAC = 10** — it *is* MAC = 10 by
  construction (Fabs/10, r = 0.89 against HIPS on the same base filters).
- **The Adama bridge turns the MAC fork into a three-way test.** Implied MAC ≈ 47 m²/g under
  Adama-like composition is unphysical for EC, so the data force one of: (a) genuinely
  extreme Addis EC (OC/EC ≈ 1, the low-OC/EC-cohort world), (b) dominant non-EC absorption in
  Addis HIPS Fabs (BrC/dust/artifact — the original HIPS-offset hypothesis, and the natural
  reading of ftir_15's season-stable offset), or (c) a large FTIR OC underestimate. A quartz
  TOR campaign measures (a) directly and, combined with HIPS on paired Teflon, apportions (b).
- **Campaign spec: ~12 days × 3 seasons of co-located quartz at ETAD** (or Bishoftu/ETBI,
  which now has HIPS data), analyzed IMPROVE_A TOR like Adama Batch-54. Statistics are not
  the constraint (~3σ/day); seasonal and protocol systematics are — spread days across
  seasons rather than concentrating them.
- **Adama itself argues against the 'simple charcoal EC' story**: its wet-season OC/EC
  (median 6.1) sits at the IMPROVE median, echoing the meeting's observation that Adama does
  not look charcoal-heavy in OC/EC space. If Addis winter differs, that is precisely what the
  campaign's Dry-season leg tests.
- **The IMPROVE bridge undercuts the raw-at-MAC6 reading.** On 151,843 IMPROVE filters,
  HIPS-vs-TOR implied MAC at Addis-like OC/EC is centered on 10 (median 10.05), not 6; the
  mild downward trend toward low OC/EC (12 mid-range → 10.4 lowest decile) and the inflated
  high-OC/EC tail (median 19.8) both fit Fabs carrying non-EC absorption that scales with
  organics. Caveat: IMPROVE and SPARTAN HIPS protocol comparability remains an assumption
  (as in ftir_08)."""


TLDR["17"] = """\
The three missing meeting figures now exist, and two of them sharpen existing conclusions.
**(1) The all-data cross plot, new orientation** (deployed predictions exist only for the
fixed 190-filter cohort, so "all data" *is* that cohort): deployed SPARTAN FTIR EC vs HIPS
reads **y = 1.90x − 4.17** (R² 0.764, RMSE 1.49) at MAC = 10 and **y = 1.14x − 4.17**
(RMSE 3.25) at MAC = 6 — both panels are in `output/plots/deck/`. **(2) A protocol-matched
no-cleaning calibration is a slope trap at full scale**: training on all **13,010** eligible
lot-248/251 filters (158 sites, no selection of any kind) yields seductive intercepts
(−1.33 raw, −0.61 corrected) but collapsed Addis slopes (**0.66 / 0.40**) and the worst
held-out TOR tests of phase 3 (R² **0.53 / 0.63**, slope ≈ 0.70) — ftir_11's "intercept
alone is not an acceptance criterion" warning, demonstrated on the whole pool. **(3) The
side-by-side full-range spectra** relocate Addis's strangeness: in CH-normalized corrected
space Addis sits *below* every IMPROVE cohort — roughly **half the broad O–H/N–H
(3000–3600 cm⁻¹) and carbonyl absorption per unit CH** (≈2.0 vs 4.6–5.2 at the 3200 peak;
≈0.9 vs 1.9–3.4 at carbonyl) and below the full pool's IQR — i.e. Addis is missing
oxygenated-organic absorption relative to IMPROVE, the spectra-level face of the low-OC/EC
ranking. **(4) Naveed's seasons split**: seasonal differences are **loading, not shape** —
CH (0.005→0.013), carbonyl, 1600-band height, deployed EC (3.3→7.2 µg/m³) and Fabs (43→56)
all peak in Kiremt, while the 1600-band center stays at **1617–1619 cm⁻¹ in all three
seasons** and 1600/CH moves only 0.62–0.75. The one shape effect is the broad O–H region,
relatively strongest in the Dry season (0.020 vs 0.016 Kiremt in absolute corrected median
despite half the CH)."""

TAKEAWAYS["17"] = """\
- **Deck gap closed.** `deployed_alldata_crossplot.png` and
  `no_cleaning_fullpool_crossplots.png` are in `output/plots/deck/` alongside the intercept
  ladder; the original calibration's real slope/intercept (1.90/−4.17 at MAC = 10,
  1.14/−4.17 at MAC = 6) can be read straight off the plot.
- **"No cleaning" is now a measured baseline, not a hypothetical.** Its near-zero intercepts
  come with collapsed slopes and degraded TOR tests — the full-pool version of the
  random-cohort trap — so the OCEC-800 selection is defended from both directions
  (better than smoke-906 *and* better than no selection at all).
- **Addis's spectral signature is a deficit, not an exotic peak.** Per unit CH it carries
  about half the oxygenated (O–H, carbonyl) absorption of any IMPROVE cohort and falls
  below the 13.6k pool's IQR — supporting the view that IMPROVE simply lacks
  charcoal-like, oxygenation-poor samples rather than Addis having features IMPROVE spectra
  can't express.
- **The 1600-band identity is not a seasonal artifact**: its center is 1617–1619 cm⁻¹ in
  Dry, Belg, and Kiremt alike, and composition ratios barely move across seasons — consistent
  with ftir_15's season-stable corrected offset. Season mainly modulates loading (all bands,
  EC, and Fabs peak in Kiremt), with a relative Dry-season enhancement of the broad O–H band
  as the one shape change.
- **Caveat**: the full-pool models use the locked k from ftir_11/13 (6 raw, 5 corrected) as
  fixed-k sensitivity, not a fresh CV selection; and 43 of 296 Addis spectra lack sampling
  dates and are excluded from the seasonal split."""


TLDR["18"] = """\
Answers the deck-review question — the ftir_08 transfer trains on the **916** HIPS-matched
IMPROVE filters; did we also test a model trained on **all** sites, since TOR exists for the
whole pool? — by putting every calibration family on one set of axes (same 239 Addis filters,
HIPS EC-equivalent orientation, MAC = 10; MAC = 6 in the tables). Three results. **(1) A
lineage audit**: ftir_09's "Current IMPROVE TOR EC" is **byte-identical to the smoke-906
calibration** (max |Δ| = 0 across all 239 predictions; the deployed SPARTAN EC is a different
model, corr 0.963) — so phase 2 never actually drew a full-pool TOR transfer on these axes,
and any citation of the ftir_09 numbers (slope 2.30, R² 0.607) should say smoke-906.
**(2) Yes — and it transfers no better**: the no-cleaning full-pool model (13,010 filters,
158 sites, k = 6 raw; retrained per-filter here, reproducing ftir_17's held-out metrics to
1e-9) reads Addis slope **0.60**, intercept **−1.09**, R² **0.691**, RMSE **3.11**,
bias **−3.0 µg/m³** on the 239 available pairs. Thirteen thousand TOR filters buy tracking
(R² 0.69 vs 0.26 for the HIPS-916 transfer) but not calibration: the slope collapses and a
−3 µg/m³ offset remains. **(3) The roundup figure** (`output/plots/deck/transfer_roundup.png`)
shows the three as-is transfers failing in three distinct ways — flat and uninformative
(HIPS-916: 0.22x, R² 0.26), steep with a deep offset (smoke-906: 2.30x − 5.38, the deck's
−6.91 being the fixed-cohort row), compressed low (full pool: 0.60x − 1.09) — while the
bottom row walks toward the local ceiling: deployed 1.90x − 4.17 (R² 0.764), OCEC-800 +
AIRSpec 0.78x − 1.28 (RMSE 2.46), Addis-only nested CV **0.91x + 0.43** (R² 0.883,
RMSE 0.38). No IMPROVE-trained model — 906, 916, or 13,010 samples, TOR or HIPS target —
gets slope and intercept simultaneously right at Addis."""

TAKEAWAYS["18"] = """\
- **The answer to the meeting question is "yes, and it doesn't help."** A TOR-target model
  on the full pool improves R² over the HIPS-916 transfer (0.69 vs 0.26) simply because 13k
  training filters beat 916, but slope (0.60) and bias (−3.0 µg/m³) stay wrong. More sites is
  not a substitute for local calibration or an explicit transfer step.
- **Correct the record on ftir_09**: its "Current IMPROVE TOR EC" panel *is* the smoke-906
  model (audited: EC_current ≡ EC_smoke_906). The full-pool TOR transfer had never been drawn
  per-filter until now.
- **The failure modes argue for slope-and-bias correction, not naive transfer.** Every TOR
  transfer is roughly linear at Addis (R² 0.55–0.76) with the wrong gain and offset —
  correctable with a small local anchor — whereas the HIPS-916 transfer has no usable gain
  (slope 0.22 available pairs, 0.15 fixed cohort). If a domain-adaptation step is pursued,
  TOR-target models are the ones worth adapting.
- **One deck figure now carries the whole training-set story** (`transfer_roundup.png`):
  naive transfers on top, deployment → targeted cohort → local calibration on the bottom,
  every panel on identical axes with n, slope, intercept, R², RMSE readable per panel.
- **Caveats**: the full-pool panel uses the locked k (6 raw / 5 corrected) rather than a
  fresh CV sweep; the deployed panel exists only for the fixed 190-filter cohort; the
  HIPS-native panels are MAC-invariant by construction, so their MAC = 6 table rows are pure
  rescalings; and IMPROVE-vs-SPARTAN HIPS protocol comparability remains an assumption, as
  in ftir_08/ftir_16."""


TLDR["19"] = """\
The deck question — apply the HIPS "MAC fix" (Fabs/6 instead of Fabs/10) to every
calibration setup — has a structural answer, shown here per-filter for all six setups in
`calibration_setup_matrix` on the fixed 190-filter cohort. Because HIPS enters every
crossplot as x = Fabs/MAC, switching MAC rescales x by a constant, so **every setup keeps
its intercept and R² exactly and its slope scales by exactly 0.6** (audited: the
recomputed fixed-cohort fits match the committed phase-2/ftir_13 metrics to 1e-9, and
intercept@MAC6 ≡ intercept@MAC10 for all six). The matrix's intercept column is therefore
MAC-proof — no MAC choice moves −4.17 / −6.91 / −3.69 / −3.22 / −1.62 — and the MAC fork
is fought entirely on **slopes**: MAC = 6 makes the raw models self-consistent
(lowest-OCEC 800 **0.95**, Ethiopia-shaped smoke **1.05**, deployed 1.14) while MAC = 10
is where the AIRSpec model lands closest (**0.86**) — the raw-at-MAC6 vs
corrected-at-MAC10 fork of ftir_13/ftir_16, now visible setup by setup as a pivot around
each fixed intercept (`output/plots/deck/mac_effect_all_calibrations.png`, slope summary
in `mac_slope_pivot.png`). One deck erratum found and fixed: the matrix quoted the
AIRSpec intercept as −1.61, but the committed value is −1.6151 → **−1.62** (ftir_13's
tl;dr had it right; `build_deck_figures.py` and the deck PNGs are corrected)."""

TAKEAWAYS["19"] = """\
- **The MAC fix cannot repair any intercept.** Changing the assumed HIPS MAC rescales the
  x-axis only, so each calibration pivots around its intercept — the offsets in the setup
  matrix survive any MAC choice, and arguments about the intercept and arguments about
  MAC are fully separable.
- **The MAC fork is a slope contest, and the sides are unchanged**: raw-spectra models
  look self-consistent at MAC = 6 (OCEC-800 0.95, Ethiopia-shaped 1.05, deployed 1.14),
  the AIRSpec OCEC-800 model at MAC = 10 (0.86). External evidence (ftir_16's IMPROVE
  implied-MAC bridge, median 10.05 at Addis-like OC/EC; ftir_15's season-stable corrected
  residuals) still tilts the fork toward MAC ≈ 10 + the corrected model.
- **R² is MAC-invariant too, so "fit quality" cannot arbitrate MAC** — RMSE and bias do
  swing with MAC, but only as re-expressions of the slope change, not as independent
  evidence.
- **Deck correction**: AIRSpec intercept is −1.62 (−1.6151), not −1.61;
  `calibration_setup_matrix.png` and `intercept_ladder.png` regenerated.
- **Caveats**: headline panels are the fixed 190-filter cohort (available-pairs rows in
  the metrics CSV); the spectral-analogs setup is shown only for completeness (fails the
  held-out TOR test); and IMPROVE-vs-SPARTAN HIPS protocol comparability remains an
  assumption, as in ftir_08/ftir_16/ftir_18."""


TLDR["20"] = """\
The two projects choose k differently in **two** places at once — the CV scheme and the
stopping rule — and separating them explains both the deployed family's large k and where
the real damage is. The AQRC Shiny app uses 10-fold **interleaved** CV (no site grouping,
`segment.type = "interleaved"`) with k read off the RMSEP curve; phase 3 uses
**site-grouped** 5-fold CV with the first-major-minimum rule. Crossed over all six
setup-matrix cohorts and raw vs second-derivative spectra: **(1) the protocols disagree on
k by up to ~3×** — on raw spectra the Calibration app protocol picks
27 / 19 / 17 / 9 / 17 / 19 components against the site-held-out protocol's
10 / 7 / 21 / 9 / 9 / 6 — but **not always in the same
direction** (Ethiopia-shaped smoke goes *up*, 17 → 21; the analogs tie at 9), so "the app
picks more components" is not the right statement. **(2) The real finding is the error floor,
not k**: holding whole sites out raises the %RMSECV floor by **×1.59 on smoke-906** and
**×1.41 on Ethiopia-shaped smoke**, but only **×1.07 / ×1.02 / ×1.01** on the full pool,
lowest-OC/EC and lowest-OC/EC + AIRSpec. Interleaved CV flatters precisely the
smoke-selected cohorts the deployed family is built from — those cohorts concentrate
repeat sampling at a few sites, so same-site filters straddle folds — while the
composition-selected cohort (126 sites, 800 filters) is essentially indifferent to the CV
scheme. **(3) Second-derivative preprocessing helps exactly where raw spectra are being
used to soak up baseline**: under honest site-grouped CV it cuts the floor on smoke-906
(155% → 100%), the full pool (110% → 96%) and Ethiopia-shaped smoke (53% → 43%), but makes
the targeted cohorts *worse* (lowest-OC/EC 62% → 77%), consistent with ftir_13/17's finding
that the AIRSpec baseline already removes what the derivative would. Absolute floors are
sobering under site-grouped CV: the full-pool and smoke-906 raw calibrations sit at
RMSECV ≈ the mean EC loading itself. **(4) The leaked quantity is baseline, not
chemistry**: on second-derivative spectra every optimism ratio collapses to **0.89–1.06**
(smoke-906 ×1.59 → ×1.00), so what an interleaved fold hands the model is the site's
characteristic background — the same structure ftir_13/ftir_19 identify as the reason raw
models fail to transfer to Addis."""

TAKEAWAYS["20"] = """\
- **Report the CV scheme with every k.** The same cohort and the same spectra give k = 27
  or k = 10 depending only on whether folds respect site boundaries. Any comparison of
  component counts across the two projects is meaningless without it.
- **The leakage penalty is cohort-specific, and it indicts the smoke lineage specifically.**
  A ×1.59 (smoke-906) and ×1.41 (Ethiopia-shaped) inflation from interleaved folds, versus
  ×1.01–1.07 for the pool and the lowest-OC/EC cohorts, says the deployed family's apparent
  skill leans on having seen its own sites. This is an independent, protocol-level reason
  to distrust the smoke-906 calibration at Addis — a site it has never seen — alongside
  ftir_13's finding that it collapses on corrected spectra.
- **The lowest-OC/EC cohort is robust to this whole argument** (×1.02 raw, ×1.01 AIRSpec):
  its skill does not depend on the fold structure, which is what you want from a cohort
  meant to transfer to a new site.
- **Second-derivative preprocessing is not a free win.** It rescues the baseline-dominated
  cohorts and degrades the AIRSpec-corrected one — the two are alternative fixes for the
  same problem, so applying both is not additive. Weakley's argument for choosing k on
  2nd-derivative spectra applies to the raw-spectra calibrations, not to phase 3's.
- **What the site leakage actually buys is baseline structure.** On second-derivative
  spectra the optimism ratio collapses to **0.89–1.06 for every cohort** — including
  smoke-906, which falls from ×1.59 to ×1.00. Removing the smooth baseline removes the
  advantage of having seen a site before, so the thing an interleaved fold leaks is the
  site's characteristic background, not its chemistry. That is the same quantity ftir_19's
  AIRSpec explainer shows Addis carries in excess, and it closes the loop with ftir_13:
  raw-spectra models regress on site background, which is exactly what fails to transfer.
- **Caveats**: the matrix's "Deployed SPARTAN" row appears here as the **entire IMPROVE
  network** (13,010 filters, no selection) because that is what it is on the training
  side — the deployed model is a network calibration trained on IMPROVE, and the only
  SPARTAN spectra in this project are ETAD's, which are the evaluation set and never
  training data; the locked analog cohort is
  the top 500, not the matrix's superseded "400"; curves are computed on each *full*
  cohort while the locked k values were selected on training splits, so the two need not
  coincide exactly; and the first-major-minimum rule needs a per-fold standard error, which
  a pooled interleaved RMSECV curve does not have — on those curves it degenerates to the
  global minimum, which is why each rule is paired with its own scheme in the figures."""


def script_to_cells(text: str) -> list:
    cells = []
    kind, lines = None, []

    def flush():
        nonlocal kind, lines
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if not body.strip():
            kind, lines = None, []
            return
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else ("" if line == "#" else line)
                for line in body.splitlines()
            )
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
        kind, lines = None, []

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind = "markdown"
        elif line.startswith("# %%"):
            flush()
            kind = "code"
        else:
            lines.append(line)
    flush()
    return cells


def build(number: str) -> None:
    script_name, notebook_name = SCRIPTS[number]
    text = (Path("scripts") / script_name).read_text()
    for replacement, marker_context in ((TLDR[number], "## tl;dr"),
                                        (TAKEAWAYS[number], "## Takeaways")):
        # Replace the first placeholder following each section marker.
        marker_at = text.index(marker_context)
        placeholder_at = text.index(PLACEHOLDER, marker_at)
        text = text[:placeholder_at] + "\n".join(
            "# " + line if line else "#" for line in replacement.splitlines()
        ).removeprefix("# ") + text[placeholder_at + len(PLACEHOLDER):]

    cells = script_to_cells(text)
    # nbclient runs headless (Agg): switch the first code cell to the inline
    # backend so plt.show() embeds figures instead of warning.
    first_code = next(c for c in cells if c.cell_type == "code")
    first_code.source = "%matplotlib inline\n" + first_code.source
    notebook = nbformat.v4.new_notebook(cells=cells)
    client = NotebookClient(notebook, timeout=1800,
                            resources={"metadata": {"path": "."}})
    client.execute()
    nbformat.write(notebook, notebook_name)
    print(f"executed and wrote {notebook_name}")


TLDR["22"] = """\
The phase-3 figures that depend on the component choice, re-derived under both protocols so
every one has a Calibration-app version and a site-held-out version
(`K_SENSITIVITY_AUDIT.md` classifies the rest; the band-identity, implied-MAC and
spectra-comparison figures contain no calibration and need no update). Three results.
**(1) ftir_15's central finding is protocol-robust and, if anything, sharper.** The raw
model's Addis residuals track score-space distance under *both* protocols
(r = **0.87** Calibration app, **0.71** site-held-out) while the AIRSpec model's do not
(**0.33** / **0.21**, Spearman −0.05) — so "the raw model misses by extrapolating, the
corrected model misses by a constant offset" is a property of the spectra, not of the
component count. The site-held-out numbers reproduce ftir_15 exactly (0.714 vs the
committed 0.71). **(2) The bootstrap intercept CI survives the protocol change for the
AIRSpec model and not for the raw one**: AIRSpec gives [−2.07, −1.03] under the app
protocol against [−1.78, −1.17] site-held-out — heavily overlapping, both excluding zero —
whereas raw widens from [−4.46, −2.78] to [−6.17, −3.11]. The headline "the corrected
intercept is real and near −1.6" holds whoever calibrates. **(3) ftir_11's cohort-size
conclusion is protocol-dependent, and this is the one to correct.** Under the site-held-out
protocol N = 800 is the sweet spot (intercept −3.22, RMSE 1.16, held-out TOR R² 0.911) with
1600 worse on both. Under the Calibration app protocol the intercept and RMSE improve
**monotonically** with cohort size (−5.56 → −4.59 → −4.17; RMSE 2.28 → 2.03 → 1.60), so
that protocol would have chosen N = 1600. "800 is the right size" is a site-held-out
result, not a universal one."""

TLDR["22"] += """ **(4) And an unplanned finding that fell out of reconciling this
notebook with ftir_21**: the Calibration app protocol is **not reproducible across
analysts**. Interleaved CV assigns folds by position (`i % 10`), so listing the *same 800
filters* in a different order changes the answer — ranked by OC/EC gives k = 18, the
committed cohort CSV gives k = 19, a shuffle gives k = 15, with Addis intercepts moving
correspondingly. Site-grouped CV assigns folds by site label and returns **k = 5 in all
three orderings**. Two people running the app on identical data can therefore ship
different calibrations."""

TAKEAWAYS["22"] = """\
- **The two ftir_15 conclusions the deck leans on are safe.** Both the extrapolation-vs-
  offset distinction and the near-−1.6 corrected intercept survive being re-derived under
  the other protocol. That is worth saying explicitly, because they are the results the
  protocol argument would otherwise cast doubt on.
- **Retire "N = 800 is the sweet spot" as a standalone claim.** It is true under the
  site-held-out protocol and false under the Calibration app protocol, where bigger is
  monotonically better. State the protocol whenever the cohort size is defended — the
  held-out TOR test (0.911 at 800 vs 0.793 at 1600) is what actually picks 800, and that
  test only exists in one of the two protocols.
- **The raw model is the protocol-sensitive one throughout** — widest bootstrap CI, largest
  k swing (6 → 17), strongest D² coupling. The AIRSpec model is stable on every axis
  measured here and in ftir_20/21. Robustness to analyst choices is now a documented
  property of that calibration, not an assertion.
- **Interleaved CV is order-dependent, which makes the app protocol unreproducible.**
  The same 800 filters give k = 15, 18 or 19 depending only on how the rows happen to be
  sorted; site-grouped folds are defined by site label and give k = 5 every time. This is
  the cleanest single argument for the site-held-out protocol — it is about reproducibility
  rather than taste, and it explains the k = 18 vs 19 difference between this notebook and
  ftir_21 on identical data.
- **Caveats**: the bootstrap CIs differ slightly from ftir_15's committed values because
  the RNG stream is consumed by four model×protocol combinations here rather than two —
  differences are within bootstrap noise, not a change of result. ftir_15's hybrid-cohort
  figure is *not* re-derived: that cohort's membership is itself selected through a fitted
  PLS score space, so an app-protocol version would have to re-select the cohort, not just
  refit it."""


TLDR["23"] = """\
Shows the component decision itself, protocol by protocol: each cohort's CV curve with the
rule's own machinery drawn on it — the Calibration app's "within 5% of the minimum" band
and threshold line, and the site-held-out protocol's ±1 SE ribbon with every local minimum
marked — then the Addis crossplot the chosen model produces, so the decision and its
consequence sit on one page. The rule internals are re-derived here and asserted equal to
the production selectors, so the drawings cannot drift from what actually runs. Three
things become visible that a k-value alone hides. **(1) Both rules deliberately stop short
of the curve minimum, and by a lot.** Selected k vs where the curve actually bottoms:
Calibration app 27/30, 19/21, 17/19, **9/26**, 17/20, 19/25; site-held-out 15/17, 4/4,
**10/26**, **4/29**, 6/9, 5/5. The spectral-analog curve keeps falling to k = 26–29 under
both protocols and both rules refuse to follow it — which is the rules working, since that
cohort has no held-out TOR skill. **(2) The two curves are different objects, not two reads
of one curve.** The app's pooled RMSECV has no fold-to-fold spread, so it cannot support an
error band; the site-grouped curve carries one, and on the smoke and IMPROVE-network
cohorts that ±1 SE ribbon is enormous — the honest uncertainty on where the minimum even
is. The app curve looks decisive precisely because it discards that information.
**(3) Where the curve genuinely bottoms early, the protocols agree**: biomass-smoke (4/4)
and lowest-OC/EC + AIRSpec (5/5) pick their own global minimum under the site-held-out
rule, so the disagreement elsewhere is about flat, noisy tails rather than about the
optimum."""

TAKEAWAYS["23"] = """\
- **A selected k is a claim about a curve, and the curves differ.** The app protocol and
  the site-held-out protocol are not two readings of the same evidence — one pools PRESS
  across position-based folds, the other averages across site-based folds and keeps the
  spread. Comparing their k values without showing both curves hides that.
- **The ±1 SE ribbon is the argument.** On the IMPROVE-network and smoke cohorts it spans
  several µg/filter, so "the minimum is at k = 17" is not a statement the data supports.
  The site-held-out rule stopping early is a response to that width; the app rule has no
  way to see it.
- **Both rules protect against the same failure and it shows most on the analogs.** Its
  curve falls monotonically to k ≈ 26–29 and both rules stop at 9 and 4 respectively —
  exactly the cohort that turns out to have no held-out TOR skill. Following a CV curve to
  its minimum would have picked the worst calibration in the set.
- **Agreement where it matters is reassuring**: the two AIRSpec-corrected and smoke curves
  bottom early and the site-held-out rule takes their true minimum (5/5 and 4/4), so the
  protocol argument is not a disagreement about every cohort — it is specifically about
  long flat tails, where the app rule's 5% band admits far more components than the
  evidence separates.
- **Caveat**: the app-protocol curves inherit the row-order dependence documented in
  ftir_22 — these were computed in the cohort orders used by ftir_21/23, and a different
  sort would move the app k by a few components (not the site-held-out k)."""


if __name__ == "__main__":
    for number in (sys.argv[1:] or list(SCRIPTS)):
        build(number)
