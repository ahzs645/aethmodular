# Attacking the intercept — candidate analyses, ranked

**Live document, last updated 2026-08-12.** This was written before any of it ran. Every
item now carries its outcome, and the ranking at the bottom has been replaced: the original
"suggested order" is spent, and what is actually next is not on the original list.

The target, stated precisely. After the best locked calibration (lowest-OC/EC 800 +
AIRSpec), the Addis residual is a season-stable **−2.0 to −2.6 µg/m³ constant**,
independent of score-space extrapolation (ftir_15/22). An additive constant in y-vs-x with
x = Fabs/MAC means a constant absorption excess C = |intercept|·MAC/slope sitting in Fabs
with no FTIR-EC counterpart.

**The headline of this reframe, as originally written, was wrong.** It read:

> - corrected branch (MAC 10): 1.62 × 10 / 0.86 ≈ 19 Mm⁻¹
> - raw branch (MAC 6): 3.22 × 6 / 0.95 ≈ 20 Mm⁻¹
> — both MAC branches imply the same ~19–20 Mm⁻¹

Those two branches do not independently imply ~19–20 Mm⁻¹, because C is **exactly
MAC-invariant**. `ftir_19` establishes that the intercept does not move with MAC and that
`a@MAC 6 = 0.6 × a@MAC 10`; together these force `|b|·6/(0.6a) ≡ |b|·10/a`. The two figures
above come from two different *models* — raw lowest-OC/EC and its AIRSpec-corrected
counterpart — and each of them returns the same C at either MAC. There is no convergence of
branches to report.

The true claim is stronger than the one it replaces: **the target survives the MAC fork
entirely.** Across all six setup-matrix calibrations C = **18.8–26.1 Mm⁻¹, median 21.5**,
≈ 46% of median Addis Fabs (47.11 Mm⁻¹) — while the slopes those same calibrations produce
span **3.4×**. The reframe therefore stands, and stands whether or not the MAC question is
ever settled: the intercept problem is "find (or rule out) ~21 Mm⁻¹ of constant non-EC
absorption at Addis at the HIPS wavelength (~633 nm — but see the open wavelength question
in `docs/filter-optics-reference.md`)." Written up in `ftir_25_intercept_invariant.md`,
which also records why the stability is weaker evidence than it looks: C is blind to any
multiplicative rescaling of the EC axis, so part of the cross-setup agreement is structural.

Candidate owners: brown carbon, dust, a HIPS-generic artifact, or an FTIR-side zero error.
Each analysis below was chosen to discriminate among them, uses only data we already hold
(resolve via `ftir_hips_chem/scripts/data_paths.py` — never hardcode Drive paths), and names
the lineage it extends.

**And the standing caveat, unchanged by any of what follows:** C is a restatement of the
intercept in absorption units, not evidence about its cause.

| # | Analysis | Status |
|---|---|---|
| 1 | IMPROVE HIPS Fabs at EC = 0 | **DONE** — runs through the origin; offset is Addis-specific (ftir_26, in prep) |
| 2 | Free-offset refit `y = a(x − c)` | **DONE** — a reparameterization; the x-side premise survives its one real test |
| 3 | Deming / errors-in-variables bound | **RESOLVED** — HIPS uncertainties exist; λ\* ≈ 2.96, not 1 |
| 4 | MA350 spectral decomposition | **CLOSED, falsified** — the MA350 cannot answer this question (ftir_28, in prep) |
| 5 | Subtract BrC and re-crossplot | **MOOT** — nothing trustworthy to subtract |
| 6 | Localize the offset by third instrument | **CLOSED, positive** — FTIR axis exonerated (ftir_28, in prep) |
| 7 | Dust's share | **OPEN** — both ChemSpec reference columns are circular; speciation elements remain |
| 8 | FTIR zero: blanks | **DONE, decisive negative** — blanks cannot explain it |
| 9 | AERONET column check | **BLOCKED** — no data locally; needs a portal download |
| 10 | ETBI as background probe | **PRE-REGISTERED**, unchanged (one correction to the comparison) |

## Tier 1 — cheap, in-hand, do first

### 1. Does IMPROVE HIPS itself have a Fabs offset at EC = 0? — **DONE. It does not.**
`local_db/tables/results_hips.csv` + `results_tor.csv` (the 151,843 matches ftir_16
already built). Regress Fabs on TOR EC — pooled, by site, and in the Addis-like
OC/EC ≤ 2.27 subset — and read the intercept **in Mm⁻¹**. If HIPS generically reads
several Mm⁻¹ at zero EC (filter scattering, tau→Fabs conversion, loading correction),
part of the Addis offset is instrument-generic and scales to Addis conditions; if IMPROVE
runs through zero, the offset is Addis-specific. *Extends ftir_16's implied-MAC machinery.*

**Answer: IMPROVE HIPS runs through the origin.** Pooled OLS intercept **+1.345 Mm⁻¹**
[1.195, 1.485]; trimmed to EC ≤ p95 it falls to **+0.200**; in the Addis-like subset,
trimmed, **+0.097** [−0.034, +0.218] — quote that site-cluster interval, not the i.i.d.-row
[0.069, 0.124], which is ~4× too tight because the Addis-like cohort concentrates in few
sites (`ftir_26`). It includes zero, which is the point. At EC ≤ 0 (n = 4,247) the median
Fabs is **+0.12** with
**23.8% negative** — zero at zero with symmetric noise. No individual site reaches 5 Mm⁻¹
(largest 4.64). And the arithmetic that settles the magnitude: only **186 of 160,023**
IMPROVE filters carry 21.5 Mm⁻¹ of *total* absorption, so a generic offset of that size
cannot exist in this instrument. **Bound: the instrument-generic share of C is ≤ 1%.**

So the offset is Addis-specific — but this analysis also raised the two non-absorber
explanations that now sit at the top of the queue (loading dependence and curve geometry);
see [What is actually next](#what-is-actually-next). Being committed as **ftir_26** (in
preparation) — cite it that way, not as this plan.

### 2. Fit the offset as a parameter and demand consistency — **DONE. It was a reparameterization.**
Refit all six setup-matrix calibrations as y = a·(x − c), c a free additive Fabs offset
(site-cluster bootstrap for CI, reusing ftir_15's machinery). *Extends ftir_15/19;
`calibration_modes.py` as-is.*

**There is no new fit to do:** `c = |b|/a` identically — confirmed to 1×10⁻¹⁰ against a
nonlinear least-squares fit, and even SE(c) agrees to 4 dp between the NLS Jacobian and the
delta method on the existing fit (0.1173 both ways). What the reparameterization still buys
is a bootstrap CI *directly on c*, and that needs **no refitting either**: ftir_15's
committed `addis_bootstrap_draws.csv` carries per-draw `slope` and `intercept`, so
`c_b = −b_b/a_b` per draw gives the interval as arithmetic. Per-season c is the same trick.

**The inference this item was built on is weak.** "If every setup recovers the same c, the
offset lives in x" does not follow: c is invariant to any multiplicative rescaling of the EC
axis, so cross-setup agreement is partly structural rather than evidential (ftir_25).

**What did test it — and it passed.** AIRSpec baselining changes only the **y-side**: it
transforms the spectra the model is built from and evaluated on, and never touches Fabs. It
halves the intercept, −3.2215 → −1.6151, which looks like a refutation (an artifact living
in x should not be halvable by a y-side transform). It is not, because it halves the slope
in the same proportion — intercept ratio **0.5013** against slope ratio **0.5406** — leaving
c at **2.0320 → 1.8846, −7.3%**, where a and b each move ~50%. Under `y = a·(x − c)` the
intercept *is* `−a·c`, so this is exactly the behaviour the x-side model predicts.
**The x-side premise survives the one test in hand that could have killed it cheaply.**

### 3. Errors-in-variables sensitivity (the Deming bound) — **RESOLVED. The λ premise here was wrong.**
This item asserted, following `docs/open-items.md`, that HIPS Fabs has no uncertainty
estimates and so fits must assume λ = 1. **HIPS uncertainty is available.**
`HIPS_Uncertainty` is its own **parameter row**, populated **190/190 at ETAD**, median
**2.9075 Mm⁻¹** (`HIPS_MDL` likewise). What misled the docs — and AGENTS.md, since
corrected — is that the `Uncertainty` *column* on `HIPS_Fabs` rows is empty: the value lives
in a sibling row, not in that column, so any probe that reads the column concludes it is
missing.

With it: σ_x = **0.308 µg/m³** (that uncertainty ÷ MAC) against σ_y ≈ **0.531** from the
AIRSpec held-out TOR RMSE, so λ\* = (σ_y/σ_x)² ≈ **2.96, not 1.0**. Assuming λ = 1
**overstates the AIRSpec EIV intercept correction by ~55%** (−2.66 against the correct
−2.09). *Extends the `calculate_regression_stats` Deming path already in `src/`; the read
recipe is in AGENTS.md.*

Two notes for anyone re-running this:

- **The direction claim is a theorem, not a finding.** OLS and Deming both pass through the
  centroid, so `b = ȳ − a·x̄` and therefore `Δb = −x̄·Δa` exactly, with x̄ = **+4.887**. The
  intercept trajectory is a rigid affine image of the slope trajectory; reporting them as two
  results is double-counting one.
- **Unresolved bonus, flagged not closed.** At λ = 1 the AIRSpec EIV slope reaches **1.071**
  at MAC 10, which would reopen the MAC fork (it clears ftir_19's |slope − 1| ≤ 0.1 bar from
  the other side). At λ\* ≈ 2.96 the effect is smaller. This has **not** been done properly at
  the correct λ, and it should be.

## Tier 2 — the decisive physics, needs the MA350 leg

**Closed with numbers, not abandoned.** All three ran; results below are preliminary,
being committed as **ftir_28** (in preparation).

### 4. MA350 spectral decomposition: how much 633-nm absorption is not BC? — **CLOSED, falsified.**
Using collocated filter-day averages (minute files under `aethalometry_dir()`; processing
lineage in `src/analysis/bc/` and ftir_hips_chem): anchor AAE_BC ≈ 1 at 880 nm, attribute the
excess at 625 nm to BrC, and get Babs_BrC(≈633) per filter day. *Extends ftir_hips_chem
aethalometer processing + phase-3 crossplots.*

**There is no red excess to attribute.** AAE(625,880) = **0.944 ± 0.060**, giving an implied
Babs_BrC of **−2.06 Mm⁻¹** — negative on **84.5% of days** — against the **+21.7 Mm⁻¹** the
intercept needs. Closing the gap would require AAE_BC = **0.316**, which is unphysical.

Channel health explains why this cannot be pushed further: **Green** gives unphysical
(negative) AAE, **UV clips on 35% of days**, **Red sits on IR within channel
reproducibility** — only **IR is trustworthy**. So the instrument has one usable channel and
no lever arm.

**Phrase the conclusion as "the MA350 cannot answer this question." Never "there is no BrC at
Addis."** The measurement rules out the instrument, not the absorber.

### 5. Subtract it and re-cross-plot — **MOOT.**
Conditional on (4) being plausible in magnitude. It is not, and there is no trustworthy
Babs_BrC(633) to subtract, so the pre-registered success criterion (intercept CI includes 0
while the held-out TOR test is untouched) cannot be evaluated. The criterion stays on the
books for the day a usable BrC estimate exists. *Numbering note: this item pencilled in
"ftir_25"; that number went to the intercept-invariance write-up, so this work takes the next
free number if it ever runs.*

### 6. Localize the offset by third instrument — **CLOSED. The day's best positive result.**
Fit FTIR-EC against MA350 BC(880) — where BrC contamination is minimal — on the fixed cohort
days, and compare to the HIPS comparison. *This is also the AAAR three-way-comparison bridge.*

**Intercept +0.285 [−0.022, 0.593], R² 0.870** (against 0.743 for the HIPS comparison). The
CI includes zero, so by this item's own pre-registered rule **the FTIR axis is exonerated**
and the additive offset localizes to the HIPS side. Tier-3 item 8 was run anyway and agrees
(below). Note the ~2.2× absolute scale gap that surfaced en route (HIPS Fabs mean 49.7 vs
MA350 b_ATN(625) mean 111.6) is mostly the multiple-scattering C-factor, i.e. expected; the
*additive* result is the news.

## Tier 3 — supporting discriminants

### 7. Dust's share of the ~21 Mm⁻¹ — **OPEN, and this item named the hazard wrongly in both directions.**
Dust absorbs at 633 nm with AAE ≈ 2–3, partially degenerate with BrC in (4). Two independent
handles: SPARTAN chemical speciation for ETAD regressed against the per-filter residual; and
Dry-season behaviour — dust should peak Dry (consistent with ftir_17's relatively-more-O–H in
Dry) while the residual is season-stable, so a large dust share is already disfavoured.
Deliverable: an upper bound on dust's contribution. *Extends ftir_15 residuals + context
tables.*

**Correction to the reference-column warning.** This item banned `ChemSpec_EC` as
"Fabs-derived". Wrong column, and the ban is now needed in both directions:

- **`ChemSpec_BC` is x-circular**: it is Fabs/10 rounded to 2 dp (R² 0.9982 against Fabs/10,
  implied MAC median 10.0003, 86.7% of filters within 0.005).
- **`ChemSpec_EC` is also unusable, for the opposite reason — y-circular**: it reproduces
  `EC_ftir` at r² = **0.999693**, ratio median **1.0000**, median |Δ| **0.0030** µg/m³, the
  2-dp rounding half-width. It *is* the FTIR-EC product, routed through the speciation table.

**Neither can arbitrate anything.** Not-x-circular does not imply independent.

What is left: **speciation elements** (Al/Si/Ca/Fe/Ti/Mg, **188 ETAD filters**) for a dust
proxy. There is **no RCFM/dust column**, so IMPROVE soil must be constructed from the
elements rather than read off. Two join traps, both of which fail silently:

- each ChemSpec filter carries a **second ~0.07 µg/m³ floor row**, so averaging halves the
  values and doubles any implied MAC; and
- `config.BASE_FILTER_ID_PATTERN` matches only the **suffixed** form (`ETAD-0001-1`),
  returning NaN for the unsuffixed ChemSpec ids and emptying the join entirely.

### 8. The FTIR zero: blanks and low-EC behaviour — **DONE. Decisive negative.**
**28 ETAD field blanks exist** (`FilterType == 'FB'`) and were predicted with the locked
models. Blank medians: **FTIR EC 0.269 µg/m³** against **0.024** on the Fabs side — an order
of magnitude apart, i.e. the FTIR side reads *higher* on blanks, the wrong direction to
manufacture a negative intercept. Blank-correcting **both** axes moves the deployed intercept
**−4.170 → −4.393, more negative**. **Blanks cannot explain the intercept.** *Extends ftir_13.*

### 9. AERONET column check — **BLOCKED.**
`aeronet.aeronet_dir()`: Addis absorption AOD + columnar AAE by season, as supporting context
for (4) rather than a quantitative anchor. **No AERONET data is present locally in any form.**
This needs a portal download, not code — it is a data-acquisition task, not an analysis one.

### 10. ETBI as the background probe — **BLOCKED, still pre-registered, one correction.**
If ~21 Mm⁻¹ were a regional/persistent background, Bishoftu (median Fabs 26.9 Mm⁻¹) would be
*mostly* background: its FTIR-EC should come out very low once spectra arrive. The prediction
stands as a clean out-of-sample test the day the INDH/CHTS/ETBI pull lands.

**Correction for like-for-like**: compare 26.9 against Addis's **Dry-season median Fabs 43.2**
(ftir_17), not the all-season 47.1 — ETBI's window is Oct–Dec, i.e. Dry only.

## What would settle it regardless
The quartz-TOR campaign (ftir_16 spec: 11–13 days/season, **~36 filters in total**, quartz
only) is the direct measurement. When this plan was written that was one option among
several, with (4)+(5) as the likely closer. It is now **the decisive measurement**: (4) is
falsified, and every EC reference in the committed dataset is circular with one axis or the
other, so nothing in hand can discriminate the surviving explanations. The ask is written up
in `quartz_tor_campaign_onepager.md`.

## What is actually next

Replacing the original suggested order, which is spent. In priority order:

**(a) The loading-dependent HIPS artifact.** Item 1 rules out an *additive*,
loading-independent instrument offset. It says nothing about one that grows with loading —
and in IMPROVE, per-site intercept **does** grow with site loading (**r = 0.689**).
Extrapolating that relation to Fabs = 47.11 gives **16.2 Mm⁻¹, 75% of C**. But the
extrapolation runs **6× beyond IMPROVE's largest site median (8.1 Mm⁻¹)**, so IMPROVE cannot
close it in either direction. This is the single largest unquantified share of C.

**(b) Curve geometry.** Fabs is **concave** in EC: `Fabs = 7.50·EC^0.796` (R² 0.713, through
zero). A straight line fitted through a concave relation manufactures a negative intercept
with no offset present at all. The scale is right, too: IMPROVE per-site intercepts sit at a
median **35% of site mean Fabs** (IQR 27–50%) against Addis's **46%**, with **29% of sites at
or above the Addis fraction** — in data where the true offset at zero is zero. (The 0.796
exponent is itself EIV-attenuated, so treat it as an upper bound on the curvature and hence
on how much of the intercept curvature could explain.) **Testing (b) on Addis directly is
BLOCKED**: it requires an EC reference derived from neither Fabs nor FTIR, and after item 7
no such column exists in hand.

**(c) Therefore quartz TOR is the decisive measurement**, not one option among several. It is
the only route to an EC reference that is circular with neither axis, and it is what
separates (a), (b), and real non-EC absorption.

Also outstanding, cheap, and unblocked: the bootstrap CI on c from ftir_15's committed draws
(item 2), the Deming sweep re-run at λ\* ≈ 2.96 including the slope-crosses-1 question
(item 3), and the dust proxy from speciation elements (item 7).
