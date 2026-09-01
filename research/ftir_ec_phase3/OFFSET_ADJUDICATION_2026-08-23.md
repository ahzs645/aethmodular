# The offset question, adjudicated — 2026-08-23

> **Correction, 2026-09-01 (ftir_47).** Two headline numbers in §5 and §5b were
> produced by pooling every blank in a lot into one quadratic. Lot 251's blanks
> belong to **three deployed calibration lines** with disjoint R1 ranges, so the
> pooled curve is not a blank response; keyed per (lot, Intercept, Slope) the
> shipped Fabs reproduce exactly and: **Pasadena does not dissolve** (3.15x →
> 3.04 ± 0.19x per-line quadratic; the 0.91x is the pooling artifact inflating
> 56/158 filters by ~80%), and **Addis's −1.27 ± 0.17 was the same artifact**
> (per-line quadratic −1.48 ± 0.18 vs deployed −1.51; the blank-line shape moves
> Addis by ≤ 0.03 µg/m³, not 0.24, so the "~15% blank-line share" is ≤ 2%). 44%
> of Addis filters (not 36%) sit below their own line's blank R1 range. Both
> slope anomalies (Delhi, Pasadena) are real. The refuted text is left in place
> below, marked. See `ftir_47_blank_lines_per_deployed_line.ipynb`.

Three documents written within 24 hours of each other reach different
conclusions about the FTIR–HIPS intercept:

- `calibration_explorer/CROSS_SITE_EVALUATION_2026-08-22.md`: "compositional
  signature — Addis AND Delhi, the aerosol not the instrument."
- `DUST_FE_TEST_2026-08-22.md` (concurrent session): per-site `Fabs = α·EC^β`
  curvature explains ~100% of the Addis and Beijing offsets → additive offset
  vs curvature is **degenerate on FTIR EC alone**; leans loading-artifact.
- `AERONET_SPARTAN_2026-08-22.md` (concurrent session, self-flagged as
  provisional): Addis surface absorption looks high for its column → leans
  artifact.

This file collects the new evidence (York/EIV re-fit with real HIPS
uncertainties, the 1617-band comparison, literature research) and states what
we currently believe and what decides the rest.

## 1. York re-fit: the table that should replace the Deming one

Per-filter weighted errors-in-variables (York 2004) using the empirical
per-site σ(Fabs) models from `unified_filter_dataset.pkl` (σ² = a² + (b·F)²
fits; USPA has 10 sub-MDL filters with up to 442% relative uncertainty).
σ_y inflated per site until MSWD = 1, which absorbs lack-of-fit (including
curvature) into the error bars — conservative. Winner config, x = Fabs/10.
Script: `scripts/york_cross_site.py` (needs the explorer server).

| site | n | κ | free York fit | z(intercept) |
|---|---|---|---|---|
| Addis | 239 | 0.92 | 0.92x − **1.38 ± 0.18** | **−7.7** |
| Bishoftu | 26 | 0.91 | 0.86x − 0.39 ± 0.45 | −0.9 |
| Beijing | 192 | 0.97 | **0.98x** + 0.64 ± 0.15 | +4.3 |
| Delhi | 152 | 0.99 | **1.83x** − 1.32 ± 0.86 | −1.5 |
| Pasadena | 158 | 0.86 | **3.15x** − 0.20 ± 0.09 | −2.2 |

Revisions this forces:

1. **Beijing was never a slope anomaly.** The unweighted 1.48x was leverage
   from its noisiest high-loading points; weighted, Beijing sits on the 1:1
   line. Its 31.8% extrapolation flag still stands, but the readout is clean.
2. **Delhi's offset softens; its slope doesn't.** Point estimate −1.32 is
   essentially Addis's −1.38, but the wide Fabs range (up to 225 Mm⁻¹) leaves
   the intercept poorly constrained: consistent with Addis-sized *and* with
   zero. Delhi's robust anomaly is the **slope** (1.83 ± 0.11).
3. **Addis is the only site with a certain negative intercept.** −1.38 ± 0.18
   survives per-filter weighting, MSWD-inflated errors, and (from yesterday)
   the extrapolation check.
4. **A common slope is rejected** (0.86–3.15 with small SEs), so shared-slope
   intercept comparisons mislead — the per-site free fits above are the
   quotable numbers. κ ≥ 0.86 everywhere: even Pasadena's slope is
   identified; 3.15x is its actual behavior, not noise amplification.

So the five-site pattern is **two separate anomalies**: an *intercept* anomaly
(Addis, certainly; Delhi, possibly) and a *slope* anomaly (Delhi 1.8x,
Pasadena 3.2x — the two carbonyl-rich aerosols, see §2; implied site MAC ~5.5
and ~3.2, or FTIR-EC over-prediction from oxidized organics).

## 2. The spectral evidence (2026-08-23, in CROSS_SITE doc)

- **1620 cm⁻¹ band: Addis-specific** (CORRECTED same day). The first-pass
  claim that Delhi matches Addis was a metric artifact — the linear-baseline
  prominence reads Delhi's carbonyl flank as "band"; peak-shape analysis
  (`BAND1617_LEAD_2026-08-23.md` correction section) shows only Addis has a
  genuine interior maximum (~1620, +0.0035). Under the unweighted table that
  killed "band tracks offset" (Delhi had the biggest intercept, no band); under
  the York re-fit it *re-aligns*: the only site with the band is the only site
  with a certain intercept. With n=5 that is consistent, not probative.
- **Carbonyl tracks the slope**: within-site r(residual, carbonyl/m³) ≈ 0.9
  at Delhi and Pasadena but collapses when detrended by each site's own line —
  carbonyl identifies *which sites* have inflated slopes, not which filters.

This matters for the degeneracy claim in DUST_FE_TEST: curvature and additive
offset are degenerate *on FTIR EC alone*, but the 1617 band is an independent
spectral axis, and it co-varies with the offset across five sites. A pure
loading artifact has no reason to follow a char band; note, though, that
char-rich sites are also heavily loaded sites, so the confound is reduced,
not eliminated (Beijing — heavily loaded, weaker band, no offset — is the
best single counterexample to the pure-loading reading).

## 2b. AERONET cross-checks (2026-08-23, in-hand data)

Two new tests on the existing AERONET pull (`output/tables/aeronet/`),
extending `AERONET_SPARTAN_2026-08-22.md`:

**Filter organic chemistry predicts the sun photometer's column absorption —
but the carrier is the 1500–1700 envelope, not the discrete 1620 peak.**
Joining per-filter corrected spectra to AERONET-matched days (n=125 Addis
filters), controlling CH loading + the filter's own Fabs + month fixed
effects: the broad corrected 1500–1700 absorbance predicts AAOD675 at
r = **+0.28** (perm p = 0.002); carbonyl alone +0.18 (p = 0.06); the tight
peak-specific 1620 metric **+0.05 (ns)** — the discrete char-band version of
this claim died under the month control, consistent with the BAND1617
correction. O–H (3100–3400) shows nothing (specificity control). The
photometer never touches the filter, so no filter artifact generates the
surviving envelope correlation: filters richer in oxygenated/aromatic organic
absorbance are matched to genuinely more-absorbing columns per unit loading,
within months. Evidence for real composition-linked absorption, but it no
longer names char specifically. Caveats: Level 1.5, same-day pairing
(mispairing biases toward zero), n=125.

**The AAE apportionment bound is weak at 675 nm — by construction.** Under
the textbook BC-AAE=1 anchor, the non-BC share of AAOD675 is Addis **4.2%**
(the only clearly positive site; Beijing 0.5%, Delhi −0.4%, Pasadena −5.7% —
the cross-site ordering matches the intercept/band ordering yet again).
Surface-equivalent ≈ 1.9 Mm⁻¹ at Addis — far below the ~14–22 Mm⁻¹ offset.
But the bound is anchor-sensitive (using Pasadena's empirical 0.78 as the BC
anchor triples it) and *blind to AAE≈1–2 absorbers* — char and dark BrC,
exactly our candidate — for the same reason ftir_28 found the MA350 cannot
resolve a red excess: two-wavelength apportionment near the BC slope has no
lever arm at red wavelengths. Do not quote the 1.9 Mm⁻¹ as a refutation; it
constrains only *steep-spectrum* BrC, which was never the candidate.

## 3. What the literature says (full citations: `ABSORPTION_LITERATURE_633NM_2026-08-23.md`)

- **Tar balls/char absorb at 633 nm.** Hoffer et al. AMT 2017: tar-ball
  absorption at 880 nm is >10% of its 470 nm value — red absorption is not
  soot-only. Char-dominated biomass BC is increasingly reported for Asia.
  Classic water-soluble BrC does fade by ~600 nm; tar/char does not.
- **Delhi numbers bracket ours**: published Delhi BrC ≈ 12 Mm⁻¹ at 660 nm
  (primary BrC 87→12 Mm⁻¹ over 370→660), against winter b_abs(880) ≈ 190
  Mm⁻¹. Tens of Mm⁻¹ of non-BC absorption at 633 nm is literature-consistent.
- **The artifact counter-hypothesis is fiber-specific.** Lack/Cappa 2008
  organic-enhancement biases arise from organics wicking into fibrous mats;
  on PTFE membranes the analogous effect is weaker. And an artifact scaling
  with *organic loading* should appear at Beijing's OM-rich winter haze — it
  doesn't.
- **The decisive experiment is cheap and uses filters we have**: Kirchstetter
  2004-style solvent extraction (water, then methanol) of archived Addis +
  Delhi PTFE filters with HIPS re-measurement at 633 nm, Beijing/Pasadena as
  controls. Absorption drop only at the anomalous sites → real BrC/char;
  methanol-resistant dark residue → tar/char specifically. Second choice:
  TEM tar-ball counts on a few filters.

## 4. Statistical discipline for the writeup (agent recommendations)

- Report the κ (range-to-noise) table alongside any EIV fit; state that
  Pasadena's earlier 4.22x *Deming* slope was an unweighted-fit artifact and
  the weighted slope is 3.15 ± 0.19.
- Frame the 16k-variant sweep as locked discovery/confirmation: headline
  numbers only from held-out + out-of-country sets; show the full screening
  score distribution; one winner's-curse sentence (Taylor & Tibshirani 2015).
- Small-n (ETBI): analytic York SEs with t(n−2); BCa bootstrap only as a
  sensitivity check. A hierarchical Bayesian EIV (Stan) is the eventual
  publication-grade model but not needed to act.

## 5. Current belief and what decides it

> ⚠ **Superseded 2026-09-01 (ftir_47):** the "~15% instrument-calibration component", the −1.27 ± 0.17, and "Pasadena's was the instrument (3.15x → 0.91x)" below all came from the lot-pooled quadratic and are wrong; per deployed line Addis is −1.48 ± 0.18 (deployed −1.51) and Pasadena 3.04 ± 0.19x. Belief as of 2026-09-01: the Addis intercept is real aerosol with a blank-line share ≤ 2%; **both** slope anomalies are real.

**Belief (final for today, after the blank-line test ran):** the Addis
intercept is **real aerosol, with a small (~15%) instrument-calibration
component**: it survives per-filter EIV weighting, extrapolation
certification, dust closure, the PTFE-zero bound, and now the blank-line
recompute (−1.27 ± 0.17 under the quadratic line, z ≈ −7). The composition
evidence is the AERONET envelope link (§2b), BC/PM2.5 = 23% (3× any other
site), and literature plausibility — no longer the 1617 band (§5d). The
slope anomalies split: **Pasadena's was the instrument** (blank-line shape
error at very low loading — 3.15x → 0.91x under the quadratic line);
**Delhi's 1.8x is real** and still awaits the MA350 anchor to separate low
site MAC from FTIR organic interference.

**Still open:** offset-vs-curvature functional form (needs an independent EC:
quartz TOR); Delhi and Pasadena slopes = low site MAC vs FTIR-EC organic
interference (needs the same, or the extraction test).

**Deciders, in order of decisiveness per unit effort:**
1. Solvent-extraction + HIPS re-measurement (Addis/Delhi filters, Beijing/
   Pasadena controls) — user-side ask to Ann/Davis.
2. Quartz TOR EC on collocated filters (already the known MAC-fork blocker).
3. TEM tar-ball counts on a handful of Addis/Delhi filters.
4. In-repo: HIPS (R,T) reflectance-channel inversion for a per-filter
   scattering term — the most direct *instrument-side* test we can run
   without new lab work (data populated at four sites, still untouched).

## 5b. Data-inventory sweep (2026-08-23): in-hand tests, ranked

A full audit of repo + Drive assets found the offset can be attacked much
harder without new lab work. Key discoveries: the HIPS math is fully decodable
from the SPARTAN batch export (`SPARTAN_HIPS_Batch1-51.v2.csv`, 3,963 rows,
27 sites, 575 blanks): τ = ln((Intercept + Slope·R1)/T1) where
(Intercept, Slope) is the **lot field-blank regression line** — so the
reflectance channel IS the scattering correction, and the untested assumption
is its validity outside the blanks' range. Ranked tests:

1. **Blank-line extrapolation + common/quadratic blank-line Fabs recompute —
   RUN (same day; now a live app feature, see §5c).** York fits per site with
   Fabs recomputed under {deployed, lot-common linear, lot quadratic} blank
   lines (matched-filter subsets):

   | site | matched | R1<blanks | deployed | lot-linear | quadratic |
   |---|---|---|---|---|---|
   | Addis | 233 | 36% | 0.95x−1.51±0.19 | 0.97x−1.29±0.19 | **0.91x−1.27±0.17** |
   | Bishoftu | 26 | 0% | 0.86x−0.39 | 0.80x−0.24 | 0.83x−0.31 |
   | Beijing | 184 | 0% | 0.97x+0.63 | 0.91x+0.63 | 0.88x+0.72 |
   | Delhi | 152 | 14% | 1.83x−1.32 | 1.80x−1.13 | **1.78x−1.05** |
   | Pasadena | 158 | 0% | 3.15x−0.20 | 1.47x+0.35 | **0.91x+0.67** |

   > ⚠ **Superseded 2026-09-01 (ftir_47):** the "quadratic" column above is the lot-pooled fit. Per deployed line: Addis 0.95x−1.48±0.18, Bishoftu 0.86x−0.36, Beijing 0.96x+0.66, Delhi 1.82x−1.25, **Pasadena 3.04x−0.15**. Outcome (a) survives more strongly (shift ≤ 0.03, not 0.24); outcome (b) is **refuted** — Pasadena's slope is real; the "~40 counts rms on the big lots" was the pooling residual (per line: 5–10 counts, lot 253 is not special).

   Three headline outcomes: **(a) the Addis intercept survives** — the
   blank-line correction shaves only ~0.2 µg/m³ (−1.51 → −1.27, z ≈ −7
   throughout). Scope caveat (raised by the concurrent session, and correct):
   this bounds the **blank-line mechanism specifically** — the recompute uses
   only raw optics (R1/T1/blanks), never EC, so it is independent of the
   offset-vs-curvature degeneracy — but blanks sample only the unloaded
   regime (blank τ ≈ 0), so any response nonlinearity *at high loading*
   remains unprobed by this test and degenerate with real absorption on
   FTIR-EC axes (DUST_FE_TEST curvature section). Read "~15%" as "the
   blank-line share," not "the instrument share." Three separate lines (MAC
   fork, ETBI contrast, this degeneracy) now terminate at the same place:
   **quartz TOR**; **(b) Pasadena's slope anomaly dissolves entirely** (3.15 → 0.91
   under the quadratic line) — at τ ~0.09 the blank-line shape error (~40
   counts rms on the big lots) dominates, so USPA was never an aerosol
   anomaly; **(c) Delhi's 1.8x slope is blank-line-robust** — that one is
   real (low site MAC or FTIR organic interference; still needs the MA350
   anchor to split). Also quotable: blank-τ PTFE zero = +0.32 ± 0.46 Mm⁻¹
   (dead branch), and lot 253's blanks are ~10× tighter (rms 3.8 vs 31–42)
   than every other lot's.
2. **MA350 IR-880 as independent per-filter BC anchor, all four sites**
   (~1 day). 172/155/63/130 filter-day matches in
   `ftir_hips_chem/processed_sites/*.pkl`; run_ftir_28.py is the template but
   has only ever run at Addis. The only in-hand axis circular with neither
   FTIR nor HIPS; splits hypothesis (B): Delhi/Pasadena Fabs/BC880 normal +
   EC_ftir/BC880 anomalous → FTIR organic interference; Fabs/BC880 low →
   genuinely low MAC. (Check first: the CHTS/INDH/USPA pkls stamp 15:00 not
   09:00 — window alignment unverified.)
3. **IMPROVE per-filter-blank τ bias transfer function** (~1 day). The FED
   export (`ahzs645_20260422_*.xlsx`) has RefI/TransI *pre-sampling* readings
   for 185k filters — a true per-filter blank SPARTAN lacks. Measures the
   population-blank-line error vs loading directly; pairs with per-site
   Fabs=α·EC^β on the 151k TOR matches (159 sites n≥300 — but IMPROVE p95
   Fabs ≈ 23 Mm⁻¹, so it constrains shape/mechanism, not Addis's magnitude).
4. **3-wavelength AAE (440-675, 675-870) per day, four cities** (~0.5 day).
   The inversion pulls hold 4-λ AAOD at 100% completeness (Addis's file is on
   Drive, not in output/tables/aeronet/); flat AAE(675-870) with elevated
   AAE(440-675) is the specific dark-BrC signature the single 440-870 exponent
   cannot show. Run with the MA350 diurnal (24h vs AERONET-hours) ratio that
   the raw Jackros minute file supports.
5. **K⁺ + Al/Si/Ti residual regression** (~0.5 day). K⁺ is the best-covered
   unused biomass tracer (200/392/47/289); DUST_FE_TEST's own unrun item 3.
   ChemSpec traps: ng/m³ units, base-FilterId join, duplicate floor rows.

Controls worth pulling later via the existing SQL pipeline: AEAZ Abu Dhabi
and IDBD Bandung sit in Delhi-like optical regimes with entirely different
chemistry (dust / non-charcoal) — their HIPS R/T/τ is already in the batch
CSV today.

## 5c. The HIPS tab (2026-08-23)

Item 1 is now a first-class explorer feature: `calibration_explorer/hips_lab.py`
(blueprint registered at the end of app.py) serves `/api/hips_york` — the
current configuration evaluated on every SPARTAN target, York-fitted with
per-filter σ under all three blank-line variants — and `/api/hips_blanks`,
the blank ledger. UI: the **HIPS** tab (table + intercept-sensitivity ladder
+ blank ledger). Targets' reference.csv now carry `ExternalFilterId` (the
batch-export FilterId bridge). The app's port is now env-overridable
(`PORT`, default 5058) so two sessions can run instances side by side.

## 5d. Third revision of the band story (same day, concurrent session)

Under a neutral baseline (pybaselines pspline_arpls — AIRSpec's spline
*anchors at 1520–1600 cm⁻¹, directly under the band*, so corrected-space
band amplitudes are suppressed by construction), the ~1617 band appears at
**ETAD and ETBI both**, not Addis alone: it is an **Ethiopian/regional
marker**. Since Bishoftu has the band but no offset, **the band does not
carry the offset** — full arc: "band tracks offset across five sites" →
"band is Addis-only" → "band is Ethiopian, decoupled from the offset."
Composition evidence for the offset now rests on the AERONET envelope link
(§2b — itself to be re-checked against the AIRSpec anchor caveat), the
BC/PM2.5 = 23% extremity, and the char literature — not on the 1617 band.

## 6. Reprioritized in-repo queue (post-agent synthesis)

1. ~~York/EIV weighted fits~~ — done (this doc, `scripts/york_cross_site.py`).
   Port into the app's readout when `app.py` is free (concurrent session).
2. ftir_14 score-space comparison, now unblocked: per-site distance
   *distributions* + Q-residuals (extrap% is score-distance only) for all
   five targets. ~0.5 day, no app.py contention.
3. Char-typology join extended to Delhi (char06 classes × INDH residuals) —
   the mechanism hunt, now the only live route to splitting the §5 readings
   in-repo.
4. HIPS reflectance channel inversion (item 4 above).
5. Backfill `extrap_pct` into `batch_results.jsonl` (README currently
   overstates) and add `target` to the batch grid.
6. Same-filter EC_ftir↔Fabs four-site constant on the NEW reference files
   (Delhi 153 pairs vs the pkl's 63).
7. Re-run `build_spartan_target.py ETBI` periodically — 23 scanned filters
   await HIPS; turns n=26 dry-season into a seasonal set for free.

Doc hygiene done today: ETBI_FIRST_LOOK gets a superseded banner;
docs/open-items stale HIPS-uncertainty line fixed; CROSS_SITE gets a York
revision note. Still owed: merge `claude/data-analysis-u9r3h7` (main-branch
docs cite ftir_26/ftir_28 which only exist there).
