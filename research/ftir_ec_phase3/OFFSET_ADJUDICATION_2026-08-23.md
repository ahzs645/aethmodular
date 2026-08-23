# The offset question, adjudicated — 2026-08-23

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

- **1617 cm⁻¹ band (char signature) tracks the intercept ordering**: Delhi
  matches Addis exactly in absolute prominence (0.0025 both), zero-offset
  sites 2–4× weaker. Per-filter 1617-vs-residual positive within 3 of 4 new
  sites.
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

**Belief:** the Addis intercept is real and at least partly compositional
(char/tar absorption at 633 nm). Evidence: survives per-filter EIV weighting,
extrapolation certification, dust closure (Fe explains 7%), the 1617-band
association, and literature plausibility. The curvature degeneracy is real on
the regression axis alone but does not explain why the residual follows a
char band, and the loading-artifact reading fails at Beijing.

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
