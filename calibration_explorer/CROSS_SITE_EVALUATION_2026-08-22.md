# Five-site cross-evaluation — 2026-08-22

> **Revision 2026-08-23:** the Deming table below is superseded by the
> per-filter weighted York/EIV fits in
> `research/ftir_ec_phase3/OFFSET_ADJUDICATION_2026-08-23.md` — headline
> changes: Beijing's slope is 0.98 (the 1.48 was unweighted leverage),
> Delhi's offset is Addis-sized in point estimate but not distinguishable
> from zero (−1.32 ± 0.86; its robust anomaly is the 1.83x slope), and
> Addis is the only site with a certain negative intercept (−1.38 ± 0.18).
> Quote the adjudication doc's table, not this one.

Every SPARTAN site with FTIR spectra + HIPS Fabs, evaluated against the same
three calibrations. Spectra pulled from Networks_1_0 (`build_spartan_target.py`
per site), AIRSpec-df6 corrected targets included. All rows: Deming, MAC 10,
all pairs. Targets live in the app dropdown: addis / etbi / chts / indh / uspa.

| site (n) | ocec-450×AIRSpec k=9 | ocec-800×AIRSpec k=5 | ocec-800×raw k=6 |
|---|---|---|---|
| **Addis** ETAD (239) | 0.93x−1.42, R² 0.71 | 0.87x−1.71, R² 0.63 | 1.67x−3.76, R² 0.69 |
| **Bishoftu** ETBI (26) | 0.93x−0.56, R² 0.51 | 0.73x−0.59, R² 0.38 | 0.78x−0.65, R² 0.86 |
| **Beijing** CHTS (192) | 1.48x−0.08, R² 0.36 | 1.28x+0.10, R² 0.57 | 1.46x−0.26, R² 0.70 |
| **Delhi** INDH (152) | 2.36x−4.70, R² 0.60 | 2.01x−2.79, R² 0.78 | 2.02x−3.66, R² 0.87 |
| **Pasadena** USPA (158) | 4.22x−0.70, R² 0.58 | 2.62x−0.22, R² 0.54 | 1.78x−0.10, R² 0.51 |

## The reframe

Yesterday's binary ("the offset is Addis-specific") becomes an **ordering**:

- **Large negative intercept: Addis AND Delhi** (−1.4…−4.7). Delhi's raw
  readout (2.02x−3.66) practically mirrors Addis raw (1.67x−3.76).
- **Near-zero intercept: Beijing, Pasadena, Bishoftu** (−0.7…+0.10).

So the offset is not one city's quirk and not the instrument — it appears at
exactly the two heavily polluted, biomass/incomplete-combustion-influenced
megacities and is absent at the coal/dust-dominated (Beijing), clean (Pasadena)
and lightly loaded satellite (Bishoftu) sites. That is a **compositional
signature** — the strongest evidence yet for the real-non-EC-absorption branch
(BrC/char-like absorbers at 633 nm), with a possible loading dependence mixed
in (the offset tracks the most heavily loaded biomass sites; ETBI's Fabs only
reaches 42 Mm⁻¹ vs Addis's ~87).

Also note: AIRSpec correction *halves* the Addis intercept (−3.8→−1.7) but
only trims Delhi's (−3.7→−2.8) — Delhi's offset is less baseline-removable,
worth its own look.

## Slopes: site-implied MAC ordering (handle with care)

Corrected-model slopes order Addis/Bishoftu ≈0.9 < Beijing 1.3–1.5 < Delhi
2.0–2.4 < Pasadena 1.8–4.2. Read as implied MAC (≈10/slope at b≈0): Addis ~11,
Beijing ~7–8, Delhi ~5, Pasadena ~4–6. Two caveats before quoting: slope
entangles true site MAC with calibration extrapolation error (Weakley's
atypical-site lesson), and Pasadena's slope is unstable across calibrations
(1.78→4.22) — its low loadings likely sit at the training edge; treat USPA
slopes as unreliable. The robust statement is the *intercept* ordering, not
the slope one.

## Extrapolation certification (added 2026-08-23)

The Reggente-2016 diagnostic is now computed with every fit (whitened distance
in the model's score space; **extrap % = share of target filters beyond the
training p95**) and the cross-site table is a one-click feature — the new
**Sites tab** evaluates the current configuration against every target, with
out-of-domain rows flagged. For ocec-450 × AIRSpec k=9:

| site | extrap % | verdict |
|---|---|---|
| Addis | 0.4% | fully in-domain (by construction) |
| Pasadena | 0.0% | in-domain — its wild slope (4.22x) is NOT extrapolation; it is an ill-conditioned Deming slope on a narrow, low Fabs range |
| Bishoftu | 19.2% | mostly in-domain — the 0.93x transfer is credible |
| **Delhi** | **23.7%** | **mostly in-domain — the −4.70 intercept is NOT an extrapolation artifact; the Delhi offset is real within the model's domain** |
| Beijing | 31.8% | flagged — partially out-of-domain; quote its slope with caution |

The certification strengthens the headline: the Addis+Delhi offset survives
the domain check, and the one slope that looked craziest (Pasadena) turns out
to be a range problem, not a domain problem.

## Caveats

ETBI n=26 dry-season only; Fabs-only references everywhere (MAC fork applies
at every site); HIPS uncertainties not propagated; site-implied MAC needs the
extrapolation diagnostic (Mahalanobis distance of each target to the training
cloud — the analog-lab machinery can produce this per site) before it's quoted.

## Mechanism first look (2026-08-23): the 1617 band follows the offset

> **CORRECTED (same day, peak-shape analysis in
> `research/ftir_ec_phase3/BAND1617_LEAD_2026-08-23.md`):** the "Delhi matches
> Addis exactly" claim below is an artifact of the linear-baseline prominence
> metric — at Delhi, A@1617 sits on the flank of its large 1700–1720 carbonyl
> band, with **no discrete interior peak**. Only Addis has a genuine local
> maximum (~1620 cm⁻¹, +0.0035). So the discrete band is **Addis-specific** —
> which no longer tracks the *unweighted* intercept ordering, but does match
> the York re-fit where Addis is the only certain intercept. The carbonyl-vs-
> slope decomposition further down is unaffected.

Corrected-space median spectra, five sites: **Delhi's 1617 cm⁻¹ band matches
Addis's exactly in absolute prominence (0.0025 vs 0.0025)** while the
zero-offset sites are 2–4× weaker (Beijing 0.0015, Bishoftu 0.0010, Pasadena
0.0007) — the band ordering tracks the intercept ordering. Delhi's mixture is
otherwise different (carbonyl ~3× Addis; overall shape r≈0.95 for all sites),
but the char-signature band (ftir_12's aromatic-C=C/carboxylate 1617) is what
the two offset sites share.

Per-filter (n=528 across ETBI/INDH/CHTS/USPA, winner calibration): 1617-band
per m³ vs residual is positive within 3 of 4 sites (ETBI r=0.64, USPA 0.63,
CHTS 0.43; INDH 0.09). Pooled carbonyl-per-m³ vs residual reaches r=0.90 —
BUT this pooled number is confounded: it is dominated by *between-site* slope
differences (Delhi's slope 2.36 makes residual grow with loading, and Delhi is
carbonyl-rich), so it conflates two hypotheses — real site-MAC differences vs
**FTIR EC over-prediction from oxidized-organic interference**. The decomposition
(same day): within-site r(residual, carbonyl/m³) is 0.92 at Delhi and 0.90 at
Pasadena (0.24 Beijing, 0.02 ETBI), surviving control for CH loading (partial
r ≈ 0.87–0.89) — but **detrending by each site's own fit line collapses it**
(0.28 / 0.06 / −0.01 / 0.04). So carbonyl doesn't explain scatter *within* a
site; it explains **which sites have inflated slopes**: the two carbonyl-rich
aerosols (Delhi's mixed haze, Pasadena's aged SOA) are exactly the sites where
FTIR-EC ≫ Fabs/10. Two readings survive — genuinely low site MAC (<10), or
**OC interference inflating FTIR-EC where oxidized organics dominate** —
and only an independent EC (quartz TOR, or the same-filter deployed-EC_ftir
four-site comparison) separates them. Delhi's weak detrended r=0.28 is the
only within-site hint for the interference reading.

Summary of the mechanism picture: **the 1617 char band tracks the intercept
(who has excess absorption); carbonyl tracks the slope (who has inflated
FTIR/absorption ratios).** Two different axes, two different chemistries.

## One-line version for the deck

> The FTIR–HIPS intercept follows the aerosol, not the instrument: large at
> Addis and Delhi, absent at Beijing, Pasadena and Bishoftu — across five
> sites, one filter lot family, and one HIPS instrument.
