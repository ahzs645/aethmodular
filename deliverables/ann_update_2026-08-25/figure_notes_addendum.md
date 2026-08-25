# Addendum — two figures added after the deck was built (2026-08-25)

The deck (`ann_update_2026-08-25.pptx`, 21 slides, commit 1c70a3f) was assembled
before these two existed. They are title-free, white, 168 dpi, house palette, and
ready to drop in. Claim-titles and NOTES below follow the house style.

---

## `f_framing_stability_deming.png`

**CLAIM TITLE:** Every MAC-10 framing picks the same calibration — and lands on
the independent York fit

**SAY:** Four Deming framings (fixed 190 / all-pairs × eval all-lots / eval lot
251) all select **ocec-440 × AIRSpec, k = 8** at held-out TOR R² 0.92, with
intercepts −1.19 to −1.55 µg/m³. The York/EIV band is −1.27 to −1.51. MAC 6 sits
at −2.42 to −2.87 — about 1.2 µg/m³ further out. Within a MAC branch the answer
is stable; the MAC choice is the only thing that moves it.

**NOTES — three corrections that had to be made to get this right:**

1. **The ranking is meaningless without a held-out gate.** Scored on
   `|intercept| + 0.5·|slope−1|` with only the slope box (0.85–1.18), the winner
   is `ocec-300 × deriv2` at intercept **+0.20 with held-out TOR R² = 0.14** —
   near-zero intercept, no predictive skill. This figure gates at held-out
   **≥ 0.90**. State the gate on any slide that uses it.
2. **"Rank 1 under every MAC-10 framing" is only true for Deming.** Under OLS the
   winner moves to `ocec-210` or `ocec-840 × deriv2` with *smaller* intercepts
   (−0.26 to −1.08). The deck's "wins 62.5% of bootstraps" (5 of 8) is the honest
   number. This figure drops OLS at the user's request, so it is internally
   consistent — just don't pair it with an "every framing" claim.
3. **"MAC-6's best is twice as bad however it's framed" is also Deming-only.**
   MAC-6 OLS all-pairs gives −0.95, better than MAC-10 Deming's −1.19.

York band = span of the three blank-line variants in
`output/plots/offset_story/york_variants_cache.json` (deployed −1.51 ± 0.19,
lot-linear −1.29 ± 0.19, lot-quadratic −1.27 ± 0.17). σ_y is inflated per site
until MSWD = 1, so curvature is absorbed into the error bars — conservative. The
adjudication doc's headline −1.38 ± 0.18 sits mid-band.

Source: 21,955 scored Addis rows in `cache/batch_results.jsonl`, cutoffs swept
every 10 filters, Option A (site-grouped 5-fold, first major minimum).

---

## `f_1617_three_baselines.png`

**CLAIM TITLE:** Three baselines, three answers — the band is Ethiopian fuel
chemistry, not the offset

**SAY:** Raw spectra see a peak only at Addis and Pasadena. AIRSpec anchors
segment 2 at the minimum in 1520–1600 cm⁻¹ and gives that window zero spline
weight — directly under the band, so it cannot see it. Only the neutral baseline
(`pspline_arpls`) resolves all five sites.

**NOTES:** The discriminator is peak **position**, not height. An amplitude
metric puts Pasadena highest (0.59) and Bishoftu *lowest* (0.09) — the exact
opposite of the finding. On position, under the neutral baseline:

| site | peak | prominence |
|---|---|---|
| Addis | **1617** | 0.106 |
| Bishoftu | **1621** | 0.033 |
| Delhi | 1625 | 0.018 |
| Beijing | 1635 | 0.014 |
| Pasadena | 1635 | 0.270 |

The two Ethiopian sites sit in a tight 1617–1621 window; every other site peaks
at 1625–1635. Pasadena's 1635 is the water assignment.

Two details in the circulating prose do **not** reproduce: Bishoftu is not
stronger than Addis (Addis is 3× more prominent), and Delhi/Beijing do not peak
at 1679 (weak peaks at 1625/1635). Neither changes the conclusion, which rests on
**Bishoftu having the band and no offset**.

Markers on the figure are the strongest local maximum in 1560–1680 cm⁻¹
(prominence ≥ 0.008); outlined markers are the Ethiopian sites. Curves are
CH-normalised with a local linear baseline removed across 1500–1760 cm⁻¹, so what
is plotted is band *shape*, not absolute absorbance.
