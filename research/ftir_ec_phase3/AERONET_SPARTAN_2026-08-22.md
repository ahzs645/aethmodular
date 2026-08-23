# AERONET vs SPARTAN filter absorption — first multi-site test, 2026-08-22

An **independent optical check** on the Addis intercept, using column absorption
retrieved by sun photometer against surface absorption measured on the filter.
Script: `scripts/run_aeronet_spartan.py`; tables in
`output/tables/aeronet/`.

## Why this test

Phase 3 finds an Addis-specific additive offset on the HIPS axis (~21.5 Mm⁻¹
absorption-equivalent) that Bishoftu does **not** share
(`calibration_explorer/ETBI_FIRST_LOOK_2026-08-22.md`). Two live readings:

- **(a) measurement** — the Addis HIPS Fabs is biased high;
- **(b) aerosol** — Addis genuinely carries extra non-EC absorption.

AERONET measures the same air independently, so two diagnostics separate them:

- **Effective absorption scale height** `H = AAOD / (Fabs × 1e-6)` [m]. A
  surface Fabs that is too high makes H implausibly small. H is *not* a physical
  boundary-layer height, so it is only meaningful **compared across sites
  measured the same way**.
- **Absorption Ångström Exponent (440–870 nm)** attributes the absorber:
  ≈1 BC-dominated, >1.5 BrC/dust-influenced, >2 dust-like.

**Wavelength**: HIPS is 632.8 nm (He–Ne); the nearest AERONET inversion channel
is **675 nm**, used throughout. An earlier repo analysis assumed 405 nm, from
before the HIPS wavelength was resolved — those numbers are superseded.

## Results (Level 1.5 almucantar inversion, daily; same-day filter matches)

| site | city | n | median Fabs (Mm⁻¹) | median AAOD675 | **H (m)** | H IQR | r(AAOD,Fabs) | AAE | frac AAE>1.5 | SSA675 |
|---|---|---|---|---|---|---|---|---|---|---|
| **ETAD** | **Addis** | 125 | **46.2** | 0.0101 | **232** | 121–396 | 0.29 | **1.17** | **0.26** | 0.916 |
| INDH | Delhi | 66 | 75.0 | 0.0435 | 729 | 299–1152 | −0.21 | 0.98 | 0.06 | 0.903 |
| CHTS | Beijing | 71 | 13.4 | 0.0125 | 894 | 658–1229 | 0.58 | 1.02 | 0.17 | 0.924 |
| USPA | Pasadena | 80 | 4.9 | 0.0023 | 544 | 276–804 | 0.14 | 0.78 | 0.03 | 0.964 |

AERONET sites: AAU_Jackros_ET (co-located with our MA350), Amity_Univ_Gurgaon,
Beijing, CalTech.

## Two findings

**1. Addis is a factor ~3 outlier on column/surface consistency.** The other
three cities cluster at H = 544–894 m; Addis sits at 232 m. The sharpest
contrast is **Delhi**: it has *1.6× Addis's surface Fabs* but *4.3× the column
AAOD*. Delhi's column scales with its surface; Addis's does not. Read plainly:
the Addis filter reports more absorption than the overhead column can account
for, by the standard the other SPARTAN cities set. That supports reading (a).

**2. Addis has the most short-wavelength-enhanced absorption.** AAE 1.17 with
26% of days above 1.5, versus Delhi 0.98/6%, Beijing 1.02/17%, Pasadena
0.78/2.5%. In absolute terms Addis is still BC-dominated (consistent with
ftir_28's MA350 no-red-excess result), but *relative to the other cities* it
carries a real BrC/dust-like component. That is a modest point for reading (b),
and the first evidence Addis differs **compositionally** in its absorption, not
only in magnitude.

## The confound that must be closed before this is presented

Filters integrate **24 h**, including the nocturnal inversion; AERONET retrieves
only in **clear daytime**, when the boundary layer is deep and pollution is
diluted. Any city whose day–night absorption contrast is unusually large will
show a depressed H for entirely physical reasons — and Addis (high-altitude
basin, charcoal cooking peaks at dawn/dusk, strong nighttime inversions) is
exactly that profile.

Working against the confound: Addis's *daytime* boundary layer should be
unusually **deep** (tropical, high-altitude, strongly convective), which would
push H up, not down.

Altitude accounts for little of the gap: Addis at 2,355 m has ~75% of the
sea-level column, a ~25% effect, not a factor of 3–4.

**Decisive follow-up (data in hand):** the MA350 at Jackros is *co-located with
the AERONET site*. Compute the ratio of 24-h-mean to AERONET-hours-mean surface
absorption from the **raw minute-resolution MA350 files** (Drive; the
`processed_sites/*_9am_resampled.pkl` copies are daily aggregates and cannot
answer this), then rescale H. If the corrected Addis H still sits far below
544–894 m, reading (a) stands on its own.

## Method correction found after the first run (2026-08-22, literature check)

**Same-day matching is wrong, and the AERONET sampling times are not what a
naive reading assumes.** Two corrections, both material:

1. **AERONET almucantar retrievals are bimodal and never midday.** At Jackros
   (n = 2,294 retrievals): local-time peaks at **08–09** (674 retrievals) and
   **15–16** (560), with essentially nothing 11:00–13:00 — the almucantar
   geometry requires solar zenith angle ≥ 50°. Median local hour 09:00; only
   65% fall in 09:00–16:00. Because the SZA constraint is latitude- and
   season-dependent, **the retrieval-hour distribution differs by city**, so
   cross-site H comparison is not automatically apples-to-apples. Checked
   empirically (2023, local time): **Addis median hour 09, 65% in 09–16**;
   Beijing median 11, 68%; Pasadena median 12, 61%. All three are bimodal with
   the same IQR (08–15), so the distributions are **broadly comparable** —
   Addis is shifted ~2 h earlier (equatorial: the sun clears SZA 50° sooner and
   the midday gap is wider). That earlier weighting samples the morning
   cooking/inversion period more heavily at Addis, which if anything biases
   Addis's surface-vs-column ratio in the direction observed — i.e. the H
   contrast is *not* explained by retrieval timing, though the ~2 h shift is
   not negligible and should be handled by window-matched compositing.
2. **A SPARTAN filter is not a 24-h sample of one day.** The design is **eight
   staggered 3-h windows spread across ~9 days** (Snider et al. 2015, App. A3);
   consecutive ETAD/CHTS filter dates sit a **median 3 days** apart, confirming
   filters do not represent single days. The correct AERONET composite averages
   retrievals **only within the 3-h windows the filter actually sampled**, not
   a same-day mean — and roughly half of each filter's sampled air was
   collected while the photometer could not see at all. The H values above
   therefore use the wrong temporal pairing and should be treated as
   provisional until rebuilt on the true sampling windows.

Both corrections are implementable with data in hand; neither has been applied
to the table above.

## Caveats

Level 1.5 (not quality-assured 2.0) — AERONET's inversion SSA/AAOD are formally
reliable only above AOD440 ≈ 0.4, which many days here will not meet; the
comparison is therefore indicative and should be repeated with a
AOD-thresholded, Level 2.0 subset. Same-day matching only (no sub-daily
alignment). Nearest-AERONET-site distances differ by city (Gurgaon is ~30 km
from the Delhi SPARTAN site). n = 66–125 per city.
