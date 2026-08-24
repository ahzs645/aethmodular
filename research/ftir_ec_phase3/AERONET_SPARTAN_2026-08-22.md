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

## The 24-h / daytime confound — **CLOSED 2026-08-23**

Filters integrate 24 h; AERONET retrieves only when solar zenith angle ≥ 50°.
The worry was that a city with a large day–night absorption contrast would show
a depressed H for entirely physical reasons — and Addis (high-altitude basin,
charcoal cooking peaks, strong nocturnal inversion) fits that profile exactly.

Measured it directly from the **raw 1-min MA350** at Jackros (co-located with the
AERONET site) and at Pasadena, using **Red BCc — the 625 nm channel, effectively
the HIPS wavelength**. Script: `scripts/ma350_diurnal_coverage.py`.

Addis does have a very strong diurnal cycle: hourly mean Red BCc runs from
3,987 ng/m³ at 00:00 to **19,096 at 04:00** (a 4.8× swing), with a second peak at
17–18. But the AERONET hours (07–10, 14–17) *straddle* it — they miss the pre-dawn
maximum and catch the afternoon ramp, and the two biases very nearly cancel.

Both ratio columns are the **median of per-day ratios**; they differ only in
whether each day is summarised by its mean or its median minute (the mean is
sensitive to Addis's sharp pre-dawn spike, so the pair brackets the answer).

| site | AERONET hours | days | ratio, daily means | daily medians | H | **H corrected** |
|---|---|---|---|---|---|---|
| **Addis** | 07–10, 14–17 (observed) | 751 | **1.05** | 0.92 | 190 | **175–200 m** |
| **Pasadena** | 06–18 (SZA 50–80°) | 327 | **1.09** | 1.07 | 544 | **580–594 m** |

![diurnal coverage](output/plots/aeronet_diurnal_coverage.png)

The figure shows the mechanism: at Addis the two retrieval windows *straddle*
the cycle — the morning one sits below the 24-h mean, the afternoon one above —
while Pasadena's single wide window sits entirely inside its midday minimum.

**Addis's ratio straddles 1 (0.92–1.05); Pasadena's sits consistently above it
(1.07–1.09), so applying the correction widens the gap rather than closing it**
— the Addis/Pasadena separation goes from 2.9× to 3.0–3.3×. Pasadena's photometer window is centred
on midday — exactly when its BC sits at the daily minimum (396 ng/m³ at 12:00 vs
762 at 05:00) — so its column-hours under-sample its own 24-h load. Addis's
bimodal window does not have that problem.

**The Addis H outlier survives the confound that was flagged as disqualifying.**
Reading (a) — the Addis HIPS Fabs is biased high — now stands on independent
optical evidence with its main alternative explanation measured and excluded.

Two caveats retained: this assumes the column tracks the surface in *relative*
diurnal shape (it need not, if aloft layers decouple from the surface), and
Delhi/Beijing have no co-located MA350, so their H values stay uncorrected.

Altitude accounts for little of the gap either: Addis at 2,355 m has ~75% of the
sea-level column — a ~25% effect, not a factor of 3–4.

## REBUILD 2026-08-23 — the temporal pairing was right; the correction below was wrong

The "method correction" in the next section claimed same-day matching was invalid
because a SPARTAN filter is *eight staggered 3-h windows over ~9 days*
(Snider et al. 2015 App. A3, via literature survey). **Checked against the actual
filter metadata — that is not what these sites do.**

`ETAD_metadata.csv` carries true `SamplingStartDate`/`SamplingEndDate`:
**every clean ETAD filter is exactly 24.0 h, local midnight to midnight**, with
196 of 253 consecutive gaps exactly 3 days (the rest 9 d = cartridge changes).
ETBI is the same shape, every 2 days. So a filter *is* a single calendar day.

Rebuilt the match on the true windows anyway — AERONET per-retrieval timestamps
(n = 2,294, not the daily product), converted to Addis local time (UTC+3), and
kept only retrievals falling inside each filter's actual `[start, end)`:

| | n | H median | notes |
|---|---|---|---|
| **true-window match** | 95 | **190 m** (IQR 109–338) | 1–10 retrievals per filter, median 3 |
| naive same-UTC-day match | 95 | **190 m** | identical filter set |

**Median \|H_true − H_naive\| = 0 m (0.0%).** The two agree exactly, because every
AERONET retrieval is daytime (07–17 local = 04–14 UTC) and therefore always
lands on the same UTC calendar day as the local filter day — the timezone edge
never bites at any of these longitudes. The original table's pairing was sound.

(H is 190 m here vs 232 m in the table below only because this subset is
restricted to filters with a verified clean 24-h window and averages retrievals
*within* the window rather than using AERONET's daily product. Same conclusion.)

### What the rebuild does establish — the real limitation, now quantified

AERONET retrievals at Jackros fall in **8 of 24 local hours: 07–10 and 14–17**.
Never midday (almucantar needs SZA ≥ 50°), never overnight. So each 24-hour
filter is optically observed during roughly 8 daytime hours and the other ~16 —
including the entire nocturnal inversion and both cooking peaks — are
**unobserved**.

This is a *coverage* problem, not a pairing problem — the one real limitation
the rebuild exposed. It would bias H downward at any city whose 24-h-mean
surface absorption exceeds its daytime-mean. Now measured at two sites and
found small at Addis (see the CLOSED section above), but it is the right thing
to have worried about.

The daytime window is **not** the same width at every site. Solar geometry
(SZA 50–80°, computed per latitude) gives Addis at 9°N only ~5–8 eligible hours,
in a tight morning/late-afternoon pair, versus 11 at Delhi and 13 at both
Pasadena and Beijing — near-equatorial sites lose the most midday. An earlier
note in this file called the distributions "broadly comparable" from medians and
IQRs alone; the hour *counts* differ substantially and that phrasing was too
generous.

Whether the asymmetry actually matters is now **measured rather than assumed**
— see "The 24-h / daytime confound — CLOSED" above. It does matter, but in the
direction that strengthens the Addis result.

**Closed the same day.** The MA350 at Jackros is co-located with the AERONET
site, so `Fabs_24h / Fabs_(07–10, 14–17)` came straight off the **raw
minute-resolution** file on Drive. (The `processed_sites/*_9am_resampled.pkl`
copies are daily aggregates and cannot answer it — the raw 1-min files are
required, and their `Time local` column is 12-hour with AM/PM.)

## Method correction found after the first run (2026-08-22, literature check) — SUPERSEDED, see above

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
AOD-thresholded, Level 2.0 subset. Same-day matching in the table above; verified
equivalent to true-window matching (see REBUILD) and diurnally corrected via
MA350 at Addis and Pasadena. Nearest-AERONET-site distances differ by city (Gurgaon is ~30 km
from the Delhi SPARTAN site). n = 66–125 per city.
