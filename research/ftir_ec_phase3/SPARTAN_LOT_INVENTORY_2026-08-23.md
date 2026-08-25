# SPARTAN filter lots — network inventory, 2026-08-23

Source: `Spartan/SPARTAN_HIPS_Batch1-51.v2.csv` (3,963 unique filters, 27 sites,
2022-04 → 2026-02). Lot is `LotId`; periods are `SampleDate` ranges.

## Every lot in the network

| lot | filters | with Fabs | sites | first | last | span |
|---|---|---|---|---|---|---|
| 248 | 332 | 274 | 10 | 2022-04-26 | 2023-05-14 | 383 d |
| 241a | 42 | 14 | 4 | 2022-06-08 | 2024-12-08 | 914 d |
| **251** | **2,635** | **2,082** | **27** | 2022-06-24 | 2025-12-29 | 1,284 d |
| 241 | 32 | 0 | 3 | 2022-07-01 | 2023-01-14 | 197 d |
| 250 | 489 | 373 | 10 | 2022-07-26 | 2024-08-13 | 749 d |
| **253** | **433** | **330** | **13** | 2025-03-17 | 2026-02-25 | 345 d |

Lot **251 is the backbone** — 66% of all filters, the only lot present at all 27
sites. Lot 241 has no Fabs at all. **26 of 27 sites mix lots** (max 4, at AUMN),
so lot is a within-site variable everywhere, not a site label.

## Lot mix at the sites we work with

| site | filters | w/Fabs | period | lot mix |
|---|---|---|---|---|
| ETAD (Addis) | 280 | 239 | 2022-12 → 2026-01 | 248: 40 (2022-12→2023-03) · **251: 224** (2023-03→2025-08) · 253: 16 (2025-11→2026-01) |
| ETBI (Bishoftu) | 32 | 26 | 2025-10 → 2025-12 | **251: 32** |
| CHTS (Beijing) | 232 | 192 | 2022-07 → 2024-12 | 248: 32 · **251: 200** |
| INDH (Delhi) | 224 | 152 | 2022-06 → 2026-02 | 248: 24 · **251: 128** · **253: 72** |
| USPA (Pasadena) | 184 | 158 | 2022-07 → 2023-11 | **251: 168** · 248: 16 |

## The lot-253 gap

The calibration basis is IMPROVE **lots 248 + 251** (`local_db/spectra_248_251.csv`).
Lot 253 entered the network in **March 2025** and now carries **433 filters
(10.9% of the network) across 13 sites** — including the 16 newest Addis filters
and 72 at Delhi. **Since 2025-09, 58% of new filters are lot 253** (234 of 401;
the rest are 251). No lot-253 spectra are in the calibration set.

**The blank-line half of that gap is small and now bounded.** Blank lines are
lot-specific (`tau = ln((Intercept + Slope·R1)/T1)`):

| lot | Intercept | Slope | distinct pairs |
|---|---|---|---|
| 248 | 1383.2 | −2.544 | 3 |
| 250 | 1438.1 | −2.932 | 4 |
| 251 | 1416.2 | −2.783 | 3 |
| **253** | **1375.1** | **−2.608** | **1** |

Applying **lot 251's blank line to the 330 lot-253 filters** shifts Fabs by a
median **+0.57 Mm⁻¹ (+2.9%)**, IQR +0.40 to +0.90; at ETAD's 14 lot-253 filters,
+0.67 Mm⁻¹. That is **2.6% of the ~21.5 Mm⁻¹ Addis offset** — lot heterogeneity
in the blank line cannot explain the offset, and mis-assigning a lot line is a
~3% error, not a structural one.

Note lot 253 also has the **highest median Fabs of any lot** (17.78 vs 14.78 on 251).

It has only **one** distinct blank-line pair so far versus 3–4 for the mature
lots, and I initially read that as its blank line being under-determined. **The
database says the opposite.** `hips.CalibrationSets` shows "253 initial" was
built from **33 lab blanks** (SPARTAN BatchId 45) — against **10** for lot 248
and 16 for 241a. Lot 253's line is the *better*-determined one.

The multiple pairs for 248/250/251 are not repeated characterisation of a
drifting substrate; they are **re-calibrations forced by instrument events** —
fibre-optic reconfiguration, replacement of the reflective collimator with a
focusing lens collimator, the HIPS move from room 132C to 138, and in two cases
(lots 250 and 251) an **unexplained shift**: *"performed due to an apparent
shift in the last calibration of this lot."*

That reframes the blank line as **time-varying within a lot, tied to hardware
events** — and it is a measurement-side mechanism that could produce a
time-correlated, site-correlated offset. It does not affect the Beijing result
below (both lots were compared inside one 178-day window), but it is a live lead
for the Addis intercept and is untested. See `scripts/AQRC_DB_NOTES.md`.

## Tested: is there a lot effect, and does baselining remove it?

**Corrected 2026-08-25.** The first version of this section reported a lot effect of
~4-5 Mm-1 and attributed it to "the phase-3 calibration (ocec-450 x AIRSpec)". The
selection space was AIRSpec but the **calibration ran on raw spectra** — the API token
for baseline-corrected spectra is `spectra="airspec"`, and the value passed
(`"corrected"`, the filename convention) falls through the dispatch to raw silently.
That is exactly the select-on-corrected / calibrate-on-raw mistake Ann flagged in the
2026-08-12 1:1. Re-run in both spaces:

Method: fit the calibration, predict the site's filters, then compare lots **within one
site and within the window when both lots were in use** (CHTS 2022-10-13 -> 2023-04-09,
178 days, n = 14 vs 34), bootstrapping the mean residual against a common line.

| cohort | calibration spectra | lot 248 - lot 251 | 95% CI | |
|---|---|---|---|---|
| ocec-450 | raw | **-4.35 Mm-1** | [-7.63, -1.20] | significant |
| ocec-450 | **AIRSpec** | -1.43 Mm-1 | [-4.01, +0.94] | **n.s.** |
| ocec-800 | raw | **-4.49 Mm-1** | [-7.01, -1.96] | significant |
| ocec-800 | **AIRSpec** | -0.86 Mm-1 | [-3.66, +2.02] | **n.s.** |

**Baseline correction removes the lot difference.** On raw spectra the two lots differ by
~4.4 Mm-1, significant at both cohort sizes; on AIRSpec-corrected spectra the difference
falls to -0.9 to -1.4 Mm-1 and spans zero in both.

This confirms the mechanism Ann proposed on 2026-08-12: *"I think when you baseline
correct, that may be what is helping - you're baseline correcting and then that minimizes
the difference between [lot 248] and [lot 251]."* Baselining is not only removing Teflon
background in general; it is specifically absorbing **between-lot substrate differences**.

**It still does not explain the Addis offset.** ETBI is 100% lot 251 and ETAD is 80% lot
251, so the Addis-vs-Bishoftu contrast is a **within-lot-251** comparison, and the
residual lot term after baselining is ~1 Mm-1 against an offset of ~21.5 Mm-1.

## Not testable with current data: lot 253 at Delhi

The obvious lot-253 test — Delhi, which carries both 251 and 253 — **is not
identifiable**. Lot is perfectly confounded with period and loading:

| lot | n | predicted EC (10–90%) | median Fabs | period |
|---|---|---|---|---|
| 251 | 76 | 1.19–9.77 (med 4.26) | 39.4 | 2023-03 → 2025-09 |
| 253 | 61 | 4.41–26.59 (med **17.58**) | **107.6** | 2025-09 → 2026-02 |

Lot-253 filters are **4× more loaded** and occupy a disjoint period (Delhi's
winter smog season). Restricting to the shared EC range leaves **6** lot-253
filters, and the bootstrapped intercept difference is 95% CI [−53, +304] —
uninformative. The full-range intercepts happen to agree (26.08 vs 26.38), but
that is a coincidence of the confound, **not** evidence of no lot effect.

To test lot 253 properly you need a site where it interleaves in time with 251
at comparable loading. Lot changes are cartridge changes and therefore sequential
by construction, so this needs a site that alternated cartridges — worth
scanning the other 12 lot-253 sites (CLST 72, CLTA 64, USSL 40, INJA 32 are the
largest) for one with interleaved dates.

**Still untested: the spectral half at lot 253 specifically.** Whether a lot-253 PTFE substrate
shifts the FTIR→EC prediction is a separate question from the blank line, and it
is answerable with data in hand — **INDH carries both lot 251 (76 filters with
Fabs) and lot 253 (61) at one site**, so a within-site, between-lot residual
comparison isolates a lot effect from a site effect. Staged pulls now support it
(`DAVIS/SPARTAN FTIR pulls/INDH/`).

Caveat: `eval_lot` in the explorer resolves lots only for the built-in Addis
target (`app.py` hardcodes `Site == 'ETAD'`), so a custom target's lots must be
mapped externally via `reference.csv`'s `ExternalFilterId` → HIPS `LotId`.

## Estimator note

The lot comparisons above use plain OLS of Fabs on predicted EC. That is fine
for a **difference between two groups at one site** — both groups share the same
noise structure, and the mean-residual-vs-common-line statistic does not depend
on per-group slope estimation at all.

It is **not** fine for comparing absolute intercepts *across* sites. OLS with
error in the x variable dilutes the slope and inflates the intercept, by an
amount that scales with each site's residual scatter — which varies hugely here
(residual sd: Pasadena 1.6, Bishoftu 4.9, Beijing 5.2, **Delhi 16.3** Mm⁻¹). A
naive cross-site intercept ladder built this way is an artifact of differing
scatter and mean loading, not an offset comparison. The project's cross-site
offset numbers come from the York/EIV re-fit at fixed MAC in
`OFFSET_ADJUDICATION_2026-08-23.md`, and that remains the reference for any
statement about site offsets.
