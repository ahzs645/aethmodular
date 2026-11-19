# Linearising HIPS across SPARTAN: per filter or per site (19 November 2026)

The question (19 Nov): can we tell, per site or per filter, whether HIPS needs linearising, and should the correction go
on every site or only the affected ones? This note answers both on the 11 Nov network table, and says where each answer
lives in the gallery. It follows [hips-loading-protocol.md](hips-loading-protocol.md) and
[hips-network-2026-11-11.md](hips-network-2026-11-11.md); nothing here changes the curve, the ceiling or the statuses.

Gallery: Meeting tab, Nov 19, **"Linearising HIPS across SPARTAN"** (`#tab=meeting&meeting=hips-story`). Code:
`gallery/app/src/lib/hipsRules.ts` (the rules), `lib/hipsLinearise.ts` (the curve, statuses, site readings),
`pages/HipsStoryPage.tsx` (the page). Tests: `lib/hipsRules.test.mjs` reproduces the 11 Nov screen's saved exponent
at all 23 testable sites, as measured and after linearising, from `public/data/spartan_reference.json`.

## Identifying where it is needed

| Level | Read from | Rule | Where |
|---|---|---|---|
| Filter | its Fabs and sampled volume | expected under-read = optical depth ÷ 2.26; used up to 20 %, caution to 30 %, shape only beyond | gallery fields `HIPS optical depth` and `HIPS expected under-read` (pickers, group "HIPS loading"), on every network row with a volume |
| Site, expected | its median filter | the bend is expected past a 10 % under-read: 17 of 30 sites | recomputed in the browser from the filters loaded, so it also reads an IMPROVE site |
| Site, seen | its own filters, no curve assumed | exponent of HIPS on FTIR EC (PM2.5 as instrument) or the heavy-over-light ratio below 1 | 8 sites: Addis Ababa, Dhaka, Johannesburg, Delhi, Santiago, Pretoria, Kaohsiung, Beijing (`lib/hipsSites.json`) |

A site's median hides its filters: 48 filters past a 20 % under-read sit at the 13 sites where the bend is not
expected, and 248 filters under 10 % sit at the 17 where it is.

## Applying it: five rules on the same 3,733 filters

Central ceiling (2.26). "Towards" and "away" count the 23 sites with an exponent of their own whose distance from 1
changes by more than 0.02.

| Rule | Filters corrected | Over a tenth from their own factor | Sites towards b = 1 | Pushed away | Within 0.1 of 1 |
|---|---|---|---|---|---|
| Every filter, by its own factor | 3,733 | 0 | 9 | 8 | 6 → 8 |
| Sites where expected | 2,085 | 357 | 6 | 2 | 6 → 9 |
| Sites where seen | 1,538 | 890 | 5 | 0 | 6 → 9 |
| Filters past a 20 % cut-off | 1,173 | 1,021 | 7 | 9 | 6 → 7 |
| One factor per site (its median filter's) | 3,733 | 834 | 0 | 0 | 6 → 6 |

What follows from it:

- **Per filter is the correction.** It needs no decision about who is affected: a light filter's factor is close to 1
  (18 % of filters move by under 5 %), and the factor grows smoothly with darkness.
- **One factor per site cannot remove the bend.** Scaling every filter of a site alike moves its level and leaves its
  exponent exactly where it was (Addis stays at 0.46; per filter takes it to 0.84). At Addis the site factor lands on
  the per-filter median level but misses more than half the filters by over a tenth.
- **A cut-off on single filters puts a step in the response:** × 1.25 at 20 %, so two filters of nearly the same
  darkness end up a quarter apart, and it pushes more sites away from proportional than any other rule.
- **A site rule is a statement of trust, not a different correction.** "Where expected" and "where seen" apply the same
  per-filter factor at fewer sites. They avoid pushing sites that do not show the bend (Abu Dhabi, Mexico City, the
  light sites already at or above 1), at the price of putting corrected and measured sites on one axis: under "where
  seen", Jaipur stays at 5.3 µg/m³ and falls behind Dhaka at 5.9 only because one was corrected.

## In the gallery

- Every chart tab: side panel, Plot, HIPS, Linearised, then **Apply to** All sites, Where expected or Where seen
  (`?hips=2.2596&hipsat=seen` style keys in a link; `hipsat` absent means all sites). Units then read ", linearised
  where seen", and the panel lists the sites on each side with their median filter's under-read.
- The story page has five parts: the effect (IMPROVE), the fix (one filter, then every SPARTAN filter on the curve),
  isolating it (each site's filters by under-read; expected against seen), before and after under the chosen rule
  (the factor per filter, each site's median BC, each site's exponent), and the five rules side by side. Its ceiling
  and rule are in Page options (`?hl.ceil=`, `?hn.rule=filter|expected|seen|cut|site`, `?hn.cut=` in percent).

## Limits

- The exponent check is against FTIR EC, not thermal EC: it tests the shape of the correction, and cannot tell a bend
  in HIPS from one in FTIR EC.
- "Where seen" chooses its sites on that same exponent, so that it pushes no site away from 1 is partly by
  construction.
- "Where seen" exists only for SPARTAN sites in the 11 Nov screen; an IMPROVE site, or a SPARTAN site added since, is
  left as measured under it until the screen is rerun (`aeth hips screen`, then `gallery/data/export_hips_linearise.py`).
- The ceiling is IMPROVE's (see the protocol's limits).

## Recommendation

Keep per filter as the correction and the default. Offer "where seen" as the conservative view for a level comparison
across sites, and say so whenever it is used. Do not use one factor per site or a cut-off on filters. This bears on
decision (4) of the 11 Nov item in [open-items.md](open-items.md).
