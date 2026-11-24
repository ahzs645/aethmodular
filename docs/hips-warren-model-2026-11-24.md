# Warren White's deposit model and the gallery's HIPS curve (24 November 2026)

Gallery: Meetings, Nov 24, "Warren White's deposit model and ours" (`#tab=meeting&meeting=hips-warren`, `pages/HipsWarrenPage.tsx`,
model in `lib/hipsPattern.ts`, data from `gallery/data/export_hips_warren.py`). It follows the 19 Nov meeting's asks and gives each
claim a status. White et al. (2025), doi 10.1080/10962247.2024.2442634.

## Is ours an extreme version of Warren's?

The same mathematics, with a more uneven deposit. HIPS reports −ln of a transmittance averaged over a deposit of relative local
thickness z (mean 1): τ(x) = −ln E[exp(−xz)].

| Model | z | Variance of z |
|---|---|---|
| Screen, White 2025 Eq. 15: τ = −ln[h e^(−αx/ω) + (1−h) e^(−x/ω)], ω = 1 + (α−1)h | two thicknesses | h(1−h)(α−1)²/ω²; 0.41 at α = 10, h = 0.638 |
| Same screen, fully pixelated | dots and nothing | (1−h)/h = 0.57, the most IMPROVE's screen can make |
| Ours: τ = x/(1 + x/τ∞) | Poisson(τ∞) number of exponential clumps, exactly | 2/τ∞ = 0.89 at 2.26; 10 % of the area clear |

The variance is the early bend (τ ≈ x − var·x²/2). So at low and middle loading ours bends more than any pattern on IMPROVE's
screen can; past τ about 0.8 his fully pixelated bound corrects more than we do, and stops at τ 1.016. Fitted to the IMPROVE medians: our curve misses by
2.7 % (every month in), α = 10 by 5.9 % (ΔAIC 25), fully pixelated 3.6 % (ΔAIC 8). With the screen free the fit wants h = 0.37, not 0.638. With every month in, the screen alone
does not make the bend against TOR (November to June it nearly does, next section); the rest can be another loading effect, or dust and brown carbon on light filters.

## Applied

| Correction | Addis (189) | MAC for slope 1 | Delhi (109) | MAC |
|---|---|---|---|---|
| none | 2.31x − 6.17 | 4.3 | 1.33x − 1.66 | 7.5 |
| White α = 10 (set on CSN in four US cities) | 0.75x − 0.47 | 13.4 | 1.01x − 0.77 | 9.9 |
| our curve, τ∞ 2.26 | 0.55x + 0.01 | 18.3 | 0.82x − 0.30 | 12.2 |
| fully pixelated | 0.19x + 2.07, 76 filters have no answer | 52 | 0.55x + 2.26 | 18.2 |

## The groups by period and season

Exporter `gallery/data/export_hips_improve_periods.py`, charts in `pages/HipsWarrenPeriods.tsx` (wp-groups, wp-years, wp-time, wp-othergroups,
wp-season-addis). Standard 3.53 cm² deposits, 2008 on, groups by the previous sample's TOR EC.

| Months | τ∞ 2008–2016 | τ∞ 2017–2025 |
|---|---|---|
| All | 2.97 | 1.73 |
| November to June | 3.20 [2.52, 4.56] | 3.20 [2.00, 4.68] |
| July to October | 4.21 [2.40, 7.25] | 1.42 [1.19, 1.69] |

- The period difference is the July to October filters. It survives grouping by two or three samples back and by the next sample (1.5 to 1.8).
- Five-year windows: the July to October bend grows from about 2016 and eases in 2021–2025. November to June also bends more in 2019–2023 (τ∞ 1.95).
- 2005–2006 and, in July to October, 2021 on read low in every group: level changes, cause not known. In 2020 the darkest July to October groups fall to 4 to 6 m²/g.
- November to June medians, 2008–2025: α free gives 9.7; α = 10 misses by 2.4 %, our curve 1.9 %. July to October: α = 10 misses by 12 %.
- Addis: November to June curve (τ∞ 3.26) 0.96x − 1.82, MAC 10.4 (8.3 to 14.6 over the interval). The July to October curve over-corrects Addis to a flat line and cannot be used.

Smoke is a candidate, not shown: the split is by month, and TOR charring, droplet deposits and brown carbon are all possible.

## How Warren got there, and how we got there

Charts in `pages/HipsWarrenRoutes.tsx` (part 2 of the page: wp-routes, wp-evidence, wp-csn, wp-meet), data from `gallery/data/export_hips_csn.py`.

- White started from the deposit (micrographs, hole share known, one unknown α) and set α = 10 by making IMPROVE and CSN means agree at four cities,
  2019–2022 (8.94 and 8.97 Mm⁻¹, 1,628 pairs, no interval). We started from Addis and fitted a two-number curve to IMPROVE against thermal EC.
  They trust HIPS on light filters; we trust TOR EC. Neither uses the other's yardstick. The equation is the same kind.
- Years. White trusts HIPS from 2003 on (2016 paper uses 2005–2014, 2025 paper 2010–2020). The 22 April notes have the same cut ("likely 2003 or
  2004"); the 13 May prep note gives none, and a search of the 14 May transcript found no years for IMPROVE. Ours is 2005–2025. In those windows τ∞ is 2.42 and 2.20.
- Their test rerun: EPA AQS Fabs (63102, Met One SASS, from August 2020) against IMPROVE, 2,372 days to 2025, days sorted by IMPROVE TOR EC.

| Fitted to the 11 medians | Value (90 %, resampling site-months) | Miss | ΔAIC |
|---|---|---|---|
| Our curve, networks may differ by a constant | τ∞ 2.29 [2.09, 2.50], constant 1.10 | 7.3 % | 0 |
| Our curve, no constant | τ∞ 2.76 [2.60, 2.90] | 8.8 % | 2.3 |
| White's model, constant free / none | α 14.6 [11.9, 19.2] / 13.3 [11.5, 16.4] | 8.7 / 8.8 % | 3.9 / 2.1 |
| α = 10 | constant 0.99 | 9.3 % | 3.3 |
| No bend | constant 0.78 | 24.8 % | 25 |

- Only the straight line is excluded; the bent models cannot be told apart. Sorting the days another way moves τ∞ from about 1.5 to 3.3.
- Factor needed (constant in): about 1.3 at τ 0.52, 1.6 at 0.75, 2.0 at 1.14, 2.1 at 1.42. Our curve is right or a little low to 1.1 and over-corrects
  at 1.42 (2.7); α = 10 is low in the middle and right at 1.42. Addis is mostly 0.7 to 1.4.
- Mean test: α = 10 gives IMPROVE 1.08 × CSN in Aug 2020–2022, 0.99 in 2023–2025 (after the paper), 1.03 overall; as reported 0.72. Our curve has no
  finite mean when a filter sits at its ceiling; below τ 1.5 it gives 1.12 and α = 10 gives 0.95.
- Months: τ∞ 2.04 [1.65, 2.43] July–October, 2.44 [2.13, 2.62] November–June, overlapping, against 1.42 and 3.26 from TOR. So TOR, or smoke at
  sites the cities are not, carries the extra July to October bend.
- Addis under these: τ∞ 2.29 0.56x − 0.07 (MAC 17.8); 2.76 0.78x − 1.07 (12.9); α 14.6 0.50x + 0.82 (20); α 13.3 0.56x + 0.52 (18). Still IMPROVE's
  sampler and screen, so still not an Addis MAC.

## Our screen is not IMPROVE's

Only the hole share h and amplitude α enter the model; hole size matters through α. SPARTAN's h and α are unknown.

- 66 of 858 (h, α) screens on the grid leave |Addis intercept| under 0.5, with MACs of 13 to 119: a zero intercept does not identify a screen or a MAC.
- The IMPROVE bend splits equally well (ΔAIC under 0.5) into a screen of α 1 to 10 and a remaining saturating part. If SPARTAN had IMPROVE's
  screen the split matters little (Addis intercept −0.8 to +0.1, MAC 15 to 19). If SPARTAN's deposits had no pattern, Addis runs from 0.53x + 0.10 (MAC 19) to 1.62x − 4.15 (MAC 6).
- Addis draws 1.4 L/min per cm², IMPROVE 6.5, CSN 0.56 (nearly even). Slower air should pattern less, yet Addis bends at least as much.

## Trusted samples and the cut

Masked (2.2 cm², 6,327) and 7.92 cm² (190) filters were in the published fit as 3.53: without them τ∞ 2.23 [1.80, 2.64]. White 2025's years and sites 2.20;
2008–2016 2.97 [2.39, 4.08]; 2017–2025 1.73 [1.33, 2.26], unexplained. At τ 0.45 IMPROVE's screen can hide at most 19 % (α = 10: 12 %, ours: 25 %). Past 1.016: Addis 40 %,
rural IMPROVE 0.6 % (0.3 % in White's set), urban IMPROVE 18 %, Delhi none.

## What settles it

SPARTAN's hole share and deposit photographs; the Addis TOR filters; why IMPROVE reads about 10 % above CSN on light filters (CSN is
now run, above). Not yet claimable: an Addis MAC, or that linearising fixes the intercept.

The exporter also writes `research/hips_drain_disc/pairs.json` (private, gitignored, in no public build); its results are in no tracked file.
