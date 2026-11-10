# Which measurement carries the Addis offset, and what the existing data can correct

**10 November 2026.** The 9 Nov status page left one question open: "Which measurement carries the offset stays
open". This note answers it with data already in the repository, says what that does to earlier conclusions,
gives the correction the existing data support, and lists the pathways still open.

Everything here is exploratory and uses existing data only. No thermal (TOR) EC has been measured on an Addis
filter, so **the absolute Addis EC level is still a range, not a number**. The rules in
[current-research-summary.md](current-research-summary.md) and [AGENTS.md](../AGENTS.md) still apply: HIPS and
the aethalometer are optical, the ChemSpec carbon columns are the same FTIR product, and agreement between
methods is not accuracy.

- Code: `research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py`, helpers in
  `research/ftir_hips_chem/scripts/reference_closure.py`, tests in `tests/test_reference_closure.py`.
- Numbers: every figure below is in [addis-reference-closure-2026-11-10.json](addis-reference-closure-2026-11-10.json),
  written by the workflow's `--summary`. It reads only committed files, so it runs from a plain checkout.
- Population: the 233 counted Addis filters of the 7 Nov closeout. 189 have a SPARTAN published FTIR EC, 188 a
  PM2.5 mass, 172 an aethalometer match.

## 1. The finding: HIPS bends at Addis, FTIR EC does not

Against HIPS optical BC (Fabs / MAC 10), SPARTAN's published FTIR EC reads 2.31x − 6.17 µg/m³ (Deming, n 189).
That intercept is "the Addis offset". A straight-line fit of one instrument on another cannot say which of the
two is not proportional. A third measurement with independent errors can, and Addis has four.

The table gives the response exponent b in Y ∝ X^b. b = 1 means proportional. It is estimated with a third
measurement as an instrument, so random error in X does not flatten it the way it flattens a log-log regression.
Intervals are 95 % bootstrap over filters.

| Y | X | Exponent, by instrument (PM2.5 mass · FTIR EC or HIPS · potassium) |
|---|---|---|
| HIPS Fabs | Aethalometer IR BCc | 0.51 [0.47, 0.56] · 0.48 [0.44, 0.51] · 0.49 [0.44, 0.55] |
| HIPS Fabs | FTIR EC, published | 0.46 [0.43, 0.50] · 0.44 [0.40, 0.47] (aethalometer) · 0.41 [0.36, 0.46] |
| HIPS Fabs | PM2.5 mass | 0.63 [0.58, 0.69] (aethalometer) · 0.60 [0.55, 0.65] · 0.50 [0.43, 0.58] |
| FTIR EC, published | Aethalometer IR BCc | 1.13 [1.05, 1.23] · 1.09 [1.01, 1.18] · 1.22 [1.11, 1.38] |
| FTIR EC, PTFE-anchored VIBES | Aethalometer IR BCc | 1.12 [1.05, 1.22] · 1.09 [1.01, 1.17] · 1.22 [1.12, 1.37] |
| Aethalometer IR BCc | PM2.5 mass | 1.28 [1.18, 1.39] (FTIR) · 1.23 [1.13, 1.33] (HIPS) · 1.03 [0.89, 1.18] |

**HIPS rises as roughly the square root of every other measurement. FTIR EC and the aethalometer are
proportional to each other.** Two of the references are neither optical nor FTIR: gravimetric PM2.5 mass and
XRF potassium. They give the same answer as the aethalometer, so the result does not rest on trusting one
optical instrument over another. The bend is also there inside each season (HIPS against the aethalometer:
Bega 0.69, Belg 0.58, Kiremt 0.56), so it is not a between-season composition effect.

Three further checks:

1. **A straight line loses to every bending curve.** Fitting HIPS optical depth against the loading from each
   reference, a saturating curve beats a proportional one by ΔAIC of −229 to −409. The fitted ceiling at Addis
   is 1.6 to 2.1 depending on the reference. The value fitted on Beijing, Pasadena and Delhi with Addis held
   out (3 Nov, `run_intercept_fixes_20261103.py` stage `stress`) is 2.07 [1.75, 2.49]. Addis optical depth has
   a median of 0.96 and reaches 1.5, so most Addis filters sit on the bend.
2. **Linearising HIPS with the held-out constant removes the offset.** Nothing from Addis sets that constant.
   The published fit moves from 2.31x − 6.17 to **0.44x + 0.58** (intercept +1.96 to −0.55 across the
   constant's interval), and linearised HIPS becomes proportional to the aethalometer (exponent 1.02).
3. **Random error, with no reference** (log-space triple collocation, n 172): FTIR EC 16 % [9, 23],
   aethalometer 13 % [8, 19], HIPS 21 % [17, 24] on a common scale. Published FTIR EC is about as precise as
   the aethalometer at Addis.

The network shows the same thing where it can be tested. With PM2.5 as the instrument (no aethalometer
needed), Addis has the highest median optical depth of any SPARTAN site (0.95) and the lowest exponent of HIPS
against FTIR EC (0.46 [0.43, 0.50]); Dhaka is next on both (0.73; 0.69 [0.55, 0.88]). Light sites with tight
intervals are near 1 (Pasadena 1.04, Taipei 1.06, Beijing 0.93). Two results need a second look before anyone
quotes them: the two Chilean sites both give 0.56 [0.53, 0.59] at moderate optical depth, and several dust or
sulfate sites have intervals too wide to read.

### What was already known, and what is new

The 3–4 Nov work found the pieces: HIPS against the aethalometer has the +2.84 intercept and a log-log slope
of 0.44, a saturation curve fitted elsewhere predicts Addis, and the Experimental page says the pattern "is
consistent with HIPS range compression". Its volume test did not confirm the mechanism, so the summary and the
status page kept the question open. What is new here:

- two references that are not optical (PM2.5 mass, potassium), which remove the "one optical instrument
  against another" objection;
- exponents that are free of the noise-in-x bias, with intervals;
- the three-way error decomposition, which had been proposed (`research/ftir_ec_phase3/DATA_OPPORTUNITY_AUDIT_2026-08-24.md`)
  and not run;
- the season by lot table and the level bracket below;
- the same test across the network.

## 2. What this changes in earlier conclusions

Every result that used HIPS / MAC 10 as the Addis yardstick needs re-reading, because a compressed x axis
turns a proportional FTIR response into a steep slope and a large negative intercept.

| Earlier reading | What the data say with a linear comparator |
|---|---|
| "The problem is calibration transfer, not HIPS" (status page, item 1) | The −6 µg/m³ intercept is on the HIPS side. A level question about FTIR remains (section 3). |
| "Baseline correction is the largest single improvement" (intercept −6.2 → −1.4) | The spline calibration reads 0.25 of the aethalometer (slope 0.21, R² 0.74) against 0.65 (slope 0.65, R² 0.87) for the published product. It closed the HIPS intercept by giving up most of the response, which matches the HIPS compression rather than fixing FTIR. |
| "The seasons fail differently"; season-specific calibrations | Kiremt has the highest concentrations (aethalometer median 9.5 against 5.1 µg/m³ in Bega), so compressed HIPS reads relatively low there and high in Bega. That alone produces a seasonal sign change in FTIR − HIPS. |
| A MAC of 28 to 32 would be needed in Dry and Belg | Those are slopes against compressed HIPS. Against linearised HIPS the published product needs about 20 m²/g at every loading (17 to 25 across the constant's interval). |
| Organic deficit: partial correlation with HIPS EC held fixed | Holding a compressed variable fixed leaves real EC variation in the residual. Redo with the aethalometer held fixed before using it as evidence. |
| Addis OC/EC 0.55 to 0.87, EC 19 to 24 % of PM2.5 | Both used HIPS / 10 as EC. With FTIR EC, OC/EC is 1.34 (5th to 95th percentile 1.12 to 1.61) and EC is 22 % of PM2.5. |

**Seasons, checked against the aethalometer.** Median FTIR EC ÷ aethalometer IR BCc:

| Season | Lot | n | Published | PTFE-anchored VIBES | HIPS / 10 |
|---|---|---:|---:|---:|---:|
| Bega | 248 | 18 | 0.53 | 0.49 | 0.85 |
| Belg | 248 | 15 | 0.64 | 0.66 | 0.79 |
| Bega | 251 | 33 | 0.66 | 0.52 | 0.76 |
| Belg | 251 | 48 | 0.66 | 0.53 | 0.70 |
| Kiremt | 251 | 58 | 0.67 | 0.57 | 0.54 |

Within lot 251 the published product has **no season effect** (0.66, 0.66, 0.67). The low Bega value comes
from the 18 lot-248 filters of December 2022 to early 2023. That is a lot or early-record question (filter
blank, or the aethalometer's first months), not a seasonal chemistry result. HIPS / 10 falls from 0.85 to 0.54
across the same rows, which is the compression again.

## 3. The correction the existing data support

### FTIR EC

1. **No intercept correction.** Against three independent linear measurements FTIR EC has none to correct.
2. **No seasonal correction.** Not for the published product within lot 251. Flag the lot-248 Bega filters.
3. **A level range, stated as uncertainty and not applied silently.** The evidence on published ÷ true:

| Anchor | Published ÷ true | Limits |
|---|---|---|
| Held-out IMPROVE sites, TOR ≥ 15 µg and TOR OC/EC < 2.5 (raw spectra, k 18; FTIR ÷ TOR) | 0.75 (0.65 at OC/EC < 1.8, n 25) | Calibration set, not Addis. Addis FTIR OC/EC is lower than nearly all of it. |
| Aethalometer, scaled by Pasadena's or Delhi's published FTIR ÷ aethalometer | 0.87, 0.83 | Assumes the MA350 units share a calibration. Beijing shows they need not. |
| Linearised HIPS at the Adama MAC (Fabs ÷ TOR 19 m²/g after the same linearisation) | 0.93 (0.77 to 1.09) | Five Adama filters, another site and sampler. |
| PM2.5 mass closure | ≥ 0.67 | Published EC reconstructs 89 % of gravimetric mass; × 1.5 reaches 100 %, × 1.9 gives 108 %. FTIR OM is below FTIR OC at Addis (median OM/OC 0.91), so OM is probably low and the real room for EC is smaller. |

The source-domain bias is monotone in composition. On held-out IMPROVE sites at TOR ≥ 15 µg per filter, median
FTIR ÷ TOR is 0.99 above OC/EC 4, 0.83 at 2.5 to 4, 0.76 at 1.5 to 2.5 and 0.64 below 1.5 (n 10). Addis sits
below 1.5. The OC/EC-balanced weighting raises the last bin to 0.79.

**Working statement: Addis FTIR EC is the published value × 1.0 to 1.5, most likely about × 1.2.** The four
anchors are soft and not independent of the assumptions listed, but they agree in direction and none supports
a factor near 2. Mass closure rules out reading the aethalometer or linearised HIPS at MAC 10 as EC: that
would make EC 33 to 39 % of PM2.5 and over-fill the mass.

### HIPS optical BC

Fabs / 10 should not be used as the Addis yardstick, or as the Addis BC record, without linearising. The
per-filter table from the workflow (`addis_per_filter.csv`, git-ignored output) carries HIPS optical depth and
Fabs linearised with the held-out constant and both ends of its interval. Two cautions:

- the correction is large (median × 1.87; × 1.63 to × 2.21 across the interval) and grows without bound near the
  ceiling, so it multiplies noise on the heaviest filters;
- the constant comes from three other sites, and the curve "moves a lot with which sites go in" (3 Nov note).

### What the incoming Addis TOR filters can and cannot settle

- The proposed pass band is 0.8 to 1.25 for median FTIR ÷ TOR ([open-items.md](open-items.md)). The existing
  evidence predicts about 0.67 to 1.0, **straddling the lower edge**, so ten filters will most likely land on
  "inconclusive". Decide before they arrive whether the deliverable is pass or fail, or an estimated level
  factor with an interval. The second is what this note says the data need.
- Ask for **HIPS on the same TOR filters, with some loaded to Addis SPARTAN optical depth (0.5 to 1.5)**. The
  city filters come from a lower-flow sampler and will sit low on the curve, where HIPS is still nearly linear.
  A few heavy pairs would give the saturation curve its first thermal anchor at Addis loadings and would fix
  the MAC and the level together.

## 4. Pathways: what is done, what is stale, what is still open

### Listed as open on the 9 Nov status page, but already run

`status-20261109` listed four analyses as waiting. `run_meeting_20261028.py` ran all four on 28 Oct, with six
more from the same list (results in `gallery/app/public/data/meetings/meeting-20261028.json`):

| Item | Result on 28 Oct |
|---|---|
| Calibration filters chosen by OC/EC nearest Addis | Belg 0.32x + 1.06 → 0.72x − 1.16; Dry slope 0.35 → 0.50 |
| Does the error track organic signal per µg EC | Partial ρ 0.46 (Dry), 0.63 (Belg), 0.09 (Kiremt), HIPS held fixed |
| Dust by season | Si/EC 0.25 to 0.28 in every season; dust does not explain Dry and Belg |
| Season-specific MAC | 28 (Dry), 32 (Belg), 10.7 (Kiremt) for slope 1 |
| Optical pyrolysis = 0 on dark IMPROVE filters | 2.9 % of the library; TOR is not high there |
| Urban IMPROVE sites only | 630 filters; about the same as the low-OC/EC set |
| Loading, VIP, speciation, boundary-layer height | all run |

All ten used HIPS / 10 on the x axis, so they are done but need the re-reading in section 2. The status page
text is corrected in this change.

### Open, and possible with data in hand

Ranked by how likely each is to change what is carried forward.

1. **Re-read the 28 Oct season, cohort, deficit and MAC results against the aethalometer and linearised HIPS**
   (172 and 233 filters). If section 2 holds, season-specific calibrations are not needed and the paper's
   seasonal section changes.
2. **The lot-248 early record.** Is the low FTIR ÷ aethalometer in Bega 2022–23 a filter-lot effect or the
   aethalometer's first months? PM2.5 mass and potassium can referee, as they did here.
3. **A thermal anchor for the HIPS curve inside IMPROVE.** Fit HIPS optical depth against TOR EC loading on the
   low-OC/EC urban sites with the instrumented estimator used here (TOR on x in loading bins is biased by
   selecting on the denominator). The Drive also holds the TOR analyser's own 635 nm laser record for IMPROVE
   (initial, final and minimum reflectance and transmittance on the quartz filter,
   `Improve/IMPROVE_laser-optics-635nm_2026-04-22.xlsx`), a second optical reading on the thermal filter that
   has not been used. This replaces the three-site aethalometer constant with one that has thermal EC under it.
4. **A composition-aware level factor from the calibration set.** Predict EC and OC together, learn the
   FTIR ÷ TOR bias as a function of predicted OC/EC on held-out sites (nested by site), and read it at Addis
   with an interval. This would replace the × 1.0 to 1.5 range with a source-domain estimate.
5. **Spectral augmentation, never run.** Only 10 IMPROVE filters at ≥ 15 µg have OC/EC below 1.5. Co-adding
   blank-corrected IMPROVE spectra can build Addis-like loadings and compositions with known TOR sums, and
   tests whether the under-read at low OC/EC is a coverage problem.
6. **The same exponent test across SPARTAN, done properly.** Section 1's network pass is a first look. With
   potassium as a second instrument and the Chilean result explained, this is the "network-wide, not only
   Addis" framing Ann asked about, and it needs no new data.
7. **PLS on the masked regions** (Ann, 10 Sep) and the raw model without 3600–4000 and below 1425 cm⁻¹ (28 Oct
   "next"). Still not run; lower priority now that the intercept is not an FTIR quantity.
8. **Blind three-way split of the Addis test set** (Ann, 25 Aug and 2 Sep). The 73-filter split exists and is
   unscored. It matters only if a calibration chosen on Addis is carried forward.
9. Smaller items from the meetings that have no result in the repository: absorbance-matched narrow cohorts
   (Ann, 10 Sep), PMF combined-factor cohorts (wood + charcoal; sea salt + polluted marine), the cross-plot
   coloured by sampled volume, PurpleAir or BAM context for Adama and Bishoftu.

### Blocked on something not in hand

Addis quartz + PTFE pairs (the level); AQRC's deployed EC model file; Emily Li's cookstove charcoal spectra;
SPARTAN's hole fraction for the White patterned-deposit correction; wet-season Bishoftu filters; FTIR
functional groups after October 2024.

## 5. For the supervisor meetings

- **The result is more publishable than "the calibration does not transfer".** With existing data: (a) HIPS
  compresses at Addis loadings, shown against four independent measurements; (b) FTIR EC is linear and as
  precise as the aethalometer; (c) the remaining FTIR question is one level factor, bracketed at 1.0 to 1.5;
  (d) the Addis-specific negative intercept that drove two years of calibration work is on the optical side.
- **This is a change of headline, so it should go to Ann before it goes into a draft.** It revises the poster's
  framing (baselining as the main win, seasonal calibrations as the next step).
- **It does not reopen the analysis freeze.** Items 1 and 2 above are re-reads of saved results. Items 3 to 6
  are candidates for the paper's "what would close it" section or for the next objective.
- **Decision still needed before the TOR filters arrive:** pass or fail against 0.8 to 1.25, or an estimated
  level factor; and whether to request HIPS and a few heavy filters in the same batch.

## Claims this note does not support

- That the aethalometer, or linearised HIPS, is the true Addis EC. They are optical; the MAC at Addis is not known.
- That the saturation mechanism is established. The curve is empirical, fitted on three other sites.
- That × 1.2 is a validated correction. It is the centre of a range from four soft anchors.
- That the 3–4 Nov volume test was wrong. It did not find the loading effect where volume is independent of
  concentration; that tension is unresolved and belongs in the paper.
- That the Chilean or other network exponents are findings. They are a first pass.
