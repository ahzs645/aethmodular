# Which measurement carries the Addis offset, and what the existing data can correct

**10 November 2026, revised the same day to remove the aethalometer.** The 9 Nov status page left one question
open: "Which measurement carries the offset stays open". This note answers it with data already in hand, says
what that does to earlier conclusions, gives the correction the existing data support, and lists the pathways
still open.

**The aethalometer is not used anywhere in this account.** The Addis MA350 never had its flow split fixed and is
not a reference there. The first version of this note leaned on it, and on a saturation ceiling that the 3 Nov
work had fitted on aethalometers at three other sites. Both are gone. The argument now rests on thermal EC from
IMPROVE and on two Addis measurements that are neither optical nor FTIR. It is a weaker argument about FTIR's
linearity than the first version made, and section 2 says where.

Everything here is exploratory and uses existing data only. No thermal (TOR) EC has been measured on an Addis
filter, so **the absolute Addis EC level is still a range, not a number**. The rules in
[current-research-summary.md](current-research-summary.md) and [AGENTS.md](../AGENTS.md) still apply.

- Code: `research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py`, helpers in
  `research/ftir_hips_chem/scripts/reference_closure.py`, tests in `tests/test_reference_closure.py`.
- Numbers: every figure below is in [addis-reference-closure-2026-11-10.json](addis-reference-closure-2026-11-10.json)
  (`"aethalometer_used": false`), written by the workflow's `--summary`.
- Inputs: the IMPROVE stage reads the Drive tables of TOR EC and HIPS (`FTIR/local_db/tables`). Its stratum
  table is saved in the JSON, so every other stage, and the gallery page, run from a plain checkout.
- Population: the 233 counted Addis filters of the 7 Nov closeout (189 with a SPARTAN published FTIR EC, 188
  with PM2.5 mass), and 347,936 IMPROVE samples from 210 sites, 2005–2025.
- Gallery: Meeting tab, Nov 10.

## 1. The finding: most of the offset is HIPS compression

Against HIPS optical BC (Fabs / MAC 10), SPARTAN's published FTIR EC reads 2.31x − 6.17 µg/m³ (Deming, n 189).
A straight-line fit of one instrument on another cannot say which of the two is not proportional.

### IMPROVE thermal EC says how HIPS behaves on dark filters

IMPROVE has TOR EC (quartz) and HIPS (PTFE) on the same day for 347,936 samples. Each sample is grouped by the
*previous* sample's EC loading at the same site. That ordering has nothing to do with this sample's measurement
error in either instrument, so the group medians are free of the bias that binning on TOR or on HIPS causes.

| IMPROVE sites | Samples | Sites | Fabs ÷ TOR EC on light filters (m²/g) | Saturation ceiling (optical depth) |
|---|---:|---:|---:|---:|
| All | 347,936 | 210 | 13.1 [12.9, 13.5] | 2.26 [1.82, 2.70] |
| Sites with median OC/EC < 4 | 32,898 | 32 | 15.1 [13.5, 16.5] | 2.25 [1.74, 3.04] |
| Six urban sites | 11,020 | 6 | 16.4 [14.9, 18.4] | 2.23 [1.81, 2.96] |

Per unit of thermal EC, HIPS reads 13.1 m²/g on the lightest IMPROVE filters and 8.4 at optical depth 0.94. A
saturating curve through the group medians (optical depth = aL / (1 + aL / ceiling)) has a **ceiling of 2.26
[1.82, 2.70]**, and the three choices of sites agree. Intervals are 90 %, resampling whole sites. Addis optical
depth has a median of 0.96 and reaches 1.51: most Addis filters sit on the bend. No FTIR and no aethalometer
enter this. (The ceiling the 3 Nov work fitted on aethalometers was 2.07 [1.75, 2.49]. It is not used, but the
agreement is worth knowing.)

### At Addis the two are far from proportional, and the IMPROVE curve predicts most of it

The response exponent b in Y ∝ X^b is estimated with a third measurement as the instrument, so random error in
X does not flatten it the way it flattens a log-log regression. Intervals are 95 % bootstrap over filters.

| Y | X | Exponent, instrument PM2.5 mass | Exponent, instrument potassium |
|---|---|---|---|
| HIPS Fabs | FTIR EC, published | 0.46 [0.43, 0.50] | 0.41 [0.36, 0.46] |
| HIPS Fabs | FTIR EC, PTFE-anchored VIBES | 0.46 [0.43, 0.50] | 0.41 [0.36, 0.45] |
| HIPS Fabs | PM2.5 mass | 0.60 [0.55, 0.65] (instrument FTIR EC) | 0.50 [0.43, 0.58] |
| FTIR EC, published | PM2.5 mass | 1.29 [1.19, 1.40] (instrument HIPS) | 1.23 [1.09, 1.38] |

With no instrument at all, the exponent of HIPS against published FTIR EC is bounded to 0.39–0.54 (the log-log
slope and the reciprocal of the reverse slope). **If FTIR EC were exactly proportional to EC, the IMPROVE curve
predicts 0.56 [0.46, 0.64] at Addis optical depths.** So HIPS compression, measured against thermal EC in another
network, accounts for most of the departure from proportionality. The bend is inside each season too (0.57,
0.53, 0.59 for Bega, Belg, Kiremt).

**Linearising HIPS with the IMPROVE ceiling takes the intercept to zero.** The published fit moves from
2.31x − 6.17 to **0.55x + 0.01**. Across the ceiling's interval the intercept runs from +1.59 to −0.97 µg/m³,
so "zero" means "not distinguishable from zero", against −6.17 before.

**The same curve predicts the pattern across SPARTAN.** With PM2.5 mass as the instrument, the exponent of HIPS
against FTIR EC falls with a site's optical depth about as the IMPROVE curve says: Pasadena 1.04 (predicted
0.96), Taipei 1.06 (0.93), Beijing 0.93 (0.87), Delhi 0.80 (0.78), Dhaka 0.69 (0.62), Addis 0.46 (0.57). Addis
has the highest median optical depth of any SPARTAN site. The two Chilean sites give 0.56 against a predicted
0.78 to 0.88 and are unexplained; 11 of 23 sites have intervals too wide to read.

## 2. What is not settled without the aethalometer

The first version of this note said FTIR EC was proportional, with no seasonal effect. Those two statements
rested on the aethalometer and are withdrawn. What the remaining data say:

- **A small leftover.** Against linearised HIPS, FTIR EC is close to proportional but not exactly: linearised
  HIPS rises as FTIR EC to the power 0.85 [0.78, 0.92] at the central ceiling (1.09 at the low end, 0.74 at the
  high end). Addis's measured exponent (0.41 to 0.46) sits at the bottom of what IMPROVE predicts (0.46 to
  0.64). Either the ceiling at Addis is at the low end of its interval, or FTIR EC rises somewhat faster than
  EC. FTIR EC also rises as PM2.5 to the power 1.2 to 1.3, which could be the same thing or a real rise in EC's
  share on polluted days. The existing data cannot separate these.
- **A smaller season or loading effect remains.** Kiremt has the heaviest filters, so the two cannot be told
  apart at Addis.
- **Lot 248 reads low, and that is not a HIPS effect.** See the table.

Median ratios of the published FTIR EC, by season and filter lot:

| Season | Lot | n | HIPS optical depth | FTIR ÷ HIPS / 10 | FTIR ÷ linearised HIPS / 10 | FTIR ÷ PM2.5 mass |
|---|---|---:|---:|---:|---:|---:|
| Bega | 248 | 18 | 0.81 | 0.63 | 0.40 | 0.16 |
| Belg | 248 | 16 | 0.91 | 0.78 | 0.47 | 0.17 |
| Bega | 251 | 37 | 0.89 | 0.84 | 0.51 | 0.20 |
| Belg | 251 | 53 | 0.96 | 0.98 | 0.53 | 0.24 |
| Kiremt | 251 | 65 | 1.14 | 1.29 | 0.60 | 0.24 |

Within lot 251 the swing from Bega to Kiremt is 0.84 to 1.29 against HIPS as measured (54 %) and 0.51 to 0.60
against linearised HIPS (18 %). Lot 248 reads about 15 to 20 % below lot 251 in the same season, against
linearised HIPS and against gravimetric mass alike.

## 3. What this changes in earlier conclusions

Every result that used HIPS / MAC 10 as the Addis yardstick needs re-reading, because a compressed x axis turns
a near-proportional response into a steep slope and a large negative intercept.

| Earlier reading | With HIPS linearised |
|---|---|
| "The problem is calibration transfer, not HIPS" (status page, item 1) | Most of the −6 µg/m³ intercept is on the HIPS side. A level question about FTIR remains (section 4), with a smaller season or loading effect and a lot offset. |
| "Baseline correction is the largest single improvement" (intercept −6.2 → −1.4) | The spline calibration reads Addis at 0.38 of HIPS / 10 against 0.99 for the published product. It lowers the whole response, which narrows the gap to a compressed HIPS. Judge baselines on IMPROVE thermal EC, not on the Addis HIPS intercept. |
| "The seasons fail differently"; season-specific calibrations | Most of the seasonal swing was HIPS compression: Kiremt's filters are the heaviest. 18 % remains within lot 251, against 54 %. |
| A MAC of 28 to 32 would be needed in Dry and Belg | Those are slopes against compressed HIPS. Against linearised HIPS the published product needs about 19 m²/g (16 to 24 across the ceiling's interval). |
| Organic deficit: partial correlation with HIPS EC held fixed | Holding a compressed variable fixed leaves real EC variation in the residual. Redo with linearised HIPS held fixed. |
| Addis OC/EC 0.55 to 0.87, EC 19 to 24 % of PM2.5 | Both used HIPS / 10 as EC. With FTIR EC, OC/EC is 1.34 (5th to 95th percentile 1.12 to 1.61) and EC is 22 % of PM2.5. |

## 4. The correction the existing data support

### FTIR EC

1. **No intercept correction.** Once HIPS is linearised there is none to correct.
2. **No seasonal correction yet.** The effect that remains is 18 %, tangled with loading, and not attributable.
   Flag the lot-248 filters.
3. **A level range, stated as uncertainty and not applied silently.** After linearising, FTIR EC is 0.55 of HIPS BC
   at MAC 10. That slope is the FTIR level times 10 ÷ the real MAC, so the two sides have to be read together.

| Anchor | Published ÷ true | Limits |
|---|---|---|
| Held-out IMPROVE, six urban IMPROVE sites: raw-spectra FTIR ÷ TOR 0.91 [0.88, 0.94]; the published product reads 1.11× that calibration at Addis | 1.01 | Assumes the bias seen at TOR OC/EC near 3 holds at Addis, where OC/EC is near 1. It probably grows (section 5). |
| The raw calibration's offset, EC gain (0.78) and OC response from IMPROVE site means, read at Addis loading and OC/EC 1: 0.82 [0.75, 0.87] | 0.91 | A linear extrapolation beyond IMPROVE's composition (section 5). |
| Linearised HIPS at the IMPROVE MAC on light filters: 13.1 (all sites), 15.1 (low OC/EC sites), 16.4 (urban) m²/g | 0.70, 0.81, 0.88 | Assumes Addis soot absorbs like IMPROVE's per unit of thermal EC. |
| Linearised HIPS at the Adama MAC (Fabs ÷ TOR 18.9 m²/g after the same linearisation) | 1.01 (0.81 to 1.15) | Five Adama filters, another site and sampler. |
| PM2.5 mass closure | ≥ 0.67 | Published EC reconstructs 89 % of gravimetric mass; × 1.5 reaches 100 %, × 1.9 gives 108 %. FTIR OM is below FTIR OC at Addis (median OM/OC 0.91), so OM is probably low and the real room for EC is smaller. |

The pieces are consistent: an FTIR level near 0.9 to 1.0 with a MAC of 16 to 19 m²/g gives the 0.55 slope.

**Working statement: Addis FTIR EC is the published value × 1.0 to 1.4, most likely about × 1.1.** An earlier
version of this note said × 1.0 to 1.5, most likely × 1.2, on the strength of "FTIR ÷ TOR is 0.65 to 0.75 on heavy,
low-OC/EC IMPROVE filters". That figure selected on TOR's own error and is withdrawn (section 5).

### HIPS optical BC

Fabs / 10 should not be used as the Addis yardstick, or as the Addis BC record, without linearising. The
per-filter table from the workflow (`addis_per_filter.csv`, git-ignored output) carries HIPS optical depth and
Fabs linearised with the IMPROVE ceiling and both ends of its interval. Two cautions:

- the correction is large (median × 1.74; × 1.55 to × 2.12 across the interval) and grows without bound near the
  ceiling, so it multiplies noise on the heaviest filters;
- the ceiling comes from IMPROVE, whose heavy filters include smoke. The low-OC/EC and urban sites give the same
  ceiling, which is the main check available.

### What the incoming Addis TOR filters can and cannot settle

- The proposed pass band is 0.8 to 1.25 for median FTIR ÷ TOR ([open-items.md](open-items.md)). The most likely
  answer (about 0.9 to 1.0) is now inside it, with the lower edge still inside the plausible range (0.7 to
  1.15). Ten pairs fix a mean bias only to about ±27 % (3 Nov simulation), which is wider than the question that
  is left. Decide before they arrive whether the deliverable is pass or fail, or an estimated level factor with
  an interval.
- Ask for **HIPS on the same TOR filters, with some loaded to Addis SPARTAN optical depth (0.5 to 1.5)**. The
  city filters come from a lower-flow sampler and will sit low on the curve, where HIPS is still nearly linear.
  A few heavy pairs would test the IMPROVE ceiling at Addis, separate the two readings of the leftover in
  section 2, and fix the MAC and the level together.

## 5. Across every site and dataset: what the next steps showed

Run the same day, on everything in hand: the IMPROVE record by period, all 23 SPARTAN sites with at least 40
complete filters, the twelve Addis calibrations the closeout saved per filter, and the held-out IMPROVE
predictions. All of it is on the gallery page (Meeting tab, Nov 10) with pickers for site and comparator.

### The ceiling is the main uncertainty, and it differs by period

| IMPROVE samples | n | Ceiling | Addis exponent it predicts | Addis fit after linearising |
|---|---:|---:|---:|---|
| All years, all sites | 347,936 | 2.26 [1.82, 2.70] | 0.56 | 0.55x + 0.01 |
| 2005–2014 | 174,602 | 2.53 [2.09, 2.94] | 0.61 | 0.68x − 0.63 |
| 2015–2025 | 173,334 | 1.71 [1.34, 2.36] | 0.42 | 0.20x + 2.25 |
| Optical pyrolysis > 0 | 340,648 | 2.06 [1.68, 2.37] | 0.52 | 0.44x + 0.60 |

The later period, which contains the Addis record, has a lower ceiling. It predicts the Addis exponent that is
measured (0.41 to 0.46) and leaves a positive intercept. So the ceiling decides whether the intercept left
after linearising is slightly negative, zero or positive. It does not decide whether the −6 goes away: every
row removes it. Why IMPROVE's HIPS response changed between the two periods (instrument history, filter
supply) is now the most useful thing to find out, and it is a question for the HIPS operators at Davis.

### The FTIR side, tested against thermal EC

A raw-spectra calibration (18 components) was refitted for TOR EC and TOR OC with whole IMPROVE sites held out:
12,953 filters, 158 sites. It reproduces the closeout (FTIR ÷ TOR 0.78 on the heavy low-OC/EC subset) and the
published Addis values (r 0.99; the published product reads 1.11× the refit).

**The level bias is real and about a tenth, not a quarter.** The closeout's "heavy, low-OC/EC" subset is chosen
by each filter's own TOR EC and TOR OC/EC. That collects filters whose TOR EC happened to read high, so
FTIR ÷ TOR on it comes out too low. Two groupings cannot do that:

| How the IMPROVE filters are chosen | n | Sites | FTIR ÷ TOR | TOR OC/EC there |
|---|---:|---:|---:|---:|
| By the filter's own TOR: EC ≥ 15 µg, OC/EC < 2.5 | 175 | 30 | 0.78 [0.72, 0.84] | 2.2 |
| By the previous sample at the site: EC ≥ 15 µg, OC/EC < 2.5 | 155 | 30 | 0.91 [0.87, 0.94] | 2.9 |
| By site: six urban IMPROVE sites | 645 | 6 | 0.91 [0.88, 0.94] | 3.2 |
| By site: median OC/EC 2.5–3.5 | 763 | 10 | 0.93 [0.89, 1.02] | 3.1 |
| By site: median OC/EC 3.5–5 | 3,379 | 35 | 0.96 [0.93, 1.00] | 4.3 |
| By site: median OC/EC above 5 | 8,799 | 112 | 1.16 [1.12, 1.20] | 6.6 |

The same artefact runs the other way if filters are chosen by FTIR's own OC/EC (FTIR ÷ TOR 1.54 below 1.5), so
a level factor cannot be learned by conditioning on either instrument's own ratio.

**The saved calibrations differ little once the filters are chosen fairly.** On the urban sites: raw spectra
0.88, spline baseline 0.92, production VIBES 0.87, PTFE-anchored VIBES 0.88, OC/EC-balanced weights 0.90,
deflated VIBES 0.82. On the selected subset the same calibrations ran from 0.65 to 0.85, which made the
composition-aware variants look like a larger fix than they are. The per-filter EPO predictions are not in the
closeout export, so EPO's 0.90 is not re-scored here; it was measured on the selected subset.

**IMPROVE cannot say what the bias is at Addis.** With OC and EC calibrations fitted the same way on both
networks, Addis FTIR OC/EC has a median of 0.98 (Bega 0.97, Belg 0.73, Kiremt 1.15). IMPROVE's 5th percentile
is 1.94 and its urban median 3.65: 99 % of Addis filters are below IMPROVE's 5th percentile. The bias grows as
OC/EC falls across IMPROVE sites (1.16, 0.96, 0.93, 0.91), so it is probably larger at Addis, by an amount no
IMPROVE filter can show. This is the honest limit of the FTIR side with existing data.

IMPROVE is a mostly rural and remote network. "Urban" above means its six city sites (Atlanta, Birmingham,
Fresno, Phoenix, Pittsburgh, Seattle), which are its low-OC/EC end, near 3.

**A correction cannot be learned by binning on OC/EC.** In the lowest bin the same IMPROVE filters read 0.62 of
thermal EC when binned by TOR's own OC/EC and 1.54 when binned by FTIR's own; binned by the previous sample at
the site they read 0.86 to 1.01. Each ratio contains the error of the EC it is built from, so a low bin collects
that error.

### The two next steps, run

**EPO, re-scored fairly.** No saved EPO predictions exist in the repository or the Drive folder, so EPO was
refitted on the raw spectra with `scripts/epo_validation.py`, the projection fitted inside each training fold
(silicon stands in for potassium, which is not in this XRF pull; this is not the closeout's anchored + EPO
pipeline). On the six urban IMPROVE sites it reads 0.93 [0.88, 0.98] of thermal EC against 0.91 [0.88, 0.93]
for the plain calibration, and 0.94 against 0.91 with previous-sample grouping. A small real gain, at the cost
of a higher overall median error (1.42 against 1.27 µg per filter; RMSE improves, 5.88 against 6.56).

**What the calibration responds to.** Across IMPROVE site means (which average out TOR's per-filter error):

| Calibration | Offset (µg per filter) | Gain per µg of thermal EC | Per µg of OC | Level at Addis loading, OC/EC 1 |
|---|---:|---:|---:|---:|
| Raw spectra | 0.88 [0.48, 1.23] | 0.78 [0.69, 0.84] | 0.015 [0.00, 0.04] | 0.82 [0.75, 0.87] |
| Raw spectra + EPO | 0.97 [0.42, 1.40] | 0.89 [0.76, 0.99] | −0.008 [−0.02, 0.01] | 0.91 [0.81, 1.00] |

The calibration under-responds to EC and makes up the difference with a fixed offset. That works at IMPROVE's
loadings (4 to 15 µg) and not at Addis's (about 30 µg), where the offset no longer matters and the level falls
toward the gain. Blank filters show an offset too, a larger one: the calibration reads 2.2 µg of EC on 338
IMPROVE field blanks and 2.7 µg on 37 Addis field blanks (0.38 µg/m³ at the Addis volume). EPO raises the gain
and removes the small OC response. The last column is a linear extrapolation, not a measurement, and it is the
best existing-data estimate of the FTIR level at Addis: about 0.8 for the raw calibration, so about 0.9 for the
published product, which reads 1.11× higher.

**Spectral augmentation does not do what I proposed.** Co-adding three urban IMPROVE filters on one blank gives
53 µg of EC, an Addis-like loading, and FTIR reads 0.83 of it against 0.89 for the single filters. But for a
linear calibration that number is exactly the sum of the three predictions minus twice the blank reading
(checked: the two agree to rounding). Co-adding shows the offset and nothing else, and a sum can never have a
lower OC/EC than its parts. It cannot test Addis composition. Withdrawn as a next step.

### Can the calibration be fixed from IMPROVE alone

`workflows/run_calibration_gain_20261110.py` (helpers in `scripts/calibration_gain.py`, numbers in
[calibration-gain-2026-11-10.json](calibration-gain-2026-11-10.json)). Raw spectra, 18 components, whole IMPROVE
sites held out, 12,953 filters, 338 field blanks. Heavy and light filters are chosen by the previous sample at
the site (TOR EC ≥ 15 µg, and below 3 µg), never by their own TOR.

| Calibration | Gain | Blank reads (µg) | Heavy | Urban | Light | Light error (µg) | Overall median error (µg) | RMSE (µg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| As trained | 0.78 [0.70, 0.84] | 2.2 | 0.95 | 0.91 | 1.27 | 1.16 | 1.27 | 6.56 |
| As trained, then offset removed and divided by the gain | 0.96 [0.85, 1.04] | 1.3 | 1.03 | 1.02 | 0.98 | 1.08 | 1.29 | 7.72 |
| Weighted toward heavy filters | 0.78 [0.62, 0.87] | 4.3 | 1.03 | 0.99 | 1.77 | 2.11 | 2.01 | 8.55 |
| Blanks added at zero | 0.81 [0.73, 0.88] | 1.3 | 0.94 | 0.91 | 1.13 | 0.96 | 1.15 | 6.34 |
| Blanks at zero, then recalibrated | 0.97 [0.86, 1.07] | 0.7 | 1.02 | 1.02 | 0.93 | 1.01 | 1.24 | 7.32 |

**The recalibration works on sites it has not seen.** With the offset and gain fitted on other sites, the urban
sites go from 0.91 to 1.02 and heavy filters from 0.95 to 1.03, and light filters are not harmed (level 1.27 to
0.98, median error 1.16 to 1.08 µg). The cost is in the largest errors, which are scaled up with everything
else (RMSE 6.56 to 7.72).

**Why the gain is low.** It rises with the number of components and levels off below 1: 0.33 at 6 components,
0.64 at 12, 0.78 at 18, 0.81 at 25. Weighting the training toward heavy filters does not move it; it only
raises the offset, so a blank reads 4.3 µg and the light-filter error nearly doubles. Adding blanks at zero is a
small free improvement at the low end and leaves the gain at 0.81. So the gain is not a training choice that
can be tuned away: a linear calibration on these spectra under-responds to EC, and a correction after the fact
is the available fix.

**No baseline changes the gain.** Run on the decision bundle and the closeout's continuum arrays (12,386
filters, 157 sites, the same folds for each; `run_calibration_gain_20261110.py --bundle`):

| Baseline | Gain | Urban, as trained | Urban, recalibrated | Light error (µg) | Median error (µg) | RMSE (µg) |
|---|---:|---:|---:|---:|---:|---:|
| Raw spectra | 0.77 [0.69, 0.85] | 0.90 | 0.97 | 1.00 | 1.14 | 3.73 |
| Spline baseline | 0.75 [0.58, 0.95] | 0.94 | 1.01 | 0.91 | 1.08 | 4.46 |
| Production VIBES | 0.73 [0.64, 0.82] | 0.88 | 0.96 | 0.93 | 1.14 | 4.18 |
| PTFE-anchored VIBES | 0.76 [0.68, 0.86] | 0.88 | 0.95 | 0.93 | 1.08 | 3.78 |
| Deflated VIBES | 0.60 [0.51, 0.71] | 0.83 | 0.94 | 1.14 | 1.34 | 4.07 |

Raw, spline, production VIBES and PTFE-anchored VIBES respond to thermal EC alike (0.73 to 0.77) and the same
recalibration works on each. Deflated VIBES, which removes smooth directions, lowers the gain. The baselines
differ a little in median error and not in what limits Addis. The spline baseline looked like the largest single
improvement only because HIPS was the yardstick: its Addis intercept against HIPS / 10 is −1.4 against −4.7 for
raw spectra, and against linearised HIPS the two are +0.26 and +0.18.

**Components and cross-validation.** On raw spectra the whole-site RMSE is lowest at 17 components (3.73 µg at 18)
and the interleaved one keeps falling to 22 (3.57 at 18). On this broad set the two schemes differ by a few per
cent. The step-by-step account of every choice is the gallery page "Nov 10 · the calibration, step by step".

**At Addis** the same recalibration raises the raw calibration by × 1.25 at the median filter, which is × 1.13 on
the published product. It is fitted and checked on IMPROVE: it corrects the loading part of the bias and says
nothing about whether Addis composition adds to it.

### The MAC and the FTIR level have to be read together

After linearising HIPS, the published FTIR EC agrees with HIPS only if the MAC is 18.7 m²/g (16.1 to 24.2
across the ceiling's interval). At the conventional MAC of 10 FTIR would be reading 0.53 of true EC. Every MAC
measured against thermal EC is above 10: 13.1 on light IMPROVE filters, 16.4 at IMPROVE's six urban sites, 15.9
on the five Adama filters (18.9 after the same linearisation). At the urban IMPROVE value FTIR reads 0.88 of
true EC; at the Adama value, 1.01. The FTIR-side extrapolation above (0.91) sits between them.

### What the Adama TOR filters do and do not help with

The five Adama pairs (July 2024) are the only thermal EC in Ethiopia. They carry 3.5 to 9.7 µg of EC per filter
at TOR OC/EC 4.6 to 7.2 and optical depth 0.15 to 0.44. Addis SPARTAN filters carry about 34 µg at an OC/EC
near 1 and optical depth near 1. Adama is the opposite end from Addis.

- **They do not test the FTIR level at Addis.** At Adama's light loading the calibrations disagree by more than
  a factor of two (median FTIR ÷ TOR on the three clean pairs: production VIBES 0.96, EPO 1.02, PTFE-anchored
  VIBES 1.31, raw spectra 2.23). That is a baseline question, which is not what limits Addis.
- **They do help with the MAC.** Per unit of thermal EC, HIPS reads 14.8 to 16.8 m²/g at Adama, above IMPROVE's
  curve at the same optical depth (10 to 12.5 across all sites; about 14 at the urban sites). Ethiopian aerosol
  absorbs more per unit EC than MAC 10 assumes. That is the other half of the 0.55 slope: with a MAC near 19
  after linearising, the published FTIR level comes out near 1.0.
- **They cannot test HIPS saturation**, because they sit low on the curve.

So Adama narrows the optical side and leaves the FTIR side at Addis composition untested. Heavier pairs from
Addis itself are what would test it.

### On held-out IMPROVE sites FTIR EC is proportional to thermal EC

The exponent of the held-out FTIR prediction against TOR EC (instrument: HIPS) is 0.94 [0.90, 0.98] over all
1,817 filters and 1.04 [0.96, 1.12] at TOR OC/EC below 2.5, with the PTFE-anchored and spline calibrations
within 0.1 of that. FTIR does not rise faster than EC where EC is known. With the period result above, the
small leftover in section 2 points at the HIPS ceiling and not at FTIR. This is the calibration set: it does
not show how FTIR behaves on Addis aerosol.

### Lot 248 is not a general filter-lot effect

IMPROVE filters of lot 248 read thermal EC no lower than lot 251 (median held-out FTIR ÷ TOR 0.89 and 0.89;
0.80 and 0.74 at low OC/EC). The low Addis lot-248 readings are therefore the SPARTAN lot-248 blanks or
handling, or a real difference in the December 2022 to early 2023 aerosol.

### Every SPARTAN site

The exponent of HIPS against FTIR EC falls with a site's optical depth about as the IMPROVE curve predicts,
with PM2.5 mass or potassium as the instrument. Linearising with the one network-wide ceiling:

| Site | Optical depth | Exponent (PM2.5 · potassium) | Predicted | Intercept as measured → linearised (µg/m³) |
|---|---:|---|---:|---|
| Addis Ababa | 0.95 | 0.46 · 0.36 | 0.57 | −6.22 → −0.18 |
| Dhaka | 0.73 | 0.69 · 0.74 | 0.62 | −1.93 → +2.44 |
| Delhi | 0.47 | 0.80 · 0.87 | 0.78 | −1.77 → −0.41 |
| Johannesburg | 0.48 | 0.88 · 1.04 | 0.78 | −0.18 → +0.42 |
| Pretoria | 0.43 | 0.90 · 1.02 | 0.80 | −0.07 → +0.45 |
| Bishoftu | 0.43 | 0.89 · n/a | 0.80 | −0.47 → −0.10 |
| Santiago | 0.41 | 0.56 · 0.57 | 0.78 | −1.33 → −0.35 |
| Beijing | 0.25 | 0.93 · 1.19 | 0.87 | −0.55 → −0.05 |
| Talca | 0.20 | 0.56 · 0.54 | 0.88 | −1.07 → −0.61 |
| Taipei | 0.16 | 1.06 · 0.99 | 0.92 | 0.00 → +0.07 |
| Pasadena | 0.10 | 1.04 · 0.69 | 0.96 | −0.02 → +0.03 |

Two things to take from it. **Addis stands alone**: the heaviest site, the least proportional, and the one
where linearising matters most. **One ceiling does not suit every site**: it over-corrects where HIPS and FTIR
EC were already near-proportional at moderate optical depth (the South African sites, Jaipur, Mexico City,
Dhaka), and the two Chilean sites are far less proportional than their optical depth predicts with either
instrument. HIPS response looks site-dependent, which is what a deposit-pattern or aerosol-type effect would
do. The IMPROVE ceiling is evidence about Addis, not a correction to apply across SPARTAN.

### Every saved Addis calibration, by season

Against HIPS as measured the twelve calibrations have intercepts from −10.3 to −1.0 µg/m³; against linearised
HIPS, from −1.0 to +0.6. Kiremt ÷ Bega in median FTIR ÷ HIPS falls from 1.63 to 1.28 for the published product
and from 1.47 to 1.18 for the spline baseline. The calibrations that use the whole spectrum keep more of the
seasonal spread than the ones that do not, which fits a composition-linked part in FTIR (more organic signal
in Kiremt). Season and loading still cannot be separated at Addis.

## 6. Pathways: what is done, what is stale, what is still open

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

All ten used HIPS / 10 on the x axis, so they are done but need the re-reading in section 3. The status page
text is corrected in this change.

### Open, and possible with data in hand

Ranked by how likely each is to change what is carried forward.

1. **Re-read the 28 Oct season, cohort, deficit and MAC results against linearised HIPS.** If section 1 holds,
   the season-specific calibrations were mostly chasing HIPS compression.
2. **Why the HIPS ceiling differs between 2005–2014 and 2015–2025 in IMPROVE, and which one applies to SPARTAN's
   filters.** Section 5 shows it decides the sign of what is left. Split the IMPROVE record by HIPS instrument
   or calibration period (the Davis operators know the dates), and by filter lot.
3. **Site-dependent HIPS response.** Why the South African sites, Jaipur and Mexico City stay near-proportional
   at optical depth 0.45, and the Chilean sites do not at 0.2 to 0.4. Aerosol type, deposit pattern and sampler
   are the candidates; the spartan_reference chemistry is in hand.
3a. **Lot 248 at Addis.** IMPROVE shows no lot effect (section 5), so it is the SPARTAN lot-248 blanks or the
   2022–23 aerosol. The lot's blanks can referee.
4. **The gain test on spline, VIBES and anchored spectra.** `run_calibration_gain_20261110.py --bundle` on the
   machine that holds the decision bundle. The closeout's saved predictions suggest the baseline changes the
   gain little; this would confirm it on every filter.
5. **Decide whether the IMPROVE recalibration is the stated Addis correction** (about × 1.1 on the published
   product, section 5). It works on unseen IMPROVE sites and is untested at Addis composition. A decision for
   Ann, not an analysis.
5a. **Done here (section 5):** EPO re-scored fairly; co-added spectra (uninformative for a linear calibration);
   the recalibration from offset and gain on held-out sites; weighting, blanks at zero and component count.
6. **Done here (section 5):** FTIR against thermal EC on held-out IMPROVE sites; the exponent and the
   linearised fit at every SPARTAN site with potassium as a second instrument; every saved calibration by season.
7. **PLS on the masked regions** (Ann, 10 Sep) and the raw model without 3600–4000 and below 1425 cm⁻¹ (28 Oct
   "next"). Still not run; lower priority now that the intercept is not an FTIR quantity.
8. **Blind three-way split of the Addis test set** (Ann, 25 Aug and 2 Sep). The 73-filter split exists and is
   unscored. It matters only if a calibration chosen on Addis is carried forward.
9. Smaller items from the meetings that have no result in the repository: absorbance-matched narrow cohorts
   (Ann, 10 Sep), PMF combined-factor cohorts (wood + charcoal; sea salt + polluted marine), the cross-plot
   coloured by sampled volume, PurpleAir or BAM context for Adama and Bishoftu.

Also unused so far: the TOR analyser's own 635 nm laser record for IMPROVE
(`Improve/IMPROVE_laser-optics-635nm_2026-04-22.xlsx`), a second optical reading on the thermal filter itself.

### Blocked on something not in hand

Addis quartz + PTFE pairs (the level); AQRC's deployed EC model file; Emily Li's cookstove charcoal spectra;
SPARTAN's hole fraction for the White patterned-deposit correction; wet-season Bishoftu filters; FTIR
functional groups after October 2024.

## 7. For the supervisor meetings

- **The result is more publishable than "the calibration does not transfer".** With existing data and no
  aethalometer: (a) IMPROVE's thermal EC shows HIPS compressing at the optical depths Addis filters reach;
  (b) that curve predicts most of the Addis disagreement and the pattern across SPARTAN; (c) linearising HIPS
  takes the −6 intercept to about zero; (d) the FTIR side is a level question: about a tenth low on the most
  Addis-like IMPROVE filters, probably more at Addis, bracketed at 1.0 to 1.4 overall.
- **It is a change of headline, so it should go to Ann before it goes into a draft.** It revises the poster's
  framing (baselining as the main win, seasonal calibrations as the next step).
- **It does not reopen the analysis freeze.** Items 1 and 2 above are re-reads of saved results.
- **Decision still needed before the TOR filters arrive:** pass or fail against 0.8 to 1.25, or an estimated
  level factor; and whether to request HIPS and a few heavy filters in the same batch.

## Claims this note does not support

- That FTIR EC is exactly proportional to EC at Addis, or free of a seasonal effect. Section 2.
- That linearised HIPS is the true Addis EC. It is optical; the MAC at Addis is not known.
- That the saturation mechanism is established. The curve is empirical, from another network.
- That × 1.1 is a validated correction. It is the centre of a range from soft anchors.
- That FTIR reads a quarter low on Addis-like IMPROVE filters. That figure selected on TOR's own error.
- That the 3–4 Nov volume test was wrong. It did not find the loading effect where volume is independent of
  concentration; that tension is unresolved and belongs in the paper.
- That the Chilean or other network exponents are findings. They are a first pass.
- Anything about the aethalometer. `--with-aethalometer` exists as a side check and is not part of this account.
