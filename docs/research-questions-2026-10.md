# Research questions, October 2026

Questions raised by the 15 Oct 2026 practice talk and by the data work around it
(SPARTAN refresh, HIPS and FTIR functional groups at every site, mass per filter).

For each question: why it matters, what we already have, and the next step. Decisions that
block the code base stay in [open-items.md](open-items.md). The settled claims are in
[current-research-summary.md](current-research-summary.md).

## From the practice talk

### EC fraction and OC/EC as a way to choose the calibration set
- **Raised by:** Navid and Ann. Addis EC is 19–25% of PM2.5 by season, against about 4% for a
  typical IMPROVE filter (1–11% for most filters). Addis has the second-lowest OC/EC of any
  SPARTAN or IMPROVE site, after Abu Dhabi.
- **What we have:** these ratios are presets in every gallery picker (Ratios: EC (FTIR) / PM2.5
  mass, OC / EC, OM / EC; for IMPROVE, EC (TOR) / PM2.5 mass and OC (TOR) / EC (TOR)), across all
  SPARTAN and IMPROVE sites.
- **Next step:** a calibration run that picks IMPROVE filters by EC/mass or OC/EC nearest Addis.
  Compare it with the spectral-analog selection on the usual Addis readout (slope, intercept,
  site-grouped cross-validation).

### Is TOR EC inflated on very dark filters?
- **Why:** on heavily loaded filters the TOR laser can saturate. Optical pyrolysis (OP) then reads
  zero and EC can come out high. That would bias the IMPROVE filters that look most like Addis,
  wildfire filters included.
- **What we have:** IMPROVE samples in the gallery now carry OP, EC1–3 and OC1–4. 9,172 samples
  have OP = 0. On those, the median EC (TOR) / HIPS BC is 1.01, against 0.84 overall. The
  `meeting/op_summary.csv` of the rank-70 run may already hold part of the answer for the analog
  filters.
- **Next step:** compare OP and EC(TOR) / HIPS BC between the analog and smoke filters and the
  rest, as a function of EC loading per filter. Then decide whether OP = 0 filters should leave
  the calibration set.

### How much of the Addis slope is the MAC?
- **Why (Sean):** HIPS BC = Fabs / MAC with MAC = 10, a convention rather than a value measured in
  Addis. MAC is known to vary by site.
- **What we have:** the gallery's Filters → units → MAC choice.
  - A different MAC scales the slope of any fit against HIPS BC.
  - The intercept is unchanged for OLS. With the gallery's Deming fit it moves slightly, because
    the error ratio stays fixed (MAC 10 to 15 at Addis: −6.76 to −6.99).
  - The poster's slide notes record the same caveat.
- **Next step:** find the MAC that brings the Addis slope to 1, and judge whether it is plausible
  for charcoal and wood smoke. Then ask how much intercept is left for the FTIR calibration to
  explain. The working view is that the MAC is part of the slope problem and the FTIR
  calibration is the intercept problem.

### Compare networks as mass on the filter, not concentration
- **Why (Ann, Satoshi):** FTIR responds to the mass on the filter. IMPROVE draws about 4.5× more
  air per filter than SPARTAN.
- **What we have:** the gallery's Mass per filter switch, and the poster's per-filter panels.
  Addis EC is about 32× the IMPROVE median per m³ but about 9× per filter.
- **Next step:** recast the extrapolation argument (how far beyond the IMPROVE calibration Addis
  sits) per filter, and per area once deposit areas are settled (see
  [Sampled volumes](#sampled-volumes)).

### Can a few Addis TOR filters test the calibration?
- **Why:** funding for a year of collocated FTIR and TOR filters at Addis, Delhi and Beijing was
  lost. About 10 parallel FTIR + TOR filters from Addis city (not the SPARTAN site) are expected
  in the next few months.
- **What we have:** the handful received so far did not show the problem. They come from a
  different site, on the Colorado State sampler, at a lower flow, so they sit closer to the
  detection limits.
- **Next step:** before reading the new filters, write down what agreement would count as
  evidence. Check whether their composition (EC/mass, OC/EC, the spectral analogs) resembles the
  SPARTAN Addis filters at all.

### Urban IMPROVE sites as a calibration set
- **Why (Ann):** the handful of urban IMPROVE sites are the closest US analogue to Addis. CSN
  (urban) filters are larger and thicker and sampled at a lower flow, and haven't been usable.
- **Next step:** a calibration from only the urban IMPROVE sites, read out on Addis like the other
  cohorts.

### Does the old SPARTAN filter support grid matter?
- **Why (Sean):** early SPARTAN filters showed pixelation from the support grid. Warren's view is
  that the current grid leaves less white space than IMPROVE's.
- **Next step:** low priority. If wanted, compare HIPS on mesh and stretched filters at sites
  that changed filter type.

## Data still to get

| What | Why | Where it would come from |
|---|---|---|
| Corrected FTIR functional groups, batch 4 (Nov 2022–Mar 2024) | The lab deliverable's groups and OM run about 2× the public OM and the four-site file (OC and EC agree), so the gallery stops at Mar 2023 | AQRC lab: the correction already applied to the four-site file |
| FTIR spectra for SPARTAN sites other than Addis | Spectral comparisons beyond Addis | AQRC query (template `etad_lots_query.sql` on Drive) |
| HIPS uncertainty and MDL for 756 older filters (mostly 2019–2022) | Public BC (Fabs / 10) covers them, but uncertainty-weighted fits need the errors | An older HIPS export from AQRC |
| Fabs for 256 filters in the raw HIPS pull (lots 241–253) | They have only raw transmittance and reflectance | The HIPS instrument calibration |
| ILHA-0216 dated 2026-11-17 in the public files | A date typo; the exports leave it out | Report to SPARTAN |

## Sampled volumes

Details in [mass-per-filter.md](mass-per-filter.md).

1. **SS4i flow.** 445 SPARTAN filters (2013–2017, 13 sites) came from the SS4i sampler. Nothing
   in hand gives its flow: no HIPS volume, and no `Analytical_MDL` to infer it from. SPARTAN's
   network papers or the SPARTAN team could confirm whether it also ran at 5 L/min.
2. **IMPROVE before 2015.** 194,680 IMPROVE samples (2003–2014, and part of 2015) have no volume.
   IMPROVE's flow records (the FED archive's flow and duration fields, or the AQRC database)
   could fill them. Until then, a nominal 32.8 m³ is within about ±10% for 98% of the samples
   that do have a volume.
3. **Two IMPROVE volumes.** The poster build's IMPROVE volumes (FED export: median 21.2 L/min,
   30.5 m³) differ from the cleaned table's quartz-module volumes (22.85 L/min, 32.8 m³) by about
   7%. Find out which module or duration each uses before quoting per-filter numbers to two
   significant figures.
4. **Teflon or quartz flow.** Per-filter HIPS, mass and elements use the quartz module's flow.
   The Teflon module's is about 0.7% lower. This is small, but it could be exact.
5. **Deposit area.** Per filter equals per area only if deposit areas match. IMPROVE's canonical
   area is undecided (3.5 vs 3.53 cm², [open-items.md](open-items.md)). SPARTAN's `DepositArea`
   per filter is in the HIPS file but hasn't been compared.
