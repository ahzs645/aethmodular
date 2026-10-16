# AAAR practice talk, group meeting Thu 15 Oct 2026

Deck: `aaar_practice_2026-10-15.pptx` (11 slides, built by `build.py`, updated 16 Oct after the practice talk; every number is asserted against
`storyline.md` / the rank-70 meeting run before the deck is written).

## One breath

EC in Addis Ababa is very high (~31× typical IMPROVE). The current FTIR calibration predicts it on the wrong scale
(2.30x −6.09), and the same kind of calibration poorly predicts the 97 held-out IMPROVE filters whose spectra most
resemble Addis spectra (highest correlation across wavenumbers, excluding defined regions) (0.46x vs HIPS, R² 0.52), even though TOR and HIPS agree on them. Addis filters carry ~9× the EC per filter of
a typical IMPROVE filter, and Addis EC is above every IMPROVE site (17 of 29 SPARTAN sites are too). Baseline
correction moves the Addis intercept from −6.1 to −1.1; a calibration explorer tests thousands of calibrations;
season-specific calibrations help in Kiremt (slope 0.51 → 0.94) but not Dry or Belg.

## Slide order (after Ann's 15 Oct review)

1. Title + roadmap
2. Addis EC by season vs IMPROVE median EC, 95th percentile EC and median PM2.5; left in µg/m³, right in µg per filter
3. The problem: current SPARTAN FTIR EC vs HIPS, 2.30x −6.09
4. FTIR predicts EC well on typical IMPROVE filters
5. Current calibrations poorly predict IMPROVE FTIR EC for spectra that most resemble Addis (HIPS on both x axes)
6. ~9× more EC per filter (histogram `f4b`; the ECDF `f4` is kept but off the slide)
7. EC by site: every IMPROVE and SPARTAN site, Addis above all IMPROVE sites
8. Baseline correction: −6.1 → −1.1
9. Calibration explorer (spline only, bars; settings unnamed)
10. Season-specific calibrations
11. Conclusions

## How to present each crossplot (15 Oct feedback)

- Point at the axes before the points. Bottom: "HIPS light absorption as EC", the light the filter absorbs (Fabs)
  divided by MAC 10. Side: "FTIR-predicted EC", from a PLS calibration built on IMPROVE filters with TOR EC.
- R² is good because the two measurements are related. The slope and intercept are what is wrong, and they are
  not physically sensible (a filter with no EC would read about −6 µg/m³ on slide 3).
- Changing MAC stretches the bottom axis: it changes the slope, not the intercept. So the intercept is not a MAC
  problem (checked in `build.py`).
- Slide 2: IMPROVE samples ~22 L/min, SPARTAN ~5 L/min, so an IMPROVE filter holds ~4× more air. In the air Addis
  EC is ~32× typical IMPROVE; on the filter, which is what FTIR sees, ~9×. Typical IMPROVE EC is ~4% of PM2.5,
  Addis ~19 to 25%.
- Slide 6: bottom = µg of EC on one filter, log scale (each tick 10×); side = percent of filters, each network adds
  to 100%.

## Why calibrate with IMPROVE? (short answer)

IMPROVE's collocated FTIR + TOR quartz library is the only one that exists anywhere. There is no Addis or urban
equivalent, and CSN filters are different (bigger, thicker, lower flow).

## Plain-language glossary

- **Addis Ababa**: capital of Ethiopia, East Africa; high altitude (~2,300 m), about five million people.
- **EC / OC / OM**: elemental carbon (black, soot-like carbon from burning), organic carbon (the rest of the particle
  carbon), organic matter (organic carbon plus the hydrogen, oxygen and other atoms attached to it).
- **PM2.5**: mass of particles smaller than 2.5 µm per cubic metre of air.
- **FTIR**: Fourier-transform infrared spectroscopy. Infrared light is passed through the filter and the amount
  absorbed at each wavenumber is recorded; it does not destroy the filter.
- **Spectrum / spectra**: that absorbance-versus-wavenumber curve for one filter (plural: spectra).
- **Baseline correction of the spectra**: fitting and subtracting the smooth, sloping background (from the filter
  material and light scattering) so only the particle absorption bands remain. Say "baseline correction", not
  "baselining".
- **HIPS**: hybrid integrating plate/sphere, an optical measurement of how much light the filter absorbs.
- **Fabs**: that filter light absorption (Mm⁻¹ once divided by the volume of air).
- **MAC**: mass absorption cross-section, how much light one gram of EC absorbs (m²/g). EC = Fabs / MAC; we use 10.
- **TOR**: thermal-optical reflectance, the standard lab method: heat a punch of a quartz filter and measure the
  carbon released, using reflected laser light to split OC from EC.
- **IMPROVE**: the US network sampling air in national parks and rural areas; has both FTIR and TOR on the same
  samples.
- **SPARTAN**: a global network of filter samplers, mostly in populated, polluted cities; carbon is measured only by
  FTIR, BC by HIPS. Addis Ababa is a SPARTAN site.
- **Calibration set**: the filters with known (TOR) EC that the FTIR model learns from.
- **PLS**: partial least squares regression, the standard statistical method that turns a whole spectrum into one
  number (here EC).
- **Spectral analogs / IMPROVE spectra that most resemble Addis spectra**: IMPROVE filters whose spectra have the
  highest correlation with the Addis median spectrum across wavenumbers, excluding defined regions (e.g. CO₂ band).
- **Deming regression**: a straight-line fit that allows for error in both x and y.

## Ann's asks from the 9 Oct email

- Slides before the practice talk: this deck (late; she asked for Tuesday).
- Self-prediction test: answered (slide 5 NOTES; the grey held-out panels are that test, site-disjoint).
- Share IMPROVE + SPARTAN map view: share build exists, not yet published.

## Don't get caught

- No OLS anywhere (Ann). Say "poorly predicts", not "under-reads" or "half of TOR": FTIR is high at the low end
  and low at the high end.
- Say IMPROVE, not "US".
- No TOR at Addis; HIPS is the reference there.
- Addis n: slides 2, 3, 6, 8 and 10 all use the same 189 filters (those with a current SPARTAN FTIR EC; 44 of the
  233 have none). The explorer (slide 9) scored all 233. 190 is the older phase-3 cohort; don't quote it. Slide 7 uses
  every public filter per site (Addis 246) so all sites are treated alike.
- Slide 8 compares the current calibration with a baseline-corrected IMPROVE calibration; they differ in more
  than the baseline (training set, protocol). The slope drops to 0.56: better intercept, not solved.
- Slide 9: don't explain the settings rows or the guardrails. 3,137 of 3,138 beat the current intercept; 254 are
  within ±0.5 µg/m³ of zero, not always with a good slope.
- Slide 10: combining the three seasonal calibrations gives 1.43x −4.23 overall (each season lands at a different
  offset). Belg gets worse.
- Don't bring up Abu Dhabi (OC predictions there are unreliable: lots of dust, little organics).
- SPARTAN carbon is FTIR, not thermal; SPARTAN BC = HIPS fAbs / 10. Slide 7 uses HIPS-measured BC only (public
  values 'estimated from the HIPS-SSR curve' are left out), so 29 SPARTAN sites qualify. Data: SPARTAN public release v3.1.4.
- OC/fAbs by site, the k-sweep and VIBES are discussion material, not on slides.

## Open items raised on 15 Oct (not run)

- EC-to-mass ratio as a calibration-set selection criterion.
- TOR optical pyrolysis (OP) on fire-affected and Addis-like IMPROVE filters.

## [FILL IN]

- Poster vs talk format at AAAR, session/time.
