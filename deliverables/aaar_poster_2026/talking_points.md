# AAAR practice talk, group meeting Thu 15 Oct 2026

Deck: `aaar_practice_2026-10-15.pptx` (8 slides, built by `build.py`; every number is asserted against
`storyline.md` / the rank-70 meeting run before the deck is written).

## One breath

EC in Addis Ababa is very high (~31× typical IMPROVE). The current FTIR calibration predicts it on the wrong scale
(2.30x −6.09), and the same kind of calibration poorly predicts the 97 held-out IMPROVE filters whose spectra look like
Addis (0.46x vs HIPS, R² 0.52), even though TOR and HIPS agree on them. Addis filters carry ~9× the EC per filter of
a typical IMPROVE filter, and Addis EC is above every IMPROVE site (21 of 36 SPARTAN sites are too). Baseline
correction moves the Addis intercept from −6.1 to −1.1; a calibration explorer tests thousands of calibrations;
season-specific calibrations help in Kiremt (slope 0.51 → 0.94) but not Dry or Belg.

## Slide order (after Ann's 15 Oct review)

1. Title + roadmap
2. Addis EC by season vs typical IMPROVE (EC focus; no sulfate/dust)
3. The problem: current SPARTAN FTIR EC vs HIPS, 2.30x −6.09
4. FTIR predicts EC well on typical IMPROVE filters
5. Current calibrations poorly predict IMPROVE FTIR EC for Addis-like spectra (HIPS on both x axes)
6. ~9× more EC per filter
7. EC by site: every IMPROVE and SPARTAN site, Addis above all IMPROVE sites
8. Baseline correction: −6.1 → −1.1
9. Calibration explorer (spline only, bars; settings unnamed)
10. Season-specific calibrations
11. Conclusions

## Ann's asks from the 9 Oct email

- Slides before the practice talk: this deck (late; she asked for Tuesday).
- Self-prediction test: answered (slide 5 NOTES; the grey held-out panels are that test, site-disjoint).
- Share IMPROVE + SPARTAN map view: share build exists, not yet published.

## Don't get caught

- No OLS anywhere (Ann). Say "poorly predicts", not "under-reads" or "half of TOR": FTIR is high at the low end
  and low at the high end.
- Say IMPROVE, not "US".
- No TOR at Addis; HIPS is the reference there.
- Slide 8 compares the current calibration with a baseline-corrected IMPROVE calibration; they differ in more
  than the baseline (training set, protocol). The slope drops to 0.56: better intercept, not solved.
- Slide 9: don't explain the settings rows or the guardrails. 3,137 of 3,138 beat the current intercept; 254 are
  within ±0.5 µg/m³ of zero, not always with a good slope.
- Slide 10: combining the three seasonal calibrations gives 1.43x −4.23 overall (each season lands at a different
  offset). Belg gets worse.
- Don't bring up Abu Dhabi (OC predictions there are unreliable: lots of dust, little organics).
- SPARTAN carbon is FTIR, not thermal; SPARTAN BC = HIPS fAbs / 10. Data: SPARTAN public release v3.1.4.
- OC/fAbs by site, the k-sweep and VIBES are discussion material, not on slides.

## [FILL IN]

- Co-author list and affiliations on slide 1.
- Poster vs talk format at AAAR, session/time.
