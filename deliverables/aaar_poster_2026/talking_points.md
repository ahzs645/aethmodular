# AAAR practice talk, group meeting Thu 15 Oct 2026

Deck: `aaar_practice_2026-10-15.pptx` (8 slides, built by `build.py`; every number is asserted against
`storyline.md` / the rank-70 meeting run before the deck is written).

## One breath

An established FTIR EC calibration built on IMPROVE works on typical US filters (held-out slope 0.93, R² 0.73)
but reads only about half of TOR on the 97 held-out US filters that look most like Addis (0.48, R² 0.46), even
though TOR and HIPS agree on them (0.91, R² 0.82). Addis filters carry ~9× the EC per filter of a typical IMPROVE
filter and have OC/EC ~0.7 vs IMPROVE ~5.5 (below 99.6 % of IMPROVE). In IMPROVE itself FTIR reads ~0.7 of TOR on
filters with OC/EC below 3, at every loading. Coverage problem, not a one-site quirk.

## Slide order

1. Title + roadmap
2. Addis chemistry by season (context for a general audience)
3. The problem: Addis FTIR vs HIPS, Deming 2.30x −6.09, R² 0.78
4. Method works on held-out IMPROVE
5. Same calibration under-reads Addis-like IMPROVE filters (key slide)
6. Two extremes: ~9× more EC per filter, OC/EC 0.7 vs 5.5
7. FTIR/TOR falls with OC/EC (0.7 below OC/EC 3); Addis sits further down still
8. Takeaway

## Ann's asks from the 9 Oct email

- Slides before the practice talk: this deck (late; she asked for Tuesday).
- Self-prediction test: answered (slide 5 NOTES; the grey held-out panels are that test, site-disjoint).
- Share IMPROVE + SPARTAN map view: share build exists, not yet published.

## Don't get caught

- Title used to say "underestimates"; at Addis the slope is 2.3, so low filters read low and high filters read
  high. Say "transfers poorly", not "underestimates", for Addis. "Under-reads" is right for the IMPROVE analogs.
- No TOR at Addis; HIPS is the reference there. TOR and HIPS agree on IMPROVE analogs.
- "Most Addis-like" depends on the similarity metric; spline and VIBES pick mostly different analogs.
- Addis OC/EC is SPARTAN OC (FTIR-derived) / BC (HIPS): no TOR at Addis. OC would need ~3x higher just to reach
  IMPROVE's 5th percentile, so the conclusion survives the circularity.
- It is OC/EC, not loading: OC/EC < 3 reads lower in every loading band; robust fit gives OC/EC +0.57, loading
  -0.09. The loading trend we saw first is heavier US filters also being more EC-rich.
- No Addis points on slide 7 (no TOR), only the band. High-OC/EC ratios > 1 are partly the small intercept on
  low-EC filters.
- Don't claim Addis is "dustier than IMPROVE" outright: its Si-O band is ~2x a typical IMPROVE filter but inside
  IMPROVE's range filter by filter. Loading is the clear outlier.
- Chemistry slide: OM is FTIR-derived, BC is HIPS-based; medians don't sum to median PM2.5.
- The 14 Oct k-sweep and the VIBES material are for the discussion, not the poster.

## [FILL IN]

- Co-author list and affiliations on slide 1.
- Poster vs talk format at AAAR, session/time.
