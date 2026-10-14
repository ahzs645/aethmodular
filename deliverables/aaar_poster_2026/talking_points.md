# AAAR practice talk, group meeting Thu 15 Oct 2026

Deck: `aaar_practice_2026-10-15.pptx` (8 slides, built by `build.py`; every number is asserted against
`storyline.md` / the rank-70 meeting run before the deck is written).

## One breath

An established FTIR EC calibration built on IMPROVE works on typical US filters (held-out slope 0.93, R² 0.73)
but reads only about half of TOR on the 97 held-out US filters that look most like Addis (0.48, R² 0.46), even
though TOR and HIPS agree on them (0.91, R² 0.82). Addis filters carry ~9× the EC per filter of a typical IMPROVE
filter, and Addis OC/EC (~1.2) is lower than every one of 211 IMPROVE sites; 31 of 32 SPARTAN sites sit below
the typical US site. On US filters FTIR reads ~0.7 of TOR at OC/EC below 2.5. Coverage problem, not a one-site quirk.

## Slide order

1. Title + roadmap
2. Addis chemistry by season (context for a general audience)
3. The problem: Addis FTIR vs HIPS, Deming 2.30x −6.09, R² 0.78
4. Method works on held-out IMPROVE
5. Same calibration under-reads Addis-like IMPROVE filters (key slide)
6. Loading per filter, ~9×
7. OC/EC continuum: every IMPROVE and SPARTAN site ranked; Addis 2nd of 243
8. FTIR/TOR falls with OC/EC on US filters; Addis sits further down
9. Takeaway

## Ann's asks from the 9 Oct email

- Slides before the practice talk: this deck (late; she asked for Tuesday).
- Self-prediction test: answered (slide 5 NOTES; the grey held-out panels are that test, site-disjoint).
- Share IMPROVE + SPARTAN map view: share build exists, not yet published.

## Don't get caught

- Title used to say "underestimates"; at Addis the slope is 2.3, so low filters read low and high filters read
  high. Say "transfers poorly", not "underestimates", for Addis. "Under-reads" is right for the IMPROVE analogs.
- No TOR at Addis; HIPS is the reference there. TOR and HIPS agree on IMPROVE analogs.
- SPARTAN has no thermal carbon: public OC/EC/OM are FTIR (method 218), BC is HIPS (221). The older site plot
  that labelled SPARTAN stars "thermal" was wrong.
- OC/EC on slides 7-8 uses EC = HIPS absorption / 10 for both networks, so it does not lean on FTIR EC and x and
  y on slide 8 do not share TOR EC. SPARTAN OC is still FTIR.
- Data source: SPARTAN public release v3.1.4 (Aug 2026). The older in-house unified file has Addis OC ~2× lower
  (filter for filter); don't quote its 0.66 OC/EC.
- Slide 8 trend holds within most loading bands but flattens above ~16 µg per filter (small n). TOR's OC/EC split
  can still couple x and y a little (charring).
- Don't claim Addis is "dustier than IMPROVE" outright: its Si-O band is ~2x a typical IMPROVE filter but inside
  IMPROVE's range filter by filter.
- Chemistry slide: OM is FTIR-derived, BC is HIPS-based; medians don't sum to median PM2.5.
- The 14 Oct k-sweep and the VIBES material are for the discussion, not the poster.

## [FILL IN]

- Co-author list and affiliations on slide 1.
- Poster vs talk format at AAAR, session/time.
