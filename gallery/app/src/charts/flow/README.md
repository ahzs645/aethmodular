# flow/

react-graph-gallery's *flow* category (Sankey, Chord, Network, Arc, Edge
Bundling) has **zero** matching figures in this repo's 827 — see
`gallery/census/CENSUS.md`.

`MethodSankey.tsx` is the first one (the second is the chord view inside
`calibration/OverlapMatrix.tsx`, where cohort overlap is a flow question): filters → site → carbon-EC coverage
(FTIR + TOR / FTIR only / TOR only / neither) → HIPS coverage or season. It is
computed from `filters.json` alone, so it needs no new export.

The next stage the README originally described — calibration variant →
reported EC — needs the calibration-variant tables exported. Add a fourth
stage here when they are.
