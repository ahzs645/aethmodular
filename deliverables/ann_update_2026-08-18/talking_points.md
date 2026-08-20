# Ann 1:1 — Tue Aug 18, 2026 — prep sheet

Ann's email: she may be **late** (prior meeting can run long), wants updates on the
**committee meeting** and **finances**, and can **run long** since ours is her last
meeting. Deck: `ann_update_2026-08-18.pptx` (20 slides, script in speaker notes).

## The one-breath summary (if time collapses)
Every item from Thursday's list is answered. The big one: **selecting on
baseline-corrected (AIRSpec) spectra fixes both spectra-based cohorts** — the analogs
go −9.7 → −4.3 (Deming, MAC 10) with TOR R² 0.71 and keep only 4/477 of their old
filters (the raw similarity was matching Teflon, as Satoshi suspected); Ethiopia-shaped
keeps 285/300 filters but its dead TOR test is rescued (0.00 → 0.63). More components
(k=21) does **not** fix the analogs — the k sweep says it was never the component count.
Cutoffs: analogs-350 ≫ analogs-500 in raw space; OC/EC-800 confirmed as the sweet spot.

## Her Aug-12 asks — status
- µg/m³ everywhere; Deming (λ*=2.96, MAC²-scaled) as primary; protocol named on every
  figure; claim titles — **done throughout the deck**.
- One consistent k-rule + protocols comparison — **slide 3 ladder**, now three ways:
  **Option A** (site-grouped 5-fold, first major minimum), **Option B** (interleaved
  10-fold, within-5% — the network's current protocol; reproduces last week's numbers),
  **Option B2** (interleaved 10-fold, first major minimum — isolates rule vs folds;
  mostly lands between A and B, so the fold structure is the bigger lever).
- Slides 4–9: the six-panel MAC grid **six ways** — raw and baselined, each under
  Options B, A, and B2. Slide 10: the analog-selection explainer (both spectra
  spaces — the corrected row visibly re-ranks the pool). Slide 13: the "Teflon
  background" explainer (90% of a raw spectrum is baseline at CH; Addis median 0.17
  vs OC/EC-800's 0.10 — raw similarity is background similarity).
- The A variants carry what B structurally can't: a
  **held-out TOR R² in every panel** (network 0.70, OC/EC 0.91, AIRSpec 0.90 —
  and eth-shaped's 0.00 asterisk made visible). Baselined story is protocol-robust:
  intercepts shrink everywhere (−0.6…−2.2) but only OC/EC+AIRSpec keeps slope ≈ 1
  and TOR skill (A −2.09 / B −2.17 / B2 −1.95). Trap panel to preempt:
  eth-shaped+AIRSpec's −0.60 has slope 0.30 and TOR 0.24 — looks great, isn't.
- Overlap 300 ∩ 800 — **1 filter** (matches her own analysis; slide 5).
- Interleaved-vs-grouped side-by-side — covered by ladder + k-sweep; ftir_32 closed the
  fold-count objection (interleaved-5 ≈ interleaved-10 → the leak is structural).

## Open caveats (don't get caught)
- Analog cohort resolves to **477 not 500** — the locked analog list is not a pure sort
  of the committed rank score. Unresolved; flagged on slide 4 notes.
- Entire-network **site-held-out k=15 vs locked 10** — unexplained; don't quote that
  row's site-held-out numbers as final.
- Corrected-selection results are app-derived through the shared validated code —
  notebook write-up (ftir_34) is the next commit.

## Blocked items → asks
- **Bishoftu**: HIPS in hand (32 filters). Need the **ETAD-style FTIR spectra export
  for ETBI** (SQL template on hand; likely via Alex). Then the crossplot is one preset.
- **Lots**: in-hand half done (curves jagged in both 248 and 251; pool is 88% lot 251 —
  the "all 248" premise was backwards). Still need **Mona's PCA plots** + spectra for
  non-248/251 lots.

## Logistics (her explicit agenda — bring specifics)
- **Committee meeting**: [FILL IN — who's confirmed, dates offered, blocker] →
  commit to a dated next step (e.g., poll out today, locked by Friday).
- **Finances**: [FILL IN — funding source, runway, gap + by when, what Ann can do].
- "Something to show": AAAR abstract submitted; provisional Addis EC series (ftir_29);
  today's results all committed on the ftir-29 branch and reproducible on demand.
- Calendar: next 1:1s move around her fall travel; next week Tuesday, +1 hr.
