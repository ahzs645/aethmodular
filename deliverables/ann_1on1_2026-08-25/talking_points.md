# Ann 1:1 — Tue 25 Aug 2026 — prep sheet

Deck: `ann_1on1_2026-08-25.pptx` (13 slides, script in speaker notes).
Distinct from the 27 Aug FTIR-group deck (`deliverables/ann_update_2026-08-25/`);
overlapping figures are shared, the framing here follows her August asks.

## The one-breath summary (if time collapses)
**Her lot hypothesis was right.** On 12 Aug she guessed baselining helps because it
minimises the lot 248 / lot 251 difference. Measured at Beijing, the one site that ran
both lots at once: on **raw** spectra the lots differ by **−4.5 Mm⁻¹** (CI excludes zero
at both cohort sizes); on **AIRSpec-corrected** spectra it collapses to **−0.9** and
spans zero. Baselining specifically absorbs between-lot substrate differences. It does
**not** explain the Addis offset — ETBI is 100% lot 251 and ETAD 80%, so Addis-vs-Bishoftu
is a within-lot-251 contrast.

## Her 12 Aug asks — status
- Baseline-correct for **both** selection and calibration — **done** (shipped 18 Aug).
- Analog cutoff sweep — **done** (dense sweep, step 10; basin located).
- Focus lot 251 + evaluate-on-same-lot — **done** (`eval_lot` in the app).
- "Is baselining helping because of the lot?" — **answered, slide 4.** Yes, and it
  removes it.
- "Lot 248 count looks too small" — **answered, slide 5.** 1,362 analyses,
  2020-12-17 → 2021-02-15. Her instinct that something was odd was right; the oddity is
  that the lot itself was short, not that the pull was partial.
- Get the Bishoftu spectra yourself — **done**, and it grew: 26 → **40** filters.
- Adama 2–3 slides for Christian and Sina — **part.** Seed figure only; the TOR-vs-FTIR
  OC/EC panel she actually named is not built. Slide 10 says so plainly.

## New since, that she has not seen
- **Lot 253** is 58% of everything SPARTAN has sampled since Sept 2025 and is outside the
  calibration basis. IMPROVE holds 5,050 lot-253 filters with matched TOR EC; manifest written.
- **256 filters** exist in raw `hips.Results` but not in the shipped HIPS CSV; 54 have
  recoverable volumes (Bishoftu +14, Delhi +26, Addis +14).
- **AERONET diurnal confound closed** — correcting widens the Addis gap (2.9× → 3.0–3.3×).

## Don't get caught
- **The `"corrected"` token trap (slide 11).** `/api/run` dispatches on
  `airspec`/`neutral`/`deriv2`; anything else falls through to **raw silently**. The first
  version of the lot result was computed this way and showed a large "lot effect" that was
  really a raw-spectra effect. Caught by an anchor refusing to reproduce. Both anchors now
  match. Corrected in `SPARTAN_LOT_INVENTORY_2026-08-23.md`.
- **Recovered Fabs are reconstructed, not official** — production QC not replicated,
  DepositArea is the site median. Quote headline results with and without them.
- **Lot test caveats** — n = 14 vs 34; within-window date distributions not matched
  further; loading differs slightly between the two lot groups.
- **Delhi lot-253 test is not identifiable** — 4× more loaded, seasonally disjoint,
  6 filters in the overlap, CI [−53, +304]. Don't offer it as a test.
- **H is not a boundary-layer height** — comparable across sites only, Level 1.5.

## Asks
- **Quartz TOR at Addis** — unchanged, still the only thing separating an additive offset
  from high-loading curvature (degenerate on FTIR-EC axes).
- **IMPROVE lot-253 scan pull** — manifest written, needs a VPN session on the Windows box.
- **Adama panel** — on me, blocked on nothing.

## [FILL IN] before presenting
- Committee meeting 2 Sept 3 pm — invite resent? who has confirmed?
- AAAR acceptance + poster slot (Thu, session 9, 1–3 pm) — forwarded to Ann?
- Anything further on funding.
