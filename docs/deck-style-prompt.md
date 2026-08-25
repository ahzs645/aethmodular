# Prompt: build a meeting deck in the house style

Copy-paste this (plus the meeting-specific inputs at the bottom) to any assistant/session
to reproduce the presentation type used for the Ann/Satoshi updates
(reference deliverable: `deliverables/ann_update_2026-08-18/`).

---

Build me a PowerPoint update deck in my established style. Follow every rule below;
they come from my advisor's explicit feedback and from my previous decks.

## Visual style (non-negotiable)

- 16:9, 13.333 × 7.5 in, plain white background, no template, no logos.
- **Claim-as-title**: one bold ~20pt line, top-left, ink #22252A. The title states the
  *finding*, not the topic ("A loading-dependent HIPS artifact can't be closed with
  IMPROVE data" — approved example). Never a topic label, never bullets restating it.
- **One figure per slide**, centered under the title, nothing else on the slide.
  Figures are exported **title-free** (the slide title does the work), matplotlib,
  white background, dpi ~165–170, top/right spines off, flattened to RGB
  (RGBA PNGs break PowerPoint).
- Palette: GREY `#8F8C84`, BLUE `#2C6E9E`, PURPLE `#7A4FA3`, ACCENT `#B23327`,
  AMBER `#C49442`, INK `#22252A`. Colour encodes the setup/series, not the protocol.
- Text-only slides (status lists, blocked items, logistics): 3–4 short lines, ~17pt.
- Optional: section dividers = bare large-text white slides; backup material behind a
  "Not Ready" divider.
- **Speaker notes on every slide**, two parts:
  - `SAY:` — the spoken script, conversational, addressed to the room, walks the eye
    through the figure and lands exactly one takeaway. Slide 1's SAY is a roadmap of
    the whole deck (and a "if X joins late, resume at slide N" note when relevant).
  - `NOTES:` — background, caveats, and answers to expected questions. Every open
    discrepancy goes here explicitly; never hide one.

## Content rules (advisor's standing requirements)

- Units **µg/m³ everywhere**; Mm⁻¹ only parenthetically. (Trap: `local_db`
  `results_tor` Value is **ng/m³** — label it as such if shown.)
- **Deming regression is the primary estimator** (the intercept is the quantity of
  interest): λ\* = 2.96 at MAC 10, scaled by MAC² (λ₆ = 2.96 × 0.36). Show OLS
  alongside. State the estimator on the figure.
- **Name the protocol on every figure/table.** House naming (define once, early):
  - **Option A** — site-grouped 5-fold CV, first major minimum, site-disjoint 80/20
    fit (the only option with a held-out TOR test — show that R² whenever A is used);
  - **Option B** — interleaved 10-fold CV, first k within 5% of minimum, fit on all
    filters (the network's current protocol; reproduces the historical slide numbers);
  - **Option B2** — interleaved 10-fold CV, first major minimum (per-fold SE band).
- One consistent k-rule per protocol; quote k **with its full scheme incl. fold count**.
- Default readout: **fixed 190-filter Addis cohort, MAC 10**; MAC 6 shown as the
  open/dashed twin (the Deming intercept is MAC-invariant — say so).
- "**baseline-corrected (AIRSpec)**" spelled out on first mention.
- Spectra shown are **never averages** — one real representative filter (state the
  selection rule) or per-filter values.
- Six-panel MAC grid grammar: filled points/solid line = MAC 10, open/dashed = MAC 6,
  dotted grey 1:1, black diamond at the shared intercept, stat box = Deming intercept
  (both MACs) + Deming slopes @10/@6 + OLS intercept + k (+ held-out TOR R² under
  Option A). Per-panel square-ish limits; never shared clipped axes.

## Provenance workflow (where numbers come from)

- Every number comes from the **calibration explorer** (`calibration_explorer/app.py`,
  port 5058, run under `~/anaconda3/bin/python` — the uv venv lacks flask). Warm the
  cache first (`warm_cache.py`); pull runs via `POST /api/run`
  `{cohort, cutoff, selection_space, spectra, mode, lot, target}`; sweeps via
  `/api/sweep`; overlaps via `/api/overlap`.
- **Validate before presenting**: the locked anchors must reproduce —
  Lowest-OC/EC Option A = k 6, OLS 1.585x−3.221, held-out R² 0.911;
  +AIRSpec = k 5, OLS 0.86x−1.615, Deming 0.95x−2.09; Option B Deming@MAC10 must
  match the historical slide intercepts (−5.76 / −10.16 / −6.74 / −2.17).
- Reuse committed explainer figures where they exist
  (`research/ftir_ec_phase3/output/plots/deck/airspec_1..3*.png`,
  `deck_notes_airspec.md` has their ready-made talk tracks and caveats).
- Build the pptx with `python-pptx` (anaconda), blank layout, notes via
  `slide.notes_slide.notes_text_frame`.

## Deck skeleton (the narrative arc)

1. Title + roadmap-in-SAY.
2. **Status table**: the current to-do list, item by item, DONE/PART/BLOCKED —
   answers "what do you have to show".
3. Provenance: intercept ladder across setups × Options A/B/B2, with the
   reproduction-of-known-numbers claim.
4. The grid block: six-panel MAC grids (raw, then baseline-corrected) under each
   Option being discussed.
5. Mechanism explainers before results that need them (how a selection works;
   what "Teflon background" means).
6. Headline results (crossplot pairs, before/after), each with its own slide.
7. Diagnostics (k sweeps, cutoff sweeps, overlaps, lots).
8. Blocked items with the **precise external ask** (who, what artifact).
9. Logistics/personal items the advisor flagged — as a real slide so it can't be
   squeezed out, with [FILL IN] prompts for things only I know.

## Standing caveats — carry these until resolved (NOTES material, never hidden)

- **Analog cutoff ordering**: "top 500" resolves to 477 filters because the cutoff is
  applied before TOR-eligibility filtering (corrected selection similarly 486/500).
  Affects cutoff comparisons; the app flags it. State n as resolved (477), not nominal.
- **Entire-network Option A picks k=15** vs the locked phase-3 pool run's k=10 —
  unexplained; don't quote that cell as final.
- Ethiopia-shaped + AIRSpec under B2 (k=21) is an unstable cell — flat curve, rule
  wanders.
- App-derived results not yet in a committed notebook must be labelled as such.

## Deliverable layout

`deliverables/<meeting>_<YYYY-MM-DD>/`:
- `figures/` — all generated PNGs (f0…fN, descriptive suffixes);
- `<meeting>_<date>.pptx`;
- `talking_points.md` — one page: the one-breath summary, advisor's asks each marked
  closed, the "don't get caught" caveat list, blocked-item asks, [FILL IN] personal
  items, calendar notes.

## Meeting-specific inputs (fill these in when using this prompt)

- Meeting, date, audience, and any constraints from their latest email (late arrival,
  explicit agenda items).
- The current to-do list to close out, item by item.
- The new results since last time (with which app configs produce them).
- The personal/logistics items to reserve a slide for.
