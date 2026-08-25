---
name: ann-deck
description: Build a meeting update deck (Ann/Satoshi style) — white claim-title slides, one title-free figure each, SAY/NOTES speaker notes, numbers pulled and validated from the calibration explorer. Use when asked to prepare/update a presentation for Ann, Satoshi, or a group meeting in this repo.
---

Follow `docs/deck-style-prompt.md` exactly — it is the single source of truth for this
deck type (visual style, advisor content rules, Option A/B/B2 naming, provenance
workflow, deck skeleton, standing caveats, deliverable layout).

Process:
1. Read `docs/deck-style-prompt.md` in full, plus the reference deliverable
   `deliverables/ann_update_2026-08-18/` (builder pattern) if it still exists.
2. Collect the meeting-specific inputs listed at the bottom of the prompt from the
   user's message/transcript; ask only for what cannot be inferred.
3. Start the explorer under anaconda python if not running, warm/verify the cache,
   and validate the locked anchor numbers before generating any figure.
4. Generate figures → build the pptx with python-pptx → write talking_points.md →
   verify (slide count, every slide has notes, figures placed) → report location.
