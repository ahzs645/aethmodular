# Phase-3 deliverables (tracked)

Shipped artifacts live here, **not** in `output/`. Anything under
`research/**/output/` is gitignored (`.gitignore` line 151), which is regenerable
scratch by design. A deck that goes in front of a collaborator is not scratch, so
it belongs in a tracked directory.

This distinction is not theoretical: `satoshi_deck_2026-08-13.html` was built in
an earlier session, written to `output/`, and silently lost when that working
copy went away. It had to be re-supplied by hand. Nothing regenerates it — see
below.

## Contents

| File | What it is |
|---|---|
| `satoshi_deck_2026-08-13.html` | Phase-3 briefing deck for the Aug 13 Satoshi meeting. Standalone: 14 figures base64-embedded, no external references, renders offline. |
| `extract_deck_figures.py` | Pulls the embedded figures back out as PNGs. |

## Why the deck is tracked rather than built

There is **no builder for this deck**. Every other deck builder in the repo emits
PPTX (`research/ftir_hips_chem/workflows/build_warren_meeting_deck.py`,
`build_warren_for_warren.py`, `research/spartan_ec_2026_06_16/_build_slides.py`,
`improve_fed_missing_data_update/src/build_deck.mjs`); nothing in the repo has
ever base64-embedded an image. The deck was hand-authored, and its house style —
a bespoke design-token set with a light/dark palette, sharing `#B23327` /
`#2C6E9E` with the matplotlib palette in `scripts/build_deck_figures.py` — exists
only inside the file's inline `<style>` block. Lose the file and the style goes
with it.

The 14 figures are likewise **not committed anywhere else**, in any commit: the
only PNGs ever tracked in this repo sit under `notebooks/analysis/output/plots/`.
So this file is the sole surviving copy of those plots.

Regenerating them is currently blocked in any case. Of the six figures in
`scripts/build_deck_figures.py`, only `fig_setup_matrix` renders without external
data (it has a hardcoded fallback); the AIRSpec figures need the Google Drive
mount that `pls_transfer.FTIRTransferPaths.defaults()` resolves to a macOS
`CloudStorage` path, and `fig_filtering_strip` needs a similarity CSV that lived
in an ignored `output/` directory and is gone. The loaders in
`scripts/phase3_common.py` all route through Drive, so even the Drive-free
runners are transitively blocked.

Treat this file as a primary source, not as build output.

## Recovering the figures

```bash
uv run python research/ftir_ec_phase3/deliverables/extract_deck_figures.py
```

Writes `figures/fig01_*.png` … `fig14_*.png`, named from each image's `alt` text.
The PNGs themselves are deliberately not tracked — AGENTS.md asks that generated
PNGs stay out of git, and they are re-derivable from the deck at any time by
running the line above.

- 2026-08-12 (later): `ann_briefing_2026-08-12.html` finalized — HIPS wavelength resolved on slide 07 (632.8 ≈ 633 nm), preliminary badges upgraded to committed notebook references, speaker notes embedded per slide (collapsible, from `deck_notes_ann_2026-08-12.md`), and appendix slides R1–R3 added resurfacing the skipped July-17 figures (ftir_15 PNGs + July-17 PDF pages 23/27/30).
