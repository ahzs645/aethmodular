#!/usr/bin/env python3
"""Export the Addis specification curve for the gallery's Calibration tab.

Source: `calibration_explorer/cache/batch_results.jsonl`, selected, vetted and
decomposed into analytic choices by `research/ftir_ec_phase3/scripts/spec_curve.py`,
the same module the matplotlib figure (build_pathway_figures.fig_spec_curve,
01_specification_curve.png) uses. Every number is the explorer's own readout.

Unlike calibration.json's grid, which keeps only the rule-k readout per
configuration, this keeps every k, because the curve is the whole search.

Writes gallery/app/public/data/spec_curve.json, columnar so 20k rows stay small:
  intercept, slope, heldout      one value per specification, sorted by intercept
  ok                             "0"/"1" string: passes every guardrail
  masks                          one "0"/"1" string per analytic choice
  co, sp, sel, mo, lot           indexes into the matching category lists
  cut, k                         raw values, for the tooltip and click-through

Run:  python gallery/data/export_spec_curve.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research" / "ftir_ec_phase3" / "scripts"))
from spec_curve import (  # noqa: E402
    CHOICES, CHOICE_GROUPS, GUARDRAIL_TEXT, HELDOUT_FLOOR, RESULTS, SLOPE_RANGE,
    passes_guardrails, spec_rows, uses,
)

OUT = REPO / "gallery" / "app" / "public" / "data" / "spec_curve.json"


def bits(flags) -> str:
    return "".join("1" if f else "0" for f in flags)


def coded(values: list) -> tuple[list, list[int]]:
    """Category list plus one index per row (None becomes the string 'none')."""
    labels = sorted({str(v) if v is not None else "none" for v in values})
    index = {v: i for i, v in enumerate(labels)}
    return labels, [index[str(v) if v is not None else "none"] for v in values]


def rnd(v, d=4):
    return None if v is None else round(float(v), d)


def main():
    if not RESULTS.exists():
        print(f"! {RESULTS} not found; run a server batch in the calibration explorer first")
        return
    rows = spec_rows(RESULTS)
    ok = [passes_guardrails(r) for r in rows]
    cats = {}
    codes = {}
    for key, field in (("co", "cohort"), ("sp", "spectra"), ("sel", "selection_space"),
                       ("mo", "mode"), ("lot", "lot")):
        cats[key], codes[key] = coded([r.get(field) for r in rows])

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "calibration_explorer/cache/batch_results.jsonl via research/ftir_ec_phase3/scripts/spec_curve.py",
        "n": len(rows),
        "n_pass": sum(ok),
        "guardrails": GUARDRAIL_TEXT,
        "heldout_floor": HELDOUT_FLOOR,
        "slope_range": list(SLOPE_RANGE),
        "choices": [label for label, _ in CHOICES],
        "groups": [{"title": t, "start": a, "end": b} for t, a, b in CHOICE_GROUPS],
        "masks": [bits(uses(r, test) for r in rows) for _, test in CHOICES],
        "ok": bits(ok),
        "intercept": [rnd(r["deming_intercept"]) for r in rows],
        "slope": [rnd(r.get("deming_slope")) for r in rows],
        "heldout": [rnd(r.get("heldout_R2"), 3) for r in rows],
        "cut": [r.get("cutoff") for r in rows],
        "k": [r.get("k") for r in rows],
        "categories": cats,
        **codes,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    print(f"spec curve: {len(rows):,} Addis specifications, {sum(ok)} pass every guardrail")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
