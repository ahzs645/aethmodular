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


# Per-configuration readouts the gallery's guardrail editor can write rules on, beyond the
# curve's own intercept / slope / held-out R². Gallery key: source keys (first present wins).
# A key no row carries is left out, so each export offers only what its search recorded.
EXTRA_FIELDS = {
    "r2": ("R2", "all_R2"),
    "rmse": ("RMSE",),
    "qres": ("q_residual_pct",),
    "extrap": ("extrap_pct",),
    "neg": ("negative_pct",),
    "above8": ("above_8_pct",),
    "predMedian": ("prediction_median", "all_median_pred"),
    "seasonSpan": ("group_median_span",),
    "interceptMac6": ("deming_intercept_mac6",),
    "trainN": ("train_n",),
    "trainSites": ("train_sites",),
    "drySlope": ("dry_slope",), "dryIntercept": ("dry_intercept",),
    "belgSlope": ("belg_slope",), "belgIntercept": ("belg_intercept",),
    "kiremtSlope": ("kiremt_slope",), "kiremtIntercept": ("kiremt_intercept",),
}


def extra_columns(rows: list[dict]) -> dict[str, list]:
    """One column per EXTRA_FIELDS key that at least one row carries, in row order."""
    out = {}
    for key, sources in EXTRA_FIELDS.items():
        col = [next((r[s] for s in sources if r.get(s) is not None), None) for r in rows]
        if any(v is not None for v in col):
            out[key] = [rnd(v) for v in col]
    return out


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
        "olsIntercept": [rnd(r.get("ols_intercept")) for r in rows],
        "olsSlope": [rnd(r.get("ols_slope")) for r in rows],
        "heldout": [rnd(r.get("heldout_R2")) for r in rows],  # 4 dp: 0.8498 must not read as 0.85 against the floor
        "cut": [r.get("cutoff") for r in rows],
        "k": [r.get("k") for r in rows],
        "categories": cats,
        **codes,
        "extra": extra_columns(rows),
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    print(f"spec curve: {len(rows):,} Addis specifications, {sum(ok)} pass every guardrail")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
