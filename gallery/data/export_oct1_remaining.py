"""Export the remaining 1 Oct 2026 items for the gallery's "Satoshi 2 Oct" tab.

Run with:  uv run --no-sync python gallery/data/export_oct1_remaining.py

Reads only the saved outputs of
  research/ftir_hips_chem/workflows/run_oct1_remaining_20261007.py
and writes gallery/app/public/data/satoshi/oct1_remaining_20261007.json. Nothing
is refitted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_satoshi_followup import GROUPS, SEASON_COLORS, SEASONS, columns, records

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/ftir_hips_chem/output/tables/oct1_remaining_20261007"
OUT = ROOT / "gallery/app/public/data/satoshi"


def sig(v, n=4):
    """Significant figures, for quantities near 1e-10 that decimal rounding would zero."""
    return None if not np.isfinite(v) else float(f"{v:.{n}g}")


def sig_columns(frame):
    return {c: ([sig(v) for v in frame[c]] if pd.api.types.is_float_dtype(frame[c])
                else frame[c].tolist()) for c in frame.columns}


def repeat_summary(rep):
    rows = []
    for (b, g), d in rep.groupby(["baseline", "group"], sort=False):
        row = {"baseline": b, "group": g, "n": len(d)}
        for m in ("addis_slope", "addis_intercept", "improve_slope", "improve_intercept", "k"):
            v = d[m].dropna()
            row[f"{m}_median"] = float(v.median()) if len(v) else None
            row[f"{m}_p10"] = float(np.percentile(v, 10)) if len(v) else None
            row[f"{m}_p90"] = float(np.percentile(v, 90)) if len(v) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((SRC / "summary.json").read_text())
    read = lambda n: pd.read_csv(SRC / f"{n}.csv")  # noqa: E731
    rep = read("blank_repeats")
    ladder = read("blank_fit_ladder")
    keep = ["model_id", "family", "baseline", "group", "stage", "eval_group", "k", "n", "R2", "RMSE", "bias",
            "deming_slope", "deming_intercept", "loading_beyond_train_p95", "t2_beyond_p95", "q_beyond_p95"]
    groups = json.loads((SRC / "groups_spectra.json").read_text())
    payload = {
        "schema_version": 1,
        "generated_from": str(SRC.relative_to(ROOT)),
        "summary": summary,
        "season_colors": SEASON_COLORS, "groups": GROUPS, "seasons": SEASONS,
        "banks": records(read("blank_banks")),
        "rank_cv": sig_columns(read("blank_rank_cv")),
        "blank_residuals": records(read("blank_residuals").assign(
            rms_median=lambda d: d.rms_median.map(sig), rms_p90=lambda d: d.rms_p90.map(sig)), 14),
        "ladder": records(ladder[[c for c in keep if c in ladder]]),
        "vip_bands": records(read("blank_vip_bands")),
        "overlap": records(read("blank_analog_overlap")),
        "repeats": records(repeat_summary(rep)),
        "group_tests": records(read("groups_tests")),
        "group_summary": records(read("groups_summary")),
        "group_members": columns(read("groups_members"), 4),
        "group_spectra": {"wn": {m: [round(v, 2) for v in w] for m, w in groups["wn"].items()},
                          "median": groups["median"], "regions": groups["regions"]},
    }
    path = OUT / "oct1_remaining_20261007.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
