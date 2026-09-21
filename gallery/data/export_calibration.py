#!/usr/bin/env python3
"""Export the phase-3 calibration grid into JSON the React gallery can chart.

Source: `calibration_explorer/cache/batch_results.jsonl` — one row per
(configuration × evaluation target × k) produced by the explorer's server
batches (dense cutoff sweeps 2026-08-20, five-site grid 2026-08-23, and later).
Every number is the explorer's own readout (same shared-script estimators as
the notebooks); nothing is recomputed here.

Writes gallery/app/public/data/calibration.json:
  grid     one row per configuration × target at the *rule* k (k == auto_k),
           the number the explorer would report without a manual k
  sweeps   every k for the Addis site-held-out rows, so the k-selection
           ("optimization") curve can be drawn per configuration
  docs     the explorer's dated result write-ups, title + first paragraph

Run:  python gallery/data/export_calibration.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPLORER = REPO / "calibration_explorer"
SRC = EXPLORER / "cache" / "batch_results.jsonl"
OUT = REPO / "gallery" / "app" / "public" / "data" / "calibration.json"

SCRIPTS = REPO / "research" / "ftir_hips_chem" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import config  # noqa: E402

# Site colours come from config.SITES so the calibration tab matches every
# other tab. Bishoftu (ETBI) is not a config site; it gets a fixed colour here.
SITE_COLOR_BY_CODE = {v["code"]: v["color"] for v in config.SITES.values()}
SITE_COLOR_BY_CODE.setdefault("ETBI", "#8e44ad")


def literal_dict(source: str, name: str) -> dict:
    """Pull a top-level `NAME = {...}` literal out of app.py without importing it
    (importing loads Flask and the spectra)."""
    # single-line literal first, then a multi-line one closed by a line that is just "}"
    m = re.search(rf"^{name}\s*=\s*(\{{[^\n]*\}})\s*$", source, re.M)
    if not m:
        m = re.search(rf"^{name}\s*=\s*(\{{.*?^\}})", source, re.S | re.M)
    if not m:
        return {}
    return ast.literal_eval(m.group(1))


def compact(r: dict) -> dict:
    """Short keys: 80k rows × long keys is most of the file."""
    g = r.get
    return {
        "co": r["cohort"], "cut": r.get("cutoff"), "sel": r.get("selection_space"),
        "sp": r["spectra"], "mo": r["mode"], "tg": r["target"],
        "lot": r.get("lot", "all"), "el": r.get("eval_lot", "all"),
        "k": r.get("k"), "ak": r.get("auto_k"),
        # fixed evaluation set (Addis fixed-190 or the target's default subset)
        "om": g("ols_slope"), "ob": g("ols_intercept"),
        "dm": g("deming_slope"), "db": g("deming_intercept"),
        "ob6": g("ols_intercept_mac6"), "db6": g("deming_intercept_mac6"),
        "r2": g("R2"), "rmse": g("RMSE"),
        # all pairs
        "aom": g("all_ols_slope"), "aob": g("all_ols_intercept"),
        "adm": g("all_deming_slope"), "adb": g("all_deming_intercept"),
        "ar2": g("all_R2"),
        # held-out IMPROVE TOR test of the calibration itself
        "ho": g("heldout_R2"),
        "n": g("n_eval") or g("n"),
    }


def first_paragraph(md: str) -> str:
    body = md.split("\n", 1)[1] if md.startswith("#") else md
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if p and not p.startswith(("#", "|", "```", ">", "-", "*")):
            return re.sub(r"\s+", " ", p)[:600]
    return ""


def main():
    if not SRC.exists():
        print(f"! {SRC} not found — run a server batch in the calibration explorer first")
        return
    app_src = (EXPLORER / "app.py").read_text()
    cohorts = literal_dict(app_src, "COHORTS")
    spectra_label = literal_dict(app_src, "SPECTRA_LABEL")
    default_cutoff = literal_dict(app_src, "DEFAULT_CUTOFF")
    registry = json.loads((EXPLORER / "target_registry.json").read_text()) if (EXPLORER / "target_registry.json").exists() else {}

    grid, sweeps = [], []
    seen_grid = set()
    n_rows = 0
    with SRC.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_rows += 1
            if r.get("error"):
                continue
            c = compact(r)
            key = (c["co"], c["cut"], c["sel"], c["sp"], c["mo"], c["tg"], c["lot"], c["el"])
            # Rule-k row: later batches can re-run a configuration; keep the last.
            if c["k"] == c["ak"]:
                if key in seen_grid:
                    grid = [x for x in grid if (x["co"], x["cut"], x["sel"], x["sp"], x["mo"], x["tg"], x["lot"], x["el"]) != key]
                seen_grid.add(key)
                grid.append(c)
            if c["tg"] == "addis" and c["mo"] == "site_heldout" and c["lot"] == "all" and c["el"] == "all":
                sweeps.append({k: c[k] for k in ("co", "cut", "sel", "sp", "k", "ak", "om", "ob", "dm", "db", "r2", "ho")})

    # Dedupe sweeps on (config, k), last wins.
    dd = {}
    for s in sweeps:
        dd[(s["co"], s["cut"], s["sel"], s["sp"], s["k"])] = s
    sweeps = sorted(dd.values(), key=lambda s: (s["co"], s["cut"] or 0, s["sel"], s["sp"], s["k"]))

    targets = {}
    for tid, t in registry.items():
        code = t.get("site_code", "")
        targets[tid] = {
            "id": tid, "label": t.get("display_name", tid), "site": t.get("physical_site", tid),
            "code": code, "color": SITE_COLOR_BY_CODE.get(code, "#7b8794"),
            "role": t.get("role"), "provisional": bool(t.get("provisional")),
        }
    for tg in {c["tg"] for c in grid}:
        targets.setdefault(tg, {"id": tg, "label": tg, "site": tg, "code": tg.upper(), "color": "#7b8794", "role": None, "provisional": False})

    docs = []
    for md in sorted(EXPLORER.glob("*_20??-??-??.md")):
        text = md.read_text()
        title = text.split("\n", 1)[0].lstrip("# ").strip()
        m = re.search(r"(\d{4}-\d{2}-\d{2})", md.name)
        docs.append({"file": f"calibration_explorer/{md.name}", "title": title, "date": m.group(1) if m else None, "summary": first_paragraph(text)})
    docs.sort(key=lambda d: d["date"] or "", reverse=True)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "calibration_explorer/cache/batch_results.jsonl",
        "n_source_rows": n_rows,
        "cohorts": cohorts,
        "spectra": spectra_label,
        "modes": {"site_heldout": "A · site-grouped 5-fold, first major minimum",
                  "app": "B · interleaved 10-fold, within 5 %",
                  "app_fmm": "B2 · interleaved 10-fold, first major minimum"},
        "default_cutoff": default_cutoff,
        "mac_value": config.MAC_VALUE,
        "deming_lambda_mac10": 2.96,
        "slope_box": [0.85, 1.18],
        "heldout_floor": 0.85,
        "targets": targets,
        "grid": grid,
        "sweeps": sweeps,
        "docs": docs,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    by_target = Counter(c["tg"] for c in grid)
    by_mode = Counter(c["mo"] for c in grid)
    print(f"read {n_rows:,} rows")
    print(f"grid: {len(grid):,} rule-k rows  by target {dict(by_target)}  by mode {dict(by_mode)}")
    print(f"sweeps: {len(sweeps):,} Addis site-held-out rows across {len({(s['co'], s['cut'], s['sel'], s['sp']) for s in sweeps})} configurations")
    print(f"docs: {len(docs)}")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
