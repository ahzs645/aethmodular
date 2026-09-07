"""Re-score an existing explorer leaderboard on alternative evaluation views.

The fit cache is view-agnostic (`app.py::_eval_view`): narrowing the readout to a
lot, a season, or an equal-n half costs a regression on <=250 points, not a
refit. So every configuration already in `cache/batch_results.jsonl` can be
re-read on a blind half for effectively nothing - which is what makes Ann's
2026-08-27 question ("pick the model on half the filters, predict the other
half") answerable against the *whole* search rather than a handful of finalists.

    python research/ftir_ec_phase3/scripts/rescore_leaderboard.py \
        --views all,early,late,odd,even --out rescored.json

Requires the explorer running on :5058 (calibration_explorer/run.sh).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:5058"
RELAX_MISSING = False
RESULTS = Path(__file__).resolve().parents[3] / "calibration_explorer" / "cache" / "batch_results.jsonl"

# the explorer's default objective and guardrail box
WEIGHT, MIN_R2, SLOPE_MIN, SLOPE_MAX = 5.0, 0.85, 0.7, 1.3
MAX_EXTRAP, MAX_Q, MAX_NEG = 30.0, 30.0, 10.0
# eval_lot BELONGS here: Addis carries cached rows for both eval_lot="all" and
# eval_lot="251", and dropping it from the config admitted both as separate
# leaderboard entries that then re-ran identically at "all" - 313 phantom
# duplicates against 398 real configurations, which flattered the rank-1-to-
# rank-10 gap from 0.208 to 0.450.
CFG_FIELDS = ("cohort", "cutoff", "selection_space", "spectra", "mode", "lot",
              "eval_lot")


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return json.load(f)
    except urllib.error.HTTPError as e:
        return json.load(e)


def score(intercept, slope, weight=WEIGHT):
    return abs(intercept) + weight * abs(slope - 1.0)


def passes(heldout_r2, slope, extrap, q, negative, relax_missing=False):
    """The app's guardrail box.

    A MISSING domain diagnostic is not a passing one. Roughly 80% of the saved
    rows predate the extrapolation / Q / negative-prediction fields, and scoring
    an absent diagnostic as 0 (the original bug here) let 711 Addis rows through
    where the app itself admits 11. `relax_missing` reproduces the old, wrong
    behaviour on purpose, for comparing against pre-backfill numbers only.
    """
    if heldout_r2 is None or heldout_r2 < MIN_R2:
        return False
    if not SLOPE_MIN <= slope <= SLOPE_MAX:
        return False
    for value, limit in ((extrap, MAX_EXTRAP), (q, MAX_Q), (negative, MAX_NEG)):
        if value is None:
            if not relax_missing:
                return False          # never vetted: run /api/batch_backfill
            continue
        if value > limit:
            return False
    return True


def metric(out, evaluation_set="fixed", mac=10.0):
    """The requested readout row, falling back to all-pairs where it must.

    Only the built-in Addis target has a `fixed` subset (the deployed-EC 190).
    At etbi/chts/indh/uspa `fixed_mask` is empty, so insisting on `fixed` here
    silently dropped every row for those sites. Fall back to all-pairs and
    RECORD which readout was used, because an all-pairs number is not comparable
    with a fixed-set one and must not be tabulated as if it were.
    """
    for want in (evaluation_set, "all"):
        rows = [m for m in out.get("metrics", [])
                if m["evaluation_set"] == want and m["MAC"] == mac]
        if rows:
            return {**rows[0], "readout": want}
    return None


def load_candidates(target, limit):
    rows = []
    with RESULTS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("target") != target or r.get("eval_split") not in (None, "all"):
                continue
            if not passes(r.get("heldout_R2"), r.get("deming_slope"),
                          r.get("extrap_pct"), r.get("q_residual_pct"),
                          r.get("negative_pct"), relax_missing=RELAX_MISSING):
                continue
            rows.append(r)
    rows.sort(key=lambda r: score(r["deming_intercept"], r["deming_slope"]))
    return rows[:limit] if limit else rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="addis")
    p.add_argument("--views", default="all,early,late,odd,even",
                   help="comma-separated eval_split values, or group:<name> entries")
    p.add_argument("--limit", type=int, default=0, help="0 = every passing row")
    p.add_argument("--out", default="rescored.json")
    p.add_argument("--relax-missing", action="store_true",
                   help="admit rows whose domain diagnostics were never computed "
                        "(reproduces the pre-2026-08-27 behaviour; the counts it "
                        "produces are NOT the app's passing set)")
    a = p.parse_args()
    global RELAX_MISSING
    RELAX_MISSING = a.relax_missing

    views = [v.strip() for v in a.views.split(",") if v.strip()]
    cands = load_candidates(a.target, a.limit)
    print(f"{len(cands)} passing configurations on {a.target}; "
          f"re-scoring on {len(views)} views", flush=True)

    out_rows, t0 = [], time.time()
    for i, r in enumerate(cands, 1):
        cfg = {f: r.get(f) for f in CFG_FIELDS}
        cfg.update(target=a.target, k=r.get("k"))
        rec = {**cfg, "cohort_label": r.get("cohort_label"),
               "auto_k": r.get("auto_k"), "views": {}}
        for v in views:
            body = dict(cfg)
            if v.startswith("group:"):
                body["eval_group"] = v.split(":", 1)[1]
            else:
                body["eval_split"] = v
            o = post("/api/run", body)
            if "error" in o:
                rec["views"][v] = {"error": o["error"]}
                continue
            m = metric(o)
            if m is None:
                rec["views"][v] = {"error": "no fixed/MAC-10 metric row"}
                continue
            rec["views"][v] = {
                "n": o["target"]["n_eval"], "n_fit": m["n"],
                "readout": m.get("readout", "fixed"),
                "slope": m["deming_slope"], "intercept": m["deming_intercept"],
                "R2": m["R2"], "RMSE": m["RMSE"],
                "score": round(score(m["deming_intercept"], m["deming_slope"]), 4),
                "heldout_R2": (o["heldout"] or {}).get("R2"),
                "extrap_pct": o["target"].get("extrap_pct"),
                "q_residual_pct": o["target"].get("q_residual_pct"),
            }
        out_rows.append(rec)
        if i % 50 == 0 or i == len(cands):
            print(f"  {i}/{len(cands)}  ({time.time() - t0:.0f}s)", flush=True)

    Path(a.out).write_text(json.dumps(
        {"target": a.target, "views": views, "weight": WEIGHT,
         "guardrails": {"min_r2": MIN_R2, "slope": [SLOPE_MIN, SLOPE_MAX],
                        "max_extrap": MAX_EXTRAP, "max_q": MAX_Q,
                        "max_negative": MAX_NEG},
         "rows": out_rows}, indent=1))
    print(f"wrote {a.out}  ({len(out_rows)} configurations, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
