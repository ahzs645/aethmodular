"""Re-derive the domain diagnostics for saved leaderboard rows that predate them.

About 80% of `cache/batch_results.jsonl` was written before the extrapolation /
Q-residual / negative-prediction fields existed, and the app refuses to let an
un-vetted row pass its guardrails. `/api/batch_backfill` fixes that for the whole
file, but the whole file is ~6.7 hours: legacy *fit cache* entries also lack the
diagnostics, so each row is a real PLS refit, not a cache hit.

It is also unnecessary. A row already outside the held-out-R2 or slope box can
never pass whatever its diagnostics turn out to be, so only rows inside that box
change any answer - 2,182 of 71,263 here, about twelve minutes.

    python research/ftir_ec_phase3/scripts/backfill_diagnostics.py [--dry-run]

Rewrites the file atomically, preserving row order and every untouched row
byte-for-byte. Requires the explorer running on :5058.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:5058"
RESULTS = (Path(__file__).resolve().parents[3]
           / "calibration_explorer" / "cache" / "batch_results.jsonl")
MIN_R2, SLOPE_MIN, SLOPE_MAX = 0.85, 0.7, 1.3
DIAGNOSTICS = ("extrap_pct", "q_residual_pct", "negative_pct",
               "above_8_pct", "prediction_median", "group_median_span")
CFG = ("cohort", "cutoff", "selection_space", "spectra", "mode", "lot",
       "target", "eval_lot", "eval_group", "eval_split", "group_scheme")


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return json.load(f)
    except urllib.error.HTTPError as e:
        return json.load(e)
    except Exception as exc:                                   # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def needs_backfill(r):
    return (r.get("q_residual_pct") is None
            and r.get("heldout_R2") is not None and r["heldout_R2"] >= MIN_R2
            and SLOPE_MIN <= r.get("deming_slope", 99) <= SLOPE_MAX)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    lines = RESULTS.read_text().splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        rows.append(json.loads(line) if line else None)
    todo = [i for i, r in enumerate(rows) if r and needs_backfill(r)]
    print(f"{len(lines)} rows, {len(todo)} need the diagnostics re-derived")
    if a.dry_run or not todo:
        return

    t0, filled, failed = time.time(), 0, 0
    for n, i in enumerate(todo, 1):
        r = rows[i]
        body = {f: r.get(f) for f in CFG if r.get(f) is not None}
        body["k"] = r.get("k")
        out = post("/api/run", body)
        if "error" in out:
            failed += 1
        else:
            t, pl = out["target"], out.get("plausibility", {})
            r.update(extrap_pct=t.get("extrap_pct"),
                     q_residual_pct=t.get("q_residual_pct"),
                     negative_pct=pl.get("negative_pct"),
                     above_8_pct=pl.get("above_8_pct"),
                     prediction_median=pl.get("median"),
                     group_medians=pl.get("group_medians", {}),
                     group_median_span=pl.get("group_median_span"))
            filled += 1
        if n % 200 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  filled {filled} failed {failed}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    tmp = RESULTS.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for original, r in zip(lines, rows):
            f.write((json.dumps(r) if r is not None else original.strip()) + "\n")
    os.replace(tmp, RESULTS)
    print(f"rewrote {RESULTS.name}: {filled} rows gained diagnostics, "
          f"{failed} failed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
