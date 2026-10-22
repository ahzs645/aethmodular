"""Export the decision workbook's experimental flag study for the gallery's "Experimental" tab.

Run with:  uv run --no-sync python gallery/data/export_experimental.py [--results DIR]

Reads one run of ``research/ftir_hips_chem/decision_workbook_20261022.ipynb`` (Section 6b: what the
"impacted" flag actually sees) from ``<Drive>/Aethmodular VIBES/decision_workbook_20261022/results/<run>``,
the newest run by default, and writes ``gallery/app/public/data/experimental/flag_detection.json``:

  ladder          AUC and paired gain over measured loading for each set of flag inputs (site bootstrap)
  within          AUC inside each IMPROVE loading decile (does distance rank errors at fixed loading?)
  coverage        error among the filters a score keeps, as the kept share falls (with oracle and random)
  site_keep       per held-out IMPROVE site: share kept at the 10 % flag rate, RMSE before and after
  missed          who the best SPARTAN-usable flag lets through (summary and the 30 largest misses)
  map_improve     every IMPROVE filter on the loading × distance plane, with its flag outcome
  map_spartan     Addis (frozen set) and the SPARTAN network on the same plane
  grid            5 × 5 loading × distance cells: IMPROVE failure and catch rates, Addis counts

Nothing is refitted here; every number is the workbook's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from data_paths import drive_root  # noqa: E402

HOME = drive_root() / "Aethmodular VIBES/decision_workbook_20261022"
OUT = ROOT / "gallery/app/public/data/experimental/flag_detection.json"
FILES = ["section6b_ladder", "section6b_within_loading", "section6b_coverage", "section6b_site_keep",
         "section6b_missed_summary", "section6b_missed_top", "section6b_map_improve", "section6b_map_spartan",
         "section6b_grid", "section5_flag_scores"]
CATEGORIES = ["fine", "false alarm", "caught failure", "missed failure"]


def latest_run() -> Path:
    runs = sorted(p for p in (HOME / "results").iterdir() if (p / "section6b_ladder.csv").exists())
    if not runs:
        raise FileNotFoundError(f"no workbook run with Section 6b under {HOME / 'results'}; execute the workbook first")
    return runs[-1]


def clean(v):
    """JSON-safe scalars: NaN/inf -> None, numpy -> python, floats to 4 significant figures."""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(f"{v:.4g}")
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def columns(df: pd.DataFrame, digits: dict[str, int] | None = None) -> dict[str, list]:
    out = {}
    for c in df.columns:
        s = df[c]
        if digits and c in digits and pd.api.types.is_numeric_dtype(s):
            out[c] = [None if not np.isfinite(v) else round(float(v), digits[c]) for v in s.astype(float)]
        else:
            out[c] = [clean(v) if not (isinstance(v, float) and np.isnan(v)) else None for v in s]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, help="one workbook results folder (default: the newest with Section 6b)")
    run = p.parse_args().results or latest_run()
    t = {name: pd.read_csv(run / f"{name}.csv") for name in FILES}
    settings = json.loads((run / "settings.json").read_text())
    names = pd.read_csv(HOME / "bundle/site_seasons.csv").drop_duplicates("code").set_index("code").site_name.to_dict()

    cov = t["section6b_coverage"]
    coverage = {score: {"kept share": g["kept share"].round(2).tolist(),
                        **{c: [clean(v) for v in g[c]] for c in ["RMSE kept", "bias kept", "impacted among kept", "impacted let through"]}}
                for score, g in cov.groupby("score", sort=False)}

    mi = t["section6b_map_improve"]
    mi["category"] = mi.category.map(CATEGORIES.index)
    ms = t["section6b_map_spartan"]
    ms["site"] = ms.code.map(names).fillna(ms.code)
    ms["date"] = ms.date.astype(str)
    ms["season"] = ms.season.fillna("")

    # the base rate and threshold the categories were cut at, recovered from the map itself
    flagged = mi.category.isin([CATEGORIES.index("false alarm"), CATEGORIES.index("caught failure")])
    failed = mi.category.isin([CATEGORIES.index("caught failure"), CATEGORIES.index("missed failure")])
    out = dict(
        schema_version=1,
        generated=datetime.now(timezone.utc).isoformat(),
        source=str(run.relative_to(drive_root())),
        source_sha256={f"{n}.csv": hashlib.sha256((run / f"{n}.csv").read_bytes()).hexdigest() for n in FILES},
        settings={k: settings.get(k) for k in ["MAIN", "ABS_TOL", "REL_TOL", "FLAG_RATE", "BEST", "N_FOLDS", "SEED", "k_selected", "bundle"]},
        counts={"improve": int(len(mi)), "impacted": int(failed.sum()), "flagged": int(flagged.sum()),
                "missed": int((mi.category == CATEGORIES.index("missed failure")).sum()),
                "addis": int((ms.population == "Addis").sum()), "network": int((ms.population == "network").sum()),
                "network_sites": int(ms[ms.population == "network"].code.nunique())},
        categories=CATEGORIES,
        ladder=records(t["section6b_ladder"]),
        flag_scores=records(t["section5_flag_scores"]),
        within=records(t["section6b_within_loading"]),
        coverage=coverage,
        site_keep=records(t["section6b_site_keep"]),
        missed_summary=records(t["section6b_missed_summary"]),
        missed_top=records(t["section6b_missed_top"].assign(date=lambda d: d.date.astype(str))),
        map_improve=columns(mi[["filter_id", "site", "loading percentile", "distance percentile", "category", "error µg", "risk"]],
                            {"loading percentile": 1, "distance percentile": 1, "error µg": 2, "risk": 3}),
        map_spartan=columns(ms[["population", "filter_id", "code", "site", "date", "season", "loading percentile", "distance percentile",
                                "PM2.5 µg", "risk", "flagged", "beyond IMPROVE loading"]],
                            {"loading percentile": 1, "distance percentile": 1, "PM2.5 µg": 1, "risk": 3}),
        grid=records(t["section6b_grid"]),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB) from {run.name}: "
          f"best usable flag '{settings.get('BEST')}', {out['counts']['missed']} of {out['counts']['impacted']} impacted filters missed")


if __name__ == "__main__":
    main()
