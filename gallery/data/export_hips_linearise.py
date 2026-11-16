"""What "HIPS linearised" means at Addis: the IMPROVE bend, the inversion, and what it does to the FTIR EC fit.

Run from the repo root with the repo environment:

    uv run python gallery/data/export_hips_linearise.py

The data behind the Meeting-tab page "Linearising HIPS at Addis" (gallery/app/src/pages/
HipsLinearisePage.tsx). Nothing is refitted here: the per-filter table is the 10 Nov closure workflow's own
(research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py, load()), and every summary
(IMPROVE strata and ceilings, the Addis fits, the network exponents, season and lot ratios) is read from the
summary that workflow committed, docs/addis-reference-closure-2026-11-10.json. The page linearises each
filter in the browser at whichever ceiling the reader picks, with the same formula as reference_closure.linearise:

    tau_lin = tau / (1 - tau / tau_inf)

All inputs are committed (the closeout, filters.json, the unified filter dataset, the closure summary); no
Drive and no aethalometer data are needed.

Writes gallery/app/public/data/hips_linearise/linearise.json. With --improve-groups, exports just the
group members from the local TOR and HIPS tables, under the private research data folder.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/workflows"), str(ROOT / "research/ftir_hips_chem/scripts")]

import run_addis_reference_closure_20261110 as closure  # noqa: E402

OUT = ROOT / "gallery/app/public/data/hips_linearise"
GROUP_OUT = OUT.parent / "research/hips_linearise_groups"
# the ceilings the page offers: the central IMPROVE fit, both ends of its 90 % interval, and the later era
ERA = "2015–2025"


def clean(v):
    """JSON-safe: NaN -> None, numpy scalars -> Python, floats rounded to what the page shows and a bit more."""
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def records(rows) -> list[dict]:
    return [{k: clean(v) for k, v in r.items()} for r in rows]


def export_improve_groups(saved: list[dict]) -> None:
    samples = closure.improve_samples()
    if samples is None:
        raise FileNotFoundError("IMPROVE TOR and HIPS tables are required; set AETHMODULAR_FTIR_LOCAL_DB")
    strata = closure.improve_strata(samples)
    if list(strata["n"]) != [s["n"] for s in saved] or any(
        not np.allclose(strata[field], [s[field] for s in saved], atol=0.00005, rtol=0)
        for field in ("loading_ug", "tau")
    ):
        raise ValueError("Rebuilt IMPROVE groups do not match the saved counts and medians")

    groups = closure.improve_groups(samples)
    GROUP_OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=GROUP_OUT) as staging:
        for key, bin_index in enumerate(strata.index, start=1):
            rows = groups.get_group(bin_index)
            # The analysis pairs TOR and HIPS by site and sampling timestamp.
            dates = rows["date"].dt.strftime("%Y-%m-%d %H:%M:%S").str.replace(" 00:00:00$", "", regex=True)
            values = rows[["previous_loading_ug", "loading_ug", "tau"]].round(6).to_numpy().tolist()
            payload = {
                "lower": closure.IMPROVE_EDGES[bin_index - 1],
                "upper": closure.IMPROVE_EDGES[bin_index] if bin_index < len(closure.IMPROVE_EDGES) else None,
                "rows": [[site, date, *numbers] for site, date, numbers in zip(rows["Site"], dates, values, strict=True)],
            }
            (Path(staging) / f"{key}.json.gz").write_bytes(
                gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0))
        for file in Path(staging).iterdir():
            file.replace(GROUP_OUT / file.name)
    print(f"wrote {len(strata)} IMPROVE groups with {strata['n'].sum():,} samples")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--improve-groups", action="store_true", help="Export the private IMPROVE group members")
    args = parser.parse_args()
    summary = json.loads(closure.SUMMARY.read_text())
    strata = [r for r in summary["improve_strata"] if r["population"] == closure.IMPROVE_PRIMARY]
    if args.improve_groups:
        export_improve_groups(strata)
        return 0
    table, closeout = closure.load()

    filters = pd.DataFrame({
        "id": table.index, "season": table["season"], "lot": table["lot"].map(lambda v: None if pd.isna(v) else str(int(v))),
        "tau": table["tau"], "volume_m3": table["volume_m3"], "volume_known": table["volume_known"],
        "hips_bc": table["hips_bc"], "ftir": table[closure.PUBLISHED], "pm25": table["pm25"],
    }).dropna(subset=["tau", "hips_bc"])

    ceilings = pd.DataFrame(summary["improve_ceiling"])
    primary = ceilings[ceilings.population.eq(closure.IMPROVE_PRIMARY)].iloc[0]
    era = ceilings[ceilings.population.eq(ERA)].iloc[0]

    raw = next(r for r in summary["raw_hips"] if r["model"] == closure.LABEL[closure.PUBLISHED])
    linearised = [r for r in summary["linearised_hips"] if r["model"] == closure.LABEL[closure.PUBLISHED]]
    network = [r for r in summary["network_exponents"] if r.get("hips_exponent_vs_ftir_ec") is not None]

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py (load) and docs/addis-reference-closure-2026-11-10.json",
        "deposit_area_cm2": closure.DEPOSIT_AREA_CM2,
        "mac": float(closure.MAC_VALUE),
        "lambda": float(closeout["settings"]["LAMBDA_ADDIS"]),
        # tau per ug of EC on light IMPROVE filters: the unsaturated Fabs / EC as a per-filter slope
        "a_per_ug": float(primary["unsaturated_fabs_over_ec"]) / (100.0 * closure.DEPOSIT_AREA_CM2),
        "ceilings": [
            {"key": "central", "label": f"τ∞ {primary['tau_inf']:.2f} (central)", "tau_inf": primary["tau_inf"]},
            {"key": "lo", "label": f"{primary['tau_inf_lo']:.2f} (90 % low)", "tau_inf": primary["tau_inf_lo"]},
            {"key": "hi", "label": f"{primary['tau_inf_hi']:.2f} (90 % high)", "tau_inf": primary["tau_inf_hi"]},
            {"key": "era", "label": f"{era['tau_inf']:.2f} ({ERA})", "tau_inf": era["tau_inf"]},
        ],
        "improve_ceilings": records(summary["improve_ceiling"]),
        "loading_edges_ug": closure.IMPROVE_EDGES,
        "strata": records(strata),
        "filters": records(filters.to_dict("records")),
        "fit_measured": {"slope": raw["slope_vs_hips"], "intercept": raw["intercept_vs_hips"], "n": raw["n"]},
        "fits_linearised": records(linearised),
        "network": records({"site": r["site"], "code": r["code"], "n": r["n"], "median_optical_depth": r["median_optical_depth"],
                            "exponent": r["hips_exponent_vs_ftir_ec"], "lo": r["lo"], "hi": r["hi"],
                            "predicted": r["predicted_from_improve"]} for r in network),
        "season_lot": records(summary["season_lot"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "linearise.json").write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    with_ftir = int(filters["ftir"].notna().sum())
    print(f"wrote {OUT / 'linearise.json'}: {len(filters)} Addis filters ({with_ftir} with published FTIR EC), "
          f"{len(strata)} IMPROVE strata, {len(network)} network sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
