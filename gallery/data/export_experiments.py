"""Export experiments_20261022.ipynb for the gallery's Experimental tab (tests 3-6, quick fixes, Addis TOR study).

Run with:  uv run --no-sync python gallery/data/export_experiments.py [--results DIR]

Reads one run of ``research/ftir_hips_chem/experiments_20261022.ipynb`` from
``<Drive>/Aethmodular VIBES/decision_workbook_20261022/experiments/<run>`` (the newest by default) and writes one file
per subpage under ``gallery/app/public/data/experimental/``:

  analogs.json      test 3: local calibrations with shape, shape + loading and loading-only neighbours
  attribution.json  test 4: preprocessing × window × pool × k grid, factorial effects
  selector.json     test 5: error-predicting selector, abstention, winner classifier, oracle
  forward.json      test 6: later-year, new-site and random splits by k
  fixes.json        ETAD-0241/0248 exclusion check (read from the current exports) and the February calendar check
  tor.json          Adama TOR study: the pilot against IMPROVE, the SPARTAN network and SPARTAN Addis, where it would need
                    to sit (targets, volume, season), nearest TOR sites, precision, draft contract
  plan.json         status and one-line headline per test (also from experimental/flag_detection.json)

Nothing is refitted here; every number is the notebook's.
"""
from __future__ import annotations

import argparse
import gzip
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
DATA = ROOT / "gallery/app/public/data"
OUT = DATA / "experimental"


def latest_run() -> Path:
    runs = sorted(p for p in (HOME / "experiments").iterdir() if (p / "e8_contract.json").exists())
    if not runs:
        raise FileNotFoundError(f"no complete experiments run under {HOME / 'experiments'}; execute experiments_20261022.ipynb first")
    return runs[-1]


def clean(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(f"{v:.5g}")
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def columns(df: pd.DataFrame, digits: int = 4) -> dict[str, list]:
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = [bool(v) for v in s]
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = [None if not np.isfinite(v) else float(f"{v:.{digits}g}") for v in s.astype(float)]
        else:
            out[c] = [None if (isinstance(v, float) and np.isnan(v)) else str(v) for v in s]
    return out


def write(name: str, payload: dict, run: Path, files: list[str]):
    payload = dict(schema_version=1, generated=datetime.now(timezone.utc).isoformat(), source=str(run.relative_to(drive_root())),
                   source_sha256={f: hashlib.sha256((run / f).read_bytes()).hexdigest() for f in files}, **payload)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"  {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")
    return payload


def etad_check() -> dict:
    """Where ETAD-0241 and ETAD-0248 stand in each current export (the 22 Oct exclusion fix)."""
    out = {}
    ref = json.loads((DATA / "spartan_reference.json").read_text())
    out["spartan_reference.json"] = {r["id"]: r["excluded"] for r in ref["rows"] if r["id"] in ("ETAD-0241", "ETAD-0248")}
    focus = json.loads((DATA / "filters.json").read_text())
    out["filters.json"] = {r["id"]: r.get("excluded", False) for r in focus["rows"] if r["id"] in ("ETAD-0241", "ETAD-0248")}
    idx = json.loads((DATA / "spartan_network/index.json").read_text())
    etad = next(s for s in idx["sites"] if s["code"] == "ETAD")
    rows = json.loads(gzip.decompress((DATA / etad["filters"]).read_bytes()))["rows"]
    out["spartan_network (ETAD site file)"] = {r["id"]: r["excluded"] for r in rows if r["id"] in ("ETAD-0241", "ETAD-0248")}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, help="one experiments results folder (default: the newest complete run)")
    run = p.parse_args().results or latest_run()
    t = {f.stem: pd.read_csv(f) for f in sorted(run.glob("e*.csv"))}
    settings = json.loads((run / "settings.json").read_text())
    contract = json.loads((run / "e8_contract.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"from {run.name}:")

    # ---------------------------------------------------------------- 3 analogs
    analogs = write("analogs", dict(settings=settings, improve=records(t["e3_improve"]), addis=records(t["e3_addis"]),
                                    addis_filters=columns(t["e3_addis_filters"]), improve_filters=columns(t["e3_improve_filters"])),
                    run, ["e3_improve.csv", "e3_addis.csv", "e3_addis_filters.csv", "e3_improve_filters.csv"])

    # ---------------------------------------------------------------- 4 attribution
    attribution = write("attribution", dict(settings=settings, grid=records(t["e4_grid"]), effects=records(t["e4_effects"]),
                                            published=records(t["e4_published"])[0]), run, ["e4_grid.csv", "e4_effects.csv", "e4_published.csv"])

    # ---------------------------------------------------------------- 5 selector
    selector = write("selector", dict(settings=settings, strategies=records(t["e5_strategies"]), coverage=records(t["e5_coverage"]),
                                      calibration=records(t["e5_calibration"]), picks=records(t["e5_picks"]), filters=columns(t["e5_filters"])),
                     run, ["e5_strategies.csv", "e5_coverage.csv", "e5_calibration.csv", "e5_picks.csv", "e5_filters.csv"])

    # ---------------------------------------------------------------- 6 forward
    forward = write("forward", dict(settings=settings, curves=records(t["e6_curves"]), summary=records(t["e6_summary"]), years=records(t["e6_years"])),
                    run, ["e6_curves.csv", "e6_summary.csv", "e6_years.csv"])

    # ---------------------------------------------------------------- quick fixes
    fixes = write("fixes", dict(settings=settings, seasons=records(t["e7_seasons"]), february=records(t["e7_february"].assign(date=lambda d: d.date.astype(str))),
                                etad=etad_check()), run, ["e7_seasons.csv", "e7_february.csv"])

    # ---------------------------------------------------------------- Addis TOR study
    smap = {s["code"]: s for s in json.loads((DATA / "improve_map.json").read_text())["sites"]}
    spartan = {s["code"]: s for s in json.loads((DATA / "spartan_map.json").read_text())["sites"]}
    net = t["e8_network"]
    net["lat"] = [smap.get(s, {}).get("lat") for s in net.site]
    net["lon"] = [smap.get(s, {}).get("lon") for s in net.site]
    net["name"] = [smap.get(s, {}).get("name", s) for s in net.site]
    for col, val in (("lat", spartan["ETAD"]["lat"]), ("lon", spartan["ETAD"]["lon"]), ("name", "Addis Ababa (SPARTAN)")):
        net.loc[net.site.eq("ETAD"), col] = val
    ad = t["e8_adama_pairs"]
    adama_latlon = [float(ad.lat.median()), float(ad.lon.median())]           # the sampler's GPS position (lab carbon file)
    for col, val in (("lat", adama_latlon[0]), ("lon", adama_latlon[1]), ("name", "Adama (CSU AMOD pilot)")):
        net.loc[net.site.eq("ADAMA"), col] = val
    sp = net.population.eq("SPARTAN network")
    net.loc[sp, "lat"] = [spartan.get(c, {}).get("lat") for c in net.site[sp]]
    net.loc[sp, "lon"] = [spartan.get(c, {}).get("lon") for c in net.site[sp]]
    net.loc[sp, "name"] = [spartan.get(c, {}).get("name", c) for c in net.site[sp]]
    pts = t["e8_points"]
    # HIPS BC per filter where the run has it (runs before 2026-10-28 carried it for Addis only, and it was not exported)
    keep = ["population", "site", "id", "EC µg/filter", "EC µg/m³", "PM2.5 µg/filter", "OC/EC", "distance percentile", "volume m³", "month"] + [c for c in ["HIPS BC µg/m³"] if c in pts]
    tor = write("tor", dict(settings=settings, contract=contract, network=records(net), points=columns(pts[keep]),
                            neighbour_sites=records(t["e8_neighbour_sites"].assign(name=lambda d: d.site.map(lambda s: smap.get(s, {}).get("name", s)),
                                                                                   lat=lambda d: d.site.map(lambda s: smap.get(s, {}).get("lat")),
                                                                                   lon=lambda d: d.site.map(lambda s: smap.get(s, {}).get("lon")))),
                            precision=records(t["e8_precision"]), pilot=records(t["e8_pilot"]),
                            adama_pairs=records(t["e8_adama_pairs"].drop(columns=["pairing_status"]).assign(date=lambda d: d.date.astype(str))),
                            adama_latlon=adama_latlon, target=records(t["e8_target"]), volume=records(t["e8_volume"]), seasons=records(t["e8_seasons"])),
              run, ["e8_network.csv", "e8_points.csv", "e8_neighbour_sites.csv", "e8_precision.csv", "e8_pilot.csv", "e8_adama_pairs.csv", "e8_contract.json",
                    "e8_target.csv", "e8_volume.csv", "e8_seasons.csv"])

    # ---------------------------------------------------------------- plan: status and headlines
    items = headlines(analogs, attribution, selector, forward, fixes, tor)
    (OUT / "plan.json").write_text(json.dumps(dict(schema_version=1, generated=datetime.now(timezone.utc).isoformat(), items=items), ensure_ascii=False))
    print(f"  {(OUT / 'plan.json').relative_to(ROOT)}")


def headlines(analogs, attribution, selector, forward, fixes, tor) -> dict:
    items = {}
    flag_path = OUT / "flag_detection.json"
    if flag_path.exists():
        f = json.loads(flag_path.read_text())
        L = {r["rung"]: r for r in f["ladder"]}
        d, load = L.get("Distance: T² + Q"), L.get("Loading: weighed PM2.5")
        items["flag"] = {"status": "done", "headline": f"Distance alone AUC {d['AUC']:.2f} vs loading {load['AUC']:.2f}; best usable flag catches "
                                                        f"{(f['counts']['impacted'] - f['counts']['missed']) / f['counts']['impacted']:.0%} of failures at a 10 % flag rate; "
                                                        "Addis is the least-flagged SPARTAN site."}
    A = {r["model"]: r for r in analogs["addis"]}
    pub, sl = A["SPARTAN published"], A["local: shape + loading"]
    items["analogs"] = {"status": "done", "headline": f"Addis intercept vs HIPS: shape + loading {sl['intercept']:+.2f}, shape {A['local: shape']['intercept']:+.2f}, "
                                                      f"published {pub['intercept']:+.2f} µg/m³ (PM ÷ neighbours {sl.get('PM ÷ neighbours (median)', float('nan')):.1f}×)."}
    e = pd.DataFrame(attribution["effects"])
    if len(e):
        top = e.reindex(e["Addis intercept"].abs().sort_values(ascending=False).index).iloc[0]
        items["attribution"] = {"status": "done", "headline": f"Largest effect on the Addis intercept: {top['term']} ({top['contrast']}), {top['Addis intercept']:+.2f} µg/m³."}
    s = {r["strategy"]: r for r in selector["strategies"]}
    best_single = min((s[k] for k in ("full spline", "full raw", "local raw")), key=lambda r: r["RMSE"])
    items["selector"] = {"status": "done", "headline": f"Selector RMSE {s['error-predicting selector']['RMSE']:.2f} vs best single model {best_single['RMSE']:.2f} "
                                                       f"({best_single['strategy']}); worst error {s['error-predicting selector']['max |error|']:.1f} vs "
                                                       f"{s['winner classifier']['max |error|']:.1f} µg for the winner classifier."}
    fs = pd.DataFrame(forward["summary"])
    sp = fs[fs.preprocessing == "spline"].set_index("split")
    items["forward"] = {"status": "done", "headline": f"Spline RMSE: later year {sp.loc[sp.index.str.startswith('later'), 'RMSE'].iloc[0]:.2f}, new sites "
                                                      f"{sp.loc[sp.index.str.startswith('new sites ('), 'RMSE'].iloc[0]:.2f}, random {sp.loc[sp.index.str.startswith('random'), 'RMSE'].iloc[0]:.2f} µg/filter."}
    se = pd.DataFrame(fixes["seasons"])
    pubs = se[se.model == "SPARTAN published"].pivot(index="season", columns="calendar", values="slope")
    etad_ok = all(v.get("ETAD-0241") is False for v in fixes["etad"].values() if "ETAD-0241" in v)
    items["fixes"] = {"status": "done", "headline": f"ETAD-0241 kept in every export: {'yes' if etad_ok else 'no'}. Published slope Belg "
                                                    + ", ".join(f"{c.split(' (')[0]} {pubs.loc['Belg', c]:.2f}" for c in pubs.columns) + "."}
    pr = pd.DataFrame(tor["precision"])
    hw = pr[(pr.pairs == 10) & pr.population.str.startswith("Addis")]["half width (relative bias)"].iloc[0]
    tg = pd.DataFrame(tor["target"]).set_index("axis")
    ec_ug, ocec = tg.loc["EC µg/filter"], tg.loc["OC/EC"]
    items["tor"] = {"status": "partial", "headline": f"Adama: {ec_ug['Adama median']:.1f} µg EC per filter against {ec_ug['Addis median']:.0f} at SPARTAN Addis, "
                                                     f"OC/EC {ocec['Adama median']:.1f} against {ocec['Addis median']:.1f}, one season only; "
                                                     f"10 pairs know a mean bias to about ±{hw:.0%}. Pass rules drafted."}
    return items


if __name__ == "__main__":
    main()
