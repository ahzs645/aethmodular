#!/usr/bin/env python3
"""Export per-filter readouts and cohort diagnostics from the calibration
explorer for the gallery's Calibration tab.

`export_calibration.py` covers what the batch results summarise (one fit per
configuration). This script covers what the explorer computes *per filter* and
per cohort: the crossplot of predicted EC against the reference, residuals,
the dated series, the blind-half check, the RMSECV curve, the cohort's OC/EC
composition, the selection rankings, cohort and site spectra, the analog lab
and the HIPS York diagnostics.

It imports the explorer (`calibration_explorer/app.py`) in-process, waits for
its data to load, and calls its own HTTP handlers through Flask's test client —
so every number here is exactly what the explorer would show. Nothing is
recomputed.

Run (the explorer's interpreter, which has flask/sklearn/pybaselines):
    ~/anaconda3/bin/python gallery/data/export_calibration_runs.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPLORER = REPO / "calibration_explorer"
OUT = REPO / "gallery" / "app" / "public" / "data" / "calibration_runs.json"

sys.path.insert(0, str(EXPLORER))
sys.path.insert(0, str(REPO / "research" / "ftir_hips_chem" / "scripts"))
import config  # noqa: E402  (site colours, filter-id pattern)

t0 = time.time()
import app as explorer  # noqa: E402  (starts the data-loading thread on import)

# The configurations the readouts are exported for. The first six are the
# explorer's built-in presets (the July-17 setup matrix); the rest are the
# dense-sweep basin and the meeting's k=9 analog run.
PRESETS = [
    {"key": "pool_raw", "label": "Entire IMPROVE network × raw", "cohort": "pool", "cutoff": None, "spectra": "raw"},
    {"key": "smoke_raw", "label": "Biomass-smoke (906) × raw", "cohort": "smoke", "cutoff": None, "spectra": "raw"},
    {"key": "eth300_raw", "label": "Ethiopia-shaped (300) × raw", "cohort": "eth_shaped", "cutoff": 300, "spectra": "raw"},
    {"key": "analog500_raw", "label": "Spectral analogs (500) × raw", "cohort": "analogs", "cutoff": 500, "spectra": "raw"},
    {"key": "ocec800_raw", "label": "Lowest-OC/EC (800) × raw (locked)", "cohort": "ocec", "cutoff": 800, "spectra": "raw"},
    {"key": "ocec800_airspec", "label": "Lowest-OC/EC (800) × AIRSpec (locked)", "cohort": "ocec", "cutoff": 800, "spectra": "airspec"},
    {"key": "ocec440_airspec", "label": "Lowest-OC/EC (440) × AIRSpec (dense-sweep basin)", "cohort": "ocec", "cutoff": 440, "spectra": "airspec"},
    {"key": "ocec450_deriv2", "label": "Lowest-OC/EC (450) × 2nd derivative", "cohort": "ocec", "cutoff": 450, "spectra": "deriv2"},
    {"key": "eth300_airspec", "label": "Ethiopia-shaped (300) × AIRSpec", "cohort": "eth_shaped", "cutoff": 300, "spectra": "airspec"},
]
MODE = "site_heldout"
SITE_COLOR_BY_CODE = {v["code"]: v["color"] for v in config.SITES.values()}
SITE_COLOR_BY_CODE.setdefault("ETBI", "#8e44ad")
BASE_ID = re.compile(getattr(config, "BASE_FILTER_ID_PATTERN", r"^(.*?)(?:-\d+)?$"))


def base_id(fid):
    if fid is None:
        return None
    m = BASE_ID.match(str(fid))
    return m.group(1) if m else str(fid)


def wait_ready(timeout=1800):
    last = None
    while not explorer.STATE["ready"]:
        if explorer.STATE.get("error"):
            raise SystemExit(f"explorer failed to load: {explorer.STATE['error']}")
        msg = explorer.STATE.get("message")
        if msg != last:
            print(f"  [{time.time() - t0:5.0f}s] {msg}")
            last = msg
        if time.time() - t0 > timeout:
            raise SystemExit("explorer did not become ready in time")
        time.sleep(2)
    print(f"explorer ready after {time.time() - t0:.0f}s")


def post(client, path, body):
    r = client.post(path, json=body)
    d = r.get_json()
    if r.status_code != 200 or (isinstance(d, dict) and d.get("error")):
        raise RuntimeError(f"{path} {body}: {r.status_code} {d.get('error') if isinstance(d, dict) else d}")
    return d


def decimate(xs, stride):
    return xs[::stride]


def main():
    wait_ready()
    client = explorer.app.test_client()
    targets = explorer.list_targets(cross_site_only=True)
    print("targets:", list(targets))

    # ---- per-filter runs -------------------------------------------------
    runs = []
    for p in PRESETS:
        cfg = {"cohort": p["cohort"], "cutoff": p["cutoff"], "spectra": p["spectra"],
               "selection_space": "raw", "mode": MODE}
        for tname in targets:
            body = {**cfg, "target": tname}
            try:
                t1 = time.time()
                d = post(client, "/api/run", body)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {p['key']} @ {tname}: {exc}")
                continue
            tgt = explorer.get_target(tname)
            fids = tgt.get("filter_ids")
            ev = d["eval"]
            n = len(ev["pred"])
            ids = [base_id(f) for f in fids] if fids and len(fids) == n else [None] * n
            runs.append({
                "preset": p["key"], "target": tname, "k": d["k"], "auto_k": d["auto_k"],
                "n_cohort": d["n_cohort"], "n_train": d["n_train"], "n_train_sites": d["n_train_sites"],
                "rmsecv_floor": d["rmsecv_floor"], "pct_rmsecv_floor": d["pct_rmsecv_floor"],
                "curve": [{"k": c["n_components"], "rmsecv": c.get("rmsecv"), "se": c.get("rmse_se")} for c in d["curve"]],
                "heldout": d["heldout"],
                "metrics": d["metrics"],
                "split_check": d.get("split_check", []),
                "plausibility": d.get("plausibility"),
                "extrap_pct": d["target"].get("extrap_pct"),
                "q_residual_pct": d["target"].get("q_residual_pct"),
                "ref_kind": d["target"]["ref_kind"],
                "eval": {"id": ids, "ref": ev["ref"], "pred": ev["pred"], "group": ev["group"],
                         "date": ev["date"], "deployed": ev["deployed"], "fixed": ev["fixed"]},
            })
            print(f"  {p['key']:18} @ {tname:6} k={d['k']:2} n={n:4}  {time.time() - t1:5.1f}s")

    # ---- cohort composition (target-independent) --------------------------
    cohorts = {}
    for p in PRESETS:
        try:
            d = post(client, "/api/cohort_info", {"cohort": p["cohort"], "cutoff": p["cutoff"],
                                                   "spectra": p["spectra"], "selection_space": "raw", "mode": MODE})
        except Exception as exc:  # noqa: BLE001
            print(f"  ! cohort_info {p['key']}: {exc}")
            continue
        cohorts[p["key"]] = {k: v for k, v in d.items()}
        print(f"  cohort_info {p['key']:18} n={d.get('n')}")

    # ---- selection rankings ------------------------------------------------
    rankings = []
    for cohort in explorer.RANKED_COHORTS:
        for sel in (("raw", "airspec") if cohort != "ocec" else ("raw",)):
            try:
                d = post(client, "/api/ranking", {"cohort": cohort, "selection_space": sel})
            except Exception as exc:  # noqa: BLE001
                print(f"  ! ranking {cohort}/{sel}: {exc}")
                continue
            rankings.append({"cohort": cohort, "selection_space": sel, **d})
            print(f"  ranking {cohort}/{sel}: {d['n_total']} candidates")

    # ---- overlap ------------------------------------------------------------
    overlap = post(client, "/api/overlap", {}).get("rows", [])
    print(f"  overlap: {len(overlap)} pairs")

    # ---- spectra: cohorts vs Addis, and every site ------------------------
    cohort_spectra = {}
    for space in ("raw", "airspec"):
        try:
            d = post(client, "/api/spectra", {"compare": True, "spectra": space, "selection_space": "raw",
                                              "cohort": "ocec", "cutoff": 800, "mode": MODE, "target": "addis"})
            cohort_spectra[space] = d
            print(f"  cohort spectra {space}: {len(d['wn'])} wavenumbers, {len(d['series'])} cohorts")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! spectra {space}: {exc}")
    site_spectra = {}
    for space in ("raw", "airspec"):
        try:
            d = post(client, "/api/site_spectra", {"space": space})
            series = []
            for s in d["series"]:
                stride = max(1, len(s["wn"]) // 700)
                series.append({"name": s["name"], "label": s["label"], "n": s["n"],
                               "wn": decimate(s["wn"], stride), "median": decimate(s["median"], stride),
                               "q25": decimate(s["q25"], stride), "q75": decimate(s["q75"], stride)})
            site_spectra[space] = {"space": space, "series": series}
            print(f"  site spectra {space}: {len(series)} sites")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! site_spectra {space}: {exc}")

    # ---- analog lab ---------------------------------------------------------
    analog = {}
    for space in ("raw", "airspec"):
        try:
            d = post(client, "/api/analog_lab", {"space": space})
            n = d["n"]
            stride = max(1, n // 3000)
            analog[space] = {
                "space": space, "n": n, "metrics": d["metrics"], "labels": d["labels"],
                "agreement": d["agreement"], "explained": d["explained"],
                "rank_stride": stride,
                "ranks": {k: v[::stride] for k, v in d["ranks"].items()},
                "sample_idx": d["sample_idx"], "pool_xy": d["pool_xy"], "addis_xy": d["addis_xy"],
            }
            print(f"  analog lab {space}: n={n}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! analog_lab {space}: {exc}")

    # ---- HIPS diagnostics ----------------------------------------------------
    hips = {}
    try:
        r = client.get("/api/hips_blanks")
        hips["blanks"] = r.get_json().get("lots")
        d = post(client, "/api/hips_york", {"cohort": "ocec", "cutoff": 450, "spectra": "airspec",
                                            "selection_space": "raw", "mode": MODE, "k": 9})
        hips["york"] = d
        hips["config"] = "ocec-450 × AIRSpec, k=9"
        print(f"  hips york: {len(d.get('rows', []))} sites")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! hips: {exc}")

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "calibration_explorer app (in-process), same handlers as the UI",
        "provenance": explorer.app.test_client().get("/api/status").get_json().get("provenance"),
        "mode": MODE,
        "mac_value": config.MAC_VALUE,
        "presets": PRESETS,
        "targets": {name: {"id": name, "label": label, "site": explorer.target_meta(name).get("physical_site", name),
                           "code": explorer.target_meta(name).get("site_code", ""),
                           "color": SITE_COLOR_BY_CODE.get(explorer.target_meta(name).get("site_code", ""), "#7b8794")}
                    for name, label in targets.items()},
        "runs": runs,
        "cohorts": cohorts,
        "rankings": rankings,
        "overlap": overlap,
        "cohort_spectra": cohort_spectra,
        "site_spectra": site_spectra,
        "analog": analog,
        "hips": hips,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":"), default=str))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
