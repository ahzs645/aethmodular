#!/usr/bin/env python3
"""Export the HIPS Fabs of ftir_36/ftir_39's locked holdout filters, so the Applicability page can follow the MAC.

ftir_39's per-filter table (research/spectral/applicability.json, domain_filters) carries each holdout filter's
residual against HIPS/10 but not its Fabs, and these 40 filters (14 Addis, January-February 2026; 26 Delhi) are in no
shipped HIPS batch: the notebooks reconstructed their Fabs from the raw instrument export. This does the same, with
research/ftir_ec_phase3/scripts/reconstruct_hips_fabs.py's own line schedule (the blank line active on each filter's
analysis date), its site-median deposit area and the filter's sampled volume:

    tau  = ln((Intercept + Slope * R1) / T1)
    Fabs = 100 * tau * DepositArea / Volume

A filter a shipped batch does carry keeps its shipped Fabs.

Check before writing: residual + Fabs/10 is each configuration's prediction, so an OLS fit of it on Fabs/10 must
reproduce the ordinary-PLS slope, intercept and R² ftir_36 saved for that configuration and holdout
(target_summary). The export is refused when any of the six fits misses by more than CHECK_TOL.

Inputs, under the Davis Data folder (data_paths.maia_data_root; AETHMODULAR_MAIA_DATA_ROOT overrides):
  DAVIS/SPARTAN HIPS pulls/spartan_hips_raw_all.csv   raw T1/R1 (data_paths.hips_raw_csv)
  Spartan/SPARTAN_HIPS_Batch1-51.v2.csv               shipped batches: blank lines, deposit area
  DAVIS/ETAD FTIR/ETAD_metadata.csv                   Addis MediaId, filter id, volume (data_paths.etad_dir)
  DAVIS/SPARTAN FTIR pulls/<SITE>/<SITE>_filters.csv  the other sites' MediaId, filter id, volume

Writes gallery/app/public/data/research/spectral/holdout_fabs.json:
  rows  {MediaId: {filter, site, date, lot, line, fabs}}   MediaId is domain_filters' ExternalFilterId
  check the six ordinary-PLS fits rebuilt against ftir_36's

Run:  uv run python gallery/data/export_holdout_fabs.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
from data_paths import etad_dir, hips_raw_csv, maia_data_root  # noqa: E402
from reconstruct_hips_fabs import _active_line, _calibration_schedule, _lot  # noqa: E402

SPECTRAL = ROOT / "gallery/app/public/data/research/spectral"
OUT = SPECTRAL / "holdout_fabs.json"
CHECK_TOL = 2e-3


def read_csv(path: Path, **kw) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kw)
    d.columns = [c.strip('﻿"') for c in d.columns]
    return d


def table(tables: dict, name: str) -> pd.DataFrame:
    return pd.DataFrame(tables[name])


def filter_metadata(data: Path) -> pd.DataFrame:
    """MediaId, filter id, sampling date and volume for every filter the staged pulls know."""
    cols = ["MediaId", "ExternalFilterId", "SamplingStartDate", "SampleVolume_m3"]
    parts = [read_csv(etad_dir() / "ETAD_metadata.csv")[cols]]
    parts += [read_csv(f)[cols] for f in sorted((data / "DAVIS/SPARTAN FTIR pulls").glob("*/*_filters.csv"))]
    return pd.concat(parts).drop_duplicates("MediaId").set_index("MediaId")


def main() -> None:
    data = maia_data_root()
    spectral = json.loads((SPECTRAL / "applicability.json").read_text())["tables"]
    per_filter, summary = table(spectral, "domain_filters"), table(spectral, "target_summary")
    media = sorted({int(m) for m in per_filter["ExternalFilterId"]})

    raw = read_csv(hips_raw_csv())
    sample = raw[raw["ResultTypeId"].eq(0)].drop_duplicates("ExternalFilterId").set_index("ExternalFilterId")
    shipped = pd.read_csv(data / "Spartan/SPARTAN_HIPS_Batch1-51.v2.csv", encoding="cp1252", low_memory=False).drop_duplicates("FilterId")
    schedule = _calibration_schedule(shipped, sample.reset_index())
    deposit = shipped.groupby("Site")["DepositArea"].median()
    meta = filter_metadata(data)

    rows = {}
    for m in media:
        if m not in meta.index:
            raise SystemExit(f"MediaId {m} is in no staged filter list; cannot name its filter")
        f = meta.loc[m]
        fid, site = f["ExternalFilterId"], f["ExternalFilterId"].split("-")[0]
        official = shipped[shipped["FilterId"].eq(fid)]
        if len(official):
            fabs, lot, line = float(official["Fabs"].iloc[0]), _lot(official["LotId"].iloc[0]), "shipped"
        else:
            x = sample.loc[fid]
            lot = _lot(x["ExternalLotId"])
            intercept, slope, line = _active_line(schedule, lot, x["Timestamp"], shipped)
            tau = float(np.log((intercept + slope * x["Reflectance"]) / x["Transmittance"]))
            fabs = 100.0 * tau * float(deposit[site]) / float(f["SampleVolume_m3"])
        rows[str(m)] = {"filter": fid, "site": site, "date": str(f["SamplingStartDate"])[:10], "lot": lot, "line": line, "fabs": round(fabs, 4)}

    # residual + Fabs/10 is the prediction: its OLS fit on Fabs/10 must be ftir_36's ordinary-PLS fit
    check = []
    for (config, target), g in per_filter.groupby(["config", "target"]):
        x = g["ExternalFilterId"].map(lambda v: rows[str(int(v))]["fabs"]).to_numpy() / 10
        y = g["residual_ugm3"].to_numpy() + x
        slope, intercept = np.polyfit(x, y, 1)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        saved = summary[(summary.config == config) & (summary.target == target) & (summary.variant == "ordinary_pls")].iloc[0]
        miss = max(abs(slope - saved.slope), abs(intercept - saved.intercept), abs(r2 - saved.R2))
        check.append({"config": config, "target": target, "n": len(g), "slope": round(slope, 4), "intercept": round(intercept, 4), "R2": round(r2, 4),
                      "saved_slope": saved.slope, "saved_intercept": saved.intercept, "saved_R2": saved.R2, "max_miss": round(float(miss), 5)})
        print(f"  {config:17s} {target:14s} slope {slope:.4f} ({saved.slope:.4f})  intercept {intercept:+.4f} ({saved.intercept:+.4f})  "
              f"R² {r2:.4f} ({saved.R2:.4f})")
    worst = max(c["max_miss"] for c in check)
    if worst > CHECK_TOL:
        raise SystemExit(f"refused: the rebuilt fits miss ftir_36's by {worst:.4f} (> {CHECK_TOL}); these are not the notebook's Fabs")

    payload = {
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Davis Data: DAVIS/SPARTAN HIPS pulls/spartan_hips_raw_all.csv, Spartan/SPARTAN_HIPS_Batch1-51.v2.csv, "
                  "filter volumes from DAVIS/ETAD FTIR/ETAD_metadata.csv and DAVIS/SPARTAN FTIR pulls/*/*_filters.csv",
        "method": "reconstruct_hips_fabs.py line schedule: tau = ln((Intercept + Slope·R1)/T1), Fabs = 100·tau·DepositArea/Volume "
                  "(site-median deposit area); reconstructed, not official",
        "rows": rows,
        "check": check,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    lines = pd.Series([r["line"] for r in rows.values()]).value_counts().to_dict()
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} holdout filters ({lines}); fits reproduce ftir_36's to {worst:.4f}")


if __name__ == "__main__":
    main()
