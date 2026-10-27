"""Aethalometer (MA350) daily data and each filter's overlap-weighted values, for the Instruments section.

Run with:  uv run --with timezonefinder python gallery/data/export_aethalometer.py

Inputs (in the repo):
  research/ftir_hips_chem/processed_sites/df_<Site>_9am_resampled.pkl   daily aggregates (config.SITES, plus Jacros)
  gallery/app/public/data/weather/filters.json                         each filter's local sampling window

The pickles are legacy daily aggregates (processed_sites/README.md, docs/matched-sample-audit-2026-09-10.md):
each row is a 24 h bin labelled at its start (`datetime_local`). Beijing, Delhi and JPL bins start at
15:00, Addis at 09:00; observed coverage was never verified. So a filter's value is the mean of the bins
it overlaps, weighted by the overlap in hours, and kept only when the bins cover at least 12 h of its
window. The overlap is exported with every filter, so a chart can demand more. Reading the labels as bin
starts is checked against HIPS (27 Oct 2026): r(IR BCc, HIPS Fabs) is 0.86-0.93 at all four sites as
labelled, against 0.45-0.60 with the bins shifted a day either way.

Absorption Ångström exponents use absorption (BCc × the MA350 cross-section, research/ftir_hips_chem/
scripts/optics.py), not BCc: AAE from BCc is offset by about -1 (optics.aae_from_columns explains why).

Writes to gallery/app/public/data/aethalometer/ (Instruments section; left out of a public build while
gallery/app/src/sections/sections.json marks it private):
  index.json + filters/<CODE>.json   per filter: the fields below and [overlap hours, bins]
  daily/<CODE>.json                  per site: every daily bin (start, BCc by channel, AAE, coverage, flow period)
  sites.json                         per site: instrument, span, bin start hour, how many filters matched
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "research/ftir_hips_chem"
sys.path.insert(0, str(AREA / "scripts"))
from config import SITES, WAVELENGTHS_NM  # noqa: E402
from optics import aae  # noqa: E402
from flow_periods import classify_flow_period  # noqa: E402

PROCESSED = AREA / "processed_sites"
OUT = ROOT / "gallery/app/public/data/aethalometer"
CHANNELS = ["UV", "Blue", "Green", "Red", "IR"]
# MA350 attenuation cross-sections (m²/g), src/external/calibration.py
SIGMA = {"UV": 24.069, "Blue": 19.070, "Green": 17.028, "Red": 14.091, "IR": 10.120}
MIN_OVERLAP_H = 12.0
# the gallery's site codes; Jacros is the second Addis instrument (AAU Jackros, beside the AERONET station)
SITE_FILES = {cfg["code"]: (name, cfg["file"]) for name, cfg in SITES.items()}
SITE_FILES["ETAD-JACROS"] = ("Jacros", "df_Jacros_9am_resampled.pkl")

FIELDS = {  # gallery field -> (unit, decimals)
    "UV BCc (aethalometer)": ("ug/m3", 3), "Blue BCc (aethalometer)": ("ug/m3", 3), "Green BCc (aethalometer)": ("ug/m3", 3),
    "Red BCc (aethalometer)": ("ug/m3", 3), "IR BCc (aethalometer)": ("ug/m3", 3), "IR BCc smoothed (aethalometer)": ("ug/m3", 3),
    "AAE UV-IR (aethalometer)": ("", 2), "AAE Blue-IR (aethalometer)": ("", 2),
    "Red absorption (aethalometer)": ("Mm-1", 1),
}


def column(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """The first column present (the pickles disagree on names and some carry duplicates)."""
    for name in names:
        if name in df.columns:
            col = df.loc[:, name]
            return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
    return None


def daily_frame(code: str, site_name: str, path: Path) -> pd.DataFrame:
    raw = pd.read_pickle(path)
    start = column(raw, "datetime_local")
    if start is None:   # Jacros carries only the 9 AM day
        start = pd.to_datetime(column(raw, "day_9am")) + pd.Timedelta(hours=9)
    out = pd.DataFrame({"start": pd.to_datetime(start).dt.tz_localize(None) if getattr(pd.to_datetime(start).dt, "tz", None) else pd.to_datetime(start)})
    for ch in CHANNELS:
        v = column(raw, f"{ch} BCc")
        out[ch] = pd.to_numeric(v, errors="coerce") / 1000.0 if v is not None else np.nan   # ng/m³ -> µg/m³
    sm = column(raw, "IR BCc smoothed", "IR BCc smoothed  (ng/m^3)")
    out["IR_smoothed"] = pd.to_numeric(sm, errors="coerce") / 1000.0 if sm is not None else np.nan
    cov = column(raw, "data_completeness_pct")
    out["coverage"] = pd.to_numeric(cov, errors="coerce") if cov is not None else np.nan
    babs = {ch: out[ch] * SIGMA[ch] for ch in CHANNELS}           # µg/m³ × m²/g = Mm⁻¹
    with np.errstate(divide="ignore", invalid="ignore"):
        out["AAE_UV_IR"] = aae(babs["UV"], babs["IR"], WAVELENGTHS_NM["UV"], WAVELENGTHS_NM["IR"])
        out["AAE_Blue_IR"] = aae(babs["Blue"], babs["IR"], WAVELENGTHS_NM["Blue"], WAVELENGTHS_NM["IR"])
    out["Red_babs"] = babs["Red"]
    out["flow"] = [classify_flow_period(d, site_name) if site_name in ("Beijing", "Delhi", "JPL", "Addis_Ababa") else None for d in out["start"]]
    out = out.dropna(subset=["start"]).sort_values("start").drop_duplicates("start").reset_index(drop=True)
    return out


def overlap_means(daily: pd.DataFrame, cols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, float, int]:
    """Overlap-weighted mean of the daily bins over [start, end); the hours covered and bins used."""
    a = daily["start"].to_numpy(dtype="datetime64[ns]")
    b = a + np.timedelta64(24, "h")
    lo, hi = np.datetime64(start, "ns"), np.datetime64(end, "ns")
    hours = (np.minimum(b, hi) - np.maximum(a, lo)) / np.timedelta64(1, "h")
    use = hours > 0
    if not use.any():
        return np.full(len(cols), np.nan), 0.0, 0
    w = hours[use]
    vals = daily.loc[use, cols].to_numpy(dtype=float)
    ok = np.isfinite(vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(ok.any(0), np.nansum(np.where(ok, vals, 0) * w[:, None], 0) / np.sum(ok * w[:, None], 0), np.nan)
    covered = float(np.sum(w[np.isfinite(vals[:, cols.index("IR")])])) if "IR" in cols else float(w.sum())
    return means, covered, int(use.sum())


def r(v, nd):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def main() -> None:
    for sub in ("filters", "daily"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    windows = json.loads((ROOT / "gallery/app/public/data/weather/filters.json").read_text())["windows"]
    cols = ["UV", "Blue", "Green", "Red", "IR", "IR_smoothed", "AAE_UV_IR", "AAE_Blue_IR", "Red_babs"]
    names = list(FIELDS)
    rows, counts, sites = {}, {}, {}
    for code, (site_name, fname) in SITE_FILES.items():
        path = PROCESSED / fname
        if not path.exists():
            continue
        daily = daily_frame(code, site_name, path)
        hours = daily["start"].dt.hour.value_counts()
        filter_code = code.split("-")[0]
        n = 0
        if code == filter_code:   # Jacros is a second instrument at Addis; its values stay in its daily file
            for fid, w in windows.items():
                if not fid.startswith(filter_code + "-"):
                    continue
                means, covered, bins = overlap_means(daily, cols, pd.Timestamp(w[0]), pd.Timestamp(w[1]))
                if covered < MIN_OVERLAP_H or not np.isfinite(means).any():
                    continue
                rows[fid] = [r(v, FIELDS[f][1]) for f, v in zip(names, means)]
                counts[fid] = [round(covered, 1), bins]
                n += 1
        out = {"start": daily["start"].dt.strftime("%Y-%m-%d %H:%M").tolist(), "flow": daily["flow"].tolist()}
        for c in cols + ["coverage"]:
            out[c] = [r(v, 2 if c.startswith("AAE") or c == "coverage" else 4) for v in daily[c]]
        (OUT / "daily" / f"{code}.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
        sites[code] = dict(site=site_name, file=fname, bins=len(daily), first=out["start"][0], last=out["start"][-1],
                           bin_start_hours={int(h): int(c) for h, c in hours.items()}, matched_filters=n,
                           coverage_field=bool(daily["coverage"].notna().any()))
        print(f"{code} ({site_name}): {len(daily)} daily bins, {n} filters matched")

    meta = dict(generated=datetime.now(timezone.utc).isoformat(),
                source="MA350 aethalometers, legacy daily aggregates in research/ftir_hips_chem/processed_sites (unverified 9 AM matching; see docs/matched-sample-audit-2026-09-10.md)",
                note=f"Overlap-weighted mean of the daily bins over each filter's sampling window; kept where the bins cover at least {MIN_OVERLAP_H:g} h. AAE from absorption (BCc × MA350 cross-section).")
    payload = dict(**meta, fields=names, field_units={f: FIELDS[f][0] for f in names}, count_fields=["Overlap hours", "Daily bins"])
    by_site: dict[str, dict] = {}
    for fid in rows:
        part = by_site.setdefault(fid.split("-")[0], {"rows": {}, "counts": {}})
        part["rows"][fid], part["counts"][fid] = rows[fid], counts[fid]
    for old in (OUT / "filters").glob("*.json"):
        old.unlink()
    for code, part in by_site.items():
        (OUT / "filters" / f"{code}.json").write_text(json.dumps(part, separators=(",", ":"), allow_nan=False))
    (OUT / "index.json").write_text(json.dumps(dict(**payload, sites={c: len(p["rows"]) for c, p in sorted(by_site.items())}),
                                               separators=(",", ":"), allow_nan=False))
    (OUT / "sites.json").write_text(json.dumps(dict(**meta, sites=sites), separators=(",", ":"), allow_nan=False))
    size = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(f"{len(rows)} filters with aethalometer values; {size / 1e6:.1f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
