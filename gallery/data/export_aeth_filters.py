"""Extra per-filter aethalometer values for the Aethalometer tab's "Against the filters" page.

Run with:  uv run python gallery/data/export_aeth_filters.py      (after export_aethalometer.py)

The main export (export_aethalometer.py) puts BCc by channel, IR smoothed, AAE and Red absorption on
every filter. Two notebooks compared the filters with columns it leaves out:

  research/ftir_hips_chem/HIPS_vs_Aethalometer_Optical_Comparison.ipynb   raw attenuation (``<ch> ATN1``)
      against HIPS, to ask whether the instrument's processing helps or hurts
  notebooks/analysis/absorption/ETAD_Aug13.ipynb (and _comprehensive_)     smoothed BCc at Red and the
      other channels (``<ch> BCc smoothed``), not only IR

This writes those, averaged over each filter's sampling window exactly as the main export averages
BCc (its ``overlap_means``: the daily bins weighted by their overlap in hours), and only for the
filters the main export kept (bins covering at least 12 h of the window). IR smoothed is already in
the main export and is not repeated. Smoothed Green and Red exist only in the Addis file.

The notebooks' second Addis file (df_Jacros_9am_resampled.pkl) is the same instrument (MA350-0238)
over a longer span; on the 515 days both cover every BCc column is identical, so it adds nothing here.

ATN is the attenuation of the current tape spot (100 ln(I0/I)): it climbs as the spot loads and
drops at every tape advance, so a window mean of it is not a concentration. It is exported only so
the page can show why the notebook found R² ≈ 0 against HIPS.

Writes to gallery/app/public/data/aethalometer/filters_extra/ (inside the Instruments section's
private data folder):
  index.json        fields, units, the MA350 cross-sections and channel wavelengths, filters per site
  <CODE>.json       {"rows": {filter id: values in `fields` order}}
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_aethalometer import (  # noqa: E402
    CHANNELS, OUT as AETH_OUT, PROCESSED, ROOT, SIGMA, SITE_FILES, WAVELENGTHS_NM, column, overlap_means, r,
)

OUT = AETH_OUT / "filters_extra"
SMOOTHED = [ch for ch in CHANNELS if ch != "IR"]   # IR smoothed is in the main export


def field_names() -> dict[str, tuple[str, str, int]]:
    """gallery field -> (pickle column(s) key, unit, decimals), in export order."""
    out: dict[str, tuple[str, str, int]] = {}
    for ch in SMOOTHED:
        out[f"{ch} BCc smoothed (aethalometer)"] = (f"{ch}_smoothed", "ug/m3", 3)
    for ch in CHANNELS:
        out[f"{ch} ATN1 (aethalometer)"] = (f"{ch}_ATN1", "", 1)
    return out


def bin_frame(path: Path) -> pd.DataFrame:
    """The daily bins with the extra columns, labelled at their start as export_aethalometer reads them."""
    raw = pd.read_pickle(path)
    start = pd.to_datetime(column(raw, "datetime_local"))
    if getattr(start.dt, "tz", None) is not None:
        start = start.dt.tz_localize(None)   # keep the local wall time
    out = pd.DataFrame({"start": start.to_numpy()})
    for ch in SMOOTHED:
        v = column(raw, f"{ch} BCc smoothed", f"{ch} BCc smoothed  (ng/m^3)")
        out[f"{ch}_smoothed"] = pd.to_numeric(v, errors="coerce").to_numpy() / 1000.0 if v is not None else np.nan   # ng/m³ -> µg/m³
    for ch in CHANNELS:
        v = column(raw, f"{ch} ATN1")
        out[f"{ch}_ATN1"] = pd.to_numeric(v, errors="coerce").to_numpy() if v is not None else np.nan
    return out.dropna(subset=["start"]).sort_values("start").drop_duplicates("start").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    windows = json.loads((ROOT / "gallery/app/public/data/weather/filters.json").read_text())["windows"]
    fields = field_names()
    cols = [c for c, _, _ in fields.values()]
    digits = [d for _, _, d in fields.values()]
    counts: dict[str, int] = {}
    for old in OUT.glob("*.json"):
        old.unlink()
    for code, (site_name, fname) in SITE_FILES.items():
        kept = AETH_OUT / "filters" / f"{code}.json"
        if code != code.split("-")[0] or not kept.exists() or not (PROCESSED / fname).exists():
            continue   # Jacros has no filters of its own; a site the main export did not match is skipped
        ids = json.loads(kept.read_text())["rows"]
        daily = bin_frame(PROCESSED / fname)
        rows = {}
        for fid in ids:
            w = windows.get(fid)
            if not w:
                continue
            means, _, _ = overlap_means(daily, cols, pd.Timestamp(w[0]), pd.Timestamp(w[1]))
            if np.isfinite(means).any():
                rows[fid] = [r(v, nd) for v, nd in zip(means, digits)]
        (OUT / f"{code}.json").write_text(json.dumps({"rows": rows}, separators=(",", ":"), allow_nan=False))
        counts[code] = len(rows)
        have = [f for j, f in enumerate(fields) if any(v[j] is not None for v in rows.values())]
        print(f"{code} ({site_name}): {len(rows)} filters; values for {', '.join(have)}")
    index = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        source="MA350 legacy daily aggregates in research/ftir_hips_chem/processed_sites, averaged over each filter's window as in export_aethalometer.py",
        note="Smoothed BCc (UV, Blue, Green, Red; IR smoothed is in the main export) and spot attenuation ATN1. ATN is a tape-spot loading, not a concentration.",
        fields=list(fields), field_units={f: u for f, (_, u, _) in fields.items()},
        sigma=SIGMA, wavelengths_nm={ch: WAVELENGTHS_NM[ch] for ch in CHANNELS}, sites=counts,
    )
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":"), allow_nan=False))
    size = sum(p.stat().st_size for p in OUT.glob("*.json"))
    print(f"{sum(counts.values())} filters; {size / 1e3:.0f} kB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
