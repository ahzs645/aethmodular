"""The aethalometers' own source apportionment, for the Instruments section's Sources (AAE) page.

Run with:  uv run python gallery/data/export_aeth_sources.py

Inputs:
  research/ftir_hips_chem/processed_sites/df_<Site>_9am_resampled.pkl   the daily bins export_aethalometer.py reads (in the repo)
  gallery/app/public/data/aethalometer/daily/<CODE>.json                  their bin starts, as exported
  <aethalometry dir>/Kyan Data/MAIA data/<Site>_cleaned_corrected_small_hourly.csv
      hourly MA350 with the instrument's split (data_paths.aethalometry_dir(): the Drive "Aethalometry Data"
      folder, or AETHMODULAR_AETHALOMETRY_DIR). Optional: without it only the pickles' values are written.

MA350 exports carry a two-component apportionment beside BCc: an AAE, biomass and fossil-fuel BCc, and
in the pickles ``BB (%)`` and the end-member exponents (``AAE fossil fuel`` 1, ``AAE biomass`` 2). The
gallery derives its AAE from absorption, BCc × the MA350 cross-section (export_aethalometer.py,
research/ftir_hips_chem/scripts/optics.py); the instrument's number checks that scale, because an AAE
taken from BCc sits about 1 lower.

Two sources, kept apart:
  ``daily``   the pickles' own daily values. Measured 27 Oct 2026: Delhi 37 bins, JPL 206; Beijing's file
              has the columns but no values, the Addis and Jacros files none.
  ``hourly``  the MAIA hourly files averaged over each daily bin (bins with at least 18 hours): the mean
              hourly "AAE calculated" and BB % = biomass BCc / (biomass + fossil BCc), for Beijing, Delhi and
              JPL, whose hourly IR BCc matches the daily files (r 1.00, 1.00, 0.99 over the bins). The
              MAIA "Central" file is not used for Addis: over the Addis bins its IR BCc is about 0.47 of
              the filter-site instrument's with r 0.62 at any shift, so it is another instrument or site.
              "AAE calculated" is the Blue/IR absorption AAE of the same hour (median difference 0.00 at
              Delhi; larger where an hour averages many minutes of noisy low BC, as at JPL).

Each daily bin keeps the start label export_aethalometer.py writes (local, ``%Y-%m-%d %H:%M``), so the
page joins these values to daily/<CODE>.json by that string.

Writes gallery/app/public/data/aethalometer/sources/instrument.json (private with the rest of
aethalometer/, gallery/app/src/sections/sections.json).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_aethalometer import AREA, OUT, PROCESSED, SITE_FILES, column  # noqa: E402

sys.path.insert(0, str(AREA / "scripts"))
from data_paths import aethalometry_dir  # noqa: E402

DEST = OUT / "sources" / "instrument.json"
# firmware column -> exported key (the pickles pad the unit with two spaces)
COLUMNS = {
    "AAE": ("aae", ("AAE",), 3),
    "BB (%)": ("bb_pct", ("BB (%)",), 2),
    "Biomass BCc": ("bb_bcc", ("Biomass BCc  (ng/m^3)", "Biomass BCc (ng/m^3)"), 4),
    "Fossil fuel BCc": ("ff_bcc", ("Fossil fuel BCc  (ng/m^3)", "Fossil fuel BCc (ng/m^3)"), 4),
}
ENDMEMBERS = {"AAE fossil fuel": "aae_fossil", "AAE biomass": "aae_biomass"}
# the MAIA hourly file of each daily-file code (not Addis: "Central" is not the filter-site instrument)
MAIA = {"CHTS": "Beijing", "INDH": "Delhi", "USPA": "JPL"}
MIN_HOURS = 18


def starts(raw: pd.DataFrame) -> pd.Series:
    """Each bin's local start, as export_aethalometer.daily_frame labels it."""
    start = column(raw, "datetime_local")
    if start is None:
        start = pd.to_datetime(column(raw, "day_9am")) + pd.Timedelta(hours=9)
    start = pd.to_datetime(start)
    return start.dt.tz_localize(None) if getattr(start.dt, "tz", None) else start


def instrument_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """The firmware apportionment per daily bin, one row per start, bins without an AAE dropped."""
    out = pd.DataFrame({"start": starts(raw)})
    for key, names, _ in COLUMNS.values():
        v = column(raw, *names)
        out[key] = pd.to_numeric(v, errors="coerce") if v is not None else np.nan
    out["bb_bcc"] = out["bb_bcc"] / 1000.0   # ng/m³ -> µg/m³, as the daily files
    out["ff_bcc"] = out["ff_bcc"] / 1000.0
    for name, key in ENDMEMBERS.items():
        v = column(raw, name)
        out[key] = pd.to_numeric(v, errors="coerce") if v is not None else np.nan
    out = out.dropna(subset=["start"]).sort_values("start").drop_duplicates("start")
    return out[np.isfinite(out["aae"])].reset_index(drop=True)


def maia_file(site: str) -> Path | None:
    try:
        path = aethalometry_dir() / "Kyan Data" / "MAIA data" / f"{site}_cleaned_corrected_small_hourly.csv"
    except Exception:   # no Drive and no override: the hourly source is optional
        return None
    return path if path.exists() else None


def read_hourly(path: Path) -> pd.DataFrame:
    """Hourly AAE and split, timed by local wall clock (the offset dropped, as the daily bins are)."""
    raw = pd.read_csv(path, usecols=["datetime_local", "AAE calculated", "Biomass BCc", "Fossil fuel BCc", "IR BCc"])
    return pd.DataFrame({
        "t": pd.to_datetime(raw["datetime_local"].astype(str).str.slice(0, 19)),
        "aae": pd.to_numeric(raw["AAE calculated"], errors="coerce"),
        "bb": pd.to_numeric(raw["Biomass BCc"], errors="coerce"),
        "ff": pd.to_numeric(raw["Fossil fuel BCc"], errors="coerce"),
        "ir": pd.to_numeric(raw["IR BCc"], errors="coerce") / 1000.0,
    }).dropna(subset=["t"]).sort_values("t")


def hourly_bins(hourly: pd.DataFrame, starts: list[str], min_hours: int = MIN_HOURS) -> pd.DataFrame:
    """Per daily bin [start, start + 24 h): mean hourly AAE, BB % of the summed split, mean IR BCc, hours used."""
    t = hourly["t"].to_numpy(dtype="datetime64[ns]")
    rows = []
    for s in starts:
        a = np.datetime64(pd.Timestamp(s), "ns")
        lo, hi = np.searchsorted(t, a), np.searchsorted(t, a + np.timedelta64(24, "h"))
        h = hourly.iloc[lo:hi]
        ok = h["aae"].notna()
        if ok.sum() < min_hours:
            continue
        split = h.loc[h["bb"].notna() & h["ff"].notna()]
        total = float((split["bb"] + split["ff"]).sum())
        rows.append({"start": s, "aae": float(h.loc[ok, "aae"].mean()),
                     "bb_pct": 100.0 * float(split["bb"].sum()) / total if total > 0 else np.nan,
                     "ir": float(h["ir"].mean()), "hours": int(ok.sum())})
    return pd.DataFrame(rows, columns=["start", "aae", "bb_pct", "ir", "hours"])


def endmember(series: pd.Series) -> float | None:
    """The end-member AAE the firmware assumed, when it is one value throughout."""
    vals = series.dropna().unique()
    return float(vals[0]) if len(vals) == 1 else None


def r(v, nd):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def main() -> None:
    sites = {}
    for code, (site_name, fname) in SITE_FILES.items():
        path = PROCESSED / fname
        if not path.exists():
            continue
        frame = instrument_frame(pd.read_pickle(path))
        daily = {"site": site_name, "n": int(len(frame)),
                 "aae_fossil": endmember(frame["aae_fossil"]), "aae_biomass": endmember(frame["aae_biomass"]),
                 "start": frame["start"].dt.strftime("%Y-%m-%d %H:%M").tolist()}
        for key, _, nd in COLUMNS.values():
            daily[key] = [r(v, nd) for v in frame[key]]
        entry = {"site": site_name, "daily": daily}
        msg = f"{code} ({site_name}): {len(frame)} daily bins with the pickle's AAE"
        hourly_path = maia_file(MAIA[code]) if code in MAIA else None
        daily_file = OUT / "daily" / f"{code}.json"
        if hourly_path and daily_file.exists():
            day = json.loads(daily_file.read_text())
            bins = hourly_bins(read_hourly(hourly_path), day["start"])
            entry["hourly"] = {"file": hourly_path.name, "n": int(len(bins)), "start": bins["start"].tolist(),
                               "aae": [r(v, 3) for v in bins["aae"]], "bb_pct": [r(v, 2) for v in bins["bb_pct"]],
                               "hours": bins["hours"].tolist()}
            # the hourly file is the same instrument when its IR BCc tracks the daily file's
            ir = dict(zip(day["start"], day["IR"]))
            pair = [(ir[s], v) for s, v in zip(bins["start"], bins["ir"]) if ir.get(s) is not None and np.isfinite(v)]
            rr = np.corrcoef(*zip(*pair))[0, 1] if len(pair) > 2 else np.nan
            msg += f"; {len(bins)} bins from {hourly_path.name} (r of IR BCc with the daily file {rr:.3f})"
        sites[code] = entry
        print(msg)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        source="MA350 source apportionment: the processed_sites pickles' own columns (daily) and the MAIA hourly exports averaged over each daily bin (hourly)",
        note=f"Bins labelled at their local start as in daily/<CODE>.json; hourly values kept where at least {MIN_HOURS} hours of the bin have an AAE.",
        sites=sites,
    )
    DEST.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(f"wrote {DEST.relative_to(OUT.parents[4])}")


if __name__ == "__main__":
    main()
