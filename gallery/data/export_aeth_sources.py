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
  ``hourly``  the MAIA hourly files averaged over each daily bin (at least 18 distinct hours per metric): the mean
              hourly "AAE calculated" and BB % = biomass BCc / (biomass + fossil BCc), for Beijing, Delhi and
              JPL, whose hourly IR BCc matches the daily files (r 1.00, 1.00, 0.99 over the bins). The
              MAIA "Central" file is not used for Addis: over the Addis bins its IR BCc is about 0.47 of
              the filter-site instrument's with r 0.62 at any shift, so it is another instrument or site.
              "AAE calculated" is the Blue/IR absorption AAE of the same hour (median difference 0.00 at
              Delhi; larger where an hour averages many minutes of noisy low BC, as at JPL).

Each daily bin keeps the start label export_aethalometer.py writes (local, ``%Y-%m-%d %H:%M``), so the
page joins these values to daily/<CODE>.json by that string. Its start is localized in the site's zone,
then the legacy 24-elapsed-hour window is evaluated in UTC, including across daylight-saving changes.
Coverage uses distinct one-hour UTC intervals anchored at that start, not CSV row counts. Identical
repeated metric values count once; conflicting duplicates invalidate that metric's hour. AAE and the
paired biomass/fossil split have independent coverage gates, so missing split data never removes AAE.

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
from export_aethalometer import AREA, OUT, PROCESSED, SITE_FILES, SITES, column  # noqa: E402

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
BIN_HOURS = 24
HOURLY_COLUMNS = {
    "AAE calculated": "aae", "Biomass BCc": "bb", "Fossil fuel BCc": "ff", "IR BCc": "ir",
}
BIN_COLUMNS = [
    "start", "aae", "bb_pct", "ir", "hours", "aae_hours", "split_hours", "ir_hours",
    "observed_hours", "raw_rows", "duplicate_rows", "conflict_hours", "aae_conflict_hours",
    "split_conflict_hours", "ir_conflict_hours", "start_utc", "end_utc", "bb_status",
]
DUPLICATE_POLICY = (
    "Within each UTC one-hour interval anchored at the bin start, field-equivalent repeats count once. "
    "Conflicting AAE, paired biomass/fossil values, or IR invalidate that metric for the hour; "
    "missing versus present is a conflict. No API/SD source priority is assumed."
)


def starts(raw: pd.DataFrame) -> pd.Series:
    """Each bin's local start, as export_aethalometer.daily_frame labels it."""
    start = column(raw, "datetime_local")
    if start is None:
        start = pd.to_datetime(column(raw, "day_9am")) + pd.Timedelta(hours=9)
    # Preserve the legacy wall-clock join key even when rows have different UTC offsets.
    def wall_clock(value):
        try:
            stamp = pd.Timestamp(value)
            return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp
        except (TypeError, ValueError):
            return pd.NaT
    return pd.to_datetime(start.map(wall_clock), errors="coerce")


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


def utc_timestamp(value, site_timezone: str) -> pd.Timestamp:
    """Honor an explicit offset; localize naive values without guessing a DST fold/gap."""
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return pd.NaT
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(site_timezone, ambiguous="NaT", nonexistent="NaT")
        return stamp.tz_convert("UTC") if not pd.isna(stamp) else pd.NaT
    except (TypeError, ValueError):
        return pd.NaT


def prepare_hourly(hourly: pd.DataFrame, site_timezone: str) -> pd.DataFrame:
    """Normalize timestamps/numeric validity without choosing between repeated observations."""
    out = hourly.copy()
    out["t"] = pd.to_datetime([utc_timestamp(v, site_timezone) for v in out["t"]], utc=True)
    nonfinite = {}
    for key in HOURLY_COLUMNS.values():
        values = pd.to_numeric(out[key], errors="coerce").astype(float) if key in out else pd.Series(np.nan, index=out.index)
        nonfinite[key] = int((~np.isfinite(values)).sum())
        out[key] = values.where(np.isfinite(values))
    # read_hourly's diagnostics retain the original invalid/nonfinite row counts on a second pass.
    out.attrs["input_diagnostics"] = hourly.attrs.get("input_diagnostics", {
        "rows": int(len(out)), "invalid_timestamp_rows": int(out["t"].isna().sum()),
        "nonfinite_or_missing_rows": nonfinite,
    })
    return out.sort_values("t", kind="stable")


def read_hourly(path: Path, site_timezone: str = "UTC") -> pd.DataFrame:
    """Read MAIA hours in UTC, preserving offsets; a missing split column leaves AAE usable."""
    names = {"datetime_local", *HOURLY_COLUMNS}
    raw = pd.read_csv(path, usecols=lambda name: name in names)
    if "datetime_local" not in raw:
        raise ValueError(f"{path.name}: missing datetime_local column")
    hourly = raw.rename(columns={"datetime_local": "t", **HOURLY_COLUMNS})
    hourly = prepare_hourly(hourly, site_timezone)
    hourly["ir"] = hourly["ir"] / 1000.0   # ng/m³ -> µg/m³, as the daily files
    hourly.attrs["input_diagnostics"]["missing_columns"] = sorted(names - set(raw.columns))
    return hourly


def agreed_metric(group: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, bool]:
    """One agreed metric/pair per hour; never synthesize a split from different records."""
    values = group[keys].to_numpy(dtype=float)
    first = values[0]
    same = np.all((values == first) | (np.isnan(values) & np.isnan(first)))
    return (first if same else np.full(len(keys), np.nan)), not bool(same)


def hourly_bins(hourly: pd.DataFrame, starts: list[str], min_hours: int = MIN_HOURS,
                site_timezone: str = "UTC") -> pd.DataFrame:
    """Legacy 24 h bins with independently verified AAE, paired-split, and IR UTC-hour coverage.

    ``hours`` remains the legacy alias for ``aae_hours``. Returned bins still require AAE coverage,
    but BB needs its own paired-split coverage. The IR mean used for the daily-file comparison also
    requires its own coverage. Rejected-bin and input diagnostics are in ``attrs``.
    """
    if not isinstance(min_hours, (int, np.integer)) or not 1 <= min_hours <= BIN_HOURS:
        raise ValueError(f"min_hours must be an integer from 1 to {BIN_HOURS}")
    hourly = prepare_hourly(hourly, site_timezone)
    rows, rejected, invalid_starts = [], [], []
    for s in starts:
        a = utc_timestamp(s, site_timezone)
        if pd.isna(a):
            invalid_starts.append(s)
            continue
        b = a + pd.Timedelta(hours=BIN_HOURS)
        h = hourly.loc[(hourly["t"] >= a) & (hourly["t"] < b)]
        # Anchoring the UTC intervals at the bin start also handles Kolkata's half-hour offset.
        slots = (h["t"] - a) // pd.Timedelta(hours=1)
        agreed, conflicts = [], []
        for _, group in h.groupby(slots, sort=True):
            av, ac = agreed_metric(group, ["aae"])
            sv, sc = agreed_metric(group, ["bb", "ff"])
            iv, ic = agreed_metric(group, ["ir"])
            agreed.append([av[0], *sv, iv[0]])
            conflicts.append([ac, sc, ic])
        values = np.asarray(agreed, dtype=float).reshape(-1, 4)
        conflict = np.asarray(conflicts, dtype=bool).reshape(-1, 3)
        aae_ok = np.isfinite(values[:, 0])
        split_ok = np.isfinite(values[:, 1:3]).all(axis=1)
        ir_ok = np.isfinite(values[:, 3])
        counts = {
            "aae_hours": int(aae_ok.sum()), "split_hours": int(split_ok.sum()), "ir_hours": int(ir_ok.sum()),
            "observed_hours": int(len(values)), "raw_rows": int(len(h)), "duplicate_rows": int(len(h) - len(values)),
            "conflict_hours": int(conflict.any(axis=1).sum()), "aae_conflict_hours": int(conflict[:, 0].sum()),
            "split_conflict_hours": int(conflict[:, 1].sum()), "ir_conflict_hours": int(conflict[:, 2].sum()),
        }
        if counts["aae_hours"] < min_hours:
            rejected.append({"start": s, "reason": "insufficient_aae_hours", **counts})
            continue
        split = values[split_ok, 1:3]
        # A higher precision accumulator avoids overflow from summing otherwise finite hourly values.
        total = np.sum(split, dtype=np.longdouble)
        biomass = np.sum(split[:, 0], dtype=np.longdouble)
        bb_pct, bb_status = np.nan, "insufficient_paired_hours"
        if counts["split_hours"] >= min_hours:
            if not np.isfinite(total):
                bb_status = "nonfinite_split_total"
            elif total <= 0:
                bb_status = "nonpositive_split_total"
            else:
                candidate = float(100 * (biomass / total))
                bb_pct = candidate if np.isfinite(candidate) else np.nan
                bb_status = "available" if np.isfinite(candidate) else "nonfinite_split_fraction"
        rows.append({
            "start": s, "aae": float(np.mean(values[aae_ok, 0], dtype=np.longdouble)),
            "bb_pct": bb_pct,
            "ir": float(np.mean(values[ir_ok, 3], dtype=np.longdouble)) if counts["ir_hours"] >= min_hours else np.nan,
            "hours": counts["aae_hours"], **counts, "start_utc": a.isoformat(), "end_utc": b.isoformat(),
            "bb_status": bb_status,
        })
    out = pd.DataFrame(rows, columns=BIN_COLUMNS)
    out.attrs["diagnostics"] = {"input": hourly.attrs["input_diagnostics"], "rejected_bins": rejected,
                                "invalid_bin_starts": invalid_starts}
    return out


def endmember(series: pd.Series) -> float | None:
    """The end-member AAE the firmware assumed, when it is one value throughout."""
    numeric = pd.to_numeric(series, errors="coerce")
    vals = numeric[np.isfinite(numeric)].unique()
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
            site_timezone = SITES[site_name]["timezone"]
            bins = hourly_bins(read_hourly(hourly_path, site_timezone), day["start"], site_timezone=site_timezone)
            entry["hourly"] = {"file": hourly_path.name, "n": int(len(bins)), "start": bins["start"].tolist(),
                               "aae": [r(v, 3) for v in bins["aae"]], "bb_pct": [r(v, 2) for v in bins["bb_pct"]],
                               "hours": bins["hours"].tolist(), "coverage_version": 2,
                               "timezone": site_timezone, "bin_hours": BIN_HOURS, "min_hours": MIN_HOURS,
                               "coverage_clock": "UTC one-hour intervals anchored at the localized bin start",
                               "duplicate_policy": DUPLICATE_POLICY, "diagnostics": bins.attrs["diagnostics"]}
            for key in BIN_COLUMNS:
                if key not in ("start", "aae", "bb_pct", "ir", "hours"):
                    entry["hourly"][key] = bins[key].tolist()
            # the hourly file is the same instrument when its IR BCc tracks the daily file's
            ir = dict(zip(day["start"], day["IR"]))
            pair = [(ir[s], v) for s, v in zip(bins["start"], bins["ir"]) if ir.get(s) is not None and np.isfinite(v)]
            rr = np.corrcoef(*zip(*pair))[0, 1] if len(pair) > 2 else np.nan
            msg += (f"; {len(bins)} AAE bins / {int(bins['bb_pct'].notna().sum())} split bins from {hourly_path.name} "
                    f"(r of IR BCc with the daily file {rr:.3f})")
        sites[code] = entry
        print(msg)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        source="MA350 source apportionment: the processed_sites pickles' own columns (daily) and the MAIA hourly exports averaged over each daily bin (hourly)",
        note=(f"Legacy bin labels match daily/<CODE>.json. Each bin spans {BIN_HOURS} elapsed hours from its localized start. "
              f"AAE and biomass/fossil fractions independently require at least {MIN_HOURS} finite, nonconflicting UTC hours; "
              "hours is the legacy alias for aae_hours. Conflicts and rejected bins remain in diagnostics."),
        sites=sites,
    )
    DEST.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(f"wrote {DEST.relative_to(OUT.parents[4])}")


if __name__ == "__main__":
    main()
