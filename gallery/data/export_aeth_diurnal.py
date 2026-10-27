"""The day inside a 24 h filter: hour-of-day tables for the Aethalometer tab's Diurnal page.

Run with:  uv run --with timezonefinder python gallery/data/export_aeth_diurnal.py
           [--aeronet-dir DIR] [--minute CODE=PATH ...] [--skip-aeth] [--skip-aeronet]

Three parts, each written only when its inputs are present:

1. The column (AERONET), from research/ftir_hips_chem/output/tables/aeronet_spartan/
   {aod,sda,inv}_points_lev15.parquet (made by workflows/fetch_aeronet_spartan.py): for every
   SPARTAN site with a station, each local-standard day-hour with a retrieval, holding the number of
   direct-sun retrievals and their mean AOD 500, Angstrom exponent 440-870 and fine-mode fraction,
   plus the almucantar inversions per day-hour. The photometer sees only clear daylight; this is how
   much of a 24 h filter's day that is, and the column's own hour-of-day cycle.

2. The surface (MA350), from Google Drive through research/ftir_hips_chem/scripts/data_paths.
   aethalometry_dir() (the repo .env's AETHMODULAR_MAIA_DATA_ROOT + "Aethalometry Data"; see SOURCES,
   --minute CODE=PATH overrides). Files read, relative to that directory:
     Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv                      ETAD, minutes
     Raw/Pasadena_MA350_1-min_2023-2024_Cleaned.csv                    USPA, minutes
     Kyan Data/MAIA data/{Beijing,Delhi}_cleaned_corrected_small_hourly.csv   CHTS, INDH, hourly means
     (Kyan Data/MAIA data/JPL_cleaned_corrected_small_hourly.csv        USPA if the Pasadena file is absent)
   Per local-standard day-hour the mean BCc of each channel (minute QC as in addis_05 and
   diurnal_pub_figures: negatives and values above mean + 3 SD dropped), the valid IR minutes, and
   AAE(UV, IR) and AAE(Blue, IR) from absorption (BCc x the MA350 attenuation cross-section,
   optics.aae). With the AERONET points of the same site it adds what
   notebooks/analysis/meteorology/aeronet_addis_{diagnostics,deep_dive}.ipynb matched: IR BCc
   averaged over +-15 min of each direct-sun retrieval (at least 5 valid minutes), aggregated per
   day-hour (BC, AOD, BC/AOD, surface AAE against the column Angstrom exponent), and each day's
   within-day CV of surface BC and of AOD over the matched retrievals (at least 3). Last, the
   research/ftir_ec_phase3/scripts/ma350_diurnal_coverage.py ratio for every day: mean (and median)
   Red BCc over all 24 h over the same over the AERONET inversion hours (the hours holding at least
   2 % of the site's inversions; at Addis 07-10 and 14-17, the hours that script used), on days with
   more than 1000 minutes and more than 300 inside those hours. Raw Red BCc, no QC, as the script.
   Unlike the script, hours here are local: the raw CSVs' "Time local" is the UTC clock (see
   read_minutes), so the script's 2026-08-23 ratios compared the wrong hours.

3. The brown-carbon test (research/ftir_ec_phase3/ftir_28_ma350_brc_falsification.ipynb), which
   needs only the committed daily record processed_sites/df_Addis_Ababa_9am_resampled.pkl: per
   filter-day absorption b_ATN = BCc x sigma_ATN at the five channels, AAE(625, 880) and the implied
   brown-carbon absorption Babs_BrC = b(625) - b(880) (880/625)^1, and the channel checks.

Seasons, date ranges and the 06-10 share of 24 h mass are left to the page, which bins the day-hours
by whichever season calendar is active. Hours are local standard time (the site's UTC offset outside
daylight saving), the same clock the weather and AERONET exports use.

Writes gallery/app/public/data/aethalometer/diurnal/:
  aeronet/index.json + aeronet/<CODE>.json   the column, per site
  aeth/index.json + aeth/<CODE>.json         the surface, per site where a record is found (BCc as ng/m3 integers)
  brc_test.json                               the ftir_28 per-day table and its summary
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "research/ftir_hips_chem"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(AREA / "scripts"))
from optics import aae  # noqa: E402
from config import WAVELENGTHS_NM, PROCESSED_SITES_DIR  # noqa: E402

AERONET_DIR = AREA / "output/tables/aeronet_spartan"
LEVEL = "15"
OUT = ROOT / "gallery/app/public/data/aethalometer/diurnal"
CHANNELS = ["UV", "Blue", "Green", "Red", "IR"]

# The surface records per site, relative to aethalometry_dir(), first found wins: ("minute", file) for
# MA350 minute exports, ("hourly", file) for the hourly means in Kyan Data/MAIA data. ETAD is MA350-0238
# at Jackros, the instrument behind the ETAD daily record and co-located with the AAU_Jackros_ET
# photometer (addis_05, diurnal_pub_figures, the AERONET notebooks); USPA is MA350-0229, the Pasadena
# file ma350_diurnal_coverage.py used; CHTS and INDH have hourly means only (the cleaned minute pickles
# multisite_diurnal_wavelength_analysis read are 1.5-2.8 GB each and are left alone).
SOURCES = {
    "ETAD": [("minute", "Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv")],
    "USPA": [("minute", "Raw/Pasadena_MA350_1-min_2023-2024_Cleaned.csv"), ("hourly", "Kyan Data/MAIA data/JPL_cleaned_corrected_small_hourly.csv")],
    "CHTS": [("hourly", "Kyan Data/MAIA data/Beijing_cleaned_corrected_small_hourly.csv")],
    "INDH": [("hourly", "Kyan Data/MAIA data/Delhi_cleaned_corrected_small_hourly.csv")],
}
CSV_UTC, CSV_DATE, CSV_TIME, CSV_TZ = "Time (UTC)", "Date local (yyyy/MM/dd)", "Time local (hh:mm:ss)", "Timezone offset (mins)"

MATCH_HALF = pd.Timedelta(minutes=15)   # +-15 min of each retrieval (the AERONET notebooks)
MATCH_MIN = 5                           # valid minutes a match needs
CV_MIN = 3                              # matched retrievals a day needs for a within-day CV
INV_HOUR_SHARE = 0.02                   # an hour is an AERONET hour if it holds >= 2 % of inversions
DAY_MIN, HOURS_MIN = 1000, 300          # ma350_diurnal_coverage.py's near-complete-day thresholds


def r(v, nd=3):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def col(values, nd=3) -> list:
    return [r(v, nd) for v in np.asarray(values, dtype=float)]


def sigma_atn() -> dict[str, float]:
    """The MA350 attenuation cross-sections (m2/g), read from the vendored firmware table as ftir_28 does."""
    with warnings.catch_warnings():          # the vendored file has an invalid escape the parser warns about
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse((ROOT / "src/external/calibration.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "atn_cross_section_dict_MAx" for t in node.targets):
            return {k: float(v) for k, v in ast.literal_eval(node.value).items()}
    raise RuntimeError("atn_cross_section_dict_MAx not found in src/external/calibration.py")


SIGMA = sigma_atn()


# ---------------------------------------------------------------------------------------- shared

def day_hour(local: pd.Series | pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Local calendar day (YYYY-MM-DD) and hour of each timestamp."""
    t = pd.Series(pd.DatetimeIndex(local))
    return t.dt.strftime("%Y-%m-%d"), t.dt.hour


def day_numbers(days: pd.Series, first: str) -> np.ndarray:
    """Whole days since `first` (YYYY-MM-DD) of each row's day."""
    return ((pd.to_datetime(days) - pd.Timestamp(first)) // pd.Timedelta(days=1)).to_numpy(int)


def dd(days: pd.Series, first: str) -> list[int]:
    """A sorted day column as the files store it: the first row's days since `first`, then each row's step
    from the one before (mostly 0 or 1, which keeps a few thousand rows small). The page cumulates it back."""
    n = day_numbers(days, first)
    return np.diff(n, prepend=0).tolist()


def scaled(values, scale: int) -> list:
    """Values as integers in 1/scale units (AOD x 1000 and so on), missing as null."""
    v = np.asarray(values, dtype=float)
    return [None if not np.isfinite(x) else int(round(x * scale)) for x in v]


def inversion_hours(inv_hour_counts: pd.Series, share: float = INV_HOUR_SHARE) -> list[int]:
    """The local hours holding at least `share` of a site's inversions (its AERONET hours)."""
    total = inv_hour_counts.sum()
    return [] if total == 0 else sorted(int(h) for h, n in inv_hour_counts.items() if n / total >= share)


# ---------------------------------------------------------------------------------------- column

def column_tables(aod: pd.DataFrame, sda: pd.DataFrame, inv: pd.DataFrame, offset_h: float) -> dict:
    """One site's retrievals as local day-hour rows: direct-sun counts and means, inversions per day-hour."""
    off = pd.Timedelta(hours=offset_h)
    pts = aod[["time_utc", "aod500", "ae440_870"]].merge(sda[["time_utc", "fmf500"]], on="time_utc", how="left")
    pts["day"], pts["hour"] = day_hour(pts.time_utc + off)
    g = pts.groupby(["day", "hour"]).agg(n=("aod500", "size"), aod500=("aod500", "mean"), ae=("ae440_870", "mean"),
                                         fmf=("fmf500", "mean")).reset_index()
    iv = pd.DataFrame({"time_utc": inv.time_utc})
    iv["day"], iv["hour"] = day_hour(iv.time_utc + off)
    gi = iv.groupby(["day", "hour"]).size().rename("n").reset_index()
    first = min(g.day.min(), gi.day.min()) if len(gi) else g.day.min()
    last = max(g.day.max(), gi.day.max()) if len(gi) else g.day.max()
    return dict(
        first=first, last=last, days=int(g.day.nunique()), scale=dict(aod500=1000, ae=100, fmf=100),
        aod=dict(dd=dd(g.day, first), h=g.hour.tolist(), n=g.n.tolist(), aod500=scaled(g.aod500, 1000), ae=scaled(g.ae, 100), fmf=scaled(g.fmf, 100)),
        inv=dict(dd=dd(gi.day, first), h=gi.hour.tolist(), n=gi.n.tolist()),
        inv_hours=inversion_hours(gi.groupby("hour").n.sum()),
        # for the site comparison: retrievals by hour of day, and the hours with a retrieval on a typical day
        hour_counts=dict(aod=g.groupby("hour").n.sum().reindex(range(24), fill_value=0).astype(int).tolist(),
                         inv=gi.groupby("hour").n.sum().reindex(range(24), fill_value=0).astype(int).tolist()),
        hours_per_day=float(g.groupby("day").hour.nunique().median()),
    )


# ---------------------------------------------------------------------------------------- surface

def read_minutes(path: Path, offset_h: float) -> pd.DataFrame:
    """A minute file as BCc per channel (ng/m3) on a local-standard-time index.

    The cleaned pickles carry a time-zone-aware `datetime_local`. The raw MA350 CSVs carry `Time (UTC)`
    and a "local" date and time that, in the Jacros and Pasadena exports, are the same UTC clock
    (`Timezone offset (mins)` is 0 throughout): reading `Time local` as local puts every hour 3 h
    (Addis) or 8 h (Pasadena) early. So the UTC column is used when present, then the date and time
    shifted by the recorded offset."""
    if path.suffix == ".pkl":
        raw = pd.read_pickle(path)
        t = pd.to_datetime(raw["datetime_local"])
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert("UTC").dt.tz_localize(None) + pd.Timedelta(hours=offset_h)
        cols = [f"{c} BCc" for c in CHANNELS if f"{c} BCc" in raw.columns]
        out = raw[cols].set_axis(pd.DatetimeIndex(t)).astype(float)
        del raw
    else:
        head = pd.read_csv(path, nrows=0).columns
        cols = [f"{c} BCc" for c in CHANNELS if f"{c} BCc" in head]
        when = [CSV_UTC] if CSV_UTC in head else [CSV_DATE, CSV_TIME, *([CSV_TZ] if CSV_TZ in head else [])]
        parts = []
        for ch in pd.read_csv(path, usecols=[*when, *cols], chunksize=500_000):
            if CSV_UTC in ch:
                t = pd.to_datetime(ch[CSV_UTC], utc=True, errors="coerce").dt.tz_localize(None) + pd.Timedelta(hours=offset_h)
            else:
                t = pd.to_datetime(ch[CSV_DATE].astype(str).str.strip().str.replace("/", "-") + " " + ch[CSV_TIME].astype(str).str.strip(),
                                   format="%Y-%m-%d %I:%M:%S %p", errors="coerce")
                if CSV_TZ in ch:     # clock time -> UTC -> standard time
                    t = t - pd.to_timedelta(ch[CSV_TZ], unit="m") + pd.Timedelta(hours=offset_h)
            parts.append(ch[cols].set_axis(pd.DatetimeIndex(t)))
        out = pd.concat(parts).astype(float)
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated()]


def read_hourly(path: Path, offset_h: float) -> pd.DataFrame:
    """An hourly-means CSV (Kyan Data/MAIA data: time-zone-aware `datetime_local`, BCc in ng/m3) on a
    local-standard-time index, negatives dropped (the minute QC's 3 SD cut would clip real hourly peaks)."""
    head = pd.read_csv(path, nrows=0).columns
    cols = [f"{c} BCc" for c in CHANNELS if f"{c} BCc" in head]
    raw = pd.read_csv(path, usecols=["datetime_local", *cols])
    t = pd.to_datetime(raw["datetime_local"], utc=True, errors="coerce").dt.tz_localize(None) + pd.Timedelta(hours=offset_h)
    out = raw[cols].set_axis(pd.DatetimeIndex(t)).astype(float)
    out = out[out.index.notna()].sort_index()
    return out.where(out >= 0)


def qc(minutes: pd.DataFrame) -> pd.DataFrame:
    """addis_05 / diurnal_pub_figures QC: negatives, then values above mean + 3 SD, set missing."""
    out = minutes.where(minutes >= 0)
    for c in out.columns:
        v = out[c].dropna()
        if len(v) > 1:
            out.loc[out[c] > v.mean() + 3 * v.std(), c] = np.nan
    return out


def hourly_table(minutes: pd.DataFrame) -> pd.DataFrame:
    """QC'd minutes (ng/m3) -> one row per local day-hour: valid IR minutes, mean BCc per channel
    in ug/m3, and AAE(UV, IR), AAE(Blue, IR) of the hour's mean absorption."""
    m = minutes / 1000.0
    m["day"], m["hour"] = day_hour(m.index)[0].to_numpy(), m.index.hour
    g = m.groupby(["day", "hour"])
    out = g[[c for c in m.columns if c.endswith("BCc")]].mean()
    out["n"] = g["IR BCc"].count()
    return with_aae(out[out.n > 0].reset_index())


def hours_table(hours: pd.DataFrame) -> pd.DataFrame:
    """Hourly means (ng/m3) -> the same rows as hourly_table, with no minute count (n missing)."""
    m = hours / 1000.0
    m["day"], m["hour"] = day_hour(m.index)[0].to_numpy(), m.index.hour
    out = m.groupby(["day", "hour"])[[c for c in m.columns if c.endswith("BCc")]].mean()
    out = out[out["IR BCc"].notna()].reset_index()
    out["n"] = np.nan
    return with_aae(out)


def with_aae(out: pd.DataFrame) -> pd.DataFrame:
    """AAE(UV, IR) and AAE(Blue, IR) of each row's mean absorption."""
    babs = {c: out[f"{c} BCc"] * SIGMA[c] for c in CHANNELS if f"{c} BCc" in out}
    for short in ("UV", "Blue"):
        if short in babs and "IR" in babs:
            out[f"aae_{short.lower()}_ir"] = aae(babs[short], babs["IR"], WAVELENGTHS_NM[short], WAVELENGTHS_NM["IR"])
    return out


def window_mean(times: np.ndarray, values: np.ndarray, at: np.ndarray, half: np.timedelta64) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of the finite `values` within +-half of each time in `at` (times sorted)."""
    ok = np.isfinite(values)
    t, v = times[ok], values[ok]
    cum = np.concatenate([[0.0], np.cumsum(v)])
    lo, hi = np.searchsorted(t, at - half, "left"), np.searchsorted(t, at + half, "right")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (cum[hi] - cum[lo]) / n
    return np.where(n > 0, mean, np.nan), n


def matched_tables(minutes: pd.DataFrame, aod: pd.DataFrame, offset_h: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Surface BC around each direct-sun retrieval (+-15 min, QC'd minutes, ug/m3): per day-hour
    means and each day's within-day CV of BC and of AOD."""
    t = minutes.index.to_numpy()
    at = (aod.time_utc + pd.Timedelta(hours=offset_h)).to_numpy()
    half = MATCH_HALF.to_timedelta64()
    ir, n = window_mean(t, minutes["IR BCc"].to_numpy(float) / 1000.0, at, half)
    pts = pd.DataFrame({"local": at, "bc_ir": ir, "n_min": n, "aod500": aod.aod500.to_numpy(), "ae": aod.ae440_870.to_numpy()})
    if "UV BCc" in minutes:
        uv, _ = window_mean(t, minutes["UV BCc"].to_numpy(float) / 1000.0, at, half)
        pts["aae"] = aae(uv * SIGMA["UV"], ir * SIGMA["IR"], WAVELENGTHS_NM["UV"], WAVELENGTHS_NM["IR"])
    else:
        pts["aae"] = np.nan
    pts = pts[(pts.n_min >= MATCH_MIN) & pts.bc_ir.notna()].copy()
    pts["day"], pts["hour"] = day_hour(pts.local)[0].to_numpy(), pd.DatetimeIndex(pts.local).hour
    pts["bc_aod"] = pts.bc_ir / pts.aod500.where(pts.aod500 > 0.01)
    hourly = pts.groupby(["day", "hour"]).agg(n=("bc_ir", "size"), bc_ir=("bc_ir", "mean"), aod500=("aod500", "mean"),
                                              ae=("ae", "mean"), aae=("aae", "mean"), bc_aod=("bc_aod", "mean")).reset_index()
    both = pts.dropna(subset=["aod500"])
    daily = both.groupby("day").agg(n=("bc_ir", "size"), bc_mean=("bc_ir", "mean"), bc_sd=("bc_ir", "std"),
                                    aod_mean=("aod500", "mean"), aod_sd=("aod500", "std")).reset_index()
    daily = daily[daily.n >= CV_MIN]
    daily["bc_cv"], daily["aod_cv"] = daily.bc_sd / daily.bc_mean, daily.aod_sd / daily.aod_mean
    return hourly, daily[["day", "n", "bc_cv", "aod_cv"]]


def coverage_ratios(red: pd.Series, hours: list[int], row_minutes: int = 1) -> pd.DataFrame:
    """ma350_diurnal_coverage.py per day: 24 h mean (median) Red BCc over the mean (median) inside
    the AERONET hours, on days with > DAY_MIN minutes and > HOURS_MIN of them inside the hours.
    `row_minutes` is how many minutes a row stands for (60 for hourly means, whose median is then of hours)."""
    x = pd.DataFrame({"bc": red.to_numpy(float), "day": day_hour(red.index)[0].to_numpy(), "hour": red.index.hour}).dropna()
    sub = x[x.hour.isin(hours)]
    full, inside = x.groupby("day").bc, sub.groupby("day").bc
    n_full, n_in = full.size() * row_minutes, inside.size().reindex(full.size().index, fill_value=0) * row_minutes
    ok = (n_full > DAY_MIN) & (n_in > HOURS_MIN)
    out = pd.DataFrame({"minutes": n_full, "minutes_in": n_in}).loc[ok[ok].index]
    for stat in ("mean", "median"):
        ratio = getattr(full, stat)() / getattr(inside, stat)()
        out[f"ratio_{stat}"] = ratio.replace([np.inf, -np.inf], np.nan).reindex(out.index)
    return out.reset_index()


def surface_payload(raw: pd.DataFrame, aod: pd.DataFrame | None, inv_hours: list[int], offset_h: float, resolution: str = "minute") -> dict:
    """Everything the page draws for one site's surface, from its minutes or hourly means (local
    standard time, ng/m3). Hourly means give the cycles and the coverage ratio; pairing with single
    retrievals (+-15 min) and the within-day CV need minutes."""
    minute = resolution == "minute"
    clean = qc(raw) if minute else raw
    hourly = hourly_table(clean) if minute else hours_table(clean)
    if hourly.empty:
        raise ValueError("no valid IR values")
    tables = {"hourly": hourly}
    if minute and aod is not None and len(aod):
        tables["matched"], tables["intraday"] = matched_tables(clean, aod, offset_h)
    cov = coverage_ratios(raw["Red BCc"], inv_hours, 1 if minute else 60) if inv_hours and "Red BCc" in raw else None
    if cov is not None:
        tables["coverage"] = cov
    first = min(t.day.min() for t in tables.values() if len(t))
    last = max(t.day.max() for t in tables.values() if len(t))
    out = dict(first=first, last=last, days=int(hourly.day.nunique()), resolution=resolution, bcc_unit="ng/m3",
               hourly=dict(dd=dd(hourly.day, first), h=hourly.hour.tolist(), **({"n": scaled(hourly.n, 1)} if minute else {}),
                           **{c: scaled(hourly[f"{c} BCc"], 1000) for c in CHANNELS if f"{c} BCc" in hourly},
                           **{k: col(hourly[k], 2) for k in ("aae_uv_ir", "aae_blue_ir") if k in hourly}))
    if "matched" in tables:
        m, dly = tables["matched"], tables["intraday"]
        out["matched"] = dict(dd=dd(m.day, first), h=m.hour.tolist(), n=m.n.tolist(), bc_ir=col(m.bc_ir, 4),
                              aod500=col(m.aod500, 3), ae=col(m.ae, 2), aae=col(m.aae, 3), bc_aod=col(m.bc_aod, 3))
        out["intraday"] = dict(dd=dd(dly.day, first), n=dly.n.tolist(), bc_cv=col(dly.bc_cv, 3), aod_cv=col(dly.aod_cv, 3))
    if cov is not None:
        out["coverage"] = dict(hours=inv_hours, dd=dd(cov.day, first), minutes=cov.minutes.tolist(), minutes_in=cov.minutes_in.tolist(),
                               ratio_mean=col(cov.ratio_mean, 3), ratio_median=col(cov.ratio_median, 3))
    return out


# ---------------------------------------------------------------------------------------- ftir_28

# ftir_19 cell 5's six calibration setups (intercept ug/m3, slope at MAC 10), as ftir_28 cell 9 quotes them
FTIR19_SETUPS = [("Deployed SPARTAN", -4.17, 1.90), ("Biomass-smoke (906)", -6.91, 2.65), ("Ethiopia-shaped smoke (300)", -3.69, 1.75),
                 ("Spectral analogs (400)", -6.43, 2.91), ("Lowest-OC/EC (800)", -3.22, 1.59), ("Lowest-OC/EC + AIRSpec", -1.62, 0.86)]


def brc_test(daily: pd.DataFrame) -> dict:
    """ftir_28's test on the daily MA350 record: absorption, AAE(625, 880), the implied BrC absorption
    against the intercept's target, and the channel checks."""
    b = pd.DataFrame({c: daily[f"{c} BCc"].astype(float) * SIGMA[c] * 1e-3 for c in CHANNELS})   # Mm-1, b_ATN scale
    wl = WAVELENGTHS_NM
    a = {c: aae(b[c], b["IR"], wl[c], wl["IR"]) for c in ("UV", "Blue", "Green", "Red")}
    brc = b["Red"] - b["IR"] * (wl["IR"] / wl["Red"]) ** 1.0
    targets = [abs(i) * 10 / s for _, i, s in FTIR19_SETUPS]
    target = float(np.mean(targets))
    uv_out = (daily["UV BCc"] <= 0) | (b["UV"] < b["IR"])
    aae_blue_red = -np.log(b["Blue"] / b["Red"]) / np.log(wl["Blue"] / wl["Red"])
    green_expected = b["Blue"] * (wl["Green"] / wl["Blue"]) ** (-aae_blue_red)
    rel = (b["Red"] / b["IR"]) / (b["Red"] / b["IR"]).median() - 1

    def mean_brc(aae_bc: float) -> float:
        return float((b["Red"] - b["IR"] * (wl["IR"] / wl["Red"]) ** aae_bc).mean())

    def closing(t: float) -> float:
        """The AAE_BC that makes the mean implied BrC equal t (bisection; the mean falls as AAE_BC rises)."""
        lo, hi = -3.0, 6.0
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean_brc(mid) > t else (lo, mid)
        return (lo + hi) / 2

    corr = {c: r(daily[f"{c} BCc"].astype(float).corr(daily["IR BCc"].astype(float)), 4) for c in ("UV", "Blue", "Green", "Red")}
    summary = dict(
        days=int(a["Red"].notna().sum()),
        aae_red_ir=dict(mean=r(a["Red"].mean(), 4), sd=r(a["Red"].std(), 4), median=r(a["Red"].median(), 4),
                        above_anchor_pct=r(100 * (a["Red"] > 1).mean(), 2)),
        brc=dict(mean=r(brc.mean(), 3), median=r(brc.median(), 3), negative_pct=r(100 * (brc < 0).mean(), 2),
                 at_target_pct=r(100 * (brc >= target).mean(), 2)),
        target=dict(mean=r(target, 3), min=r(min(targets), 3), max=r(max(targets), 3),
                    setups=[dict(setup=n, intercept=i, slope=s, C=r(c, 3)) for (n, i, s), c in zip(FTIR19_SETUPS, targets)]),
        closing_aae=dict(mean_target=r(closing(target), 4), per_setup=[r(closing(c), 3) for c in targets]),
        uv=dict(out_of_range_pct=r(100 * uv_out.mean(), 2), nonpositive=int((daily["UV BCc"] <= 0).sum()),
                below_ir=int(((b["UV"] < b["IR"]) & ~(daily["UV BCc"] <= 0)).sum()), r_ir=corr["UV"]),
        green=dict(aae_mean=r(a["Green"].mean(), 4), aae_median=r(a["Green"].median(), 4), negative_pct=r(100 * (a["Green"] < 0).mean(), 2),
                   deficit_pct=r(-100 * ((b["Green"] / green_expected).median() - 1), 2), r_ir=corr["Green"]),
        red=dict(iqr_halfwidth_pct=r(100 * (rel.quantile(.75) - rel.quantile(.25)) / 2, 3),
                 bcc_ratio_median=r((daily["Red BCc"] / daily["IR BCc"]).median(), 4), r_ir=corr["Red"]),
        blue=dict(aae_mean=r(a["Blue"].mean(), 4), r_ir=corr["Blue"]),
    )
    return dict(
        date=pd.to_datetime(daily["day_9am"]).dt.strftime("%Y-%m-%d").tolist(),
        b={c: col(b[c], 3) for c in CHANNELS},
        aae={c: col(a[c], 4) for c in a},
        brc=col(brc, 3), uv_out=[bool(v) for v in uv_out],
        summary=summary,
    )


# ---------------------------------------------------------------------------------------- main

def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))


def find_source(code: str, overrides: dict[str, Path]) -> tuple[str, Path] | None:
    """(resolution, path) of the first surface record found for a site."""
    if code in overrides:
        p = overrides[code]
        return ("hourly" if "hourly" in p.name else "minute"), p
    try:
        from data_paths import aethalometry_dir
        base = aethalometry_dir()
    except Exception as exc:  # no Drive, no .env: the surface part is skipped
        print(f"{code}: aethalometry_dir() did not resolve ({exc})")
        return None
    for kind, rel in SOURCES.get(code, []):
        if (base / rel).is_file():
            return kind, base / rel
    print(f"{code}: nothing under {base} ({', '.join(rel for _, rel in SOURCES.get(code, [])) or 'none registered'})")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aeronet-dir", type=Path, default=AERONET_DIR)
    ap.add_argument("--minute", action="append", default=[], metavar="CODE=PATH", help="a minute file for a site (pickle or MA350 CSV)")
    ap.add_argument("--skip-aeth", action="store_true")
    ap.add_argument("--skip-aeronet", action="store_true")
    args = ap.parse_args()
    from export_weather import std_offset_hours

    generated = datetime.now(timezone.utc).isoformat()
    smap = {s["code"]: s for s in json.loads((ROOT / "gallery/app/public/data/spartan_map.json").read_text())["sites"]}
    offsets = {c: std_offset_hours(s["lat"], s["lon"]) for c, s in smap.items() if s.get("lat") is not None}
    cache = args.aeronet_dir
    have_aeronet = (cache / f"aod_points_lev{LEVEL}.parquet").is_file()
    points = {}
    if have_aeronet:
        stations = pd.read_csv(cache / f"stations_lev{LEVEL}.csv")
        chosen = stations[stations.chosen].set_index("code").station.to_dict()
        for p, cols in (("aod", ["site", "station", "time_utc", "aod500", "ae440_870"]), ("sda", ["site", "station", "time_utc", "fmf500"]),
                        ("inv", ["site", "station", "time_utc"])):
            f = pd.read_parquet(cache / f"{p}_points_lev{LEVEL}.parquet", columns=cols)
            f["time_utc"] = pd.to_datetime(f.time_utc)
            f = f[f.station == f.site.map(chosen)]
            points[p] = {c: g.sort_values("time_utc").reset_index(drop=True) for c, g in f.groupby("site")}
    else:
        print(f"no AERONET points under {cache}: run research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py")

    # 1. the column
    inv_hours: dict[str, list[int]] = {}
    if have_aeronet and not args.skip_aeronet:
        index = {}
        for code in sorted(points["aod"]):
            if code not in offsets:
                continue
            empty = points["aod"][code].iloc[:0]
            t = column_tables(points["aod"][code], points["sda"].get(code, empty.assign(fmf500=[])),
                              points["inv"].get(code, empty[["time_utc"]]), offsets[code])
            inv_hours[code] = t["inv_hours"]
            write(OUT / "aeronet" / f"{code}.json", dict(site=code, station=points["aod"][code].station.iloc[0], offset=offsets[code], **t))
            index[code] = dict(station=points["aod"][code].station.iloc[0], offset=offsets[code], days=t["days"],
                               first=t["first"], last=t["last"], retrievals=int(sum(t["aod"]["n"])),
                               inversions=int(sum(t["inv"]["n"])), inv_hours=t["inv_hours"], hour_counts=t.pop("hour_counts"),
                               hours_per_day=t.pop("hours_per_day"))
            print(f"{code}: {index[code]['retrievals']} retrievals on {t['days']} days, inversion hours {t['inv_hours']}")
        write(OUT / "aeronet/index.json", dict(
            generated=generated, level=f"Level {LEVEL[0]}.{LEVEL[1]}", source="NASA AERONET Version 3 direct-sun, SDA and almucantar retrievals "
            "(research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py), the station chosen for each SPARTAN site",
            note=f"Local standard time. Inversion hours: the hours holding at least {INV_HOUR_SHARE:.0%} of the site's inversions.", sites=index))
    elif have_aeronet:
        for code, g in points["inv"].items():
            if code in offsets:
                inv_hours[code] = inversion_hours((g.time_utc + pd.Timedelta(hours=offsets[code])).dt.hour.value_counts())

    # 2. the surface
    if not args.skip_aeth:
        overrides = dict((k, Path(v)) for k, v in (m.split("=", 1) for m in args.minute))
        index = {}
        for code in sorted(set(SOURCES) | set(overrides)):
            found = find_source(code, overrides)
            if found is None or code not in offsets:
                continue
            kind, path = found
            print(f"{code}: reading {path.name} ({kind})")
            minutes = (read_minutes if kind == "minute" else read_hourly)(path, offsets[code])
            aod = points.get("aod", {}).get(code)
            try:
                payload = surface_payload(minutes, aod, inv_hours.get(code, []), offsets[code], kind)
            except ValueError as exc:
                print(f"{code}: {path.name} gave nothing usable ({exc})")
                continue
            finally:
                gc.collect()
            write(OUT / "aeth" / f"{code}.json", dict(site=code, file=path.name, offset=offsets[code], **payload))
            index[code] = dict(file=path.name, resolution=kind, records=int(len(minutes)), days=payload["days"], first=payload["first"],
                               last=payload["last"], matched="matched" in payload, coverage_hours=inv_hours.get(code, []))
            print(f"{code}: {len(minutes):,} {kind} records, {payload['days']} days")
            del minutes
        # the index is written either way: with no minute data it lists what was looked for, so the page
        # can say what is missing without requesting files that do not exist
        previous = OUT / "aeth/index.json"
        old = json.loads(previous.read_text())["sites"] if previous.is_file() else {}
        write(previous, dict(
            generated=generated, source="MA350 records on Google Drive (data_paths.aethalometry_dir()): minute exports where there are, hourly means otherwise",
            note="Local standard time. Hourly means of QC'd minutes (negatives and > mean + 3 SD dropped) or the hourly-means files (negatives dropped), in ug/m3; AAE from "
                 "absorption (BCc x MA350 cross-section). Matched: +-15 min of each AERONET direct-sun retrieval. Coverage: raw "
                 "Red BCc, 24 h over the AERONET inversion hours (ma350_diurnal_coverage.py).",
            searched={c: [rel for _, rel in SOURCES.get(c, [])] for c in sorted(set(SOURCES) | set(overrides))},
            sites={**old, **index}))
        if not index:
            print("no minute data found: the aethalometer hour-of-day files were not written")

    # 3. the brown-carbon test, from the committed daily record
    daily = PROCESSED_SITES_DIR / "df_Addis_Ababa_9am_resampled.pkl"
    if daily.is_file():
        t = brc_test(pd.read_pickle(daily))
        write(OUT / "brc_test.json", dict(generated=generated, source=f"research/ftir_hips_chem/processed_sites/{daily.name} (MA350-0238, ETAD filter-days)",
                                          sigma_atn=SIGMA, wavelengths_nm=WAVELENGTHS_NM, **t))
        s = t["summary"]
        print(f"brc_test: {s['days']} days, AAE(625,880) {s['aae_red_ir']['mean']} ± {s['aae_red_ir']['sd']}, "
              f"BrC {s['brc']['mean']} Mm-1 ({s['brc']['negative_pct']}% negative)")


if __name__ == "__main__":
    main()
