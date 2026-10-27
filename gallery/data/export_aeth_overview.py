"""Hourly black carbon, high-BC episodes and the excluded samples, for the Aethalometer tab's Overview page.

Run with:  uv run python gallery/data/export_aeth_overview.py [--top 16]

What addis_02_temporal_patterns.ipynb did from the Jacros minute data beyond a daily view, for every
focus site: the week inside each season (hour x weekday), when the day peaks, the hours above the 95th
percentile, and the high-BC episodes at minute resolution. Also Multi_Site_Analysis_FollowUp 7b's
excluded dates (outliers.EXCLUDED_SAMPLES), which the page looks up in the daily files.

Inputs, relative to research/ftir_hips_chem/scripts/data_paths.aethalometry_dir() (the repo .env's
AETHMODULAR_MAIA_DATA_ROOT + "Aethalometry Data"), the same files the Diurnal page's exporter reads:
  Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv                ETAD (MA350-0238), minutes: hourly series and traces
  Raw/Pasadena_MA350_1-min_2023-2024_Cleaned.csv              USPA (MA350-0229), minutes: traces only
  Kyan Data/MAIA data/{Beijing,Delhi,JPL}_cleaned_corrected_small_hourly.csv   CHTS, INDH, USPA hourly means
The raw CSVs' "Time local" is the UTC clock ("Timezone offset (mins)" is 0), so "Time (UTC)" is read
and shifted to the site's local standard time; the hourly files carry a zone-aware `datetime_local`.
JPL's hourly series is the MAIA file (Nov 2021 - Sep 2024, the filter era), not the shorter Pasadena
minutes (Jun 2023 - May 2024), which only supply episode traces; the two are compared on their overlap.

Rules (stated on the page):
  minutes   every finite minute, negatives included (minute noise averages out over an hour); an hour
            needs at least MIN_MINUTES valid IR minutes. Not the notebooks' screen (addis_02, addis_05,
            and the Diurnal export drop negatives and minutes above mean + 3 SD of the record): at Addis
            that limit is ~35 ug/m3, so it removes the tops of exactly the episodes this page is about
            and caps the hourly series near 32 ug/m3 where the unscreened hours reach ~83. The exporter
            prints what the screen would have done.
  hourly    negatives dropped (the files are already cleaned hourly means)
  clock     local standard time (no daylight saving), as the weather, AERONET and Diurnal exports
  episode   a run of at least EPISODE_MIN_HOURS consecutive hours with IR BCc above the record's
            95th-percentile hour; a missing hour ends the run

Writes gallery/app/public/data/aethalometer/overview/:
  index.json            per site: source files, span, threshold, the episodes; exclusions per site; rules
  hourly/<CODE>.json    IR BCc per local-standard hour on a contiguous grid (ng/m3 integers, null missing)
  traces/<CODE>.json    for the top episodes a minute file covers: the minute IR BCc from 6 h before to 6 h after
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "research/ftir_hips_chem"
sys.path.insert(0, str(AREA / "scripts"))
from config import SITES  # noqa: E402
from data_paths import aethalometry_dir  # noqa: E402
from outliers import EXCLUDED_SAMPLES  # noqa: E402

OUT = ROOT / "gallery/app/public/data/aethalometer/overview"
HOURLY = {
    "ETAD": ("minute", "Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv"),
    "USPA": ("hourly", "Kyan Data/MAIA data/JPL_cleaned_corrected_small_hourly.csv"),
    "CHTS": ("hourly", "Kyan Data/MAIA data/Beijing_cleaned_corrected_small_hourly.csv"),
    "INDH": ("hourly", "Kyan Data/MAIA data/Delhi_cleaned_corrected_small_hourly.csv"),
}
MINUTES = {
    "ETAD": "Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv",
    "USPA": "Raw/Pasadena_MA350_1-min_2023-2024_Cleaned.csv",
}
CODE_SITE = {cfg["code"]: (name, cfg["timezone"]) for name, cfg in SITES.items()}
MIN_MINUTES = 30
EPISODE_Q = 0.95
EPISODE_MIN_HOURS = 2
TRACE_PAD = pd.Timedelta(hours=6)


def standard_offset(tz: str) -> float:
    """The zone's UTC offset outside daylight saving, in hours (January; every focus site is northern or has none)."""
    return pd.Timestamp("2023-01-15", tz=tz).utcoffset().total_seconds() / 3600


def read_minutes(path: Path, offset_h: float, cols=("IR BCc",)) -> pd.DataFrame:
    """A raw MA350 minute CSV as BCc (ng/m3) on a local-standard-time index, from "Time (UTC)"."""
    parts = []
    for ch in pd.read_csv(path, usecols=["Time (UTC)", *cols], chunksize=500_000):
        t = pd.to_datetime(ch["Time (UTC)"], utc=True, errors="coerce").dt.tz_localize(None) + pd.Timedelta(hours=offset_h)
        parts.append(ch[list(cols)].set_axis(pd.DatetimeIndex(t)).astype(float))
    out = pd.concat(parts)
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated()]


def qc(minutes: pd.DataFrame) -> pd.DataFrame:
    """Negatives, then values above mean + 3 SD of the (non-negative) record, set missing."""
    out = minutes.where(minutes >= 0)
    for c in out.columns:
        v = out[c].dropna()
        if len(v) > 1:
            out.loc[out[c] > v.mean() + 3 * v.std(), c] = np.nan
    return out


def hourly_from_minutes(ir: pd.Series, min_minutes: int = MIN_MINUTES) -> pd.Series:
    """Mean IR per local hour where at least `min_minutes` minutes are valid; a contiguous hourly grid."""
    g = ir.groupby(ir.index.floor("h"))
    mean = g.mean().where(g.count() >= min_minutes)
    return grid(mean)


def read_hourly(path: Path, offset_h: float) -> pd.Series:
    """A MAIA hourly-means CSV (zone-aware datetime_local, ng/m3) as IR on a local-standard grid, negatives dropped."""
    raw = pd.read_csv(path, usecols=["datetime_local", "IR BCc"])
    t = pd.to_datetime(raw["datetime_local"], utc=True, errors="coerce").dt.tz_localize(None) + pd.Timedelta(hours=offset_h)
    s = pd.Series(raw["IR BCc"].astype(float).to_numpy(), index=pd.DatetimeIndex(t)).dropna()
    s = s[s.index.notna()]
    s = s.where(s >= 0).groupby(s.index.floor("h")).mean()
    return grid(s)


def grid(s: pd.Series) -> pd.Series:
    """The series on every hour from its first valid hour to its last (missing hours NaN)."""
    s = s.dropna()
    if s.empty:
        return s
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))


def episodes(hourly: pd.Series, q: float = EPISODE_Q, min_hours: int = EPISODE_MIN_HOURS) -> tuple[float, list[dict]]:
    """Runs of at least `min_hours` consecutive hours above the series' q-quantile (a missing hour ends a run).

    Each: its first hour, length, the peak hour's value and position, mean, the excess above the
    threshold summed over its hours (ng/m3 h), and the hours from start to peak and peak to end."""
    v = hourly.to_numpy(float)
    thr = float(np.nanquantile(v, q))
    above = np.isfinite(v) & (v > thr)
    out = []
    i = 0
    while i < len(v):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(v) and above[j + 1]:
            j += 1
        n = j - i + 1
        if n >= min_hours:
            seg = v[i:j + 1]
            k = int(np.argmax(seg))
            out.append(dict(start=hourly.index[i], hours=n, peak=float(seg[k]), peak_at=k, mean=float(seg.mean()),
                            excess=float((seg - thr).sum()), rise=k, fall=n - 1 - k))
        i = j + 1
    return thr, out


def trace(minutes: pd.Series, start: pd.Timestamp, hours: int) -> dict | None:
    """Minute IR from TRACE_PAD before an episode to TRACE_PAD after it, on a whole-minute grid."""
    a, b = start - TRACE_PAD, start + pd.Timedelta(hours=hours) + TRACE_PAD
    seg = minutes.loc[(minutes.index >= a) & (minutes.index < b)]
    if seg.notna().sum() < 60 * hours * 0.5:
        return None
    seg = seg.groupby(seg.index.floor("min")).mean().reindex(pd.date_range(a, b, freq="min", inclusive="left"))
    return dict(t0=a.strftime("%Y-%m-%d %H:%M"), IR=ints(seg))


def ints(values) -> list:
    return [None if not np.isfinite(x) else int(round(x)) for x in np.asarray(values, dtype=float)]


def exclusions() -> dict[str, list[dict]]:
    """outliers.EXCLUDED_SAMPLES keyed by site code."""
    code = {name: cfg["code"] for name, cfg in SITES.items()}
    return {code[site]: [dict(date=e["date"], filter_id=e.get("filter_id"), reason=e["reason"]) for e in items]
            for site, items in EXCLUDED_SAMPLES.items() if site in code}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--top", type=int, default=16, help="episodes per site with a minute trace (by peak)")
    args = ap.parse_args()
    base = aethalometry_dir()
    for sub in ("hourly", "traces"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    sites = {}
    for code, (kind, rel) in HOURLY.items():
        path = base / rel
        if not path.exists():
            print(f"{code}: {rel} not found, skipped")
            continue
        site, tz = CODE_SITE[code]
        off = standard_offset(tz)
        minutes = None
        if code in MINUTES and (base / MINUTES[code]).exists():
            minutes = read_minutes(base / MINUTES[code], off)["IR BCc"]
            screened = hourly_from_minutes(qc(minutes.to_frame())["IR BCc"])
            plain = hourly_from_minutes(minutes)
            print(f"  {code}: the notebooks' minute screen would cut {(minutes > minutes[minutes >= 0].mean() + 3 * minutes[minutes >= 0].std()).mean():.1%} "
                  f"of minutes and cap the highest hour at {screened.max():.0f} ng/m3 (unscreened {plain.max():.0f}; P95 {screened.quantile(EPISODE_Q):.0f} vs {plain.quantile(EPISODE_Q):.0f})")
        hourly = hourly_from_minutes(minutes) if kind == "minute" else read_hourly(path, off)
        thr, eps = episodes(hourly)
        if kind == "hourly" and minutes is not None:   # the Pasadena minutes against the MAIA hours, where both are
            both = pd.concat([hourly_from_minutes(minutes), hourly], axis=1, keys=["min", "hour"]).dropna()
            if len(both) > 24:
                print(f"  {code}: minute-file hours vs hourly file on {len(both)} hours: r = {both['min'].corr(both['hour']):.3f}, "
                      f"median ratio {np.median(both['min'] / both['hour']):.3f}")
        traces = []
        if minutes is not None:
            for k, e in sorted(enumerate(eps), key=lambda x: -x[1]["peak"]):
                if len(traces) >= args.top:
                    break
                tr = trace(minutes, e["start"], e["hours"])
                if tr:
                    traces.append(dict(episode=k, **tr))
        (OUT / "hourly" / f"{code}.json").write_text(json.dumps(dict(
            site=code, first=hourly.index[0].strftime("%Y-%m-%d %H:00"), offset=off, unit="ng/m3", IR=ints(hourly)), separators=(",", ":")))
        (OUT / "traces" / f"{code}.json").write_text(json.dumps(dict(site=code, unit="ng/m3", pad_hours=TRACE_PAD.total_seconds() / 3600,
                                                                     traces=traces), separators=(",", ":")))
        sites[code] = dict(
            site=site, resolution=kind, hourly_file=rel, minute_file=MINUTES.get(code) if minutes is not None else None, offset=off,
            first=hourly.index[0].strftime("%Y-%m-%d %H:00"), last=hourly.index[-1].strftime("%Y-%m-%d %H:00"),
            hours=int(len(hourly)), valid_hours=int(hourly.notna().sum()), threshold=round(thr), traced=len(traces),
            episodes=dict(start=[e["start"].strftime("%Y-%m-%d %H:00") for e in eps], hours=[e["hours"] for e in eps],
                          peak=[round(e["peak"]) for e in eps], peak_at=[e["peak_at"] for e in eps], mean=[round(e["mean"]) for e in eps],
                          excess=[round(e["excess"]) for e in eps]))
        print(f"{code} ({site}): {sites[code]['valid_hours']} of {len(hourly)} hours, P95 {thr:.0f} ng/m3, {len(eps)} episodes, {len(traces)} traced")
    index = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        source="MA350 records on the Drive mirror (data_paths.aethalometry_dir()): Jacros minute CSV for Addis, MAIA hourly means for Beijing, Delhi and JPL",
        note=(f"Local standard time. Minutes: every finite minute (no 3 SD screen, which would cut the episodes' tops), hours kept with at least {MIN_MINUTES} valid IR minutes; "
              f"hourly files: negatives dropped. Episodes: at least {EPISODE_MIN_HOURS} consecutive hours above the record's {EPISODE_Q:.0%} hour."),
        rules=dict(min_minutes=MIN_MINUTES, episode_quantile=EPISODE_Q, episode_min_hours=EPISODE_MIN_HOURS, trace_pad_hours=TRACE_PAD.total_seconds() / 3600),
        sites=sites, exclusions=exclusions())
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    size = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(f"{size / 1e6:.2f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
