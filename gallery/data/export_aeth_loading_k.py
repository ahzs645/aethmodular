"""Kyan Shlipak's manual loading compensation for the MA350, on a grid of k, for the Experimental page "Aethalometer loading k".

    uv run python gallery/data/export_aeth_loading_k.py [--minute CODE=PATH ...]

What the MA350 does (AethLabs manual; Drinovec et al. 2015): spot 1 takes most of the flow, spot 2 the rest, and the
instrument solves for the loading constant K from the two spots every minute, then reports BCc = BC1 / (1 - K * ATN1).
That needs the flows to differ (target Flow1:Flow2 = 3, 75:25). At Jacros (Addis, MA350-0238) the split sits near
55:45, so the two spots load almost alike and the live K is noisy. Kyan's fix (EC_BC deck; src/data/processors/
site_corrections.py, _apply_manual_loading_compensation_ethiopia): one fixed k for the record,

    BCc(k) = BC1 / (1 - k * ATN1)   where ATN1 > 3,   BC1 otherwise,

with k the median of the instrument's K over ATN1 > 3, or the k that leaves corrected BC uncorrelated with ATN1
(golden-section search on 0.004..0.016 in his deck; 0.01185 for IR at Jacros), or the k that leaves the FTIR EC - BC
residuals uncorrelated with ATN1 (0.01172).

This exporter evaluates that formula on a grid of k so the page can move k freely:
  per filter     mean BCc(k) over the filter's own sampling window (half its minutes needed), the instrument BCc, BC1, BC2,
                 mean ATN1, with the filter's FTIR EC and HIPS BC (spartan_reference.json)
  per k          |r| of BCc(k) with ATN1 over all minutes (Kyan's criterion, no FTIR), and the step across tape advances:
                 median of BCc(k) 5..15 min after an advance over 15..5 min before, which is 1 when k removes the loading
                 effect (the fresh spot reads what the loaded one should have)
  per hour       hour-of-day mean BCc(k) and instrument BCc (local time)
  distributions  ATN1 of every minute (the page draws 1 - k*ATN1 from it), the instrument's K and 1 - K*ATN1, Flow1/Flow2

A minute whose denominator 1 - k*ATN1 falls to 0.1 or below (a factor over 10) is left out at that k and counted.
Inputs: the raw minute CSVs of export_aeth_quality_minute.py (Drive, Aethalometry Data/Raw), weather/filters.json for
the sampling windows, spartan_reference.json. Writes gallery/app/public/data/experimental/aeth_loading_k.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_aeth_quality_minute as minute  # noqa: E402  (and, through it, research/ftir_hips_chem/scripts)

ROOT = minute.ROOT
OUT = ROOT / "gallery/app/public/data/experimental/aeth_loading_k.json"
CHANNELS = ["IR", "Red", "Blue"]
K_GRID = np.round(np.arange(0, 0.02 + 1e-9, 0.0005), 4)
ATN_MIN = 3.0          # Kyan: below ATN1 3 the minute keeps BC1
FLOOR = 0.1            # denominators at or under this are left out (factor > 10)
SITES = {   # code -> (label, CSV under aethalometry_dir(), zone, filter site)
    "ETAD-JACROS": ("Addis Ababa (Jacros, MA350-0238)", "Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv", "Africa/Addis_Ababa", "ETAD"),
    "USPA": ("Pasadena (JPL, MA350-0229)", "Raw/Pasadena_MA350_1-min_2023-2024_Cleaned.csv", "America/Los_Angeles", "USPA"),
}
# Kyan's numbers for Jacros IR (EC_BC deck, slide 5), drawn as reference marks
KYAN = {"median": 0.0129, "bc_atn": 0.011846, "residual": 0.011723}


def r(v, nd=4):
    return None if v is None or not np.isfinite(v) else float(round(float(v), nd))


def col(a, nd=4):
    return [r(v, nd) for v in a]


def hist(values, lo, hi, step):
    edges = np.round(np.arange(lo, hi + step / 2, step), 6)
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    counts, _ = np.histogram(np.clip(v, lo, hi - 1e-12), edges)
    return {"lo": lo, "step": step, "counts": counts.astype(int).tolist(), "n": int(len(v))}


def corrected(bc1: np.ndarray, atn: np.ndarray, k: float) -> np.ndarray:
    den = np.where(atn > ATN_MIN, 1 - k * atn, 1.0)
    out = bc1 / den
    out[~(den > FLOOR)] = np.nan
    return out


def site(df: pd.DataFrame, windows: dict, ref: dict, filter_site: str, zone: str) -> dict:
    df = minute.add_segments(df)
    t_local = df["t"]
    hour = t_local.dt.hour.to_numpy()
    tv = df["tu"].to_numpy()
    utc = lambda x: x.tz_localize(zone, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC").tz_localize(None)  # noqa: E731

    # sampling windows of this site's filters inside the record
    lo_t, hi_t = t_local.min(), t_local.max()
    fil = []
    for fid, w in sorted(windows.items()):
        if not fid.startswith(filter_site + "-"):
            continue
        a, b = pd.Timestamp(w[0]), pd.Timestamp(w[1])
        if b < lo_t or a > hi_t:
            continue
        i0, i1 = np.searchsorted(tv, np.datetime64(utc(a))), np.searchsorted(tv, np.datetime64(utc(b)))
        fil.append((fid, a, b, i0, i1, 0.5 * (b - a) / pd.Timedelta(minutes=1)))

    # tape advances within a session: the first minute of every segment after the first
    seg_start = df.groupby("segment")["tu"].min().iloc[1:].to_numpy()
    session = df["Session ID"].to_numpy()
    first_idx = np.searchsorted(tv, seg_start)
    same_session = session[first_idx] == session[np.maximum(first_idx - 1, 0)]
    adv = seg_start[same_session]

    out = {"minutes": int(len(df)), "span": [lo_t.strftime("%Y-%m-%d"), hi_t.strftime("%Y-%m-%d")], "advances": int(len(adv)),
           "flow_ratio": hist((df["Flow1 (mL/min)"] / df["Flow2 (mL/min)"]).to_numpy(float), 0, 5, 0.05) if "Flow2 (mL/min)" in df else None,
           "channels": {}}
    filters = {"id": [f[0] for f in fil], "date": [f[1].strftime("%Y-%m-%d") for f in fil],
               "hours": [r((f[2] - f[1]) / pd.Timedelta(hours=1), 1) for f in fil],
               "ec": [r(ref.get(f[0], {}).get("ec"), 3) for f in fil], "hips": [r(ref.get(f[0], {}).get("hips"), 3) for f in fil]}
    for ch in CHANNELS:
        bc1 = df[f"{ch} BC1"].to_numpy(float)
        atn = df[f"{ch} ATN1"].to_numpy(float)
        bcc = df[f"{ch} BCc"].to_numpy(float)
        bc2 = df[f"{ch} BC2"].to_numpy(float) if f"{ch} BC2" in df else np.full(len(df), np.nan)
        K = df[f"{ch} K"].to_numpy(float)
        ok = np.isfinite(bc1) & np.isfinite(atn)
        corr, left_out, step, step_lo, step_hi, hourly = [], [], [], [], [], []
        win = []
        for k in K_GRID:
            v = corrected(bc1, atn, k)
            m = ok & np.isfinite(v)
            corr.append(r(np.corrcoef(v[m], atn[m])[0, 1], 5) if m.sum() > 100 else None)
            left_out.append(int((ok & ~np.isfinite(v)).sum()))
            pre, post = minute.window_means(tv, v, adv)
            with np.errstate(divide="ignore", invalid="ignore"):
                rat = post / pre
            rat = rat[np.isfinite(rat) & (pre > 0) & (post > 0)]
            step.append(r(np.median(rat), 4) if len(rat) else None)
            step_lo.append(r(np.quantile(rat, 0.25), 4) if len(rat) else None)
            step_hi.append(r(np.quantile(rat, 0.75), 4) if len(rat) else None)
            hourly.append(col(pd.Series(v).groupby(hour).mean().reindex(range(24)).to_numpy() / 1000, 4))
            means = []
            for _, _, _, i0, i1, need in fil:
                s = v[i0:i1]
                g = np.isfinite(s)
                means.append(r(s[g].mean() / 1000, 4) if g.sum() >= need and need > 0 else None)
            win.append(means)
        win_arr = np.array([[np.nan if x is None else x for x in row] for row in win])     # k x filter

        def wmean(a):
            res = []
            for _, _, _, i0, i1, need in fil:
                s = a[i0:i1]
                g = np.isfinite(s)
                res.append(r(s[g].mean(), 4) if g.sum() >= need and need > 0 else None)
            return res

        kk = K[np.isfinite(K) & (atn > ATN_MIN)]
        inst_den = np.where(atn > ATN_MIN, 1 - K * atn, np.nan)
        out["channels"][ch] = {
            "corr": corr, "left_out": left_out, "step": step, "step_q25": step_lo, "step_q75": step_hi,
            "advances_used": int(np.isfinite(minute.window_means(tv, bc1, adv)[0]).sum()),
            "hourly": hourly, "hourly_instrument": col(pd.Series(bcc).groupby(hour).mean().reindex(range(24)).to_numpy() / 1000, 4),
            "corr_instrument": r(np.corrcoef(bcc[np.isfinite(bcc) & np.isfinite(atn)], atn[np.isfinite(bcc) & np.isfinite(atn)])[0, 1], 5),
            "k_median": r(np.median(kk), 6), "k_q25": r(np.quantile(kk, 0.25), 6), "k_q75": r(np.quantile(kk, 0.75), 6),
            "atn": hist(atn[ok], 0, 160, 1), "atn_over3": int((ok & (atn > ATN_MIN)).sum()),
            "k_hist": hist(kk, -0.01, 0.05, 0.0005), "den_instrument": hist(inst_den, -1, 2, 0.02),
            "filters": {
                "bc_k": [[None if not np.isfinite(x) else r(x, 4) for x in win_arr[:, j]] for j in range(len(fil))],   # filter x k
                "bc_instrument": [None if x is None else r(x / 1000, 4) for x in wmean(bcc)],
                "bc1": [None if x is None else r(x / 1000, 4) for x in wmean(bc1)],
                "bc2": [None if x is None else r(x / 1000, 4) for x in wmean(bc2)],
                "atn": wmean(np.where(ok, atn, np.nan)),
            },
        }
        print(f"    {ch}: median K {out['channels'][ch]['k_median']}, advances {out['channels'][ch]['advances_used']}", flush=True)
    out["filters"] = filters
    return out


def reference() -> dict:
    rows = json.loads((ROOT / "gallery/app/public/data/spartan_reference.json").read_text())["rows"]
    ref = {}
    for row in rows:
        ec, fabs = row.get("EC (FTIR)"), row.get("HIPS Fabs")
        ref[row["id"]] = {"ec": ec if isinstance(ec, (int, float)) else None,
                          "hips": fabs / 10 if isinstance(fabs, (int, float)) else None}
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--minute", action="append", default=[], metavar="CODE=PATH", help="a minute CSV for a site code, instead of the Drive file")
    args = ap.parse_args()
    override = dict(a.split("=", 1) for a in args.minute)
    from data_paths import aethalometry_dir
    windows = json.loads((ROOT / "gallery/app/public/data/weather/filters.json").read_text())["windows"]
    ref = reference()
    payload = {"schema_version": 1, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "source": "gallery/data/export_aeth_loading_k.py",
               "settings": {"k_grid": K_GRID.tolist(), "atn_min": ATN_MIN, "floor": FLOOR, "kyan": KYAN, "channels": CHANNELS,
                            "hips_mac": 10},
               "sites": {}}
    for code, (label, rel, zone, filter_site) in SITES.items():
        path = Path(override[code]) if code in override else aethalometry_dir() / rel
        if not path.exists():
            print(f"{code}: {path} not found, skipped")
            continue
        print(f"{code}: reading {path.name}", flush=True)
        df = minute.read_minutes(path, zone)
        payload["sites"][code] = {"label": label, "filter_site": filter_site, **site(df, windows, ref, filter_site, zone)}
        print(f"{code}: {payload['sites'][code]['minutes']} minutes, {len(payload['sites'][code]['filters']['id'])} filters", flush=True)
        del df
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
