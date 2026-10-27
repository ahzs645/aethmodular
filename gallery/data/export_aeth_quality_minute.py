"""Minute-level QC for the Data quality page: the raw MA350 records at Jacros (Addis) and Pasadena (JPL).

Run with:  uv run python gallery/data/export_aeth_quality_minute.py [--minute CODE=PATH ...]
           (after gallery/data/export_aeth_quality.py; needs the Drive mirror, see below)

Inputs, relative to research/ftir_hips_chem/scripts/data_paths.aethalometry_dir() (the repo .env's Drive root +
"Aethalometry Data"); --minute CODE=PATH overrides one:
  Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv      ETAD-JACROS, MA350-0238 (the instrument behind the Jacros pickle)
  Raw/Pasadena_MA350_1-min_2023-2024_Cleaned.csv    USPA, MA350-0229 (JPL's instrument)
  gallery/app/public/data/weather/filters.json       each filter's local sampling window (ETAD, USPA)

Time: the CSVs' "Date/Time local" columns are the UTC clock ("Timezone offset (mins)" is 0 on every row), so
"Time (UTC)" is read and converted to the site's zone (wall clock, daylight saving included), the clock the
filter windows use.

What is computed (all from the same minute rows, sorted, duplicate stamps dropped):
  flow_qc      notebooks/qc/flow_qc_analysis: R = Flow1/Flow2, f1, f2, S = (F1+F2)/total, T = total/setpoint;
               R pass/warn/fail within 5/10 % of the site's median R, S and T within 0.02 of 1; their summary,
               pass rates overall and per local day, the R distribution and a Flow2 x Flow1 2-D histogram.
  smoothing    research/ftir_hips_chem/smoothing_comparison: raw BCc (negatives dropped) against its 5/10/15/30-min
               centred rolling medians and against the smoothed BCc of the daily pickles, which is the cleaning
               pipeline's DEMA (src/data/qc/pkl_cleaning.py dema_bc_and_atn: pandas ewm, span 10, per tape spot);
               r/RMSE/MAE/bias per channel and window, hour-of-day quartiles, one example day, a 2-D histogram.
  tape         ftir_49 A2: the BCc step across each tape advance (10 min ending 5 min before vs 10 min starting
               5 min after, 8 valid minutes each) against random times at least 60 min from an advance.
  kstar        ftir_49 A1: the instrument K, the closure K* and spot 2 under K, by ATN1 bin and spot age.
  clean        ftir_49 A3: nominal flow, |dT/dt| and |dRH/dt| under their medians, UV ATN1 < 30, 30 min clear of an
               advance; AAE and channel ratios by environment bin; daily AAE on all minutes vs the clean ones.
  missing      ftir_49 B4: masked daily means on near-complete 9am days (random, contiguous, rush-hour,
               high- and low-pollution loss of 10/20/40 % of minutes).
  aae_order    ftir_49 B5: daily mean of the per-minute AAE against the AAE of daily-mean absorption.
  windows      ftir_49 B6 on every filter: the 9am day, the calendar day, the +-1-day tolerant mean and the
               filter's own sampling window, IR and Red BCc.

Writes gallery/app/public/data/aethalometer/quality/minute/<CODE>.json and index.json (private with the rest of
aethalometer/). Concentrations leave in ug/m3.
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
import export_aethalometer as aeth  # noqa: E402  (puts research/ftir_hips_chem/scripts on the path)
from config import WAVELENGTHS_NM  # noqa: E402
from optics import aae  # noqa: E402

ROOT = aeth.ROOT
OUT = aeth.OUT / "quality" / "minute"
CHANNELS = aeth.CHANNELS
SIGMA = aeth.SIGMA          # MA350 attenuation cross-sections (m2/g), src/external/calibration.py
SEED = 20260901             # ftir_49's RNG seed
SITES = {   # code -> (name, file under aethalometry_dir(), zone, filter-site code)
    "ETAD-JACROS": ("Jacros (Addis)", "Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv", "Africa/Addis_Ababa", "ETAD"),
    "USPA": ("Pasadena (JPL)", "Raw/Pasadena_MA350_1-min_2023-2024_Cleaned.csv", "America/Los_Angeles", "USPA"),
}
META = ["Session ID", "Status", "Tape position", "Flow setpoint (mL/min)", "Flow total (mL/min)", "Flow1 (mL/min)",
        "Flow2 (mL/min)", "Sample temp (C)", "Sample RH (%)", "Internal temp (C)"]
PER_CHANNEL = ["ATN1", "ATN2", "K", "BC1", "BC2", "BCc"]
WINDOWS = [5, 10, 15, 30]          # smoothing_comparison's rolling-median windows (minutes)
DEMA_SPAN = 10                     # pkl_cleaning.preprocess_data's DEMA span
ATN_BINS = [5, 10, 20, 30, 50, 80, 200]
AGE_BINS = [0, 30, 60, 180, 360, 720, 1440, 1e6]
AGE_LABELS = ["<30 min", "30–60", "1–3 h", "3–6 h", "6–12 h", "12–24 h", ">24 h"]
LEVELS = [0.10, 0.20, 0.40]
KINDS = ["random", "contiguous", "rush-hour", "high-pollution", "low-pollution"]
N_DRAWS = 20


def r(v, nd=3):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def col(values, nd=3) -> list:
    return [r(v, nd) for v in np.asarray(values, dtype=float)]


def q(v: pd.Series | np.ndarray, p: float):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, p)) if len(v) else np.nan


# ------------------------------------------------------------------------------------------- reading

def read_minutes(path: Path, zone: str) -> pd.DataFrame:
    """The QC columns of a raw MA350 CSV, one row per minute: `tu` the UTC stamp (sorted, duplicates dropped; the
    clock every minute count uses), `t` the local wall clock (days, hours, filter windows); BC in ng/m3 as recorded.
    At the autumn daylight-saving change `t` repeats an hour, so nothing that counts minutes reads it."""
    head = pd.read_csv(path, nrows=0).columns
    want = ["Time (UTC)", *[c for c in META if c in head], *[f"{ch} {v}" for ch in CHANNELS for v in PER_CHANNEL if f"{ch} {v}" in head]]
    parts = []
    for chunk in pd.read_csv(path, usecols=want, chunksize=300_000, low_memory=False):
        tu = pd.to_datetime(chunk.pop("Time (UTC)"), utc=True, errors="coerce")
        num = chunk.apply(pd.to_numeric, errors="coerce").astype("float32")
        num.insert(0, "tu", tu.dt.tz_localize(None).to_numpy())
        num.insert(1, "t", tu.dt.tz_convert(zone).dt.tz_localize(None).to_numpy())
        parts.append(num)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["tu"].notna()].sort_values("tu", kind="stable")
    return df.drop_duplicates("tu").reset_index(drop=True)


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Tape spots (ftir_49): a new segment wherever Tape position or Session ID changes; each minute's age on
    its spot and the minutes to the next advance."""
    change = df["Tape position"].ne(df["Tape position"].shift()) | df["Session ID"].ne(df["Session ID"].shift())
    change.iloc[0] = True
    df["segment"] = change.cumsum().astype(int)
    g = df.groupby("segment")["tu"]
    df["age_min"] = (df["tu"] - g.transform("min")).dt.total_seconds() / 60
    df["to_next_min"] = (g.transform("max") - df["tu"]).dt.total_seconds() / 60
    return df


def dema(values: pd.Series, segment: pd.Series, span: int = DEMA_SPAN) -> pd.Series:
    """The cleaning pipeline's smoothing (pkl_cleaning.dema_bc_and_atn): 2·EMA − EMA(EMA), restarted on each spot."""
    def one(s: pd.Series) -> pd.Series:
        ema = s.ewm(span=span, adjust=False).mean()
        return 2 * ema - ema.ewm(span=span, adjust=False).mean()
    return values.astype(float).groupby(segment, sort=False).transform(one)


def hist(values, lo: float, hi: float, bins: int) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    counts, edges = np.histogram(np.clip(v, lo, hi), bins=bins, range=(lo, hi))
    return {"lo": r(lo, 4), "hi": r(hi, 4), "counts": counts.astype(int).tolist(), "below": int((v < lo).sum()), "above": int((v > hi).sum())}


def hist2d(x, y, xr: tuple[float, float], yr: tuple[float, float], bins: int = 40) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    h, _, _ = np.histogram2d(x[ok], y[ok], bins=bins, range=[xr, yr])
    return {"x": [r(xr[0], 4), r(xr[1], 4)], "y": [r(yr[0], 4), r(yr[1], 4)], "counts": h.astype(int).T.tolist(), "n": int(ok.sum())}


# ------------------------------------------------------------------------------------------- flow QC

def flow_flags(df: pd.DataFrame) -> pd.DataFrame:
    """flow_qc_analysis.process_flow_data: the five metrics and their flags on the minutes with all four flows."""
    f = pd.DataFrame({"t": df["t"], "F1": df["Flow1 (mL/min)"], "F2": df["Flow2 (mL/min)"], "Ft": df["Flow total (mL/min)"], "Fs": df["Flow setpoint (mL/min)"]}).astype({"F1": float, "F2": float, "Ft": float, "Fs": float})
    f = f.dropna()
    with np.errstate(divide="ignore", invalid="ignore"):
        f["R"] = (f["F1"] / f["F2"]).replace([np.inf, -np.inf], np.nan)
        f["f1"], f["f2"] = f["F1"] / f["Ft"], f["F2"] / f["Ft"]
        f["S"], f["T"] = (f["F1"] + f["F2"]) / f["Ft"], f["Ft"] / f["Fs"]
    target = float(f["R"].median())
    dev = (f["R"] - target).abs()
    f["R_pass"], f["R_warn"] = dev <= 0.05 * target, (dev > 0.05 * target) & (dev <= 0.10 * target)
    f["R_fail"] = dev > 0.10 * target
    f["S_pass"], f["T_pass"] = (f["S"] - 1).abs() <= 0.02, (f["T"] - 1).abs() <= 0.02
    f["QC_pass"] = f["R_pass"] & f["S_pass"] & f["T_pass"]
    f["sum_diff"], f["set_diff"] = f["F1"] + f["F2"] - f["Ft"], f["Ft"] - f["Fs"]
    f.attrs["R_target"] = target
    return f


def flow_qc(df: pd.DataFrame) -> dict:
    f = flow_flags(df)
    stats = {m: {k: r(v, 4) for k, v in dict(mean=f[m].mean(), median=f[m].median(), std=f[m].std(), p5=q(f[m], .05), p95=q(f[m], .95),
                                               iqr=q(f[m], .75) - q(f[m], .25)).items()} for m in ("R", "f1", "f2", "S", "T")}
    rates = {k: r(f[k].mean(), 4) for k in ("R_pass", "R_warn", "R_fail", "S_pass", "T_pass", "QC_pass")}
    day = f["t"].dt.strftime("%Y-%m-%d")
    g = f.groupby(day)
    daily = pd.DataFrame({"R_med": g["R"].median(), "R_p10": g["R"].quantile(.1), "R_p90": g["R"].quantile(.9),
                          "R_pass": g["R_pass"].mean(), "QC_pass": g["QC_pass"].mean(), "f1": g["f1"].median(), "f2": g["f2"].median(),
                          "sum_diff": g["sum_diff"].median(), "set_diff": g["set_diff"].median(), "n": g.size()})
    lo, hi = q(f["R"], .005), q(f["R"], .995)
    x1, x2 = q(f["F2"], .01), q(f["F2"], .99)
    y1, y2 = q(f["F1"], .01), q(f["F1"], .99)
    pad = lambda a, b: (a - 0.1 * (b - a), b + 0.1 * (b - a))  # noqa: E731  (the notebook's 10 % padding)
    return {
        "n": len(f), "R_target": r(f.attrs["R_target"], 4), "stats": stats, "rates": rates,
        "daily": {"day": daily.index.tolist(), **{k: col(daily[k], 4) for k in daily.columns if k != "n"}, "n": daily["n"].astype(int).tolist()},
        "R_hist": hist(f["R"], lo, hi, 80),
        "flows_2d": hist2d(f["F2"], f["F1"], pad(x1, x2), pad(y1, y2), 40),
        "corr_F1_F2": r(f[["F1", "F2"]].corr().iloc[0, 1], 4),
    }


# ------------------------------------------------------------------------------------------- smoothing

def smoothing(df: pd.DataFrame) -> dict:
    out: dict = {"windows": WINDOWS, "dema_span": DEMA_SPAN, "stats": [], "diurnal": {}}
    hour = df["t"].dt.hour.to_numpy()
    variants: dict[str, dict[str, pd.Series]] = {}
    for ch in CHANNELS:
        raw = df[f"{ch} BCc"].astype(float) / 1000
        if raw.notna().sum() < 1000:
            continue
        v = {"raw": raw.where(raw >= 0), "dema": dema(raw, df["segment"])}
        for w in WINDOWS:
            v[f"rm{w}"] = v["raw"].rolling(w, center=True, min_periods=3).median()
        variants[ch] = v
        for w in WINDOWS:
            a, b = v["dema"].to_numpy(), v[f"rm{w}"].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 100:
                continue
            d = b[ok] - a[ok]
            out["stats"].append({"channel": ch, "window": w, "n": int(ok.sum()), "r": r(np.corrcoef(a[ok], b[ok])[0, 1], 4),
                                 "rmse": r(np.sqrt(np.mean(d ** 2)), 4), "mae": r(np.mean(np.abs(d)), 4), "bias": r(np.mean(d), 4)})
        if ch in ("IR", "Blue"):
            per = {}
            for name, s in v.items():
                g = pd.Series(s.to_numpy()).groupby(hour)
                per[name] = {"q25": col(g.quantile(.25).reindex(range(24)), 4), "q50": col(g.median().reindex(range(24)), 4),
                             "q75": col(g.quantile(.75).reindex(range(24)), 4)}
            out["diurnal"][ch] = per
    if "IR" in variants:
        v = variants["IR"]
        # the example day: the notebook's find_good_days, the middle of the days with >= 1300 IR minutes
        days = df["t"].dt.normalize()
        counts = v["raw"].groupby(days).count()
        good = counts[counts >= 1300].index
        day = good[len(good) // 2] if len(good) else counts.idxmax()
        sel = (days == day).to_numpy()
        minute = ((df.loc[sel, "t"] - day).dt.total_seconds() // 60).astype(int).to_numpy()
        out["example"] = {"day": day.strftime("%Y-%m-%d"), "minute": minute.tolist(),
                          **{name: col(s[sel], 3) for name, s in v.items()}}
        best = min((s for s in out["stats"] if s["channel"] == "IR"), key=lambda s: s["rmse"], default=None)
        if best:
            top = max(q(v["dema"], .99), q(v[f"rm{best['window']}"], .99))
            out["scatter_IR"] = {"window": best["window"], **hist2d(v["dema"], v[f"rm{best['window']}"], (0, top), (0, top), 40)}
    return out


# ------------------------------------------------------------------------------------------- ftir_49

def window_means(t: np.ndarray, values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `values` over [-15, -5) and [+5, +15) minutes around each time, each needing 8 valid minutes."""
    pre, post = np.full(len(times), np.nan), np.full(len(times), np.nan)
    pos = np.searchsorted(t, times)
    for i, (p, t0) in enumerate(zip(pos, times)):
        lo, hi = max(0, p - 20), min(len(values), p + 20)
        tt = (t[lo:hi] - t0) / np.timedelta64(1, "m")
        vv = values[lo:hi]
        m_pre = (tt >= -15) & (tt < -5) & ~np.isnan(vv)
        m_post = (tt >= 5) & (tt < 15) & ~np.isnan(vv)
        if m_pre.sum() >= 8 and m_post.sum() >= 8:
            pre[i], post[i] = vv[m_pre].mean(), vv[m_post].mean()
    return pre, post


def tape_steps(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    t = df["tu"].to_numpy()
    adv = df.groupby("segment")["tu"].min().iloc[1:].to_numpy()
    cand = df.loc[(df["age_min"] > 60) & (df["to_next_min"] > 60), "tu"].to_numpy()
    ctrl = rng.choice(cand, size=min(len(cand), 5 * len(adv)), replace=False) if len(cand) else cand
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan  # noqa: E731
    rows, hists, ir_points = [], {}, None
    for ch in CHANNELS:
        v = df[f"{ch} BCc"].to_numpy(dtype=float)
        pre, post = window_means(t, v, adv)
        cpre, cpost = window_means(t, v, ctrl)
        step, cstep = post - pre, cpost - cpre
        with np.errstate(divide="ignore", invalid="ignore"):
            rel, crel = step / pre, cstep / cpre
        s, cs = step[np.isfinite(step)], cstep[np.isfinite(cstep)]
        rr, cr = rel[np.isfinite(rel)], crel[np.isfinite(crel)]
        if not len(s):
            continue
        rows.append({"channel": ch, "n": len(s), "n_control": len(cs), "step_median": r(np.median(s) / 1000, 4),
                     "step_mad": r(mad(s) / 1000, 4), "control_mad": r(mad(cs) / 1000, 4),
                     "rel_median": r(100 * np.median(rr), 2), "rel_q25": r(100 * q(rr, .25), 2), "rel_q75": r(100 * q(rr, .75), 2),
                     "rel_over25": r(100 * np.mean(np.abs(rr) > .25), 2), "control_rel_over25": r(100 * np.mean(np.abs(cr) > .25), 2),
                     "positive": r(np.mean(s > 0), 3), "control_positive": r(np.mean(cs > 0), 3)})
        hists[ch] = hist(100 * rr, -100, 100, 40)
        if ch == "IR":
            hists["control"] = hist(100 * cr, -100, 100, 40)
            ok = np.isfinite(step)
            ir_points = {"pre": col(pre[ok] / 1000, 3), "step": col(step[ok] / 1000, 3)}
    return {"advances": len(adv), "segments_median_min": r(df.groupby("segment").size().median(), 0), "table": rows, "hist": hists, "ir_points": ir_points}


def kstar(df: pd.DataFrame) -> list[dict]:
    rows = []
    age = pd.cut(df["age_min"], AGE_BINS, right=False, labels=AGE_LABELS)
    for ch in CHANNELS:
        bc1, bc2, bcc = (df[f"{ch} {v}"].astype(float) for v in ("BC1", "BC2", "BCc"))
        a1, a2, K = (df[f"{ch} {v}"].astype(float) for v in ("ATN1", "ATN2", "K"))
        valid = bc1.notna() & bc2.notna() & bcc.notna() & K.notna() & (a1 >= 5) & (bc1 > 0) & (bc2 > 0)
        den = bc1 * a2 - bc2 * a1
        ks = ((bc1 - bc2) / den.where(den.abs() > 1e-9)).where(valid)
        sub = pd.DataFrame({"K": K, "Kstar": ks, "spot2": (bc2 / (1 - K * a2) / bcc).where(valid),
                            "bcc_star": ((1 - K * a1) / (1 - ks * a1)), "atn": pd.cut(a1, ATN_BINS, right=False), "age": age})
        sub = sub[sub["Kstar"].notna()]
        for by, key in (("ATN1", "atn"), ("spot age", "age")):
            for b, part in sub.groupby(key, observed=True):
                rows.append({"channel": ch, "by": by, "bin": str(b), "n": len(part), "K": r(part["K"].median(), 5),
                             "Kstar": r(part["Kstar"].median(), 5), "Kstar_q25": r(q(part["Kstar"], .25), 5), "Kstar_q75": r(q(part["Kstar"], .75), 5),
                             "spot2_over_bcc": r(part["spot2"].median(), 4), "bcc_star_over_bcc": r(part["bcc_star"].median(), 4)})
    return rows


def absorption(df: pd.DataFrame) -> pd.DataFrame:
    """b = BCc × sigma_ATN, Mm-1 on the attenuation scale (ftir_49)."""
    return pd.DataFrame({ch: df[f"{ch} BCc"].astype(float) * SIGMA[ch] * 1e-3 for ch in CHANNELS})


def aae_pair(b: pd.DataFrame | pd.Series, short: str):
    return aae(b[short], b["IR"], WAVELENGTHS_NM[short], WAVELENGTHS_NM["IR"])


def clean_mask(df: pd.DataFrame) -> dict:
    step_ok = df["tu"].diff().dt.total_seconds().eq(60)
    for c, name in (("Sample temp (C)", "dT"), ("Sample RH (%)", "dRH")):
        df[name] = df[c].astype(float).diff().where(step_ok).rolling(10, min_periods=5).mean().abs()
    ratio = df["Flow1 (mL/min)"].astype(float) / df["Flow2 (mL/min)"].astype(float)
    nominal_ratio = float(ratio.median())
    flow_dev = (df["Flow total (mL/min)"].astype(float) - df["Flow setpoint (mL/min)"]) / df["Flow setpoint (mL/min)"]
    near = (df["age_min"] < 30) | (df["to_next_min"] < 30)
    b = absorption(df)
    base = b.notna().all(axis=1)
    med_dT, med_dRH = float(df.loc[base, "dT"].median()), float(df.loc[base, "dRH"].median())
    parts = {"nominal flow": (flow_dev.abs() < 0.05) & ((ratio / nominal_ratio - 1).abs() < 0.10),
             "|dT/dt| < median": df["dT"] < med_dT, "|dRH/dt| < median": df["dRH"] < med_dRH,
             "UV ATN1 < 30": df["UV ATN1"] < 30, ">30 min from an advance": ~near}
    clean = pd.concat(parts, axis=1).all(axis=1)

    def bins(key: pd.Series, cuts, labels=None, quantiles=False):
        k = pd.qcut(key[base], cuts, duplicates="drop") if quantiles else pd.cut(key[base], cuts, labels=labels, right=False)
        rows = []
        for name, part in b[base].groupby(k, observed=True):
            s = part.sum()
            rows.append({"bin": str(name), "n": len(part), "aae_blue": r(aae_pair(s, "Blue"), 4), "aae_red": r(aae_pair(s, "Red"), 4),
                         "red_ir": r((s["Red"] / SIGMA["Red"]) / (s["IR"] / SIGMA["IR"]), 4), "uv_ir": r((s["UV"] / SIGMA["UV"]) / (s["IR"] / SIGMA["IR"]), 4),
                         "ir_mean": r(part["IR"].mean() / SIGMA["IR"], 3)})
        return rows
    env = {
        "|dT/dt| quintile (°C/min, 10-min mean)": bins(df["dT"], 5, quantiles=True),
        "|dRH/dt| quintile (%/min, 10-min mean)": bins(df["dRH"], 5, quantiles=True),
        "Flow1/Flow2 off its median": bins((ratio / nominal_ratio - 1).abs(), [0, .02, .05, .10, .25, 10], ["<2%", "2–5%", "5–10%", "10–25%", ">25%"]),
        "Total flow off its setpoint": bins(flow_dev.abs(), [0, .02, .05, .10, 10], ["<2%", "2–5%", "5–10%", ">10%"]),
        "IR ATN1": bins(df["IR ATN1"].astype(float), [0, 5, 10, 20, 30, 50, 80, 300]),
        "UV ATN1": bins(df["UV ATN1"].astype(float), [0, 5, 10, 20, 30, 50, 80, 300]),
        "Spot age": bins(df["age_min"], AGE_BINS, AGE_LABELS),
    }
    day9 = (df["t"] - pd.Timedelta(hours=9)).dt.normalize()
    per_day = {}
    for label, sel in (("all", base), ("clean", base & clean)):
        g = b[sel].groupby(day9[sel])
        s = g.mean()
        per_day[label] = pd.DataFrame({"n": g.size(), "red": aae_pair(s, "Red"), "green": aae(s["Green"], s["IR"], WAVELENGTHS_NM["Green"], WAVELENGTHS_NM["IR"]),
                                       "blue": aae_pair(s, "Blue")})
    a, c = per_day["all"], per_day["clean"]
    common = a.index[a["n"] >= 720].intersection(c.index[c["n"] >= 180])
    summary = []
    for label, d in (("all minutes", a.loc[common]), ("clean mask", c.loc[common])):
        summary.append({"minutes": label, "n_days": len(common), **{f"{k}_mean": r(d[k].mean(), 4) for k in ("red", "green", "blue")},
                        **{f"{k}_sd": r(d[k].std(), 4) for k in ("red", "green", "blue")}, "red_over1": r(100 * (d["red"] > 1).mean(), 2)})
    paired = c.loc[common, "red"] - a.loc[common, "red"]
    return {"fraction": r(clean[base].mean(), 4), "n": int(base.sum()), "median_dT": r(med_dT, 5), "median_dRH": r(med_dRH, 5),
            "nominal_ratio": r(nominal_ratio, 4), "parts": {k: r(v[base].mean(), 4) for k, v in parts.items()},
            "env": env, "daily_aae": summary, "paired_red_mean": r(paired.mean(), 4), "paired_red_sd": r(paired.std(), 4),
            "hist_red_all": hist(a.loc[common, "red"], 0, 2, 40), "hist_red_clean": hist(c.loc[common, "red"], 0, 2, 40)}


def day_matrix(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """9am days (local 09:00 to 09:00, labelled at the start) by minute of the day."""
    day9 = (df["t"] - pd.Timedelta(hours=9)).dt.normalize()
    minute = ((df["t"] - pd.Timedelta(hours=9)) - day9).dt.total_seconds() // 60
    sub = pd.DataFrame({"d": day9, "m": minute.astype(int), "v": df[column].astype(float)}).dropna()
    return sub.pivot_table(index="d", columns="m", values="v", aggfunc="mean").reindex(columns=np.arange(1440))


def masked_errors(M: np.ndarray, kind: str, frac: float, rng: np.random.Generator, rush: np.ndarray) -> np.ndarray:
    """ftir_49 B4: percent error of the masked daily mean, one value per (day, draw)."""
    D, N = M.shape
    L = int(round(frac * N))
    true = M.mean(axis=1)
    errs = []
    for _ in range(N_DRAWS if kind in ("random", "contiguous", "rush-hour") else 1):
        keep = np.ones((D, N), dtype=bool)
        if kind == "random":
            keep[rng.random((D, N)).argsort(axis=1) < L] = False
        elif kind == "contiguous":
            s = rng.integers(0, N - L + 1, size=D)
            idx = np.arange(N)[None, :]
            keep = ~((idx >= s[:, None]) & (idx < (s + L)[:, None]))
        elif kind == "rush-hour":
            rush_idx, other = np.flatnonzero(rush), np.flatnonzero(~rush)
            for d in range(D):
                drop = rng.choice(rush_idx, L, replace=False) if L <= len(rush_idx) else np.concatenate([rush_idx, rng.choice(other, L - len(rush_idx), replace=False)])
                keep[d, drop] = False
        else:
            order = np.argsort(-M if kind == "high-pollution" else M, axis=1)
            keep[np.arange(D)[:, None], order[:, :L]] = False
        errs.append(100 * (np.nanmean(np.where(keep, M, np.nan), axis=1) / true - 1))
    return np.concatenate(errs)


def missingness(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    mats, days = {}, None
    for ch in ("IR", "Red"):
        m = day_matrix(df, f"{ch} BCc")
        m = m[m.notna().mean(axis=1) >= 0.95].interpolate(axis=1, limit_direction="both")
        mats[ch] = m
        days = m.index if days is None else days.intersection(m.index)
    rush = np.isin(((9 * 60 + np.arange(1440)) // 60) % 24, [7, 8, 9, 17, 18, 19])
    rows = []
    for ch in ("IR", "Red"):
        M = mats[ch].loc[days].to_numpy(dtype=float)
        for kind in KINDS:
            for frac in LEVELS:
                e = masked_errors(M, kind, frac, rng, rush) if len(M) else np.array([])
                rows.append({"channel": ch, "mask": kind, "loss": int(frac * 100), "bias": r(np.mean(e), 3) if len(e) else None,
                             "median": r(q(e, .5), 3), "q25": r(q(e, .25), 3), "q75": r(q(e, .75), 3), "p2_5": r(q(e, .025), 3),
                             "p97_5": r(q(e, .975), 3), "over25": r(100 * np.mean(np.abs(e) > 25), 2) if len(e) else None})
    return {"n_days": len(days), "first": days.min().strftime("%Y-%m-%d") if len(days) else None,
            "last": days.max().strftime("%Y-%m-%d") if len(days) else None, "table": rows,
            "ir_mean": r(mats["IR"].loc[days].mean(axis=1).mean() / 1000, 3) if len(days) else None}


def aae_order(df: pd.DataFrame) -> dict:
    b = absorption(df)
    day9 = (df["t"] - pd.Timedelta(hours=9)).dt.normalize()
    out = {}
    for short in ("Blue", "Red"):
        ok = (b[short] > 0) & (b["IR"] > 0)
        inst = aae(b[short].where(ok), b["IR"].where(ok), WAVELENGTHS_NM[short], WAVELENGTHS_NM["IR"])
        g = pd.DataFrame({"inst": inst, "s": b[short], "l": b["IR"], "n": b["IR"].notna()}).groupby(day9)
        n = g["n"].sum()
        keep = n >= 1368      # the near-complete days of the missingness benchmark (95 % of 1440)
        agg = aae(g["s"].mean(), g["l"].mean(), WAVELENGTHS_NM[short], WAVELENGTHS_NM["IR"])[keep]
        diff = (g["inst"].mean()[keep] - agg).dropna()
        dmed = (g["inst"].median()[keep] - agg).dropna()
        out[f"AAE({WAVELENGTHS_NM[short]},880)"] = {
            "n_days": len(diff), "undefined_pct": r(100 * inst.isna().mean(), 2), "agg_mean": r(agg.mean(), 4), "agg_sd": r(agg.std(), 4),
            "diff_median": r(diff.median(), 4), "diff_q25": r(q(diff, .25), 4), "diff_q75": r(q(diff, .75), 4),
            "diff_p2_5": r(q(diff, .025), 4), "diff_p97_5": r(q(diff, .975), 4), "over01": r(100 * (diff.abs() > 0.1).mean(), 2),
            "median_diff_median": r(dmed.median(), 4), "hist": hist(diff, -1.5, 1.5, 40)}
    return out


def filter_windows(df: pd.DataFrame, windows: dict[str, list], site: str, zone: str) -> dict:
    """Per filter of `site`: IR and Red BCc (ug/m3) over the 9am day starting on its start date, the calendar day,
    the mean of the 9am days d-1..d+1 (each needing 720 minutes, as all day means here), and its sampling window
    (needing half the window's minutes)."""
    t = df["t"]
    day9 = (t - pd.Timedelta(hours=9)).dt.normalize()
    cal = t.dt.normalize()
    out = {"id": [], "date": [], "hours": []}
    vals = {f"{ch}_{k}": [] for ch in ("IR", "Red") for k in ("9am", "calendar", "tolerant", "window")}
    g9 = {ch: df[f"{ch} BCc"].astype(float).groupby(day9).agg(["mean", "count"]) for ch in ("IR", "Red")}
    gc_ = {ch: df[f"{ch} BCc"].astype(float).groupby(cal).agg(["mean", "count"]) for ch in ("IR", "Red")}
    tv = df["tu"].to_numpy()
    lo_t, hi_t = t.min(), t.max()
    utc = lambda x: x.tz_localize(zone, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC").tz_localize(None)  # noqa: E731
    for fid, w in sorted(windows.items()):
        if not fid.startswith(site + "-"):
            continue
        a, b = pd.Timestamp(w[0]), pd.Timestamp(w[1])
        if b < lo_t or a > hi_t:
            continue
        d = a.normalize()
        out["id"].append(fid)
        out["date"].append(d.strftime("%Y-%m-%d"))
        out["hours"].append(r((b - a) / pd.Timedelta(hours=1), 1))
        i0, i1 = np.searchsorted(tv, np.datetime64(utc(a))), np.searchsorted(tv, np.datetime64(utc(b)))
        need = 0.5 * (b - a) / pd.Timedelta(minutes=1)
        for ch in ("IR", "Red"):
            def day_mean(tab, key):
                if key in tab.index and tab.at[key, "count"] >= 720:
                    return tab.at[key, "mean"] / 1000
                return np.nan
            tol = [day_mean(g9[ch], d + pd.Timedelta(days=k)) for k in (-1, 0, 1)]
            tol = [x for x in tol if np.isfinite(x)]
            seg = df[f"{ch} BCc"].to_numpy(dtype=float)[i0:i1]
            ok = np.isfinite(seg)
            vals[f"{ch}_9am"].append(r(day_mean(g9[ch], d), 4))
            vals[f"{ch}_calendar"].append(r(day_mean(gc_[ch], d), 4))
            vals[f"{ch}_tolerant"].append(r(np.mean(tol), 4) if tol else None)
            vals[f"{ch}_window"].append(r(seg[ok].mean() / 1000, 4) if ok.sum() >= need and need > 0 else None)
    return {**out, **vals}


# ------------------------------------------------------------------------------------------- main

def site_payload(df: pd.DataFrame, windows: dict, filter_site: str, zone: str) -> dict:
    df = add_segments(df)
    rng = np.random.default_rng(SEED)
    span = [df["t"].min().strftime("%Y-%m-%d %H:%M"), df["t"].max().strftime("%Y-%m-%d %H:%M")]
    payload = {"minutes": len(df), "span": span, "segments": int(df["segment"].nunique())}
    for name, fn in (("flow_qc", lambda: flow_qc(df)), ("smoothing", lambda: smoothing(df)), ("tape", lambda: tape_steps(df, rng)),
                     ("kstar", lambda: kstar(df)), ("clean", lambda: clean_mask(df)), ("missing", lambda: missingness(df, rng)),
                     ("aae_order", lambda: aae_order(df)), ("windows", lambda: filter_windows(df, windows, filter_site, zone))):
        payload[name] = fn()
        print(f"  {name} done", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--minute", action="append", default=[], metavar="CODE=PATH", help="a minute CSV for a site code, instead of the Drive file")
    args = ap.parse_args()
    override = dict(a.split("=", 1) for a in args.minute)
    from data_paths import aethalometry_dir
    windows = json.loads((ROOT / "gallery/app/public/data/weather/filters.json").read_text())["windows"]
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"generated": datetime.now(timezone.utc).isoformat(), "seed": SEED, "sites": {}}
    for code, (name, rel, zone, filter_site) in SITES.items():
        path = Path(override[code]) if code in override else aethalometry_dir() / rel
        if not path.exists():
            print(f"{code}: {path} not found, skipped")
            continue
        print(f"{code}: reading {path.name}", flush=True)
        df = read_minutes(path, zone)
        payload = site_payload(df, windows, filter_site, zone)
        (OUT / f"{code}.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        index["sites"][code] = {"name": name, "file": rel, "zone": zone, "minutes": payload["minutes"], "span": payload["span"], "filter_site": filter_site}
        print(f"{code}: {payload['minutes']} minutes, {(OUT / f'{code}.json').stat().st_size / 1e3:.0f} kB")
        del df
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":"), allow_nan=False))
    size = sum(p.stat().st_size for p in OUT.glob("*.json"))
    print(f"{size / 1e6:.2f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
