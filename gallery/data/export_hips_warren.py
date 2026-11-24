"""Warren White's deposit model beside the IMPROVE curve, and what the IMPROVE fit does in his trusted samples.

Run from the repo root (needs the FTIR local database's TOR and HIPS tables; the Phoenix part also needs the cleaned
IMPROVE FED cache):

    uv run python gallery/data/export_hips_warren.py
    uv run python gallery/data/export_hips_warren.py --cache /tmp/improve_all.parquet   # shared with export_hips_ceiling_checks

White et al. (2025, doi 10.1080/10962247.2024.2442634) describe a deposit patterned by the support screen: a share h of the
area (0.638 on IMPROVE; McDade, Dillner & Indresand 2009) is alpha times as thick as the rest, and the reported optical depth is

    tau = -ln[ h exp(-alpha x / w) + (1 - h) exp(-x / w) ],   w = 1 + (alpha - 1) h,

with x the mean absorption density on the filter (their Eq. 15 and 18). Writes

* gallery/app/public/data/hips_linearise/warren.json: the same loading groups as the published curve without the filters
  of another deposit area (masked filters until 2007), the models fitted to them, the curve split between a screen part and
  a remaining part, and the ceiling in the samples White et al. (2016, 2025) use.
* gallery/app/public/data/research/hips_drain_disc/pairs.json: PRIVATE and gitignored. The Phoenix collocated pairs, where
  one sampler had drain discs for a time (White et al. 2025, section 5). Unpublished UC Davis data: never commit or publish.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research/ftir_hips_chem/workflows"), str(ROOT / "research/ftir_hips_chem/scripts")]

import data_paths  # noqa: E402
import export_hips_ceiling_checks as checks  # noqa: E402
import improve_io  # noqa: E402
import reference_closure as rc  # noqa: E402

OUT = ROOT / "gallery/app/public/data/hips_linearise/warren.json"
PRIVATE = ROOT / "gallery/app/public/data/research/hips_drain_disc/pairs.json"
H = 0.638                                       # IMPROVE screen hole fraction (McDade, Dillner & Indresand 2009)
AREA = 3.53                                     # cm2, the standard IMPROVE deposit
LIMIT = float(-np.log(1 - H))                   # the pixelation limit in optical depth: absorptance = h
CUT = 0.45
URBAN = ["ATLA1", "BIRM1", "FRES1", "PHOE1", "PITT1", "PUSO1"]
SPLIT_ALPHAS = (1, 2, 3, 5, 10)
SEED = 20261124
# the Phoenix window and the one day a routine filter's own comment mentions a drain disc (see pairs.json note)
WINDOW = ("2024-08-22", "2025-05-19")
AMBIGUOUS = "2025-02-09"


def two_zone(x, alpha, h=H):
    x = np.asarray(x, dtype=float)
    w = 1 + (alpha - 1) * h
    return -np.log(h * np.exp(-alpha * x / w) + (1 - h) * np.exp(-x / w))


def composite(x, alpha, c, h=H):
    """A screen pattern whose zones each follow the saturating curve instead of Beer-Lambert."""
    x = np.asarray(x, dtype=float)
    w = 1 + (alpha - 1) * h
    curve = lambda u: u / (1 + u / c)  # noqa: E731
    return -np.log(h * np.exp(-curve(alpha * x / w)) + (1 - h) * np.exp(-curve(x / w)))


def with_area(samples: pd.DataFrame) -> pd.DataFrame:
    hips = pd.read_csv(data_paths.ftir_local_db() / "tables/results_hips.csv", encoding="utf-8-sig", low_memory=False,
                       usecols=["Site", "SampleDate", "Parameter", "SampleDepositArea"])
    hips = hips[hips["Parameter"] == "fAbs"]
    hips["date"] = pd.to_datetime(hips["SampleDate"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    area = hips.groupby(["Site", "date"])["SampleDepositArea"].first().round(2).rename("area").reset_index()
    return samples.merge(area, on=["Site", "date"], how="left")


def ceiling(frame: pd.DataFrame):
    s = checks.strata(frame, "previous_ug")
    (a, c), rms = checks.fit_medians(s["L"].to_numpy(), s["t"].to_numpy())
    return float(a), float(c), float(rms), s


def sample_sets(samples: pd.DataFrame, n_boot: int) -> list[dict]:
    year = samples["date"].dt.year
    plain = samples["area"] == AREA
    window = plain & year.between(2010, 2020)
    per_site = samples[window]["Site"].value_counts()
    busy = window & samples["Site"].isin(per_site[per_site >= 1000].index)
    urban = samples["Site"].isin(URBAN)
    sets = [
        ("published", "The published set, 2005–2025", "as used", pd.Series(True, samples.index)),
        ("unmasked", "Without filters of another deposit area", "White 2016: masked filters excluded", plain),
        ("white2016", "2005–2014, unmasked", "White 2016: the years of that paper", plain & year.between(2005, 2014)),
        ("white2025", "2010–2020, sites with 1,000 samples", "White 2025: the samples of that paper", busy),
        ("rural", "  of those, rural sites", "White 2025", busy & ~urban),
        ("urban", "  of those, urban sites", "White 2025", busy & urban),
        ("old", "2008–2016, unmasked, two-inch sphere", "before the HIPS rebuild", plain & year.between(2008, 2016)),
        ("new", "2017–2025, four-inch sphere", "after the HIPS rebuild", plain & (year >= 2017)),
    ]
    rows = []
    for key, label, why, keep in sets:
        part = samples[keep]
        a, c, _, s = ceiling(part)
        by = {k: v for k, v in part.groupby("Site")}
        names = np.array(sorted(by))
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(n_boot):
            b = checks.strata(pd.concat([by[k] for k in rng.choice(names, names.size)], ignore_index=True), "previous_ug")
            if len(b) >= 5:
                boot.append(checks.fit_medians(b["L"].to_numpy(), b["t"].to_numpy())[0][1])
        lo, hi = np.percentile(boot, [5, 95])
        rows.append({"key": key, "label": label, "why": why, "filters": int(len(part)), "sites": int(part["Site"].nunique()), "groups": int(len(s)),
                     "fabs_per_ec": round(a * 100 * AREA, 3), "tau_inf": round(c, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                     "past_cut": round(float((part["tau"] > CUT).mean()), 5), "past_limit": round(float((part["tau"] > LIMIT).mean()), 5)})
        print(f"{label:44s} n={len(part):>7,} τ∞ {c:.2f} [{lo:.2f}, {hi:.2f}] past the limit {100 * rows[-1]['past_limit']:.2f} %")
    return rows


def model_fits(medians: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    L, t = medians["L"].to_numpy(), medians["t"].to_numpy()

    def fit(key, label, model, start, lower, upper, free):
        p, rms = checks.fit_medians(L, t, model, start, lower, upper)
        return {"key": key, "label": label, "free": free, "params": [round(float(v), 5) for v in p], "rms_log": round(rms, 5),
                "aic": round(float(len(L) * np.log(rms ** 2) + 2 * len(p)), 2),
                "residual_pct": [round(100 * float(v), 2) for v in t / model(L, *p) - 1]}

    fits = [
        fit("curve", "The saturating curve (in use)", rc.hyperbolic, (0.04, 2.0), (0, 0.2), (1, 100), "a, τ∞"),
        fit("alpha10", "White 2025, α = 10, IMPROVE screen", lambda L, a: two_zone(a * L, 10), (0.04,), (0,), (1,), "a"),
        fit("full", "White 2025, fully pixelated, IMPROVE screen", lambda L, a: rc.patterned(L, a, H), (0.04,), (0,), (1,), "a"),
        fit("free", "White 2025, α free, IMPROVE screen", lambda L, a, al: two_zone(a * L, al), (0.04, 10.0), (0, 1.0), (1, 1e4), "a, α"),
        fit("free_h", "White 2025, α and the screen both free", lambda L, a, al, h: two_zone(a * L, al, h), (0.04, 10.0, 0.6), (0, 1.0, 0.05), (1, 1e4, 0.99), "a, α, h"),
    ]
    splits = []
    for al in SPLIT_ALPHAS:
        r = fit(f"split{al}", f"α = {al}", lambda L, a, c, al=al: composite(a * L, al, c), (0.04, 3.0), (0, 0.2), (1, 100), "a, τ∞ of the rest")
        splits.append({"alpha": al, "c": r["params"][1], "rms_log": r["rms_log"], "aic": r["aic"]})
    for r in fits + splits:
        print(r)
    return fits, splits


# ----------------------------------------------------------------------------- private: the Phoenix drain-disc pairs
def phoenix(n_boot: int) -> dict:
    columns = ["SiteCode", "POC", "Date", "fAbs_Val", "ECf_Val", "MF_Val", "FEf_Val", "FlowRate_A_Val", "SampDur_A_Val", "FlowRate_C_Val",
               "SampDur_C_Val", "FlowRate_Val", "SampDur_Val"]
    d = improve_io.load_improve_clean(usecols=columns)
    d["date"] = pd.to_datetime(d["Date"], errors="coerce")

    def side(site, poc):
        x = d[(d["SiteCode"] == site) & (d["POC"] == poc)].copy()
        va = x["FlowRate_A_Val"].fillna(x["FlowRate_Val"]) * x["SampDur_A_Val"].fillna(x["SampDur_Val"]) / 1000
        vc = x["FlowRate_C_Val"].fillna(x["FlowRate_Val"]) * x["SampDur_C_Val"].fillna(x["SampDur_Val"]) / 1000
        x["tau"], x["L"] = x["fAbs_Val"] * va / (100 * AREA), x["ECf_Val"] * vc
        return x[["date", "tau", "L", "fAbs_Val", "ECf_Val", "MF_Val", "FEf_Val"]]

    m = side("PHOE1", 1).merge(side("PHOE5", 2), on="date", suffixes=("1", "5"))
    m = m[(m["tau1"] > 0) & (m["tau5"] > 0) & (m["date"] != pd.Timestamp(AMBIGUOUS))].copy()
    for k, c in {"fabs": "fAbs_Val", "mass": "MF_Val", "ec": "ECf_Val", "fe": "FEf_Val"}.items():
        m[k] = m[f"{c}5"] / m[f"{c}1"]
    m["drain"] = m["date"].between(*WINDOW)
    x = m[m["drain"]]

    def alpha_of(p):
        f = lambda q: np.log(two_zone(p["tau5"].to_numpy() * np.exp(q[1]), np.exp(q[0]))) - np.log(p["tau1"].to_numpy())  # noqa: E731
        return float(np.exp(optimize.least_squares(f, [0.5, 0.0], loss="soft_l1", f_scale=0.1).x[0]))

    def groups(p, s, o, k):
        q = p.groupby(pd.qcut(p[f"L{o}"], k), observed=True)
        return q[f"L{s}"].median().to_numpy(), q[f"tau{s}"].median().to_numpy()

    def bend(L, t):
        f = lambda q: np.log(np.exp(q[0]) * L / (1 + np.exp(q[0]) * L / np.exp(q[1]))) - np.log(t)  # noqa: E731
        return float(np.exp(optimize.least_squares(f, [np.log(0.04), np.log(2.0)], loss="soft_l1", f_scale=0.15).x[1]))

    pct = lambda b: [round(float(v), 3) for v in np.percentile(b, [5, 95])]  # noqa: E731
    boots = [x.sample(len(x), replace=True, random_state=i) for i in range(n_boot)]
    dark = x[x["tau1"] >= 1]
    other = m[~m["drain"] & (m["tau1"] >= 1)]
    ceilings = []
    for label, part, k in (("Drain-disc window", x, 7), ("October to March, other years", m[~m["drain"] & m["date"].dt.month.isin([10, 11, 12, 1, 2, 3])], 12)):
        part = part[(part["L1"] > 0) & (part["L5"] > 0)]
        for s, o, who in (("1", "5", "Routine sampler"), ("5", "1", "Collocated sampler")):
            b = [bend(*groups(part.sample(len(part), replace=True, random_state=i), s, o, k)) for i in range(n_boot)]
            ceilings.append({"dates": label, "sampler": who, "pairs": int(len(part)), "tau_inf": round(bend(*groups(part, s, o, k)), 3), "ci": pct(b)})
    # HIPS precision without the drain-disc months, as the error model of export_hips_ceiling_checks would see it
    def relative(p):
        v1, v2 = p["fAbs_Val1"].to_numpy(), p["fAbs_Val5"].to_numpy()
        mean, var = (v1 + v2) / 2, (v1 - v2) ** 2 / 2
        r = optimize.least_squares(lambda q: (q[0] ** 2 + (q[1] * mean) ** 2) - var, [np.median(mean) * 0.05, 0.1], loss="soft_l1", f_scale=np.median(var))
        return round(abs(float(r.x[1])), 4)

    rows = m.assign(day=m["date"].dt.strftime("%Y-%m-%d"))
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "PRIVATE: unpublished UC Davis experiment. Window inferred from filter IDs (13 boxes of 7 at 21 days), not from dated tags; "
                f"{AMBIGUOUS} left out because the routine filter's comment mentions a drain disc.",
        "window": list(WINDOW), "pairs": [
            {"d": r.day, "t1": round(r.tau1, 4), "t5": round(r.tau5, 4), "r": round(r.fabs, 4), "mass": round(r.mass, 4) if np.isfinite(r.mass) else None, "ec": round(r.ec, 4) if np.isfinite(r.ec) else None,
             "fe": round(r.fe, 4) if np.isfinite(r.fe) else None, "drain": bool(r.drain)} for r in rows.itertuples()],
        "dark": {"n": int(len(dark)), "ratio": round(float(dark["fabs"].median()), 4),
                 "ci": pct([dark["fabs"].sample(len(dark), replace=True, random_state=i).median() for i in range(2000)]),
                 "other_n": int(len(other)), "other_ratio": round(float(other["fabs"].median()), 4)},
        "alpha": {"value": round(alpha_of(x), 3), "ci": pct([alpha_of(b) for b in boots])},
        "ceilings": ceilings,
        "hips_relative_error": {"all_pairs": relative(m), "without_drain": relative(m[~m["drain"]])},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, help="parquet of every IMPROVE sample (export_hips_ceiling_checks --cache)")
    parser.add_argument("--boot", type=int, default=100, help="site resamples (default %(default)s)")
    parser.add_argument("--no-private", action="store_true", help="leave the private Phoenix file alone")
    args = parser.parse_args()
    samples = with_area(checks.improve_all(args.cache)).dropna(subset=["previous_ug"])
    plain = samples[samples["area"] == AREA]
    sets = sample_sets(samples, args.boot)
    medians = checks.strata(plain, "previous_ug")
    fits, splits = model_fits(medians)
    urban = samples["Site"].isin(URBAN)
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "gallery/data/export_hips_warren.py; IMPROVE TOR and HIPS tables; White et al. (2016, 2025); McDade, Dillner & Indresand (2009)",
        "n_boot": args.boot, "hole_fraction": H, "limit_tau": round(LIMIT, 4), "cut": CUT, "deposit_area_cm2": AREA,
        "areas": [{"area_cm2": float(a), "filters": int(n)} for a, n in samples["area"].value_counts().items()],
        "sets": sets,
        "medians": [{"n": int(r.n), "loading_ug": round(float(r.L), 4), "tau": round(float(r.t), 5)} for r in medians.itertuples()],
        "fits": fits, "splits": splits,
        "shares": [{"population": name, "filters": int(k.sum()), "past_cut": round(float((samples.loc[k, "tau"] > CUT).mean()), 5),
                    "past_limit": round(float((samples.loc[k, "tau"] > LIMIT).mean()), 5)}
                   for name, k in (("IMPROVE, rural", (samples["area"] == AREA) & ~urban), ("IMPROVE, six urban sites", (samples["area"] == AREA) & urban))],
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.1f} kB")
    if not args.no_private:
        PRIVATE.parent.mkdir(parents=True, exist_ok=True)
        PRIVATE.write_text(json.dumps(phoenix(300), separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        print(f"wrote {PRIVATE.relative_to(ROOT)} (private): {PRIVATE.stat().st_size / 1024:.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
