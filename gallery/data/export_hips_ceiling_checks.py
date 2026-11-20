"""Three checks that the IMPROVE HIPS ceiling is defensible, and what each does to Addis.

Run from the repo root with the repo environment (needs the FTIR local database's TOR and HIPS tables and the cleaned
IMPROVE FED cache, research/ftir_hips_chem/output/improve_high_fabs_comparison/improve_valid_cleaned.csv):

    uv run python gallery/data/export_hips_ceiling_checks.py
    uv run python gallery/data/export_hips_ceiling_checks.py --cache /tmp/improve_all.parquet

The published ceiling τ∞ = 2.26 is ``tau = aL / (1 + aL/τ∞)`` fitted through 17 group medians, IMPROVE filters grouped
by the previous sample's TOR EC loading at the same site (run_addis_reference_closure_20261110). Before calling it
defensible (Ann, 19 Nov 2026), three checks:

1. **Errors in both axes, single filters.** No grouping at all: the curve fitted to every filter by orthogonal distance
   regression, with each filter's measurement error on TOR EC and on HIPS from IMPROVE's collocated samplers at Phoenix
   (PHOE1 and PHOE5, 1,118 days), and with the filter-to-filter spread in absorption per EC that is real, not
   measurement error, estimated on light filters and added to the τ side. Variants show what each assumption does.
2. **The instrument.** Grouping by the previous sample assumes its loading carries none of this filter's errors. If
   errors ran in batches (one HIPS lot, one analysis run) that would fail, so the ceiling is refitted grouping by the
   sample two and three back, the next sample, and the previous sample only when it is from a different HIPS lot.
3. **Extrapolation and the shape of the curve.** Addis filters sit at and beyond the darkest IMPROVE group, so the
   correction there depends on the curve's shape past the data. The group medians are refitted with an exponential
   approach and with patterned deposits (White et al. 2025), and every ceiling above is carried to the Addis fit of
   FTIR EC on linearised HIPS BC: slope, intercept, the MAC for slope 1, and the filters past the curve's ceiling.

Writes gallery/app/public/data/hips_linearise/ceiling_checks.json (a few kB).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import odr, optimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/workflows"), str(ROOT / "research/ftir_hips_chem/scripts")]

import data_paths  # noqa: E402
import improve_io  # noqa: E402
import reference_closure as rc  # noqa: E402
import run_addis_reference_closure_20261110 as closure  # noqa: E402
from pls_transfer import deming_regression  # noqa: E402

OUT = ROOT / "gallery/app/public/data/hips_linearise/ceiling_checks.json"
LINEARISE = ROOT / "gallery/app/public/data/hips_linearise/linearise.json"
SEED = 20261120
N_BOOT = 100
AREA = closure.DEPOSIT_AREA_CM2
COLLOCATED = ("PHOE1", 1, "PHOE5", 2)          # site and POC of the two Phoenix samplers that ran side by side
LIGHT_UG = (2.0, 8.0)                         # light filters for the real spread in absorption per EC: past the MDL, before the bend


# ----------------------------------------------------------------------------- data
def improve_all(cache: Path | None) -> pd.DataFrame:
    """Every IMPROVE sample with TOR EC and HIPS, as improve_samples builds them but keeping all of them (not only those
    with a previous sample) and the HIPS lot, with the previous, two-back, three-back and next samples' loadings."""
    if cache and cache.is_file():
        return pd.read_parquet(cache)
    tables = data_paths.ftir_local_db() / "tables"

    def read(name, parameters, extra=()):
        frame = pd.read_csv(tables / name, encoding="utf-8-sig", low_memory=False,
                            usecols=["Site", "SampleDate", "Parameter", "Value", "AverageFlowRate", "ElapsedTime", *extra])
        frame = frame[frame["Parameter"].isin(parameters)].copy()
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
        wide = frame.pivot_table(index=["Site", "SampleDate"], columns="Parameter", values="Value", aggfunc="first")
        first = frame.groupby(["Site", "SampleDate"])[["AverageFlowRate", "ElapsedTime", *extra]].first()
        wide["volume_m3"] = first["AverageFlowRate"] * first["ElapsedTime"] / 1000.0
        for column in extra:
            wide[column] = first[column]
        return wide

    tor, hips = read("results_tor.csv", ["EC", "OC", "OPTR"]), read("results_hips.csv", ["fAbs"], ("LotNumber",))
    both = tor.join(hips, how="inner", lsuffix="_quartz", rsuffix="_ptfe").reset_index()
    both["date"] = pd.to_datetime(both["SampleDate"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    ok = (both["EC"] > 0) & (both["fAbs"] > 0) & both["date"].notna()
    ok &= both["volume_m3_quartz"].between(20, 45) & both["volume_m3_ptfe"].between(20, 45)
    both = both[ok].sort_values(["Site", "date"]).reset_index(drop=True)
    both["loading_ug"] = both["EC"] / 1000.0 * both["volume_m3_quartz"]
    both["tau"] = both["fAbs"] * both["volume_m3_ptfe"] / (100.0 * AREA)
    both["lot"] = both["LotNumber"].astype("string")
    by = both.groupby("Site")
    for name, k, (lo, hi) in (("previous", 1, (1, 7)), ("two_back", 2, (2, 14)), ("three_back", 3, (3, 21)), ("next", -1, (1, 7))):
        gap = (both["date"] - by["date"].shift(k)).dt.days.abs()
        both[f"{name}_ug"] = by["loading_ug"].shift(k).where(gap.between(lo, hi))
    both["previous_lot"] = by["lot"].shift(1)
    out = both[["Site", "date", "EC", "fAbs", "volume_m3_ptfe", "loading_ug", "tau", "lot", "previous_lot",
                "previous_ug", "two_back_ug", "three_back_ug", "next_ug"]]
    if cache:
        out.to_parquet(cache)
    return out


# ----------------------------------------------------------------------------- fits
def fit_medians(loading, tau, model=rc.hyperbolic, start=(0.04, 2.0), lower=(0.0, 0.2), upper=(1.0, 100.0)):
    residual = lambda p: np.log(model(loading, *p)) - np.log(tau)  # noqa: E731
    result = optimize.least_squares(residual, list(start), bounds=(list(lower), list(upper)))
    return result.x, float(np.sqrt(np.mean(result.fun ** 2)))


def strata(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    g = frame.groupby(np.digitize(frame[column], closure.IMPROVE_EDGES))
    t = pd.DataFrame({"n": g.size(), "L": g["loading_ug"].median(), "t": g["tau"].median()})
    return t[t["n"] >= closure.IMPROVE_MIN_STRATUM]


def instruments(samples: pd.DataFrame, n_boot: int) -> list[dict]:
    """The published grouping with another instrument for this filter's loading; intervals resample whole sites."""
    variants = {
        "previous": ("previous_ug", None, "The previous sample (published)"),
        "two_back": ("two_back_ug", None, "The sample two back"),
        "three_back": ("three_back_ug", None, "The sample three back"),
        "next": ("next_ug", None, "The next sample"),
        "other_lot": ("previous_ug", "other", "The previous sample, when its HIPS lot differs"),
        "same_lot": ("previous_ug", "same", "The previous sample, when its HIPS lot is the same"),
    }
    frames = {}
    for key, (column, lot, label) in variants.items():
        part = samples.dropna(subset=[column])
        if lot:
            known = (part["lot"].notna() & part["previous_lot"].notna()).astype(bool)
            same = (part["lot"] == part["previous_lot"]).fillna(False).astype(bool)
            part = part[known & (same if lot == "same" else ~same)]
        frames[key] = part
    rows = []
    for key, part in frames.items():
        column, _, label = variants[key]
        s = strata(part, column)
        (a, tau_inf), rms = fit_medians(s["L"].to_numpy(), s["t"].to_numpy())
        rows.append({"key": key, "label": label, "samples": int(len(part)), "sites": int(part["Site"].nunique()), "groups": int(len(s)),
                     "a_per_ug": round(float(a), 6), "tau_inf": round(float(tau_inf), 4), "boot": []})
    rng = np.random.default_rng(SEED)
    sites = np.array(sorted(samples["Site"].unique()))
    by_site = {k: {s: f for s, f in part.groupby("Site")} for k, part in frames.items()}
    for i in range(n_boot):
        draw = rng.choice(sites, sites.size)
        for row in rows:
            own = by_site[row["key"]]
            part = pd.concat([own[s] for s in draw if s in own], ignore_index=True)
            s = strata(part, variants[row["key"]][0])
            if len(s) >= 5:
                row["boot"].append(fit_medians(s["L"].to_numpy(), s["t"].to_numpy())[0][1])
        if (i + 1) % 20 == 0:
            print(f"  instruments: resample {i + 1}/{n_boot}")
    for row in rows:
        lo, hi = np.percentile(row.pop("boot"), [5, 95])
        row["tau_inf_lo"], row["tau_inf_hi"] = round(float(lo), 4), round(float(hi), 4)
        print(f"instrument {row['key']:11s} n={row['samples']:>7,} groups={row['groups']} τ∞ {row['tau_inf']:.2f} [{lo:.2f}, {hi:.2f}]")
    return rows


def error_model(samples: pd.DataFrame) -> dict:
    """Measurement error of TOR EC and HIPS Fabs per filter, from the Phoenix collocated samplers, and the real spread in
    absorption per EC on light filters. Each error is sd = sqrt(floor² + (relative × value)²), fitted to the pairs."""
    clean = improve_io.load_improve_clean(usecols=["SiteCode", "POC", "Date", "ECf_Val", "ECf_Unc", "fAbs_Val", "fAbs_Unc"])
    clean["date"] = pd.to_datetime(clean["Date"], errors="coerce")
    a_site, a_poc, b_site, b_poc = COLLOCATED
    a = clean[(clean["SiteCode"] == a_site) & (clean["POC"] == a_poc)]
    b = clean[(clean["SiteCode"] == b_site) & (clean["POC"] == b_poc)]
    pairs = a.merge(b, on="date", suffixes=("_1", "_2"))
    out = {"pairs": int(len(pairs)), "sites": [a_site, b_site], "first": str(pairs["date"].min().date()), "last": str(pairs["date"].max().date())}
    for key in ("ECf", "fAbs"):
        d = pairs[[f"{key}_Val_1", f"{key}_Val_2", f"{key}_Unc_1", f"{key}_Unc_2"]].dropna()
        v1, v2 = d[f"{key}_Val_1"].to_numpy(), d[f"{key}_Val_2"].to_numpy()
        mean, var = (v1 + v2) / 2, (v1 - v2) ** 2 / 2                  # one sample's error variance, from each pair
        # floor and relative error by least squares on the pair variances (absolute residuals: the mean of d²/2 is σ²)
        res = optimize.least_squares(lambda p: (p[0] ** 2 + (p[1] * mean) ** 2) - var, [np.median(mean) * 0.05, 0.1], loss="soft_l1",
                                     f_scale=np.median(var))
        floor, rel = abs(float(res.x[0])), abs(float(res.x[1]))
        z = (v1 - v2) / np.sqrt(d[f"{key}_Unc_1"] ** 2 + d[f"{key}_Unc_2"] ** 2)
        out[key] = {"n": int(len(d)), "floor": round(floor, 5), "relative": round(rel, 4),
                    "median_reported_relative": round(float((d[f"{key}_Unc_1"] / v1).median()), 4),
                    "z_sd": round(float(np.std(z)), 3), "z_robust_sd": round(float(1.4826 * np.median(np.abs(z - np.median(z)))), 3)}
    # the real spread in absorption per EC: on light filters, the variance of log(τ / L) less what measurement adds
    light = samples[samples["loading_ug"].between(*LIGHT_UG)]
    ec, fabs = light["EC"].to_numpy() / 1000.0, light["fAbs"].to_numpy()
    rel_ec = np.sqrt(out["ECf"]["floor"] ** 2 + (out["ECf"]["relative"] * ec) ** 2) / ec
    rel_fabs = np.sqrt(out["fAbs"]["floor"] ** 2 + (out["fAbs"]["relative"] * fabs) ** 2) / fabs
    ratio = np.log(light["tau"].to_numpy() / light["loading_ug"].to_numpy())
    spread = 1.4826 * np.median(np.abs(ratio - np.median(ratio)))       # robust: a few odd filters do not set it
    real = float(np.sqrt(max(spread ** 2 - np.median(rel_ec ** 2 + rel_fabs ** 2), 0.0)))
    out["light"] = {"from_ug": LIGHT_UG[0], "to_ug": LIGHT_UG[1], "n": int(len(light)), "log_spread": round(float(spread), 4),
                    "measurement_part": round(float(np.sqrt(np.median(rel_ec ** 2 + rel_fabs ** 2))), 4), "real_spread": round(real, 4)}
    print(f"collocated: {out['pairs']} pairs; EC {out['ECf']}; fAbs {out['fAbs']}; light {out['light']}")
    return out


def eiv(samples: pd.DataFrame, errors: dict, n_boot: int) -> list[dict]:
    """Orthogonal distance regression of the curve on every filter, under each assumption about the errors."""
    ec, fabs = samples["EC"].to_numpy() / 1000.0, samples["fAbs"].to_numpy()
    L, T = samples["loading_ug"].to_numpy(), samples["tau"].to_numpy()
    sd = lambda e, v: np.sqrt(e["floor"] ** 2 + (e["relative"] * v) ** 2)  # noqa: E731
    sx_meas = sd(errors["ECf"], ec) / ec * L
    sy_meas = sd(errors["fAbs"], fabs) / fabs * T
    real = errors["light"]["real_spread"] * T
    variants = [
        ("tau_only", "Error on τ only (an ordinary weighted fit)", None, np.sqrt(sy_meas ** 2 + real ** 2)),
        ("measured", "Measurement error on both, from the collocated pairs", sx_meas, sy_meas),
        ("measured_real", "Measurement error on both, and the real spread in absorption per EC on τ", sx_meas, np.sqrt(sy_meas ** 2 + real ** 2)),
        ("half_tor", "As above, with TOR's error halved", sx_meas / 2, np.sqrt(sy_meas ** 2 + real ** 2)),
        ("double_tor", "As above, with TOR's error doubled", sx_meas * 2, np.sqrt(sy_meas ** 2 + real ** 2)),
    ]
    model = odr.Model(lambda b, x: rc.hyperbolic(x, b[0], b[1]))

    def run(index, sx, sy):
        data = odr.RealData(L[index], T[index], sx=None if sx is None else sx[index], sy=sy[index])
        return odr.ODR(data, model, beta0=[0.037, 2.26], maxit=200).run().beta

    sites = samples["Site"].to_numpy()
    by_site = {s: np.flatnonzero(sites == s) for s in np.unique(sites)}
    names = np.array(sorted(by_site))
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([by_site[s] for s in rng.choice(names, names.size)]) for _ in range(n_boot)]
    everything = np.arange(len(L))
    rows = []
    for key, label, sx, sy in variants:
        a, tau_inf = run(everything, sx, sy)
        # only the main assumption gets an interval: each resample is a full fit on ~350,000 filters
        boot = [run(d, sx, sy)[1] for d in draws] if key == "measured_real" else []
        lo, hi = (np.percentile(boot, [5, 95]) if boot else (np.nan, np.nan))
        rows.append({"key": key, "label": label, "samples": int(len(L)), "a_per_ug": round(float(a), 6), "tau_inf": round(float(tau_inf), 4),
                     "tau_inf_lo": None if not boot else round(float(lo), 4), "tau_inf_hi": None if not boot else round(float(hi), 4)})
        print(f"eiv {key:14s} τ∞ {tau_inf:.2f}" + (f" [{lo:.2f}, {hi:.2f}]" if boot else ""))
    return rows


def within_site(samples: pd.DataFrame, n_boot: int, min_n: int = 30) -> dict:
    """One ceiling with each site its own absorption per EC: groups by the previous sample's loading within each site,
    every site its own a, one τ∞ shared. A bend that came from heavy filters being from sites with less absorption per EC
    would vanish here; a loading effect would not."""
    def cells(frame):
        g = frame.groupby(["Site", np.digitize(frame["previous_ug"], closure.IMPROVE_EDGES)])
        t = pd.DataFrame({"n": g.size(), "L": g["loading_ug"].median(), "t": g["tau"].median()}).reset_index()
        t = t[(t["n"] >= min_n) & (t["L"] > 0) & (t["t"] > 0)]
        return t[t.groupby("Site")["Site"].transform("size") >= 3]       # a site needs three groups to say anything about the bend

    def fit(t):
        codes, index = np.unique(t["Site"].to_numpy(), return_inverse=True)
        L, T = t["L"].to_numpy(), t["t"].to_numpy()
        start = np.r_[np.full(codes.size, 0.04), 2.0]
        residual = lambda p: np.log(rc.hyperbolic(L, p[index], p[-1])) - np.log(T)  # noqa: E731
        lower, upper = np.r_[np.zeros(codes.size), 0.2], np.r_[np.ones(codes.size), 100.0]
        return float(optimize.least_squares(residual, start, bounds=(lower, upper)).x[-1]), int(codes.size), int(len(t))

    part = samples.dropna(subset=["previous_ug"])
    tau_inf, sites, n_cells = fit(cells(part))
    rng = np.random.default_rng(SEED)
    by_site = {s: f for s, f in part.groupby("Site")}
    names = np.array(sorted(by_site))
    boot = []
    for i in range(n_boot):
        # a site drawn twice is a second site: relabel so each copy gets its own a
        draw = pd.concat([by_site[s].assign(Site=f"{s}#{k}") for k, s in enumerate(rng.choice(names, names.size))], ignore_index=True)
        boot.append(fit(cells(draw))[0])
        if (i + 1) % 20 == 0:
            print(f"  within sites: resample {i + 1}/{n_boot}")
    lo, hi = np.percentile(boot, [5, 95])
    print(f"within sites: {sites} sites, {n_cells} groups, τ∞ {tau_inf:.2f} [{lo:.2f}, {hi:.2f}]")
    return {"key": "within_site", "label": "Within sites: each site its own absorption per EC, one shared ceiling", "sites": sites,
            "groups": n_cells, "min_n": min_n, "tau_inf": round(tau_inf, 4), "tau_inf_lo": round(float(lo), 4), "tau_inf_hi": round(float(hi), 4)}


# ----------------------------------------------------------------------------- shapes and Addis
def exponential(loading, a, tau_inf):
    """``tau = τ∞ (1 − exp(−aL/τ∞))``: proportional on light filters, approaching τ∞ faster than the hyperbola."""
    x = a * np.asarray(loading, dtype=float)
    return tau_inf * (1.0 - np.exp(-x / tau_inf))


SHAPES = {
    # name: (label, response, inverse(τ, p) → aL, ceiling(p), start, lower, upper)
    "hyperbolic": ("Hyperbolic (published)", rc.hyperbolic, lambda t, p: t / (1 - t / p[1]), lambda p: p[1], (0.04, 2.0), (0.0, 0.2), (1.0, 100.0)),
    "exponential": ("Exponential approach", exponential, lambda t, p: -p[1] * np.log(1 - t / p[1]), lambda p: p[1], (0.04, 2.0), (0.0, 0.2), (1.0, 100.0)),
    "patterned": ("Patterned deposit, fitted coverage (White et al. 2025)", rc.patterned,
                  lambda t, p: -p[1] * np.log((np.exp(-t) - (1 - p[1])) / p[1]), lambda p: -np.log(1 - p[1]), (0.04, 0.8), (0.0, 0.05), (1.0, 0.9999)),
}


def addis_outcome(lin: dict, inverse, params) -> dict:
    """The Addis fit of FTIR EC on HIPS BC linearised through a curve: each filter's HIPS BC × (aL / τ)."""
    f = pd.DataFrame(lin["filters"])
    f = f[f["ftir"].notna()]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = inverse(f["tau"].to_numpy(), params)
    keep = np.isfinite(u) & (u > 0)
    x = f["hips_bc"].to_numpy()[keep] * u[keep] / f["tau"].to_numpy()[keep]
    d = deming_regression(x, f["ftir"].to_numpy()[keep], lin["lambda"])
    return {"filters": int(keep.sum()), "past_ceiling": int((~keep).sum()), "slope": round(d["slope"], 4), "intercept": round(d["intercept"], 4),
            "mac_for_slope_one": round(lin["mac"] / d["slope"], 3), "median_factor": round(float(np.median(u[keep] / f["tau"].to_numpy()[keep])), 3)}


def support(samples: pd.DataFrame, lin: dict, darkest_group: float) -> dict:
    """How far the Addis filters sit beyond the IMPROVE filters the curve is fitted on."""
    taus = np.sort(samples["tau"].to_numpy())
    addis = np.array([f["tau"] for f in lin["filters"] if f["ftir"] is not None])
    darker = len(taus) - np.searchsorted(taus, addis)                    # IMPROVE filters at least as dark as each Addis filter
    return {"improve_filters": int(len(taus)), "darkest_group_tau": round(darkest_group, 4), "addis_filters": int(len(addis)),
            "addis_beyond_darkest_group": round(float(np.mean(addis > darkest_group)), 4),
            "addis_with_under_1000_darker": round(float(np.mean(darker < 1000)), 4), "addis_with_under_100_darker": round(float(np.mean(darker < 100)), 4),
            "improve_darker_than_addis_median": int(len(taus) - np.searchsorted(taus, np.median(addis))),
            "improve_tau_quantiles": {str(q): round(float(np.quantile(taus, q)), 4) for q in (0.9, 0.99, 0.999, 0.9999)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, help="parquet of every IMPROVE sample: read if present, else written after the read")
    parser.add_argument("--boot", type=int, default=N_BOOT, help="site resamples (default %(default)s)")
    args = parser.parse_args()
    samples = improve_all(args.cache)
    published = samples.dropna(subset=["previous_ug"])
    print(f"{len(samples):,} IMPROVE samples; {len(published):,} with a previous sample (the published set has 347,936)")
    lin = json.loads(LINEARISE.read_text())
    summary = json.loads(closure.SUMMARY.read_text())
    medians = pd.DataFrame([r for r in summary["improve_strata"] if r["population"] == closure.IMPROVE_PRIMARY])

    errors = error_model(published)
    eiv_rows = eiv(published, errors, args.boot)
    inst_rows = instruments(samples, args.boot)
    site_row = within_site(samples, args.boot)

    shapes = []
    for key, (label, response, inverse, ceiling, start, lower, upper) in SHAPES.items():
        p, rms = fit_medians(medians["loading_ug"].to_numpy(), medians["tau"].to_numpy(), response, start, lower, upper)
        shapes.append({"key": key, "label": label, "params": [round(float(v), 6) for v in p], "ceiling": round(float(ceiling(p)), 4),
                       "rms_log": round(rms, 5), "aic": round(float(len(medians) * np.log(rms ** 2) + 2 * len(p)), 3)})
    # Warren's fully pixelated limit: the deposit covers the support-screen holes, 0.638 of the area (McDade et al. 2009)
    p, rms = fit_medians(medians["loading_ug"].to_numpy(), medians["tau"].to_numpy(), lambda L, a: rc.patterned(L, a, 0.638), (0.04,), (0.0,), (1.0,))
    shapes.append({"key": "pixelated", "label": "Fully pixelated deposit, coverage 0.638 (White et al. 2025)", "params": [round(float(p[0]), 6), 0.638],
                   "ceiling": round(float(-np.log(1 - 0.638)), 4), "rms_log": round(rms, 5), "aic": round(float(len(medians) * np.log(rms ** 2) + 2), 3)})
    for s in shapes:
        print(f"shape {s['key']:12s} ceiling {s['ceiling']:.2f} rms {s['rms_log']:.4f} aic {s['aic']:.1f}")

    hyper_inverse = SHAPES["hyperbolic"][2]
    patterned_inverse = SHAPES["patterned"][2]
    addis = []
    published_row = next(r for r in summary["improve_ceiling"] if r["population"] == closure.IMPROVE_PRIMARY)
    for label, tau_inf, group in (("Published, central", published_row["tau_inf"], "published"), ("Published, 90 % low", published_row["tau_inf_lo"], "published"),
                                  ("Published, 90 % high", published_row["tau_inf_hi"], "published")):
        addis.append({"label": label, "group": group, "ceiling": tau_inf, **addis_outcome(lin, hyper_inverse, [None, tau_inf])})
    for population, label in (("2005–2014", "Published grouping, 2005–2014 filters only"), ("2015–2025", "Published grouping, 2015–2025 filters only")):
        r = next(x for x in summary["improve_ceiling"] if x["population"] == population)
        addis.append({"label": label, "group": "era", "ceiling": r["tau_inf"], **addis_outcome(lin, hyper_inverse, [None, r["tau_inf"]])})
    addis.append({"label": site_row["label"], "group": "within_site", "ceiling": site_row["tau_inf"], **addis_outcome(lin, hyper_inverse, [None, site_row["tau_inf"]])})
    for r in eiv_rows:
        addis.append({"label": f"Errors in both axes: {r['label']}", "group": "eiv", "ceiling": r["tau_inf"], **addis_outcome(lin, hyper_inverse, [None, r["tau_inf"]])})
    for r in inst_rows:
        addis.append({"label": f"Instrument: {r['label']}", "group": "instrument", "ceiling": r["tau_inf"], **addis_outcome(lin, hyper_inverse, [None, r["tau_inf"]])})
    for s in shapes:
        inverse = {"hyperbolic": hyper_inverse, "exponential": SHAPES["exponential"][2]}.get(s["key"], patterned_inverse)
        addis.append({"label": f"Shape: {s['label']}", "group": "shape", "ceiling": s["ceiling"], **addis_outcome(lin, inverse, s["params"])})
    addis.append({"label": "As measured", "group": "measured", "ceiling": None, **addis_outcome(lin, lambda t, p: t, None)})
    for r in addis:
        print(f"addis {r['label'][:60]:60s} ceiling {r['ceiling'] if r['ceiling'] is None else round(r['ceiling'], 2)!s:>6} "
              f"{r['slope']:.2f}x{r['intercept']:+.2f} MAC1 {r['mac_for_slope_one']:.1f} past {r['past_ceiling']}")

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "gallery/data/export_hips_ceiling_checks.py; IMPROVE TOR and HIPS tables, the cleaned FED cache, linearise.json",
        "published": {"tau_inf": published_row["tau_inf"], "lo": published_row["tau_inf_lo"], "hi": published_row["tau_inf_hi"]},
        "n_boot": args.boot, "errors": errors, "eiv": eiv_rows, "instruments": inst_rows, "within_site": site_row, "shapes": shapes,
        "support": support(published, lin, float(medians["tau"].max())), "addis": addis,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
