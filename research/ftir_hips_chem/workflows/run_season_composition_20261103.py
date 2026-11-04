"""Is the Addis Bega < Belg < Kiremt pattern in FTIR EC a season effect or a composition effect? 3 Nov 2026.

Follows ``run_transfer_diagnosis_20261102.py`` and reads its outputs, so it adds no new fits:

* ``network_filters.csv``: per-filter FTIR EC / HIPS for every model at k 18 (k 12 for the OC/EC-balanced model), plus
  the IMPROVE TOR yardstick ``exp_tor_hips`` (expected TOR/HIPS at the filter's loading and OC/BC).
* ``predictions.npz``: IMPROVE out-of-site predictions, used for the IMPROVE organic-sensitivity check.
* ``addis_residuals.csv``: Addis filter lots.
* the 22 Oct decision bundle: ``improve_meta.csv`` (TOR OC/EC) and ``network_meta.csv`` (PM2.5, SO4, Si, Al, Fe, K).

Seasons use the ``local_2026_10`` calendar (``belg_feb``): Bega Oct-Jan, Belg Feb-May, Kiremt Jun-Sep.

The response is log(FTIR EC / TOR yardstick) per filter. Season effects are OLS coefficients on season dummies, reported
as % relative to Bega, with log HIPS loading as a covariate and then with organic covariates:

* ``loc``  FTIR OC (SPARTAN's product). Comes from the same spectrum as FTIR EC, so it can share the error.
* ``lrm``  residual mass, PM2.5 − 1.375 SO4 − (2.20 Al + 2.49 Si + 2.42 Fe) − HIPS BC, in µg/m³. It doesn't use the
           spectrum. It also contains nitrate, ammonium beyond (NH4)2SO4, Ca/Ti/Na dust and water, so it is a rough
           organic proxy.
* ``lnsk`` non-soil K (K − 0.6 Fe), a biomass-burning marker.

The IMPROVE check repeats the organic regression with TOR OC, which is independent of the FTIR spectrum.

    ../../.venv/bin/python workflows/run_season_composition_20261103.py [--bundle DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path[:0] = [str(AREA / "workflows"), str(AREA / "scripts"), str(ROOT)]

SRC = AREA / "output/tables/transfer_diagnosis_20261102"
OUT = AREA / "output/tables/season_composition_20261103"
SEASONS = ["Bega", "Belg", "Kiremt"]
K_IMPROVE = 18
COVARIATE_SETS = {
    "season only": "s",
    "+ loading": "s + lh",
    "+ loading + residual mass": "s + lh + lrm",
    "+ loading + residual mass + nsK": "s + lh + lrm + lnsk",
    "+ loading + residual mass + nsK + lot": "s + lh + lrm + lnsk + C(lot)",
    "+ loading + FTIR OC": "s + lh + loc",
}


def season_of(dates: pd.Series) -> pd.Series:
    m = pd.to_datetime(dates).dt.month
    return pd.Series(np.select([m.isin([10, 11, 12, 1]), m.isin([2, 3, 4, 5])], ["Bega", "Belg"], "Kiremt"), index=dates.index)


def model_columns(nf: pd.DataFrame) -> list[str]:
    return ["published"] + [c for c in nf.columns if " k" in c]


def default_bundle() -> Path:
    import run_transfer_diagnosis_20261102 as td

    return td.default_bundle()


def load_sites(bundle: Path) -> tuple[pd.DataFrame, list[str]]:
    nf = pd.read_csv(SRC / "network_filters.csv")
    nm = pd.read_csv(bundle / "network_meta.csv")
    lots = pd.read_csv(SRC / "addis_residuals.csv").set_index("filter_id").lot
    d = nf[nf.code.isin(["ETAD", "ETBI"])].merge(
        nm[["filter_id", "volume_m3", "pm25_ugm3", "so4_ugm3", "si_ugm3", "al_ugm3", "fe_ugm3", "k_xrf_ugm3", "oc_ftir_ugm3"]],
        on="filter_id", how="left")
    d["season"] = season_of(d.date)
    d["lot"] = d.filter_id.map(lots).fillna(0).astype(int).astype(str)
    v = d.volume_m3
    d["bc_ugm3"] = d.hips_ug / v
    d["rm_ugm3"] = d.pm25_ugm3 - 1.375 * d.so4_ugm3 - (2.20 * d.al_ugm3 + 2.49 * d.si_ugm3 + 2.42 * d.fe_ugm3) - d.bc_ugm3
    d["nsk_ugm3"] = (d.k_xrf_ugm3 - 0.6 * d.fe_ugm3).clip(lower=0.005)
    d["oc_over_bc"] = d.oc_ftir_ugm3 / d.bc_ugm3
    d["rm_over_bc"] = d.rm_ugm3 / d.bc_ugm3
    d["lh"] = np.log(d.hips_ug)
    d["loc"] = np.log(d.oc_ftir_ugm3 * v)
    d["lrm"] = np.log(d.rm_ugm3.clip(lower=0.3) * v)
    d["lnsk"] = np.log(d.nsk_ugm3 * v)
    models = model_columns(nf)
    for m in models:
        d[f"y|{m}"] = np.log(d[m] / d.exp_tor_hips)
    return d, models


def pct(b: float) -> float:
    return 100 * (np.exp(b) - 1)


def season_tables(d: pd.DataFrame, models: list[str]) -> dict[str, pd.DataFrame]:
    a = d[d.code == "ETAD"].copy()
    a["s"] = pd.Categorical(a.season, SEASONS)
    a["tertile"] = pd.qcut(a.hips_ug, 3, labels=["light", "mid", "heavy"]).astype(str)
    ok = a.so4_ugm3.notna() & (a.rm_ugm3 > 0.3)

    med, tert, eff, lso = [], [], [], []
    for m in models:
        a["y"] = a[f"y|{m}"]
        trend = np.polyfit(a.lh, a.y, 1)
        a["y_adj"] = a.y - np.polyval(trend, a.lh)
        for s, g in a.groupby("season"):
            med.append(dict(model=m, season=s, n=len(g), vs_hips=g[m].median(), vs_tor=np.exp(g.y.median()),
                            vs_tor_loading_adjusted=np.exp(g.y_adj.median())))
        for (t, s), g in a.groupby(["tertile", "season"]):
            tert.append(dict(model=m, tertile=t, season=s, n=len(g), vs_tor=np.exp(g.y.median())))
        for label, rhs in COVARIATE_SETS.items():
            r = smf.ols(f"y ~ {rhs}", a[ok]).fit()
            ci = r.conf_int()
            row = dict(model=m, covariates=label, n=int(r.nobs), r2=r.rsquared)
            for s in ("Belg", "Kiremt"):
                k = f"s[T.{s}]"
                row |= {f"{s}_pct": pct(r.params[k]), f"{s}_lo": pct(ci.loc[k, 0]), f"{s}_hi": pct(ci.loc[k, 1]),
                        f"{s}_p": r.pvalues[k]}
            for c in ("lrm", "loc", "lnsk"):
                row[f"coef_{c}"] = r.params.get(c, np.nan)
            eff.append(row)
        for proxy in ("lrm", "loc"):
            for s in SEASONS:
                train, test = a[ok & (a.season != s)], a[ok & (a.season == s)]
                r = smf.ols(f"y ~ lh + {proxy}", train).fit()
                lso.append(dict(model=m, proxy=proxy, held_out=s, n=len(test), observed=np.exp(test.y.median()),
                                predicted=np.exp(np.median(r.predict(test)))))
    comp = a.groupby("season").agg(n=("filter_id", "size"), hips_ug=("hips_ug", "median"), oc_over_bc=("oc_over_bc", "median"),
                                   rm_over_bc=("rm_over_bc", "median"), pm25_ugm3=("pm25_ugm3", "median"),
                                   so4_ugm3=("so4_ugm3", "median"), nsk_ugm3=("nsk_ugm3", "median")).reset_index()
    return dict(season_medians=pd.DataFrame(med), loading_tertiles=pd.DataFrame(tert), season_effects=pd.DataFrame(eff),
                leave_season_out=pd.DataFrame(lso), composition=comp,
                addis_filters=a.drop(columns=["s", "y", "y_adj"]))


def improve_sensitivity(bundle: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    im = pd.read_csv(bundle / "improve_meta.csv")
    P = np.load(SRC / "predictions.npz")
    im["hips"] = im.fabs_Mm1 * im.volume_m3 / 10
    im["oc"] = im.tor_oc_ugm3 * im.volume_m3
    im["tor"] = im.ec_ug
    im["ocec"] = im.tor_oc_ugm3 / im.tor_ec_ugm3
    base = ((im.hips > 5) & (im.oc > 0) & (im.tor > 2) & (im.tor < 150)).to_numpy()
    names = [k.split("|")[0] for k in P.files if k.endswith("|improve_oof")]
    rows, bins = [], []
    for nm in names:
        k = 12 if nm == "anchored_balanced" else K_IMPROVE
        f = P[f"{nm}|improve_oof"][:, k - 1]
        ok = base & (f > 0)
        g = im[ok].assign(y=np.log(f[ok] / im.tor[ok]), loc=np.log(im.oc[ok]), lh=np.log(im.hips[ok]), lt=np.log(im.tor[ok]))
        for rhs in ("loc + lt", "loc + lh", "loc + lh + lt"):
            r = smf.ols(f"y ~ {rhs}", g).fit()
            rows.append(dict(model=nm, k=k, covariates=rhs, n=int(r.nobs), coef_loc=r.params["loc"], se=r.bse["loc"]))
        cut = pd.cut(g.ocec, [0, 1.5, 2.5, 4, 6, 10, 100])
        for b, h in g.groupby(cut, observed=True):
            bins.append(dict(model=nm, k=k, ocec_bin=str(b), n=len(h), ftir_over_tor=np.exp(h.y.median())))
    return pd.DataFrame(rows), pd.DataFrame(bins)


def bishoftu(d: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    a = d[(d.code == "ETAD") & d.so4_ugm3.notna() & (d.rm_ugm3 > 0.3)]
    e = d[d.code == "ETBI"]
    rows = []
    for m in models:
        for proxy in ("lrm", "loc"):
            ee = e[e.rm_ugm3 > 0.3] if proxy == "lrm" else e
            r = smf.ols(f"Q('y|{m}') ~ lh + {proxy}", a).fit()
            rows.append(dict(model=m, proxy=proxy, n=len(ee), months=", ".join(sorted(pd.to_datetime(ee.date).dt.strftime("%Y-%m").unique())),
                             observed=np.exp(ee[f"y|{m}"].median()), predicted_from_addis=np.exp(np.median(r.predict(ee))),
                             addis_bega_observed=np.exp(a.loc[a.season == "Bega", f"y|{m}"].median()),
                             oc_over_bc=ee.oc_over_bc.median(), rm_over_bc=ee.rm_over_bc.median()))
    return pd.DataFrame(rows)


WINDOW_GROUPS = [("4000-3400 O-H", 4000, 3400), ("3400-3000 N-H / O-H / aromatic", 3400, 3000),
                 ("3000-2800 aliphatic C-H", 3000, 2800), ("2800-2000", 2800, 2000), ("2000-1800", 2000, 1800),
                 ("1800-1500 C=O, C=C, N-H", 1800, 1500), ("1500-1425", 1500, 1425), ("below 1425", 1425, 499)]


def window_attribution(bundle: Path, addis: pd.DataFrame) -> pd.DataFrame:
    """Split each filter's FTIR EC / TOR yardstick into spectral-window shares of the k 18 PLS prediction.

    Linear, so the window shares plus the intercept share add up to the ratio. ``kiremt_minus_bega`` is the difference
    of season means; ``kiremt_adjusted`` is the Kiremt coefficient of ``share ~ season + log HIPS + log residual mass``.
    """
    nm = pd.read_csv(bundle / "network_meta.csv")
    row = pd.Series(np.arange(len(nm)), index=nm.filter_id)
    a = addis[addis.so4_ugm3.notna() & (addis.rm_ugm3 > 0.3)].reset_index(drop=True)
    idx = row.reindex(a.filter_id).to_numpy()
    coef = np.load(SRC / "coefficients.npz")
    wn, wns = np.load(bundle / "wn_full.npy"), np.load(bundle / "wn_spline.npy")
    spectra = {"raw": (bundle / "network_raw.npy", wn), "anchored": (SRC / "anchored_network.npy", wn),
               "vibes": (bundle / "network_vibes.npy", wn), "spline": (bundle / "network_spline.npy", wns)}
    exp_tor = (a.exp_tor_hips * a.hips_ug).to_numpy()
    formula, term = 'share ~ C(season, Treatment("Bega")) + lh + lrm', 'C(season, Treatment("Bega"))[T.Kiremt]'
    rows = []
    for name, (path, grid) in spectra.items():
        X = np.asarray(np.load(path, mmap_mode="r")[idx], float)
        share = (X - coef[f"{name}|xm"]) * coef[f"{name}|B"][:, K_IMPROVE - 1] / exp_tor[:, None]
        total = a[f"{name} k{K_IMPROVE}"].to_numpy() * a.hips_ug.to_numpy() / exp_tor
        parts = [(w, share[:, (grid <= hi) & (grid > lo)].sum(1)) for w, hi, lo in WINDOW_GROUPS if ((grid <= hi) & (grid > lo)).any()]
        parts += [("intercept", total - share.sum(1)), ("TOTAL", total)]
        for w, v in parts:
            d = a[["season", "lh", "lrm"]].assign(share=v)
            rows.append(dict(model=name, window=w,
                             kiremt_minus_bega=d.share[d.season == "Kiremt"].mean() - d.share[d.season == "Bega"].mean(),
                             kiremt_adjusted=smf.ols(formula, d).fit().params[term]))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=None, help="22 Oct decision bundle (default: the Drive copy)")
    a = p.parse_args()
    bundle = a.bundle or default_bundle()
    t = time.time()
    d, models = load_sites(bundle)
    out = season_tables(d, models)
    out["improve_organic_slope"], out["improve_ocec_bins"] = improve_sensitivity(bundle)
    out["bishoftu"] = bishoftu(d, models)
    out["season_windows"] = window_attribution(bundle, out["addis_filters"])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "receipt.json").write_text(json.dumps({
        "source": str(SRC.relative_to(ROOT)), "bundle": str(bundle), "models": models, "calendar": "local_2026_10 (belg_feb)",
        "response": "log(FTIR EC / IMPROVE TOR yardstick)", "improve_k": K_IMPROVE,
        "residual_mass": "PM2.5 - 1.375 SO4 - (2.20 Al + 2.49 Si + 2.42 Fe) - HIPS BC (ug/m3), kept if > 0.3",
        "n_addis": int((d.code == "ETAD").sum()), "n_bishoftu": int((d.code == "ETBI").sum()),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    print(f"wrote {len(out)} tables to {OUT} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
