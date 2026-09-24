"""IMPROVE-analog bias diagnostics (Ann/Satoshi meeting, 2026-09-17).

The question: are the IMPROVE filters selected as spectral analogs of each Addis
season mispredicted the same way Addis is? If they are, the model is seeing the
same thing we are; if not, the difference points at the references (TOR EC vs
HIPS Fabs) or at something in the spectra the correlation cannot see.

Both baseline methods come from the same frozen full-profile Colab run
(``vibes_colab_cloud/.../full-83dcf32e86bc0f09``): identical IMPROVE filters,
identical Addis filters, identical 2,002-channel grid. Its AIRSpec arrays match
the phase-3 cache the meeting's seasonal calibrations were fitted on to ~1e-7
relative, so AIRSpec results here reproduce that pipeline.

Units, stated once because the source table misleads: ``results_tor.csv``
``Value`` is **ng/m³** (loading = Value × volume / 1000 reproduces the run's
µg/filter y exactly). ``phase3_common.load_tor_loadings`` names that column
``TOR_EC_ugm3``; its OC/EC ratio is unaffected, but anything dividing Fabs by it
would be off by 1000. Concentrations here are converted explicitly.

Selection masks affect analog selection only; every PLS fit uses the full grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))

from config import ETHIOPIA_SEASONS, MAC_VALUE  # noqa: E402
from seasonal_analogs import mean_correlation_scores, spectral_region_mask  # noqa: E402
from plotting.utils import calculate_regression_stats, deming_bootstrap  # noqa: E402
from calibration_modes import (  # noqa: E402
    protocol_cv_curve,
    protocol_select_k,
    protocol_train_mask,
)

RUN = AREA / "output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
ADDIS_EVALUATION = AREA / "output/tables/ann_weekly_20260910/addis_evaluation.csv"
METHODS = ("AIRSpec", "VIBES")
# Full-range VIBES (4000-500 cm-1, rank cap 30), assembled by
# workflows/assemble_vibes_fullrange.py from the 2026-09-23 Colab run. When present
# it adds two baselines on the same cases: all 2722 channels ("VIBES-full") and
# the same correction restricted to AIRSpec's 4000-1425 grid ("VIBES-full-cut"),
# which separates "fitted on a wider range" from "used a wider range".
FULLRANGE = AREA / "output/tables/vibes_fullrange/rank30"
FULLRANGE_METHODS = ("VIBES-full", "VIBES-full-cut")
SEASONS = list(ETHIOPIA_SEASONS)
# Locked at the meeting: always drop CO2 (1800-2500) and everything above 3600.
LOCKED_MASK = (1800, 3600)
# What the meeting's seasonal slide was fitted with (CO2 only), for reproduction.
MEETING_MASK = (1800, None)
N_ANALOGS = 500
MAX_COMPONENTS = 30
# sigma_y is the prior held-out TOR RMSE proxy used for every Addis Deming fit
# in the phase-3 and Sept-10 work; sigma_x comes from HIPS_Uncertainty rows.
SIGMA_Y_PROXY = 0.531
TOR_PARAMETERS = ("EC", "OC", "EC1", "OPTR", "OPTT")


@dataclass
class Context:
    wn: np.ndarray
    lib: pd.DataFrame                 # IMPROVE calibration filters + references
    X: dict                           # method -> (n_lib, n_wn) corrected spectra
    addis: pd.DataFrame               # evaluation filters present in both methods
    XA: dict                          # method -> (n_addis, n_wn)
    sigma_x: float                    # HIPS Fabs/MAC uncertainty, µg/m³
    notes: list = field(default_factory=list)
    methods: tuple = METHODS          # baselines available in X/XA
    wn_by: dict = field(default_factory=dict)   # method -> its wavenumber grid

    def wn_of(self, method):
        return self.wn_by.get(method, self.wn)

    @property
    def lam(self) -> float:
        return (SIGMA_Y_PROXY / self.sigma_x) ** 2


def _ftir_tables():
    from phase3_common import PATHS

    return PATHS.ftir_dir / "local_db/tables"


def load_tor_fractions(path=None) -> pd.DataFrame:
    """One row per (Site, date): TOR EC/OC/EC1/OPTR/OPTT in µg/m³ plus volume."""
    path = Path(path) if path else _ftir_tables() / "results_tor.csv"
    tor = pd.read_csv(
        path,
        usecols=["Site", "SampleDate", "Parameter", "Value", "AverageFlowRate", "ElapsedTime"],
    )
    tor = tor.loc[tor.Parameter.isin(TOR_PARAMETERS)].copy()
    tor["date"] = pd.to_datetime(tor.SampleDate, format="mixed", errors="coerce").dt.normalize()
    tor = tor.drop_duplicates(["Site", "date", "Parameter"])
    wide = tor.pivot(index=["Site", "date"], columns="Parameter", values="Value") / 1000.0
    wide.columns = [f"tor_{c.lower()}_ugm3" for c in wide.columns]
    vol = (
        tor.drop_duplicates(["Site", "date"])
        .assign(tor_volume_m3=lambda d: d.AverageFlowRate / 1000 * d.ElapsedTime)
        .set_index(["Site", "date"])["tor_volume_m3"]
    )
    return wide.join(vol).reset_index()


def load_improve_hips(path=None) -> pd.DataFrame:
    """HIPS fAbs (Mm⁻¹) keyed by the PTFE filter id the spectra carry."""
    path = Path(path) if path else _ftir_tables() / "results_hips.csv"
    hips = pd.read_csv(path, usecols=["MatchedFilterId", "Parameter", "Value"])
    hips = hips.loc[hips.Parameter.str.casefold().eq("fabs")].drop_duplicates("MatchedFilterId")
    return hips.rename(columns={"MatchedFilterId": "filter_id", "Value": "fabs_Mm1"})[
        ["filter_id", "fabs_Mm1"]
    ]


def _fullrange_arrays(cases):
    """Full-range VIBES arrays aligned to the earlier run's rows, or None."""
    info = FULLRANGE / "ASSEMBLY.json"
    if not info.exists():
        return None
    meta = json.loads(info.read_text())
    if meta["n_corrected"] != meta["n_cases"]:
        return None
    fcases = pd.read_csv(FULLRANGE / "cases.csv")
    if not (fcases.sample_id.to_numpy() == cases.sample_id.to_numpy()).all():
        raise ValueError("full-range cases are not aligned with the earlier run")
    return np.load(FULLRANGE / "corrected_VIBES_full.npy"), np.load(FULLRANGE / "wn.npy")


def load_context(tor_path=None, hips_path=None, sigma_x=None, fullrange=True) -> Context:
    cases = pd.read_csv(RUN / "cases.csv")
    wn = np.load(RUN / "wn.npy")
    arrays = {m: np.load(RUN / f"corrected_{m}.npy") for m in METHODS}
    wn_by = {m: wn for m in METHODS}
    full = _fullrange_arrays(cases) if fullrange else None
    if full is not None:
        fa, fwn = full
        window = np.array([int(np.argmin(np.abs(fwn - w))) for w in wn])
        if not np.allclose(fwn[window], wn):
            raise ValueError("AIRSpec grid is not a subset of the full-range grid")
        arrays["VIBES-full"], wn_by["VIBES-full"] = fa, fwn
        arrays["VIBES-full-cut"], wn_by["VIBES-full-cut"] = fa[:, window], wn

    lib = cases.loc[cases.kind.eq("calibration")].copy()
    lib["row"] = lib.index.to_numpy()
    lib["filter_id"] = lib.filter_id.astype(int)
    lib["date"] = pd.to_datetime(lib.date).dt.normalize()
    lib["lot"] = lib.lot.astype(int)
    lib = lib.merge(load_tor_fractions(tor_path), on=["Site", "date"], how="left",
                    validate="many_to_one")
    lib = lib.merge(load_improve_hips(hips_path), on="filter_id", how="left",
                    validate="many_to_one")
    lib["tor_oc_ec"] = lib.tor_oc_ugm3 / lib.tor_ec_ugm3
    lib["ftir_y_ugm3"] = lib.y / lib.tor_volume_m3        # TOR EC as the run fitted it
    lib["mac_ratio"] = lib.fabs_Mm1 / lib.tor_ec_ugm3
    lib["op_tor_frac"] = lib.tor_optr_ugm3 / (lib.tor_ec_ugm3 + lib.tor_optr_ugm3)
    lib = lib.reset_index(drop=True)
    # y is µg/filter; y/volume must reproduce TOR EC µg/m³ if the ng/m³ reading is right
    agree = np.nanmedian(np.abs(lib.ftir_y_ugm3 / lib.tor_ec_ugm3 - 1))
    if not agree < 1e-3:
        raise ValueError(f"TOR unit check failed: median |y/vol / EC - 1| = {agree:.3g}")

    ev = pd.read_csv(ADDIS_EVALUATION)
    targets = cases.loc[cases.kind.eq("target")].copy()
    targets["MediaId"] = targets.sample_id.str.split(":").str[1].astype(int)
    targets["row"] = targets.index.to_numpy()
    addis = ev.merge(targets[["MediaId", "row"]], on="MediaId", how="inner", validate="one_to_one")
    addis["date"] = pd.to_datetime(addis.date)
    missing = sorted(set(ev.MediaId) - set(addis.MediaId))

    if sigma_x is None:
        from data_matching import load_filter_data

        f = load_filter_data()
        u = f.loc[f.Site.eq("ETAD") & f.Parameter.eq("HIPS_Uncertainty"), "Uncertainty"]
        sigma_x = float(u.median() / MAC_VALUE)

    ctx = Context(
        wn=wn,
        lib=lib,
        X={m: arrays[m][lib.row.to_numpy()].astype(float) for m in arrays},
        addis=addis.reset_index(drop=True),
        XA={m: arrays[m][addis.row.to_numpy()].astype(float) for m in arrays},
        sigma_x=sigma_x,
        methods=tuple(arrays),
        wn_by=wn_by,
    )
    ctx.notes.append(
        f"Addis: {len(addis)} of {len(ev)} evaluation filters are in the frozen run "
        f"(absent MediaIds: {missing})"
    )
    return ctx


# --------------------------------------------------------------------------- selection
def correlation_analogs(ctx, method, target_mask, mask=LOCKED_MASK, n=N_ANALOGS,
                        aggregate="median"):
    """Top-n IMPROVE filters by Pearson r to the target group's median spectrum.

    Same recipe as the Sept-10 seasonal selections (one scan per physical filter
    is already guaranteed by the frozen run). Returns lib positions and scores.
    """
    X, XA = ctx.X[method], ctx.XA[method][np.asarray(target_mask, bool)]
    target = np.median(XA, axis=0)[None, :] if aggregate == "median" else XA
    scores = mean_correlation_scores(X, target, spectral_region_mask(ctx.wn_of(method), *mask))
    order = np.lexsort((ctx.lib.filter_id.to_numpy(), -scores))
    return order[:n], scores[order[:n]]


def group_masks(ctx):
    g = {"All Addis": np.ones(len(ctx.addis), bool)}
    g.update({s: ctx.addis.season.eq(s).to_numpy() for s in SEASONS})
    return g


# --------------------------------------------------------------------------- fitting
@dataclass
class Fit:
    k: int
    train: np.ndarray
    model: PLSRegression
    curve: pd.DataFrame

    def predict(self, X):
        return self.model.predict(np.asarray(X, float)).ravel()


def fit_cohort(X, y, sites, mode="site_heldout", train=None, k=None,
               max_components=MAX_COMPONENTS):
    """Fit one cohort under a protocol; ``train`` overrides the protocol split."""
    X, y, sites = np.asarray(X, float), np.asarray(y, float), np.asarray(sites)
    if train is None:
        train = protocol_train_mask(mode, X, y, sites)
    curve = protocol_cv_curve(mode, X, y, sites, train,
                              max_components=min(max_components, int(train.sum()) - 2))
    if k is None:
        k = protocol_select_k(mode, curve)
    model = PLSRegression(n_components=int(k), scale=False).fit(X[train], y[train])
    return Fit(k=int(k), train=train, model=model, curve=curve)


def repeated_site_splits(sites, n_repeats, test_size=0.2, seed=20260917):
    """Independent site-disjoint train masks (the protocol split is one of these)."""
    sites = np.asarray(sites)
    splitter = GroupShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
    for train_pos, _ in splitter.split(sites, groups=sites):
        train = np.zeros(len(sites), bool)
        train[train_pos] = True
        yield train


# --------------------------------------------------------------------------- readouts
def xy_stats(x, y, lam=1.0, sigma_x=None, sigma_y=None, groups=None, boot=False, seed=20260917):
    """OLS + Deming for a comparison panel; y is the measurement being judged."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    if sigma_x is not None and sigma_y is not None:
        st = calculate_regression_stats(x[ok], y[ok], errors_in_variables=True,
                                        sigma_x=sigma_x, sigma_y=sigma_y)
    else:
        st = calculate_regression_stats(x[ok], y[ok], errors_in_variables=True)
    out = {
        "n": int(st["n"]),
        "R2": float(st["R2"]),
        "ols_slope": float(st["slope"]),
        "ols_intercept": float(st["intercept"]),
        "deming_slope": float(st["deming_slope"]),
        "deming_intercept": float(st["deming_intercept"]),
        "RMSE": float(st["RMSE"]),
        "bias": float(st["bias"]),
        "x_median": float(np.median(x[ok])),
        "y_median": float(np.median(y[ok])),
    }
    if boot:
        g = None if groups is None else np.asarray(groups)[ok]
        try:
            ci = deming_bootstrap(x[ok], y[ok], lam, groups=g, seed=seed, n_boot=1000)
            out.update({k: float(v) for k, v in ci.items() if k.endswith(("_low", "_high"))})
        except ValueError:          # too few independent groups for a block bootstrap
            pass
    return out


def mac_percentile(ctx, fabs, ec, reference=None):
    """Where each (fAbs, EC) pair's MAC ratio sits in the IMPROVE pool's distribution."""
    pool = ctx.lib if reference is None else reference
    ref = (pool.fabs_Mm1 / pool.tor_ec_ugm3).to_numpy(float)
    ref = np.sort(ref[np.isfinite(ref) & (pool.tor_ec_ugm3.to_numpy(float) > 0.05)])
    ratio = np.asarray(fabs, float) / np.asarray(ec, float)
    return 100.0 * np.searchsorted(ref, ratio) / len(ref)


# --------------------------------------------------------------------------- category rankings
def _analysis_to_filter():
    p = np.load(ROOT / "research/ftir_ec_phase3/output/corrected/improve_pool_corrected_df6.npz",
                allow_pickle=True)
    return dict(zip(p["analysis_id"].astype(int), p["filter_id"].astype(int)))


def _positions(ctx, filter_ids):
    f2pos = pd.Series(np.arange(len(ctx.lib)), index=ctx.lib.filter_id)
    kept = [f for f in dict.fromkeys(int(v) for v in filter_ids) if f in f2pos.index]
    return f2pos.loc[kept].to_numpy(int)


def vip_analog_ranking(ctx):
    """Earlier VIP/Euclidean analogs, AIRSpec-corrected space (explorer cache ranking)."""
    z = np.load(ROOT / "calibration_explorer/cache/analog_corrected_ranking.npz")
    a2f = _analysis_to_filter()
    return _positions(ctx, [a2f[int(a)] for a in z["ids"] if int(a) in a2f])


def smoke_positions(ctx):
    smoke = pd.read_csv(AREA / "output/tables/pls_calibration_phase2/smoke_cohort_spectral_selection.csv")
    return _positions(ctx, smoke.FilterId)


def eth_shaped_ranking(ctx, method):
    """Ethiopia-shaped smoke: band-feature distance to Addis, in this method's space."""
    from pls_transfer import band_feature_distance

    pos = smoke_positions(ctx)
    d = band_feature_distance(ctx.X[method][pos], ctx.wn_of(method), ctx.XA[method])
    return pos[np.argsort(d, kind="stable")]


def ocec_ranking(ctx):
    lib = ctx.lib
    ok = (lib.tor_ec_ugm3 > 0) & (lib.tor_oc_ugm3 > 0) & lib.tor_oc_ec.notna()
    frame = lib.loc[ok, ["tor_oc_ec", "filter_id"]]
    return frame.sort_values(["tor_oc_ec", "filter_id"]).index.to_numpy(int)


def correlation_ranking(ctx, method, target_mask, mask=LOCKED_MASK):
    pos, _ = correlation_analogs(ctx, method, target_mask, mask, n=len(ctx.lib))
    return pos
