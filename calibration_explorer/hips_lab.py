"""HIPS lab — blank-line diagnostics and per-filter weighted York/EIV fits.

The HIPS scattering correction is a *lot field-blank regression line*
(tau = ln((Intercept + Slope*R1)/T1), Fabs = 100*tau*DepositArea/Volume —
decoded and verified against the SPARTAN batch export, see
research/ftir_ec_phase3/OFFSET_ADJUDICATION_2026-08-23.md sec 5b). Heavily
loaded filters sit below the R1 range of the blanks that define that line, so
their Fabs rests on an extrapolated calibration. This module quantifies both
problems for the explorer:

- per-target York (heteroscedastic errors-in-variables) fits of prediction vs
  Fabs/MAC using the per-site HIPS uncertainty models, replacing pooled-lambda
  Deming (York et al. 2004, Am. J. Phys. 72:367);
- the same fit under alternative blank lines (lot-common linear, lot
  quadratic), so the sensitivity of each site's intercept to the blank-line
  extrapolation is a number, not an argument;
- the blank ledger per lot (n, refit rms linear vs quadratic, R1 range, and
  the blank-tau zero bound).

Registered from app.py via hips_lab.register(app, ctx) — ctx carries the app
internals so this file never imports app (which may run as __main__).
Standalone science run against a live server: python hips_lab.py
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PKL = REPO / "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl"
SITE_CODE = {"addis": "ETAD", "etbi": "ETBI", "chts": "CHTS",
             "indh": "INDH", "uspa": "USPA"}
MAC = 10.0

_BATCH_PATH: list[Path] = []          # set by register() / _default_batch_path


def _default_batch_path() -> Path:
    import sys
    sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
    from pls_transfer import FTIRTransferPaths
    return FTIRTransferPaths.defaults().spartan_hips_primary


@lru_cache(maxsize=1)
def batch() -> pd.DataFrame:
    """The SPARTAN HIPS batch export: per-filter R1/T1, the deployed blank
    line (Intercept, Slope), lot, geometry, and shipped Fabs."""
    path = _BATCH_PATH[0] if _BATCH_PATH else _default_batch_path()
    b = pd.read_csv(path, encoding="cp1252",
                    usecols=["Site", "FilterId", "FilterType", "LotId", "T1",
                             "R1", "Intercept", "Slope", "tau", "DepositArea",
                             "Volume", "Fabs"])
    b["LotId"] = b["LotId"].astype(str).str.strip()
    return b


@lru_cache(maxsize=1)
def blank_lines() -> dict:
    """Per lot: pooled field+lab blank refits (linear and quadratic), the
    blanks' R1 range, and the blank-tau zero bound."""
    b = batch()
    blanks = b[b["FilterType"].isin(["FB", "LB"])].dropna(
        subset=["R1", "T1"]).query("T1 > 0")
    out = {}
    for lot, g in blanks.groupby("LotId"):
        if len(g) < 5:
            continue
        R, T = g["R1"].to_numpy(float), g["T1"].to_numpy(float)
        lin = np.polyfit(R, T, 1)                     # [a1, a0]
        quad = np.polyfit(R, T, 2)                    # [a2, a1, a0]
        tau0 = g["tau"].dropna()
        out[lot] = {
            "n": int(len(g)),
            "lin": [float(v) for v in lin],
            "quad": [float(v) for v in quad],
            "rms_lin": float(np.std(T - np.polyval(lin, R))),
            "rms_quad": float(np.std(T - np.polyval(quad, R))),
            "r1_min": float(R.min()), "r1_max": float(R.max()),
            "tau0_mean": float(tau0.mean()) if len(tau0) else None,
            "tau0_sd": float(tau0.std()) if len(tau0) else None,
        }
    return out


@lru_cache(maxsize=1)
def sigma_models() -> dict:
    """Per-site sigma_Fabs(Fabs): sigma^2 = a^2 + (b*F)^2 fitted to the
    HIPS_Uncertainty parameter rows. ETBI has no rows -> pooled non-ETAD."""
    with open(PKL, "rb") as f:
        d = pickle.load(f)
    fab = d[d.Parameter == "HIPS_Fabs"][["Site", "FilterId", "Concentration"]]
    unc = d[d.Parameter == "HIPS_Uncertainty"][["Site", "FilterId", "Concentration"]]
    j = fab.merge(unc, on=["Site", "FilterId"], suffixes=("", "_u")).dropna()

    def fit(g):
        F, s = g["Concentration"].to_numpy(float), g["Concentration_u"].to_numpy(float)
        A = np.vstack([np.ones_like(F), F ** 2]).T
        coef, *_ = np.linalg.lstsq(A, s ** 2, rcond=None)
        a2, b2 = max(float(coef[0]), 1e-4), max(float(coef[1]), 0.0)
        return a2, b2

    out = {site: fit(g) for site, g in j.groupby("Site")}
    out["ETBI"] = fit(j[j.Site != "ETAD"])
    return out


def sigma_fabs(site_code: str, fabs: np.ndarray) -> np.ndarray:
    a2, b2 = sigma_models().get(site_code, sigma_models()["ETBI"])
    return np.sqrt(a2 + b2 * np.asarray(fabs, float) ** 2)


def york(x, y, sx, sy, tol=1e-12, itmax=200):
    """York et al. 2004 (uncorrelated errors). Returns b, a, se_b, se_a, mswd."""
    wx, wy = 1.0 / sx ** 2, 1.0 / sy ** 2
    b = np.polyfit(x, y, 1)[0]
    beta = Xb = Yb = None
    for _ in range(itmax):
        W = wx * wy / (wx + b ** 2 * wy)
        Xb, Yb = np.sum(W * x) / np.sum(W), np.sum(W * y) / np.sum(W)
        U, V = x - Xb, y - Yb
        beta = W * (U / wy + b * V / wx)
        b_new = np.sum(W * beta * V) / np.sum(W * beta * U)
        done = abs(b_new - b) < tol
        b = b_new
        if done:
            break
    W = wx * wy / (wx + b ** 2 * wy)
    a = Yb - b * Xb
    xi = Xb + beta
    xbar = np.sum(W * xi) / np.sum(W)
    u = xi - xbar
    se_b = float(np.sqrt(1.0 / np.sum(W * u ** 2)))
    se_a = float(np.sqrt(1.0 / np.sum(W) + xbar ** 2 * se_b ** 2))
    mswd = float(np.sum(W * (y - b * x - a) ** 2) / max(len(x) - 2, 1))
    return float(b), float(a), se_b, se_a, mswd


def york_site(pred, fabs, site_code):
    """Free York fit of prediction vs fabs/MAC with the site's per-filter
    sigma_x; sigma_y inflated until MSWD = 1 (absorbs lack-of-fit — the
    conservative choice). Returns the fit dict + kappa."""
    x = np.asarray(fabs, float) / MAC
    y = np.asarray(pred, float)
    sx = sigma_fabs(site_code, fabs) / MAC
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    sy = np.full_like(x, max(float(np.std(resid)), 1e-3))
    b = a = se_b = se_a = mswd = None
    for _ in range(6):
        b, a, se_b, se_a, mswd = york(x, y, sx, sy)
        sy = sy * np.sqrt(max(mswd, 1e-6))
    kappa = max(0.0, 1.0 - float(np.mean(sx ** 2)) / float(np.var(x)))
    return {"slope": round(b, 3), "slope_se": round(se_b, 3),
            "intercept": round(a, 3), "intercept_se": round(se_a, 3),
            "kappa": round(kappa, 3)}


def tau_variant(rows: pd.DataFrame, kind: str) -> np.ndarray:
    """tau under a blank-line variant: 'deployed' uses the shipped line,
    'lot_lin'/'lot_quad' re-derive it from the pooled lot blanks."""
    lines = blank_lines()
    R, T = rows["R1"].to_numpy(float), rows["T1"].to_numpy(float)
    if kind == "deployed":
        top = rows["Intercept"].to_numpy(float) + rows["Slope"].to_numpy(float) * R
    else:
        key = "lin" if kind == "lot_lin" else "quad"
        top = np.array([np.polyval(lines[lot][key], r) if lot in lines else np.nan
                        for lot, r in zip(rows["LotId"], R)])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.where((top > 0) & (T > 0), top / T, np.nan))


def site_rows(pred, ref_fabs, filter_ids, site_code):
    """One target's full HIPS-lab readout: York fits under each blank-line
    variant, the below-blank-range extrapolation fraction, and match counts."""
    b = batch()
    sub = (b[(b["Site"] == site_code) & (b["FilterType"] == "PM2.5")]
           .drop_duplicates("FilterId").set_index("FilterId"))
    ids = pd.Index([str(f) for f in filter_ids])
    have = ids.isin(sub.index)
    rows = sub.loc[ids[have]]
    pred = np.asarray(pred, float)[have]
    ref = np.asarray(ref_fabs, float)[have]
    dv = 100.0 * rows["DepositArea"].to_numpy(float) / rows["Volume"].to_numpy(float)

    lines = blank_lines()
    r1min = np.array([lines[lot]["r1_min"] if lot in lines else np.nan
                      for lot in rows["LotId"]])
    frac_below = float(np.nanmean(rows["R1"].to_numpy(float) < r1min))

    out = {"n": int(len(ids)), "n_matched": int(have.sum()),
           "frac_below_blank_r1": round(frac_below, 3), "fits": {}}
    for kind in ("deployed", "lot_lin", "lot_quad"):
        fabs_v = tau_variant(rows, kind) * dv
        ok = np.isfinite(fabs_v) & np.isfinite(pred)
        if ok.sum() < 8:
            out["fits"][kind] = {"error": f"only {int(ok.sum())} usable filters"}
            continue
        fit = york_site(pred[ok], fabs_v[ok], site_code)
        fit["median_fabs"] = round(float(np.nanmedian(fabs_v[ok])), 2)
        out["fits"][kind] = fit
    # sanity: shipped Fabs vs deployed-line recompute (should be ~identical)
    ship = rows["Fabs"].to_numpy(float)
    dep = tau_variant(rows, "deployed") * dv
    ok = np.isfinite(ship) & np.isfinite(dep)
    if ok.sum():
        out["recompute_max_diff"] = round(float(np.nanmax(np.abs(ship[ok] - dep[ok]))), 4)
    return out


# ----------------------------------------------------------------------------- #
# Flask wiring
# ----------------------------------------------------------------------------- #
def register(app, ctx: dict) -> None:
    """Attach the HIPS-lab routes. ctx: STATE, COMPUTE_LOCK, run_config,
    list_targets, get_target, _config_from, spartan_hips_path."""
    from flask import jsonify, request

    _BATCH_PATH.clear()
    _BATCH_PATH.append(Path(ctx["spartan_hips_path"]))

    @app.route("/api/hips_blanks")
    def api_hips_blanks():
        try:
            return jsonify({"lots": blank_lines()})
        except Exception as exc:                       # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/api/hips_york", methods=["POST"])
    def api_hips_york():
        if not ctx["STATE"]["ready"]:
            return jsonify({"error": "data still loading"}), 503
        b = request.get_json(force=True) or {}
        cfg = ctx["_config_from"](b)
        cfg.pop("eval_lot", None)
        cfg.pop("target", None)
        rows = []
        for name in ctx["list_targets"]():
            code = SITE_CODE.get(name)
            try:
                tgt = ctx["get_target"](name)
                fids = tgt.get("filter_ids")
                if code is None or not fids:
                    continue          # only SPARTAN targets with an id bridge
                if tgt.get("ref_kind") != "fabs":
                    continue
                with ctx["COMPUTE_LOCK"]:
                    out = ctx["run_config"](k_override=b.get("k"), **cfg,
                                            target=name, eval_lot="all")
                row = site_rows(out["eval"]["pred"], out["eval"]["ref"],
                                fids, code)
                row.update({"site": name, "code": code, "k": out["k"],
                            "label": tgt["label"]})
                rows.append(row)
            except Exception as exc:                   # noqa: BLE001
                rows.append({"site": name, "error": f"{type(exc).__name__}: {exc}"})
        return jsonify({"rows": rows, "mac": MAC})


# ----------------------------------------------------------------------------- #
# standalone science run (against a live server on :5058)
# ----------------------------------------------------------------------------- #
def main():
    import json
    import urllib.request
    targets = {"addis": "ETAD", "etbi": "ETBI", "chts": "CHTS",
               "indh": "INDH", "uspa": "USPA"}
    print("blank ledger:")
    for lot, L in sorted(blank_lines().items()):
        print(f"  lot {lot:5s} n={L['n']:3d} rms lin/quad {L['rms_lin']:5.1f}/"
              f"{L['rms_quad']:5.1f}  R1 {L['r1_min']:.0f}-{L['r1_max']:.0f}  "
              f"tau0 {L['tau0_mean']:+.4f}±{L['tau0_sd']:.4f}")
    print(f"\n{'site':6s} {'match':>5s} {'<blank':>6s} | "
          f"{'deployed':>16s} | {'lot-linear':>16s} | {'lot-quadratic':>16s}")
    for name, code in targets.items():
        body = {"cohort": "ocec", "cutoff": 450, "spectra": "airspec",
                "mode": "site_heldout", "target": name, "k": 9}
        req = urllib.request.Request("http://127.0.0.1:5058/api/run",
                                     json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=900))
        if name == "addis":
            import sys
            sys.path.insert(0, str(REPO / "research/ftir_ec_phase3/scripts"))
            import phase3_common as pc
            ev = next(x for x in pc.load_addis_evaluation()
                      if isinstance(x, pd.DataFrame) and "MediaId" in x.columns)
            fids = ev["ExternalFilterId"].astype(str).tolist()
        else:
            ref = pd.read_csv(REPO / f"calibration_explorer/targets/{name}/reference.csv")
            fids = ref["ExternalFilterId"].astype(str).tolist()
        row = site_rows(d["eval"]["pred"], d["eval"]["ref"], fids, code)
        cells = []
        for kind in ("deployed", "lot_lin", "lot_quad"):
            f = row["fits"][kind]
            cells.append(f"{f['slope']:.2f}x{f['intercept']:+.2f}±{f['intercept_se']:.2f}"
                         if "error" not in f else f["error"])
        print(f"{name:6s} {row['n_matched']:5d} {row['frac_below_blank_r1']*100:5.0f}% | "
              + " | ".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
