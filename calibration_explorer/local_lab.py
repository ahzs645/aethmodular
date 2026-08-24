"""Local (per-filter) calibration: the LOCAL/LWR lineage, testable per site.

Instead of selecting ONE cohort and fitting ONE global PLS, fit a small PLS
per target filter on its k most-similar library spectra (Shenk & Westerhaus
1997 LOCAL; Naes & Isaksson 1990 LWR; Reggente/Dillner/Takahama applied the
kNN idea to this exact FTIR problem). Two properties make this the right
"go further" for the spectral-analog thread:

- it is unsupervised on the target (similarity only: no target y anywhere);
- it attacks the failure mode the five-site grid re-audit exposed: globally
  selected analog cohorts win screening while 96% extrapolated. A local model
  is fitted inside the target filter's own neighborhood, and the per-filter
  neighbor-similarity statistic is an applicability-domain diagnostic that
  travels with every prediction.

Similarity is Pearson correlation in a chosen spectral space (deriv2 is the
LOCAL-literature default: the derivative kills the PTFE baseline; airspec
and raw are available for comparison). Optional tricube weighting is applied
via sqrt-weight row scaling of the centered neighborhood (the standard
weighted-PLS approximation; sklearn PLS has no sample_weight).

Registered from app.py via local_lab.register(app, ctx). Standalone science
run against a live server is NOT needed: this module computes directly from
the app's in-memory arrays.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

_CTX: dict = {}
_LIB: dict = {}          # space -> dict(ids, X (row-standardized), Xc (raw), y)


def _library(space: str) -> dict:
    """TOR-eligible library in the chosen space, built once per space."""
    if space in _LIB:
        return _LIB[space]
    D = _CTX["D"]
    ids = D["pool"].index.to_numpy()
    if space == "airspec":
        rows = [D["corr_row"].get(int(i)) for i in ids]
        keep = np.array([r is not None for r in rows])
        ids = ids[keep]
        X = D["corr_pool"][[D["corr_row"][int(i)] for i in ids]].astype(np.float32)
    elif space == "deriv2":
        X = savgol_filter(
            D["pool_raw"].loc[ids, D["wcols"]].to_numpy(np.float32),
            axis=1, **_CTX["SAVGOL"]).astype(np.float32)
    else:
        X = D["pool_raw"].loc[ids, D["wcols"]].to_numpy(np.float32)
    y = D["pool"].loc[ids, "TOR_EC_loading_ug"].to_numpy(float)
    sites = D["pool"].loc[ids, "Site"].to_numpy()
    # row-standardize once for Pearson-by-matmul
    Xs = X - X.mean(axis=1, keepdims=True)
    Xs /= (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
    _LIB[space] = {"ids": ids, "X": X, "Xs": Xs, "y": y, "sites": sites}
    return _LIB[space]


def _target_matrix(target_name: str, space: str) -> np.ndarray:
    t = _CTX["get_target"](target_name)
    return np.asarray(_CTX["_target_X"](t, space), float)


def local_predict(target_name: str, space="deriv2", k=200, ncomp=8,
                  weighting="uniform") -> dict:
    """Per-filter LOCAL predictions + the per-filter applicability diagnostic.

    Returns predictions in ug/m3 (loading / target volume, matching
    run_config's convention), per-filter mean neighbor Pearson r, the
    neighborhood's dominant library sites, and the crossplot metrics block.
    """
    lib = _library(space)
    t = _CTX["get_target"](target_name)
    Xe = _target_matrix(target_name, space)
    Es = Xe - Xe.mean(axis=1, keepdims=True)
    Es /= (np.linalg.norm(Es, axis=1, keepdims=True) + 1e-12)
    R = Es @ lib["Xs"].T                      # (n_target, n_lib) Pearson r

    preds = np.empty(len(Xe))
    sim_mean = np.empty(len(Xe))
    sim_min = np.empty(len(Xe))
    top_sites = []
    for i in range(len(Xe)):
        nn = np.argpartition(-R[i], k)[:k]
        r_nn = R[i][nn]
        Xk = lib["X"][nn].astype(float)
        yk = lib["y"][nn]
        if weighting == "tricube":
            d = 1.0 - r_nn
            w = (1.0 - (d / (d.max() + 1e-12)) ** 3) ** 3
            sw = np.sqrt(np.clip(w, 1e-6, None))
            xm = (Xk * w[:, None]).sum(0) / w.sum()
            ym = float((yk * w).sum() / w.sum())
            model = PLSRegression(n_components=ncomp, scale=False)
            model.fit((Xk - xm) * sw[:, None], (yk - ym) * sw)
            preds[i] = float(model.predict((Xe[i] - xm)[None, :])[0]) + ym
        else:
            model = PLSRegression(n_components=ncomp, scale=False)
            model.fit(Xk, yk)
            preds[i] = float(model.predict(Xe[i][None, :])[0])
        sim_mean[i] = float(r_nn.mean())
        sim_min[i] = float(r_nn.min())
        s, c = np.unique(lib["sites"][nn], return_counts=True)
        top_sites.append(s[np.argmax(c)])
    pred_ugm3 = preds / np.asarray(t["volume"], float)
    metrics = _CTX["crossplot_metrics"](t, pred_ugm3)
    return {"target": target_name, "space": space, "k": int(k),
            "ncomp": int(ncomp), "weighting": weighting,
            "n": int(len(Xe)), "n_library": int(len(lib["ids"])),
            "metrics": metrics,
            "pred": [round(float(v), 4) for v in pred_ugm3],
            "sim_mean": [round(float(v), 4) for v in sim_mean],
            "sim_min": [round(float(v), 4) for v in sim_min],
            "sim_mean_median": round(float(np.median(sim_mean)), 4),
            "dominant_site": max(set(top_sites), key=top_sites.count)}


def _cached_local(body: dict) -> dict:
    cache_dir: Path = _CTX["CACHE_DIR"]
    key = _CTX["_cache_key"]("local", body.get("target", "addis"),
                             body.get("space", "deriv2"),
                             body.get("k", 200), body.get("ncomp", 8),
                             body.get("weighting", "uniform"))
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = local_predict(body.get("target", "addis"),
                        space=body.get("space", "deriv2"),
                        k=int(body.get("k", 200)),
                        ncomp=int(body.get("ncomp", 8)),
                        weighting=body.get("weighting", "uniform"))
    path.write_text(json.dumps(out))
    return out


def register(app, ctx: dict) -> None:
    """ctx: STATE, COMPUTE_LOCK, D, get_target, _target_X, crossplot_metrics,
    _cache_key, CACHE_DIR, SAVGOL, list_targets."""
    from flask import jsonify, request

    _CTX.update(ctx)

    @app.route("/api/local_run", methods=["POST"])
    def api_local_run():
        if not ctx["STATE"]["ready"]:
            return jsonify({"error": "data still loading"}), 503
        b = request.get_json(force=True) or {}
        try:
            with ctx["COMPUTE_LOCK"]:
                return jsonify(_cached_local(b))
        except Exception as exc:                       # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/api/local_cross", methods=["POST"])
    def api_local_cross():
        """LOCAL evaluated on every SPARTAN target: the analog counterpart
        of /api/cross_site."""
        if not ctx["STATE"]["ready"]:
            return jsonify({"error": "data still loading"}), 503
        b = request.get_json(force=True) or {}
        rows = []
        for name in ctx["list_targets"]():
            try:
                tgt = ctx["get_target"](name)
                if tgt.get("ref_kind") != "fabs":
                    continue
                with ctx["COMPUTE_LOCK"]:
                    out = _cached_local({**b, "target": name})
                rows.append({k: out[k] for k in
                             ("target", "n", "metrics", "sim_mean_median",
                              "dominant_site")})
            except Exception as exc:                   # noqa: BLE001
                rows.append({"target": name,
                             "error": f"{type(exc).__name__}: {exc}"})
        return jsonify({"rows": rows, "space": b.get("space", "deriv2"),
                        "k": int(b.get("k", 200)),
                        "ncomp": int(b.get("ncomp", 8)),
                        "weighting": b.get("weighting", "uniform")})
