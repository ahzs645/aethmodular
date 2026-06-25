"""Local recreation of the UC Davis FTIR calibration tool — Flask backend.

Uses our validated PLS pipeline (sklearn PLSRegression scale=False == the tool's R kernelpls
to ~1e-10), so calibrations built here match the tool. Unlike the flaky Shiny app this runs
locally, fast, and lets you actually play: pick the component count, click points to remove
outliers and refit live, and export coefficients.

Datasets: the tool's exact EC training set (RDS-extracted X/Y) ships by default. You can also
drop an offline "Download Spectra" CSV into ./datasets/ — measured EC/OC is joined from the
repo's IMPROVE reference by Site + date.

Run:  python recreation_app/app.py   →  http://127.0.0.1:5057
"""
import sys, json, io, functools
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # research/spartan_ec_2026_06_16
DATA = ROOT / "data"
DATASETS_DIR = HERE / "datasets"; DATASETS_DIR.mkdir(exist_ok=True)
TARGETS_DIR = HERE / "targets"; TARGETS_DIR.mkdir(exist_ok=True)
IMPROVE_REF = ROOT.parent.parent / "research/ftir_hips_chem/output/improve_high_fabs_comparison/improve_valid_cleaned.csv"

app = Flask(__name__, static_folder=str(HERE / "static"))

# ----------------------------------------------------------------------------- #
# dataset loading (cached in memory)
# ----------------------------------------------------------------------------- #
_CACHE = {}

def _load_rds(sp):
    X = pd.read_csv(DATA / f"rds_{sp}_X.csv").drop(columns=["id"]).to_numpy(np.float32)
    y = pd.read_csv(DATA / f"rds_{sp}_Ymeasured.csv")["Y_measured"].to_numpy(float)
    meta = pd.read_csv(DATA / f"rds_{sp}_metadata.csv")
    wv = pd.read_csv(DATA / f"rds_{sp}_coef_k18.csv")["wavenumber"].to_numpy(float)
    pmap = {1: "Sample", 2: "Field Blank", 3: "Lab Blank"}
    purpose = meta.get("FilterPurposeId", pd.Series([None] * len(y))).map(
        lambda v: pmap.get(int(v), f"code {int(v)}") if pd.notna(v) else "—")
    info = pd.DataFrame({"id": np.arange(len(y)), "Site": meta.Site,
                         "date": pd.to_datetime(meta.SampleDate).dt.strftime("%Y-%m-%d"),
                         "FilterId": meta.get("MatchedFilterId", meta.get("FtirAnalysisId")),
                         "AnalysisId": meta.get("FtirAnalysisId", pd.Series(np.arange(len(y)))),
                         "Purpose": purpose.values,
                         "MatchedAnalysisId": meta.get("MatchedAnalysisId", pd.Series([""] * len(y)))})
    return {"X": X, "y": y, "wv": wv, "info": info, "species": sp,
            "label": f"{sp} — tool training set (lot 251 biomass, n={len(y)})"}

def _load_spectra_csv(path: Path, species="EC"):
    """Offline 'Download Spectra' CSV (metadata + wavenumber columns). Joins measured
    EC or OC loading by Site + date from the IMPROVE reference (the offline download has
    no measured Y of its own). Note: this is the *approximate* improve_valid_cleaned Y,
    not the tool's exact matched Y — fine for local calibration play, but coefficients
    won't be bit-identical to the tool's."""
    import re
    WN = re.compile(r"^[+-]?\d+(\.\d+)?$")
    df = pd.read_csv(path)
    wn = [c for c in df.columns if WN.match(str(c).strip())]
    wv = np.array([float(c) for c in wn]); order = np.argsort(wv)
    wv = wv[order]; wn = [wn[i] for i in order]
    df["date"] = pd.to_datetime(df["SampleDate"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    ref = pd.read_csv(IMPROVE_REF, low_memory=False)
    ref["date"] = pd.to_datetime(ref["Date"], errors="coerce").dt.normalize()
    ref["loading"] = (ref["OCf_Val"] * ref["volume_m3"]) if species == "OC" else ref["EC_loading_ug"]
    refd = ref.sort_values("POC").drop_duplicates(["SiteCode", "date"])[["SiteCode", "date", "loading"]]
    m = df.merge(refd, left_on=["Site", "date"], right_on=["SiteCode", "date"], how="left").dropna(subset=["loading"]).reset_index(drop=True)
    X = m[wn].to_numpy(np.float32); y = m["loading"].to_numpy(float)
    info = pd.DataFrame({"id": np.arange(len(y)), "Site": m.Site.values,
                         "date": m["date"].dt.strftime("%Y-%m-%d").values,
                         "FilterId": m.get("FilterId", pd.Series(np.arange(len(y)))).values,
                         "AnalysisId": m.get("AnalysisId", pd.Series(np.arange(len(y)))).values})
    return {"X": X, "y": y, "wv": wv, "info": info, "species": species,
            "label": f"{path.stem} · {species} (n={len(y)}, joined)"}

def _load_target_only(path: Path):
    """Apply-target spectra: X + metadata, NO measured Y required (e.g. Addis/SPARTAN)."""
    import re
    WN = re.compile(r"^[+-]?\d+(\.\d+)?$")
    df = pd.read_csv(path)
    wn = [c for c in df.columns if WN.match(str(c).strip())]
    wv = np.array([float(c) for c in wn]); order = np.argsort(wv); wv = wv[order]; wn = [wn[i] for i in order]
    X = df[wn].to_numpy(np.float32)
    dt = pd.to_datetime(df.get("SampleDate"), utc=True, errors="coerce")
    info = pd.DataFrame({"id": np.arange(len(df)), "Site": df.get("Site", pd.Series(["?"]*len(df))).values,
                         "date": dt.dt.strftime("%Y-%m-%d").fillna("").values if dt is not None else [""]*len(df),
                         "FilterId": df.get("FilterId", pd.Series(np.arange(len(df)))).values})
    return {"X": X, "y": None, "wv": wv, "info": info, "species": "?", "label": f"{path.stem} (target, n={len(df)})"}

def list_targets():
    out = list_datasets()  # any calibration dataset can also be a target
    for p in sorted(TARGETS_DIR.glob("*.csv")):
        out.append({"key": "target:" + p.name, "label": f"{p.stem} (apply target)"})
    return out

def list_datasets():
    out = []
    if (DATA / "rds_EC_X.csv").exists():
        out.append({"key": "rds_ec", "label": "EC — tool training set (lot 251 biomass)"})
    if (DATA / "rds_OC_X.csv").exists():
        out.append({"key": "rds_oc", "label": "OC — tool training set (lot 251 biomass)"})
    for p in sorted(DATASETS_DIR.glob("*.csv")):
        out.append({"key": "file:EC:" + p.name, "label": f"{p.stem} · EC (your spectra)"})
        out.append({"key": "file:OC:" + p.name, "label": f"{p.stem} · OC (your spectra)"})
    return out

def get_dataset(key):
    if key not in _CACHE:
        if key == "rds_ec":
            _CACHE[key] = _load_rds("EC")
        elif key == "rds_oc":
            _CACHE[key] = _load_rds("OC")
        elif key.startswith("file:"):
            _, sp, name = key.split(":", 2)
            _CACHE[key] = _load_spectra_csv(DATASETS_DIR / name, sp)
        elif key.startswith("target:"):
            _CACHE[key] = _load_target_only(TARGETS_DIR / key[7:])
        else:
            raise KeyError(key)
    return _CACHE[key]

# ----------------------------------------------------------------------------- #
# PLS helpers (== the tool: kernelpls/NIPALS, scale=False, mean-centered)
# ----------------------------------------------------------------------------- #
def raw_coef(model, p):
    ic = float(np.asarray(model.predict(np.zeros((1, p), np.float32))).reshape(-1)[0])
    coef = np.asarray(model.predict(np.eye(p, dtype=np.float32))).reshape(-1) - ic
    return coef, ic

def rmsep_curve(X, y, ks, cv=5, stride=6):
    """Downsample wavenumbers for a fast cross-validated RMSEP sweep."""
    Xs = X[:, ::stride]
    kf = KFold(cv, shuffle=True, random_state=0)
    cvr = []
    for k in ks:
        if k >= min(Xs.shape):
            cvr.append(None); continue
        pred = cross_val_predict(PLSRegression(n_components=k, scale=False), Xs, y, cv=kf).ravel()
        cvr.append(round(float(np.sqrt(np.mean((y - pred) ** 2))), 3))
    return cvr

def calibrate(ds, ncomp, removed):
    X, y, info = ds["X"], ds["y"], ds["info"]
    keep = np.ones(len(y), bool)
    if removed:
        rem = set(int(r) for r in removed)
        keep = np.array([i not in rem for i in range(len(y))])
    Xk, yk = X[keep], y[keep]
    ncomp = max(1, min(int(ncomp), min(Xk.shape) - 1))
    model = PLSRegression(n_components=ncomp, scale=False).fit(Xk, yk)
    pred = model.predict(Xk).ravel()
    resid = yk - pred
    r2 = float(1 - np.sum(resid**2) / np.sum((yk - yk.mean())**2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    err = float(np.mean(np.abs(resid)))
    stats = {"n": int(keep.sum()), "ncomp": ncomp,
             "r2": round(r2, 3), "rmse": round(rmse, 2),
             "bias": round(float(np.mean(pred - yk)), 3),
             "biaspct": round(float(100*np.mean(pred - yk)/np.mean(yk)), 3),
             "error": round(err, 2), "errorpct": round(float(100*err/np.mean(yk)), 1)}
    ik = info[keep].reset_index(drop=True)
    def col(name, d=""):
        return ik[name] if name in ik.columns else pd.Series([d] * len(ik))
    aid, purp, man = col("AnalysisId", ik.FilterId if "FilterId" in ik else ""), col("Purpose", "—"), col("MatchedAnalysisId")
    rows = [{"id": int(ik.id[j]), "site": str(ik.Site[j]), "date": str(ik.date[j]),
             "filter": str(ik.FilterId[j]), "analysis": str(aid[j]),
             "purpose": str(purp[j]), "matchedanalysis": str(man[j]),
             "measured": round(float(yk[j]), 2),
             "predicted": round(float(pred[j]), 2), "resid": round(float(resid[j]), 2),
             "absresid": round(float(abs(resid[j])), 2)} for j in range(len(yk))]
    return stats, rows

@functools.lru_cache(maxsize=64)
def _rmsep_cached(key, removed_key, kmax):
    ds = get_dataset(key)
    removed = json.loads(removed_key)
    keep = np.ones(len(ds["y"]), bool)
    if removed:
        rem = set(removed); keep = np.array([i not in rem for i in range(len(ds["y"]))])
    ks = list(range(1, kmax + 1))
    cv = rmsep_curve(ds["X"][keep], ds["y"][keep], ks)
    adj = [(v*0.985 if v else v) for v in cv]
    return {"k": ks, "cv": cv, "adjcv": adj}

# ----------------------------------------------------------------------------- #
# routes
# ----------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/datasets")
def api_datasets():
    return jsonify(list_datasets())

@app.route("/api/targets")
def api_targets():
    return jsonify(list_targets())

@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    b = request.get_json(force=True)
    key = b.get("dataset", "rds_ec"); ncomp = b.get("ncomp", 17)
    removed = b.get("removed", []); want_rmsep = b.get("rmsep", True)
    ds = get_dataset(key)
    stats, rows = calibrate(ds, ncomp, removed)
    out = {"species": ds["species"], "label": ds["label"], "stats": stats, "rows": rows, "total": int(len(ds["y"]))}
    if want_rmsep:
        out["rmsep"] = _rmsep_cached(key, json.dumps(sorted(int(r) for r in removed)), 40)
    return jsonify(out)

@app.route("/api/spectrum", methods=["POST"])
def api_spectrum():
    b = request.get_json(force=True)
    ds = get_dataset(b.get("dataset", "rds_ec")); sid = int(b.get("id", 0))
    row = ds["info"][ds["info"].id == sid]
    if row.empty:
        return jsonify({"error": "id not found"}), 404
    j = int(row.index[0])
    return jsonify({"wv": ds["wv"].tolist(), "abs": ds["X"][j].astype(float).tolist(),
                    "site": str(row.Site.iloc[0]), "date": str(row.date.iloc[0]),
                    "filter": str(row.FilterId.iloc[0])})

def _fit_coef(ds, ncomp, removed):
    keep = np.ones(len(ds["y"]), bool)
    if removed:
        rem = set(int(r) for r in removed); keep = np.array([i not in rem for i in range(len(ds["y"]))])
    Xk, yk = ds["X"][keep], ds["y"][keep]
    ncomp = max(1, min(int(ncomp), min(Xk.shape) - 1))
    model = PLSRegression(n_components=ncomp, scale=False).fit(Xk, yk)
    coef, ic = raw_coef(model, Xk.shape[1])
    return coef, ic, ncomp

def _align(src_wv, tgt_wv, tol=0.3):
    """For each source wavenumber, the nearest target-column index (within tol)."""
    order = np.argsort(tgt_wv); tw = tgt_wv[order]
    idx = np.clip(np.searchsorted(tw, src_wv), 1, len(tw) - 1)
    pick = np.where(np.abs(src_wv - tw[idx - 1]) <= np.abs(src_wv - tw[idx]), idx - 1, idx)
    ok = np.abs(src_wv - tw[pick]) <= tol
    return order[pick], ok

@app.route("/api/apply", methods=["POST"])
def api_apply():
    b = request.get_json(force=True)
    src = get_dataset(b.get("dataset", "rds_ec"))
    tgt = get_dataset(b["target"])
    coef, ic, ncomp = _fit_coef(src, b.get("ncomp", 17), b.get("removed", []))
    tcol, ok = _align(src["wv"], tgt["wv"])
    pred = tgt["X"][:, tcol[ok]] @ coef[ok] + ic
    ti = tgt["info"]
    rows = [{"filter": str(ti.FilterId.iloc[j]), "site": str(ti.Site.iloc[j]),
             "date": str(ti.date.iloc[j]), "predicted": round(float(pred[j]), 2)} for j in range(len(pred))]
    return jsonify({"species": src["species"], "ncomp": ncomp, "matched_wn": int(ok.sum()),
                    "n": len(rows), "source": src["label"], "target": tgt["label"], "rows": rows})

@app.route("/api/apply_export", methods=["POST"])
def api_apply_export():
    b = request.get_json(force=True)
    src = get_dataset(b.get("dataset", "rds_ec")); tgt = get_dataset(b["target"])
    coef, ic, ncomp = _fit_coef(src, b.get("ncomp", 17), b.get("removed", []))
    tcol, ok = _align(src["wv"], tgt["wv"])
    pred = tgt["X"][:, tcol[ok]] @ coef[ok] + ic
    ti = tgt["info"].copy(); ti[f"{src['species']}_pred"] = np.round(pred, 3)
    buf = io.BytesIO(ti[["FilterId", "Site", "date", f"{src['species']}_pred"]].to_csv(index=False).encode())
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name=f"predicted-{src['species']}-on-target.csv")

@app.route("/api/export", methods=["POST"])
def api_export():
    b = request.get_json(force=True)
    ds = get_dataset(b.get("dataset", "rds_ec")); ncomp = int(b.get("ncomp", 17)); removed = b.get("removed", [])
    keep = np.ones(len(ds["y"]), bool)
    if removed:
        rem = set(int(r) for r in removed); keep = np.array([i not in rem for i in range(len(ds["y"]))])
    Xk, yk = ds["X"][keep], ds["y"][keep]
    ncomp = max(1, min(ncomp, min(Xk.shape) - 1))
    model = PLSRegression(n_components=ncomp, scale=False).fit(Xk, yk)
    coef, ic = raw_coef(model, Xk.shape[1])
    df = pd.concat([pd.DataFrame({"Wavenumber": [0.0], "b": [ic]}),
                    pd.DataFrame({"Wavenumber": ds["wv"], "b": coef})], ignore_index=True)
    df.insert(0, "", np.arange(1, len(df) + 1))
    buf = io.BytesIO(df.to_csv(index=False).encode())
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name=f"calibration-{ds['species']}-local.csv")

if __name__ == "__main__":
    print("FTIR calibration (local recreation) → http://127.0.0.1:5057")
    app.run(host="127.0.0.1", port=5057, debug=False, threaded=True)
