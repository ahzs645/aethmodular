"""AIRSpec and VIBES baselines for every native SPARTAN FTIR analysis (19 Oct 2026).

Baseline correction only: no model is fitted or retrained and no EC/OC value changes.

* AIRSpec: the vendored APRLssb spline port (DF1=6, DF2=4) on each analysis's own native
  grid, 1425-4000 cm-1 as by design. Runs locally (about 15 ms a spectrum).
* VIBES: the saved blank background the gallery already uses (87 blanks, rank 70,
  ``vibes_colab_full500_rank85_resume_20261001``) applied with the same ELBO + MAP fit as
  ``scripts/vibes_fullrange_run.py``. The background is defined on its 2,722-channel grid
  (3998.4-500.1 cm-1), so each native spectrum is first resampled onto that grid: the mean of
  the native spectrum's linear interpolant over each model channel's width (area preserving,
  so finer native grids are averaged, not point sampled). Model channels past a native
  grid's end take the end value and are counted per analysis. Runs on a Colab VM.

Stages (from the repo root):

    uv run python research/ftir_hips_chem/workflows/network_baselines_20261019.py prepare
    uv run python research/ftir_hips_chem/workflows/network_baselines_20261019.py check     # vs the frozen Addis inputs
    uv run python research/ftir_hips_chem/workflows/network_baselines_20261019.py airspec
    # on the VM:   python network_baselines_20261019.py vibes --bundle DIR --root DIR --workers N
    uv run python research/ftir_hips_chem/workflows/network_baselines_20261019.py collect --result VM1/checkpoints VM2/checkpoints ...
    uv run python research/ftir_hips_chem/workflows/network_baselines_20261019.py compare   # every Addis filter vs the frozen runs

The network spectra are read from the gallery's own shards (gallery/data/export_spartan_network.py),
so the analyses, ids and raw values are exactly those the drawers show.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
NET = ROOT / "gallery/app/public/data/spartan_network"
OUT = ROOT / "research/ftir_hips_chem/output/tables/network_baselines_20261019"
MODEL = ROOT / ("research/ftir_hips_chem/output/tables/vibes_colab_full500_rank85_resume_20261001/"
                "persistent_results/full500-rank85-9345e818c61929fe/background_model.npz")
TYPE_ORDER = {"PM2.5": 0, "FB": 1, "PM10": 2}   # PM2.5 first, so a partial run covers the charts' filters


def _gz_json(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def native_analyses():
    """Every native FTIR analysis: (descriptor, wavenumbers descending, absorbance)."""
    out = []
    for f in sorted((NET / "native").glob("*.json.gz")):
        d = _gz_json(f)
        for key, s in sorted(d["spectra"].items()):
            buf = gzip.decompress((NET.parent / s["file"]).read_bytes())
            y_all = np.frombuffer(buf, dtype="<f4")
            for a in s["analyses"]:
                y = y_all[a["offset"]:a["offset"] + a["n"]].astype(float)
                wn = np.linspace(a["first"], a["last"], a["n"])
                out.append(({**a, "code": d["code"], "filter_key": key, "file": s["file"]}, wn, y))
    return out


def resample(wn_desc, y, grid_desc):
    """Mean of the linear interpolant of (wn, y) over each grid channel's width.

    Both grids are descending and uniform. Returns (values, n_extended) where n_extended counts
    grid channels whose centre lies outside the native span (they take the end value).
    """
    x, v = wn_desc[::-1], y[::-1]                     # ascending for the integral
    g = grid_desc[::-1]
    h = np.abs(np.diff(g)).mean()
    edges = np.concatenate([[g[0] - h / 2], (g[:-1] + g[1:]) / 2, [g[-1] + h / 2]])
    # extend the native curve flat past its ends so every bin is covered
    lo, hi = min(edges[0], x[0]) - h, max(edges[-1], x[-1]) + h
    xe = np.concatenate([[lo], x, [hi]])
    ve = np.concatenate([[v[0]], v, [v[-1]]])
    cum = np.concatenate([[0.0], np.cumsum(np.diff(xe) * (ve[1:] + ve[:-1]) / 2)])
    # integral of a piecewise-linear function at arbitrary points
    k = np.clip(np.searchsorted(xe, edges) - 1, 0, len(xe) - 2)
    t = edges - xe[k]
    slope = (ve[k + 1] - ve[k]) / (xe[k + 1] - xe[k])
    at = cum[k] + ve[k] * t + slope * t * t / 2
    vals = np.diff(at) / np.diff(edges)
    n_ext = int(((g < x[0]) | (g > x[-1])).sum())
    return vals[::-1], n_ext


def prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(MODEL)
    grid = z["wn"]
    rows, X = [], []
    for a, wn, y in native_analyses():
        if not np.isfinite(y).all():
            raise ValueError(f"non-finite native absorbance: analysis {a['analysis_id']}")
        v, n_ext = resample(wn, y, grid)
        rows.append({"case_row": len(rows), "analysis_id": a["analysis_id"], "media_id": a["media_id"], "code": a["code"],
                     "filter_key": a["filter_key"], "filter_id": a["filter_id"], "filter_type": a["filter_type"],
                     "date": a["date"], "qc": a["qc"], "native_first": a["first"], "native_last": a["last"], "native_n": a["n"],
                     "model_channels_extended": n_ext})
        X.append(v.astype(np.float32))
    cases = pd.DataFrame(rows)
    order = cases.filter_type.map(TYPE_ORDER).fillna(9).to_numpy()
    perm = np.lexsort((cases.case_row.to_numpy(), order))
    cases = cases.iloc[perm].reset_index(drop=True)
    cases["case_row"] = np.arange(len(cases))
    X = np.stack(X)[perm]
    bundle = OUT / "vibes_bundle"
    bundle.mkdir(exist_ok=True)
    np.save(bundle / "X_model_grid.npy", X)
    cases.to_csv(bundle / "cases.csv", index=False)
    np.savez_compressed(bundle / "background_model.npz", **{k: z[k] for k in z.files})
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(bundle.iterdir()) if p.name != "MANIFEST.json"}
    (bundle / "MANIFEST.json").write_text(json.dumps({
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_cases": len(cases),
        "by_type": cases.filter_type.value_counts().to_dict(), "sites": int(cases.code.nunique()),
        "model": str(MODEL.relative_to(ROOT)), "model_rank": int(z["components"].shape[1]),
        "model_blanks": int(len(z["blank_ids"])), "grid": [float(grid[0]), float(grid[-1]), int(len(grid))],
        "resampling": "mean of the native linear interpolant over each model channel; ends extended flat",
        "files": files,
        "content_hash": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}, indent=2) + "\n")
    print(f"{len(cases)} analyses at {cases.code.nunique()} sites -> {bundle}")
    print(cases.filter_type.value_counts().to_string())
    print("model channels extended per analysis:", cases.model_channels_extended.describe().round(2).to_dict())


def check(args):
    """The frozen Addis inputs (ETAD_FTIR_spectra.csv, 2,722 channels) against our resampled native ones."""
    sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
    import data_paths
    csv = Path(data_paths.etad_dir()) / "ETAD_FTIR_spectra.csv"
    ref = pd.read_csv(csv)
    bundle = OUT / "vibes_bundle"
    cases = pd.read_csv(bundle / "cases.csv", dtype={"media_id": str, "analysis_id": str})
    X = np.load(bundle / "X_model_grid.npy", mmap_mode="r")
    grid = np.load(bundle / "background_model.npz")["wn"]
    ref_wn = np.array([float(c) for c in ref.columns[2:]])
    print("frozen grid equals model grid:", np.allclose(ref_wn, grid))
    by = {(str(r.SampleAnalysisId)): r for r in ref.itertuples(index=False)}
    stats = []
    for c in cases[cases.code.eq("ETAD")].itertuples():
        r = by.get(str(c.analysis_id))
        if r is None:
            continue
        a = np.asarray(r[2:], dtype=float)
        b = X[c.case_row].astype(float)
        m = np.isfinite(a)
        d = b[m] - a[m]
        stats.append({"analysis_id": c.analysis_id, "rms_diff": float(np.sqrt(np.mean(d**2))), "max_abs": float(np.abs(d).max()),
                      "offset": float(np.median(d)), "r": float(np.corrcoef(a[m], b[m])[0, 1]), "ref_range": float(np.ptp(a[m]))})
    s = pd.DataFrame(stats)
    print(f"{len(s)} Addis analyses matched by analysis id")
    print(s.describe().round(6).to_string())
    s.to_csv(OUT / "check_addis_vs_frozen.csv", index=False)


def airspec(args):
    sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
    from airspec_baseline import airspec_baseline_matrix
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {}
    for a, wn, y in native_analyses():
        groups.setdefault((a["first"], a["last"], a["n"]), []).append((a, y))
    rec, start = [], time.perf_counter()
    store = {}
    for (first, last, n), items in groups.items():
        wn = np.linspace(first, last, n)
        Y = np.stack([y for _, y in items])
        base, _ = airspec_baseline_matrix(wn, Y, df1=6, df2=4, n_jobs=args.workers, chunksize=8)
        for (a, _), b in zip(items, base):
            ok = np.isfinite(b)
            i0, i1 = int(np.argmax(ok)), int(len(ok) - np.argmax(ok[::-1]))
            store[a["analysis_id"]] = (i0, b[i0:i1].astype(np.float32))
            rec.append({"analysis_id": a["analysis_id"], "code": a["code"], "filter_key": a["filter_key"], "filter_type": a["filter_type"],
                        "start": i0, "n": i1 - i0, "native_n": n, "success": bool(ok.any())})
    np.savez_compressed(OUT / "airspec_baselines.npz", **{f"a{k}_{v[0]}": v[1] for k, v in store.items()})
    pd.DataFrame(rec).to_csv(OUT / "airspec_cases.csv", index=False)
    print(f"AIRSpec: {len(rec)} analyses in {time.perf_counter() - start:.0f} s, {sum(r['success'] for r in rec)} with a baseline")


# ---------------------------------------------------------------- on the VM

def _status(root, **kw):
    path = root / "execution_status.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(kw)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(cur, indent=2))
    os.replace(tmp, path)


def vibes(args):
    sys.path.insert(0, str(args.scripts))
    from joblib import Parallel, delayed, parallel_config
    from threadpoolctl import threadpool_limits
    from vibes_baseline import VibesBackground, vibes_baseline_matrix

    root, bundle = args.root, args.bundle
    root.mkdir(parents=True, exist_ok=True)
    _status(root, state="running", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for name, sha in manifest["files"].items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"bundle file changed: {name}")
    z = np.load(bundle / "background_model.npz")
    bg = VibesBackground(z["wn"], z["mean"], z["components"], pd.DataFrame(), tuple(map(str, z["blank_ids"])), 0.0)
    X = np.load(bundle / "X_model_grid.npy")
    cases = pd.read_csv(bundle / "cases.csv", dtype=str)
    ids = [f"network:{a}" for a in cases.analysis_id]
    ck = root / "checkpoints"
    ck.mkdir(exist_ok=True)
    sig = manifest["content_hash"][:16]

    def one(i):
        with threadpool_limits(limits=1):
            base, vib, diag = vibes_baseline_matrix(bg.wavenumbers, X[i][None].astype(float), bg, sample_ids=[ids[i]],
                                                    tau=0.1, loss="PB", maxiter=10000)
        r = diag.iloc[0].to_dict()
        r["case_row"] = i
        return base[0].astype(np.float32), r

    start, B = time.perf_counter(), args.batch
    # --rows a:b splits the cases across VMs; a must be a multiple of --batch so checkpoint
    # names line up and the parts merge in `collect`
    lo, hi = (int(v) for v in args.rows.split(":")) if args.rows else (0, len(X))
    hi = min(hi, len(X))
    if lo % B:
        raise ValueError("--rows start must be a multiple of --batch")
    for first in range(lo, hi, B):
        idx = np.arange(first, min(first + B, hi))
        path = ck / f"batch_{first:06d}.npz"
        if path.exists():
            continue
        with parallel_config(backend="loky", inner_max_num_threads=1):
            res = Parallel(n_jobs=args.workers)(delayed(one)(int(i)) for i in idx)
        tmp = path.with_suffix(".partial")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, signature=np.array(sig), indices=idx, baseline=np.stack([r[0] for r in res]),
                                diagnostics=np.array(json.dumps([r[1] for r in res], default=str)))
        os.replace(tmp, path)
        _status(root, state="running", done=int(idx[-1] + 1), total=hi, first=lo, elapsed_min=round((time.perf_counter() - start) / 60, 1))
        print(f"{idx[-1] + 1}/{hi} {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    _status(root, state="complete", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_seconds=round(time.perf_counter() - start), workers=args.workers, signature=sig)


def collect(args):
    """Checkpoints (downloaded from one or more VMs) -> one baseline matrix + diagnostics."""
    bundle = OUT / "vibes_bundle"
    sig = json.loads((bundle / "MANIFEST.json").read_text())["content_hash"][:16]
    cases = pd.read_csv(bundle / "cases.csv")
    grid = np.load(bundle / "background_model.npz")["wn"]
    base = np.full((len(cases), len(grid)), np.nan, dtype=np.float32)
    diags, seen = {}, set()
    for ck in map(Path, args.result):
        for p in sorted(ck.glob("batch_*.npz")):
            if p.name in seen:   # a batch two VMs both finished: keep the first copy
                continue
            seen.add(p.name)
            with np.load(p) as z:
                if str(z["signature"]) != sig:
                    raise ValueError(f"checkpoint from another bundle: {p}")
                base[z["indices"]] = z["baseline"]
                for r in json.loads(str(z["diagnostics"])):
                    diags[r["case_row"]] = {**r, "vm": ck.parent.parent.name}
    d = pd.DataFrame(sorted(diags.values(), key=lambda r: r["case_row"]))
    np.save(OUT / "vibes_baselines_model_grid.npy", base)
    d.to_csv(OUT / "vibes_fit_diagnostics.csv", index=False)
    done = np.isfinite(base).any(axis=1)
    print(f"VIBES: {done.sum()}/{len(cases)} analyses with a baseline; {int((~d.success.astype(bool)).sum())} failed fits")


FROZEN_VIBES = MODEL.parent
FROZEN_AIRSPEC = ROOT / "research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
FROZEN_ETAD = ROOT / "research/ftir_hips_chem/output/tables/vibes_colab_bundle_full500/stage/data/etad_metadata.csv"


def _agree(ours, frozen):
    k = np.isfinite(ours) & np.isfinite(frozen)
    d = ours[k] - frozen[k]
    span = float(np.ptp(frozen[k]))
    return {"channels": int(k.sum()), "rms_diff": float(np.sqrt(np.mean(d**2))), "max_abs_diff": float(np.abs(d).max()),
            "frozen_range": span, "rel_rms": float(np.sqrt(np.mean(d**2)) / span) if span else np.nan,
            "r": float(np.corrcoef(ours[k], frozen[k])[0, 1])}


def compare(args):
    """Every Addis filter in the frozen runs: our network baselines against the frozen corrected spectra.

    The frozen runs averaged an Addis filter's repeat scans before correcting; here each analysis is
    corrected on its own, so filters with repeats are compared through the mean of their analyses and
    reported apart from the single-analysis filters, which compare directly.
    """
    bundle = OUT / "vibes_bundle"
    cases = pd.read_csv(bundle / "cases.csv", dtype={"media_id": str, "analysis_id": str})
    X = np.load(bundle / "X_model_grid.npy", mmap_mode="r").astype(float)
    grid = np.load(bundle / "background_model.npz")["wn"]
    vb = np.load(OUT / "vibes_baselines_model_grid.npy").astype(float)
    ours_vibes = X - vb
    fz = pd.read_csv(FROZEN_VIBES / "cases.csv")
    fv = np.load(FROZEN_VIBES / "corrected_VIBES_full.npy", mmap_mode="r")
    fa_cases = pd.read_csv(FROZEN_AIRSPEC / "cases.csv")
    fa = np.load(FROZEN_AIRSPEC / "corrected_AIRSpec.npy", mmap_mode="r")
    fa_wn = np.load(FROZEN_AIRSPEC / "wn.npy")
    air = np.load(OUT / "airspec_baselines.npz")
    air_by = {k.split("_")[0][1:]: (int(k.split("_")[1]), air[k]) for k in air.files}
    native = {a["analysis_id"]: (wn, y) for a, wn, y in native_analyses() if a["code"] == "ETAD"}
    meta = pd.read_csv(FROZEN_ETAD)
    rows = []
    for method, frozen_cases, frozen in (("VIBES", fz, fv), ("AIRSpec", fa_cases, fa)):
        etad = frozen_cases[frozen_cases.sample_id.str.startswith("etad:")]
        for j, c in etad.iterrows():
            media = c.sample_id.split(":")[1]
            mine = cases[cases.media_id.eq(media)]
            if mine.empty:
                continue
            if method == "VIBES":
                ours = np.nanmean(ours_vibes[mine.case_row.to_numpy()], axis=0)
            else:
                traces = []
                for aid in mine.analysis_id:
                    wn, y = native[aid]
                    i0, b = air_by[aid]
                    corr = np.full(len(y), np.nan)
                    corr[i0:i0 + len(b)] = y[i0:i0 + len(b)] - b
                    span = wn[i0:i0 + len(b)]
                    v, _ = resample(span, corr[i0:i0 + len(b)], fa_wn)
                    # drop the channels whose bins reach past the baseline's span (flat-extended there)
                    v[(fa_wn > span[0] - 2) | (fa_wn < span[-1] + 2)] = np.nan
                    traces.append(v)
                ours = np.nanmean(traces, axis=0)
            rows.append({"method": method, "media_id": media, "filter_id": c.get("filter_id"), "kind": c["kind"],
                         "analyses": len(mine), **_agree(ours, np.asarray(frozen[j], dtype=float))})
    r = pd.DataFrame(rows)
    r["repeats"] = np.where(r.analyses > 1, "repeat scans (mean of analyses)", "single analysis")
    r.to_csv(OUT / "compare_addis_vs_frozen.csv", index=False)
    summary = (r.groupby(["method", "repeats", "kind"])
               .agg(filters=("media_id", "size"), rel_rms_median=("rel_rms", "median"), rel_rms_p95=("rel_rms", lambda s: s.quantile(.95)),
                    rel_rms_max=("rel_rms", "max"), r_median=("r", "median"), r_min=("r", "min"),
                    within_1pct=("rel_rms", lambda s: float((s < 0.01).mean())))
               .reset_index())
    print(summary.round(5).to_string(index=False))
    (OUT / "compare_addis_vs_frozen_summary.json").write_text(summary.to_json(orient="records", indent=1) + "\n")
    worst = r.sort_values("rel_rms", ascending=False).groupby("method").head(5)
    print(worst[["method", "filter_id", "kind", "analyses", "rel_rms", "r"]].round(5).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("prepare")
    sub.add_parser("check")
    sub.add_parser("compare")
    p = sub.add_parser("airspec")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p = sub.add_parser("vibes")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scripts", type=Path, required=True, help="folder holding vibes_baseline.py (and ../vendor/pyvibes)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--rows", default="", help="START:STOP case rows for this VM (START a multiple of --batch)")
    p = sub.add_parser("collect")
    p.add_argument("--result", nargs="+", required=True, help="the downloaded checkpoints folder(s), one per VM")
    a = ap.parse_args()
    if a.stage == "vibes":
        try:
            vibes(a)
        except BaseException as exc:
            _status(a.root, state="failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise
    else:
        globals()[a.stage](a)


if __name__ == "__main__":
    main()
