"""Browser test of batched VIBES (webgpu_demo/): Newton ELBO + MAP in JavaScript, Khatri-Rao GEMM in WGSL.

    # write inputs and float64 reference next to a copy of the page, serve it, run it in Chrome
    python webgpu_demo.py prepare --bundle bundle_v2 --newton run_newton64/results.npz --out web/ --rows 256
    python -m http.server 8765 --bind 127.0.0.1 --directory web/
    python webgpu_demo.py run --mode webgpu --rows 256     # or --mode javascript

The page reports time and agreement with the float64 Newton + MAP run (needs playwright and Chrome).
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "webgpu_demo"


def prepare(a):
    b, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("index.html", "vibes.js"):
        shutil.copy2(HERE / f, out / f)
    m = np.load(b / "background_model.npz")
    mu, W = m["mean"].astype(float), m["components"].astype(float)
    X = np.load(b / "X_model_grid.npy", mmap_mode="r")
    saved = np.load(b / "saved_baselines.npy", mmap_mode="r")
    N = np.load(a.newton)
    rows = np.random.default_rng(5).permutation(len(X))[: a.rows]
    rows[:4] = [1278, 3802, 466, int(np.argmin(N["sigma"]))]  # typical, slowest, premature stop, smallest sigma
    Y = np.asarray(X[rows], float)
    base = N["baseline"][rows].astype(float)
    W.tofile(out / "W.f64")
    mu.tofile(out / "mu.f64")
    (Y - mu).tofile(out / "V.f64")
    base.tofile(out / "B.f64")
    crms = np.sqrt(np.mean((Y - np.asarray(saved[rows], float)) ** 2, axis=1))
    crms = np.where(np.isfinite(crms), crms, np.sqrt(np.mean((Y - base) ** 2, axis=1)))
    (out / "meta.json").write_text(json.dumps({
        "p": W.shape[0], "c": W.shape[1], "rows": len(rows), "row_ids": rows.tolist(),
        "ref": {"sigma": N["sigma"][rows].tolist(), "elbo": N["elbo"][rows].tolist(), "crms": crms.tolist()}}))


def run(a):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True, args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"])
        pg = b.new_page()
        pg.goto(f"http://127.0.0.1:{a.port}/index.html?mode={a.mode}&n={a.rows}")
        t = time.time()
        r = None
        while time.time() - t < a.timeout and not r:
            r = pg.evaluate("window.__result || null")
            pg.wait_for_timeout(1000)
        print(json.dumps(r, indent=1) if r else "TIMEOUT " + pg.inner_text("#log")[:300])
        b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--bundle", required=True)
    p.add_argument("--newton", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=256)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["webgpu", "javascript"], default="webgpu")
    r.add_argument("--rows", type=int, default=256)
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    {"prepare": prepare, "run": run}[a.stage](a)
