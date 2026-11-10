"""Browser test of the VIBES solver compiled to WebAssembly (wasm_demo/): Newton ELBO + MAP in Rust.

    # needs cargo with the wasm32-unknown-unknown target:  rustup target add wasm32-unknown-unknown
    python wasm_demo.py build                                  # --gallery also installs it in the gallery
    # write a sample of spectra and the Python float64 results next to a copy of the page
    python wasm_demo.py prepare --bundle vibes_bundle --out web/ --rows 64
    python wasm_demo.py check --dir web/                       # Node, no browser
    python -m http.server 8766 --bind 127.0.0.1 --directory web/
    python wasm_demo.py run --rows 64 --workers 4              # needs playwright

`--bundle` is a folder with background_model.npz and X_model_grid.npy (a network VIBES bundle).
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "wasm_demo"
WASM = HERE / "target" / "wasm32-unknown-unknown" / "release" / "vibes_wasm.wasm"
PAGE = ("index.html", "worker.js", "vibes_wasm.js", "check.mjs")
SOURCES = ("Cargo.toml", "Cargo.lock", "src/lib.rs")
GALLERY = ROOT.parents[1] / "gallery" / "app" / "src" / "lib" / "vibes"


def source_hash():
    """One hash over the files the module is built from, so a stale gallery copy can be detected without cargo."""
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update(name.encode())
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def build(a):
    env = {**os.environ, "RUSTFLAGS": "-C target-feature=+simd128"}
    subprocess.run(["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"], cwd=HERE, env=env, check=True)
    print(f"{WASM}  {WASM.stat().st_size:,} bytes")
    if getattr(a, "gallery", False):
        body = WASM.read_bytes()
        (GALLERY / WASM.name).write_bytes(body)
        rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True, check=True).stdout.strip()
        (GALLERY / "vibes_wasm.build.json").write_text(json.dumps({
            "source": "research/ftir_hips_chem/wasm_demo", "source_sha256": source_hash(),
            "wasm_sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "rustc": rustc,
            "flags": env["RUSTFLAGS"]}, indent=1) + "\n")
        print(f"installed in {GALLERY}")


def write_folder(out, W, mean, Y, tau=0.1, row_ids=None):
    """Write spectra, the background and the Python float64 fit of each spectrum to `out`."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import vibes_batch

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in PAGE:
        shutil.copy2(HERE / f, out / f)
    shutil.copy2(WASM, out / WASM.name)
    W, mean, Y = (np.ascontiguousarray(v, dtype="<f8") for v in (W, mean, Y))
    elbo = vibes_batch.fit_elbo_newton(Y, mean, W, tau=tau)
    fit = vibes_batch.solve_map_batch(Y, mean, W, sigma=elbo.sigma, tau=tau)
    if not (elbo.success.all() and fit.converged.all()):
        raise RuntimeError("the Python reference did not converge on every spectrum")
    for name, v in (("W.f64", W), ("mu.f64", mean), ("Y.f64", Y), ("B.f64", fit.baseline)):
        np.ascontiguousarray(v, dtype="<f8").tofile(out / name)
    (out / "meta.json").write_text(json.dumps({
        "p": W.shape[0], "c": W.shape[1], "rows": len(Y), "tau": tau, "row_ids": row_ids,
        "ref": {"sigma": elbo.sigma.tolist(), "elbo": elbo.elbo.tolist()}}))
    return out


def prepare(a):
    b = Path(a.bundle)
    m = np.load(b / "background_model.npz")
    X = np.load(b / "X_model_grid.npy", mmap_mode="r")
    rows = np.sort(np.random.default_rng(5).permutation(len(X))[: a.rows])
    write_folder(a.out, m["components"], m["mean"], np.asarray(X[rows], float), row_ids=rows.tolist())
    print(f"{len(rows)} spectra written to {a.out}")


def check(a):
    r = subprocess.run(["node", "check.mjs", "."], cwd=a.dir, check=True, capture_output=True, text=True)
    print(json.dumps(json.loads(r.stdout), indent=1))


def run(a):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True, executable_path=a.browser)
        pg = b.new_page()
        pg.goto(f"http://127.0.0.1:{a.port}/index.html?n={a.rows}" + (f"&workers={a.workers}" if a.workers else ""))
        t = time.time()
        r = None
        while time.time() - t < a.timeout and not r:
            r = pg.evaluate("window.__result || null")
            pg.wait_for_timeout(500)
        print(json.dumps(r, indent=1) if r else "TIMEOUT " + pg.inner_text("#log")[:300])
        b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    b.add_argument("--gallery", action="store_true", help="copy the module into gallery/app/src/lib/vibes with its build record")
    p = sub.add_parser("prepare")
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=64)
    c = sub.add_parser("check")
    c.add_argument("--dir", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rows", type=int, default=64)
    r.add_argument("--workers", type=int, default=0, help="0 uses the machine's logical cores")
    r.add_argument("--port", type=int, default=8766)
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--browser", default=None, help="path to a Chromium or Chrome binary; default is Playwright's own")
    a = ap.parse_args()
    {"build": build, "prepare": prepare, "check": check, "run": run}[a.stage](a)
