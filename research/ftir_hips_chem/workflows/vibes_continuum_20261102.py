"""VIBES with a continuum-protected background, on Colab (2 Nov 2026).

``run_transfer_diagnosis_20261102.py`` showed that the production VIBES background (87 blanks, rank 70) deletes the broad
black-carbon continuum: an injected nu^1 continuum comes back at 0.7 %. This reruns real VIBES (tau 0.1, PB loss,
maxiter 10000) on every IMPROVE, Addis and SPARTAN-network spectrum of the 22 Oct decision bundle with two backgrounds
that cannot absorb a smooth continuum:

* ``deflate``  the production mean, and the 70 components with the smooth directions 1, nu, nu^2 projected out (rank 67).
* ``anchorhp`` VIBES on the PTFE-anchored spectrum (scripts/continuum_anchor.py) with a zero mean and the 70 components
               high-passed (Savitzky-Golay, 41 channels), so VIBES only removes residual *sharp* blank structure.

Local stages: ``prepare`` / ``run`` / ``collect``. Every stage accepts ``--output``; the default is the new
``regeneration_v2`` child directory, so historical bundles and spectra are preserved. Preparation records the installed
Python/package versions and copies the code and repository lock. It no longer needs a missing historical requirements
file. This is a versioned rerun with the original PB/tau/maxiter/retry procedure, not a promise of byte-identical results
across different numerical-library versions. A complete collection has a hash-bound receipt; missing, overlapping or
incompatible checkpoints are rejected. Failed fits are reported and require explicit ``--allow-failures`` to retain NaNs.

    uv run --extra vibes python research/ftir_hips_chem/workflows/vibes_continuum_20261102.py prepare --source BUNDLE --background-model MODEL --output RUN
    uv run --extra vibes python research/ftir_hips_chem/workflows/vibes_continuum_20261102.py run --output RUN --workers 8 --rows 0:120
    uv run --extra vibes python research/ftir_hips_chem/workflows/vibes_continuum_20261102.py run --output RUN --workers 8
    uv run --extra vibes python research/ftir_hips_chem/workflows/vibes_continuum_20261102.py collect --output RUN

The bounded first batch is reused by the full run only after its identity is checked. Colab stages ``launch``, ``watch``,
``vm`` and ``bench`` remain available; they use the same checkpointed CPU worker. No GPU implementation is substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
try:
    AREA = HERE.parents[1]
    OUT = AREA / "output/tables/vibes_continuum_20261102"
except IndexError:   # on the VM the runner sits at /content/vc/bundle/code/
    AREA = OUT = None
VARIANTS = ("deflate", "anchorhp")
BATCH = 120
TAG = "vc"
PROTOCOL = "vibes_continuum_regeneration_v2"
SOLVER = {"adapter": "vibes_baseline_matrix", "loss": "PB", "tau": 0.1,
          "maxiter": 10000, "retry_failed": True}
PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "cvxpy", "clarabel",
            "joblib", "threadpoolctl", "tqdm")
DEFAULT_MODEL = AREA / "output/tables/network_baselines_20261019/vibes_bundle/background_model.npz" if AREA else None
DEFAULT_OUTPUT = OUT / "regeneration_v2" if OUT else None


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _file_hash(entry):
    return entry.get("sha256") if isinstance(entry, dict) else entry


def _manifest_file(directory, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe manifest file: {name}")
    return directory / relative


def _verify_manifest(directory):
    """Check each frozen byte sequence, not just the dimensions of the arrays."""
    directory = Path(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    files = manifest.get("files", {})
    if not files:
        raise ValueError("bundle manifest has no files")
    for name, entry in files.items():
        if _sha(_manifest_file(directory, name)) != _file_hash(entry):
            raise ValueError(f"bundle file changed: {name}")
    content_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if manifest.get("content_hash") != content_hash:
        raise ValueError("bundle content_hash does not match its file hashes")
    if manifest.get("schema_version") == 2:
        if manifest.get("protocol") != PROTOCOL or manifest.get("solver") != SOLVER:
            raise ValueError("bundle solver protocol differs from this runner")
    if tuple(manifest.get("variants", ())) != VARIANTS or manifest.get("batch") != BATCH:
        raise ValueError("bundle variants/batch differ from this runner")
    return manifest


def _runtime():
    return {"python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _status(root, **kw):
    p = root / "execution_status.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw)
    tmp = p.with_suffix(".partial")
    tmp.write_text(json.dumps(cur, indent=1))
    os.replace(tmp, p)


# ---------------------------------------------------------------- shared background construction (local and VM)
def backgrounds(wn, mean, W):
    """The two continuum-protected backgrounds as (mean, components) pairs."""
    from scipy.signal import savgol_filter
    x = wn / 2000.0
    Q, _ = np.linalg.qr(np.column_stack([np.ones_like(x), x, x ** 2]))
    U, s, _ = np.linalg.svd(W - Q @ (Q.T @ W), full_matrices=False)
    r = W.shape[1] - 3
    Whp = W - savgol_filter(W.T, 41, 2, axis=1).T
    return {"deflate": (mean, U[:, :r] * s[:r]), "anchorhp": (np.zeros_like(mean), Whp)}


# ---------------------------------------------------------------- prepare (local)
def prepare(args):
    sys.path.insert(0, str(AREA / "scripts"))
    import continuum_anchor as ca
    src = Path(args.source)
    bundle = OUT / "bundle"
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to replace an existing run: {OUT}; choose a new --output")
    source_manifest = json.loads((src / "MANIFEST.json").read_text())
    source_names = ["wn_full.npy", *[f"{g}_{kind}" for g in ("improve", "addis", "network")
                                    for kind in ("raw.npy", "meta.csv")]]
    source_hashes = {}
    for name in source_names:
        expected = _file_hash(source_manifest.get("files", {}).get(name))
        actual = _sha(src / name)
        if not expected or actual != expected:
            raise ValueError(f"source bundle file differs from its manifest: {name}")
        source_hashes[name] = actual
    source_hashes["MANIFEST.json"] = _sha(src / "MANIFEST.json")
    wn = np.load(src / "wn_full.npy", allow_pickle=False)
    if wn.ndim != 1 or len(wn) < 41 or not np.isfinite(wn).all() or not (np.all(np.diff(wn) > 0) or np.all(np.diff(wn) < 0)):
        raise ValueError("source grid must be finite, strictly monotonic and at least 41 channels")
    model_path = Path(getattr(args, "background_model", None) or DEFAULT_MODEL)
    model_hash = _sha(model_path)
    with np.load(model_path, allow_pickle=False) as z:
        model_grid, mean, components = z["wn"], z["mean"], z["components"]
    # The frozen decision grid is stored as float32, while the production model
    # grid is float64. Accept only that exact storage-rounding identity, not an
    # approximate grid or a resampling of the spectra.
    if model_grid.shape != wn.shape or not np.array_equal(model_grid.astype(wn.dtype), wn):
        raise ValueError("background grid differs from the source grid after its storage cast")
    if mean.shape != wn.shape or components.ndim != 2 or components.shape[0] != len(wn) or components.shape[1] <= 3:
        raise ValueError("background mean/components have incompatible dimensions")
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise ValueError("background model contains non-finite values")
    raw, rows = [], []
    for g in ("improve", "addis", "network"):
        X = np.load(src / f"{g}_raw.npy", mmap_mode="r", allow_pickle=False)
        meta = pd.read_csv(src / f"{g}_meta.csv", dtype={"filter_id": str})
        if "filter_id" not in meta or meta.filter_id.isna().any() or meta.filter_id.str.strip().eq("").any() or meta.filter_id.duplicated().any():
            raise ValueError(f"{g}: require one nonempty unique physical-filter ID per row")
        if X.shape != (len(meta), len(wn)) or not len(meta) or not np.isfinite(X).all():
            raise ValueError(f"{g}: raw spectra and physical-filter metadata are inconsistent or non-finite")
        raw.append(X.astype(np.float32))
        rows.append(pd.DataFrame({"group": g, "group_row": np.arange(len(X)), "filter_id": meta.filter_id.astype(str)}))
    raw = np.vstack(raw)
    rows = pd.concat(rows, ignore_index=True)
    rows["row"] = np.arange(len(rows))
    anchored = ca.AnchoredBackground(wn, mean, components).correct(raw)
    runtime = _runtime()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".continuum-prepare-", dir=OUT.parent) as tmp:
        staged = Path(tmp) / "bundle"
        (staged / "code/scripts").mkdir(parents=True)
        np.save(staged / "raw.npy", raw)
        np.save(staged / "anchored.npy", anchored)
        np.save(staged / "wn.npy", wn)
        shutil.copyfile(model_path, staged / "background_model.npz")
        rows.to_csv(staged / "rows.csv", index=False)
        for name in ("vibes_baseline.py", "continuum_anchor.py"):
            shutil.copyfile(AREA / "scripts" / name, staged / "code/scripts" / name)
        shutil.copytree(AREA / "vendor/pyvibes", staged / "code/vendor/pyvibes",
                        ignore=shutil.ignore_patterns("__pycache__", "*.ipynb", "data", "docs", ".git"))
        shutil.copyfile(HERE, staged / "code/vibes_continuum.py")
        (staged / "requirements.txt").write_text("".join(f"{name}=={version}\n" for name, version in runtime["packages"].items()))
        _atomic_json(staged / "environment.json", runtime)
        repo = AREA.parents[1]
        for name in ("uv.lock", "pyproject.toml"):
            if (repo / name).is_file():
                shutil.copyfile(repo / name, staged / name)
        if _sha(model_path) != model_hash or any(_sha(src / name) != value for name, value in source_hashes.items()):
            raise ValueError("an input changed while preparing the continuum bundle")
        files = {str(p.relative_to(staged)): _sha(p) for p in sorted(staged.rglob("*")) if p.is_file()}
        _atomic_json(staged / "MANIFEST.json", {
            "schema_version": 2, "protocol": PROTOCOL,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_rows": len(rows),
            "by_group": rows.group.value_counts().to_dict(), "variants": VARIANTS, "batch": BATCH, "solver": SOLVER,
            "source": {"decision_files": source_hashes, "background_model_sha256": model_hash,
                       "background_rank": int(components.shape[1]), "source_grid_dtype": str(wn.dtype),
                       "background_grid_dtype": str(model_grid.dtype),
                       "grid_identity": "exact after source-storage cast; no interpolation"},
            "files": files, "content_hash": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()})
        OUT.mkdir(exist_ok=True)
        if any(OUT.iterdir()):
            raise FileExistsError(f"output became nonempty during preparation: {OUT}")
        staged.rename(bundle)
    print(f"bundle: {len(rows)} spectra x {len(VARIANTS)} variants = {len(rows) * len(VARIANTS)} fits")


# ---------------------------------------------------------------- vm (runs on Colab)
def _jobs(n_rows):
    """Interleave variants so every VM row range does both."""
    return [(v, r) for r in range(n_rows) for v in VARIANTS]


def _run_identity(bundle, manifest):
    if manifest.get("schema_version") != 2:
        return None
    return {"protocol": PROTOCOL, "bundle_content_hash": manifest["content_hash"], "solver": SOLVER,
            "runner_sha256": _file_hash(manifest["files"]["code/vibes_continuum.py"]),
            "runtime": json.loads((bundle / "environment.json").read_text())}


def _read_checkpoint(path, manifest, n_channels, identity=None):
    """Return a complete, correctly ordered batch or reject it before reuse."""
    expected_signature = manifest["content_hash"] if manifest.get("schema_version") == 2 else manifest["content_hash"][:16]
    total = int(manifest["n_rows"]) * len(VARIANTS)
    with np.load(path, allow_pickle=False) as saved:
        required = {"signature", "first", "variant", "row", "baseline", "diagnostics"}
        if not required.issubset(saved.files):
            raise ValueError(f"{path.name}: checkpoint fields are incomplete")
        if saved["signature"].shape != () or str(saved["signature"].item()) != expected_signature:
            raise ValueError(f"{path.name}: checkpoint belongs to a different bundle signature")
        start = saved["first"]
        if start.shape != () or start.dtype.kind not in "iu":
            raise ValueError(f"{path.name}: checkpoint first must be an integer")
        first = int(start)
        if first < 0 or first >= total or first % BATCH or path.name != f"batch_{first:06d}.npz":
            raise ValueError(f"{path.name}: invalid checkpoint batch position")
        expected = _jobs(manifest["n_rows"])[first:min(first + BATCH, total)]
        variants, rows, baseline = saved["variant"], saved["row"], saved["baseline"]
        if variants.ndim != 1 or rows.ndim != 1 or rows.dtype.kind not in "iu":
            raise ValueError(f"{path.name}: invalid row/variant arrays")
        jobs = list(zip(variants.tolist(), rows.tolist()))
        if jobs != expected:
            raise ValueError(f"{path.name}: row/variant membership, order or completeness differs from its batch")
        if baseline.shape != (len(expected), n_channels) or baseline.dtype.kind != "f":
            raise ValueError(f"{path.name}: baseline dimensions/dtype differ from the frozen bundle")
        diagnostics = json.loads(str(saved["diagnostics"].item()))
        if not isinstance(diagnostics, list) or len(diagnostics) != len(expected):
            raise ValueError(f"{path.name}: diagnostics do not cover the batch")
        for (variant, row), diag, values in zip(expected, diagnostics, baseline):
            if (not isinstance(diag, dict) or diag.get("variant") != variant or diag.get("row") != row
                    or not isinstance(diag.get("row"), int) or isinstance(diag.get("row"), bool)
                    or not isinstance(diag.get("success"), bool)):
                raise ValueError(f"{path.name}: diagnostics disagree with row/variant identity or success type")
            if diag["success"] and not np.isfinite(values).all():
                raise ValueError(f"{path.name}: successful fit has non-finite baseline values")
            if not diag["success"] and not np.isnan(values).all():
                raise ValueError(f"{path.name}: failed fit must retain an all-NaN baseline")
        if identity is not None:
            if "run_identity" not in saved.files or json.loads(str(saved["run_identity"].item())) != identity:
                raise ValueError(f"{path.name}: checkpoint solver, code or runtime identity differs")
        return first, variants.copy(), rows.copy(), baseline.copy(), diagnostics


def _fit_spectrum(variant, row, y, wn, background, fit):
    """Use the original adapter and settings; never substitute another solver."""
    from threadpoolctl import threadpool_limits

    start = time.perf_counter()
    with threadpool_limits(limits=1):
        try:
            base, _, diagnostics = fit(wn, y[None], background, sample_ids=[f"{variant}:{row}"],
                                       tau=SOLVER["tau"], loss=SOLVER["loss"], maxiter=SOLVER["maxiter"],
                                       retry_failed=SOLVER["retry_failed"])
            diag = diagnostics.iloc[0].to_dict()
            diag["success"] = bool(diag["success"])
            baseline = base[0].astype(np.float32) if diag["success"] else np.full(len(wn), np.nan, np.float32)
            if diag["success"] and not np.isfinite(baseline).all():
                raise ValueError("adapter reported success with non-finite baseline")
        except Exception as exc:  # preserve every attempted fit and its failure
            diag, baseline = {"success": False, "error": repr(exc)[:300]}, np.full(len(wn), np.nan, np.float32)
    diag.update(variant=variant, row=int(row), seconds=round(time.perf_counter() - start, 2))
    return baseline, diag


def vm(args):
    bundle, root = Path(args.bundle), Path(args.root)
    manifest = _verify_manifest(bundle)
    identity = _run_identity(bundle, manifest)
    if identity is not None and (_sha(HERE) != identity["runner_sha256"] or _runtime() != identity["runtime"]):
        raise ValueError("runner code or installed package/Python versions differ from the prepared bundle")
    old_status = root / "execution_status.json"
    if old_status.exists() and identity is not None:
        if json.loads(old_status.read_text()).get("run_identity") != identity:
            raise ValueError("run directory belongs to different code, parameters, runtime or inputs")
    sys.path.insert(0, str(bundle / "code/scripts"))
    from joblib import Parallel, cpu_count, delayed, parallel_config
    from vibes_baseline import VibesBackground, vibes_baseline_matrix

    wn = np.load(bundle / "wn.npy", allow_pickle=False)
    z = np.load(bundle / "background_model.npz", allow_pickle=False)
    raw = np.load(bundle / "raw.npy", mmap_mode="r", allow_pickle=False)
    anc = np.load(bundle / "anchored.npy", mmap_mode="r", allow_pickle=False)
    if raw.shape != (manifest["n_rows"], len(wn)) or anc.shape != raw.shape:
        raise ValueError("prepared spectral dimensions differ from the bundle manifest")
    bgs = {v: VibesBackground(wn, m, W, pd.DataFrame(), tuple(), 0.0)
           for v, (m, W) in backgrounds(wn, z["mean"], z["components"]).items()}
    jobs = _jobs(manifest["n_rows"])
    lo, hi = (int(v) for v in args.rows.split(":")) if args.rows else (0, len(jobs))
    if not 0 <= lo < hi <= len(jobs) or lo % BATCH or (hi % BATCH and hi != len(jobs)):
        raise ValueError(f"--rows must cover complete {BATCH}-job batches within 0:{len(jobs)}")

    def one(v, r):
        y = np.asarray((raw if v == "deflate" else anc)[r], float)
        return _fit_spectrum(v, r, y, wn, bgs[v], vibes_baseline_matrix)

    root.mkdir(parents=True, exist_ok=True)
    _status(root, state="running", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cpus=cpu_count(), run_identity=identity, scope="requested job range", first=lo, total=hi)
    ck = root / "checkpoints"
    ck.mkdir(exist_ok=True)
    start = time.perf_counter()
    workers = args.workers or min(8, cpu_count())
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    failed, resumed = 0, 0
    for first in range(lo, hi, BATCH):
        part = jobs[first:min(first + BATCH, hi)]
        path = ck / f"batch_{first:06d}.npz"
        if path.exists():
            *_, diagnostics = _read_checkpoint(path, manifest, len(wn), identity)
            failed += sum(not d["success"] for d in diagnostics)
            resumed += len(part)
            continue
        with parallel_config(backend="loky", inner_max_num_threads=1):
            res = Parallel(n_jobs=workers)(delayed(one)(v, r) for v, r in part)
        tmp = path.with_suffix(".partial")
        with tmp.open("wb") as fh:
            signature = manifest["content_hash"] if identity is not None else manifest["content_hash"][:16]
            np.savez_compressed(fh, signature=np.array(signature), first=first,
                                variant=np.array([v for v, _ in part]), row=np.array([r for _, r in part]),
                                baseline=np.stack([x[0] for x in res]),
                                diagnostics=np.array(json.dumps([x[1] for x in res], default=str)),
                                run_identity=np.array(json.dumps(identity, sort_keys=True)))
        os.replace(tmp, path)
        *_, diagnostics = _read_checkpoint(path, manifest, len(wn), identity)
        failed += sum(not d["success"] for d in diagnostics)
        done = first + len(part)
        mins = (time.perf_counter() - start) / 60
        _status(root, state="running", done=done, first=lo, total=hi, elapsed_min=round(mins, 1),
                failed=failed, resumed=resumed, per_min=round((done - lo - resumed) / max(mins, 1e-6), 1))
        print(f"{done}/{hi} {mins:.1f} min", flush=True)
    _verify_manifest(bundle)
    _status(root, state="complete", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            done=hi, first=lo, total=hi, failed=failed, resumed=resumed,
            full_bundle=lo == 0 and hi == len(jobs), wall_seconds=round(time.perf_counter() - start), workers=workers)


def local(args):
    """The same checkpointed CPU worker used on Colab, on this machine."""
    args.bundle = OUT / "bundle"
    args.root = OUT / "results/local/run"
    vm(args)


# ---------------------------------------------------------------- Colab plumbing (local)
def _colab(*argv, timeout=180, check=True):
    p = subprocess.run(["colab", *argv], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"colab {' '.join(argv[:2])}: {p.stdout[-800:]} {p.stderr[-800:]}")
    return p


def _exec(session, code, timeout=120):
    f = OUT / "launch" / f"exec_{session}.py"
    f.write_text(code)
    return _colab("exec", "-s", session, "-f", str(f), "--timeout", str(timeout), timeout=timeout + 60, check=False)


def _retry(fn, tries=4):
    for k in range(tries):
        try:
            return fn()
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            if k == tries - 1:
                raise
            print(f"  retry after {type(exc).__name__}", flush=True)


def _tarball():
    tb = OUT / "launch/bundle.tar"
    if not tb.exists():
        with tarfile.open(tb, "w") as tf:   # npy compresses poorly; plain tar
            tf.add(OUT / "bundle", arcname="bundle")
    return tb


def _upload(name, tb):
    """Upload in 8 MiB parts (single large uploads hang), reassemble and sha-check on the VM."""
    parts_dir = OUT / "launch/parts"
    if not parts_dir.exists():
        parts_dir.mkdir()
        with open(tb, "rb") as f:
            k = 0
            while chunk := f.read(8 * 2 ** 20):
                (parts_dir / f"part_{k:04d}").write_bytes(chunk)
                k += 1
    parts = sorted(parts_dir.iterdir())
    _exec(name, "import os; os.makedirs('/content/vc/parts', exist_ok=True)", timeout=60)
    for p in parts:
        _retry(lambda p=p: _colab("upload", "-s", name, str(p), f"/content/vc/parts/{p.name}", timeout=120))
    sha = _sha(tb)
    out = _exec(name, f"""
import hashlib, pathlib, tarfile
parts = sorted(pathlib.Path('/content/vc/parts').iterdir())
with open('/content/vc/bundle.tar', 'wb') as f:
    for p in parts: f.write(p.read_bytes())
ok = hashlib.sha256(open('/content/vc/bundle.tar', 'rb').read()).hexdigest() == '{sha}'
if ok: tarfile.open('/content/vc/bundle.tar').extractall('/content/vc')
print('SHA_OK' if ok else 'SHA_BAD', len(parts))
""", timeout=300).stdout
    if "SHA_OK" not in out:
        raise RuntimeError(f"{name}: bundle upload failed: {out[-400:]}")


def _start_runner(name, rows, workers=0):
    code = f"""
import os, subprocess, sys
cmd = (f"{{sys.executable}} -m pip install -q -r /content/vc/bundle/requirements.txt && "
       f"{{sys.executable}} /content/vc/bundle/code/vibes_continuum.py vm --bundle /content/vc/bundle "
       f"--root /content/vc/run --rows {rows} --workers {workers}")
p = subprocess.Popen(['bash', '-c', cmd], stdout=open('/content/vc/run.log', 'w'), stderr=subprocess.STDOUT,
                     start_new_session=True)
print('started', p.pid, 'cpus', os.cpu_count())
"""
    return _exec(name, code, timeout=120).stdout.strip()[-200:]


def bench(args):
    """Allocate one VM, report vCPUs and fits/min on the first 2 batches, keep the VM for launch."""
    (OUT / "launch").mkdir(parents=True, exist_ok=True)
    name = f"{TAG}-0"
    p = _colab("new", "-s", name, "--gpu", args.gpu, timeout=600, check=False)
    print((p.stdout + p.stderr).strip()[-300:], flush=True)
    print(_exec(name, "import os, subprocess; print('CPUS', os.cpu_count()); "
                      "print(subprocess.run(['bash','-c','nproc; free -g | head -2; nvidia-smi -L'],capture_output=True,text=True).stdout)",
                timeout=60).stdout, flush=True)
    _upload(name, _tarball())
    print(_start_runner(name, f"0:{2 * BATCH}"), flush=True)


def launch(args):
    (OUT / "launch").mkdir(parents=True, exist_ok=True)
    tb = _tarball()
    n = json.loads((OUT / "bundle/MANIFEST.json").read_text())["n_rows"] * len(VARIANTS)
    existing = _colab("sessions", check=False).stdout
    sessions = []
    for k in range(args.vms):
        name = f"{TAG}-{k}"
        if name not in existing:
            p = _colab("new", "-s", name, "--gpu", args.gpu, timeout=600, check=False)
            if p.returncode:
                print(f"{name}: not allocated ({(p.stdout + p.stderr).strip().splitlines()[-1:]})", flush=True)
                break
            _upload(name, tb)
        sessions.append({"session": name, "t0": time.time()})
        print(f"{name}: ready", flush=True)
    if not sessions:
        raise SystemExit("no Colab runtime allocated")
    first = args.skip
    per = -(-(n - first) // len(sessions) // BATCH) * BATCH
    for k, s in enumerate(sessions):
        s["rows"] = f"{first + k * per}:{min(first + (k + 1) * per, n)}"
        # a fresh runner on a VM that already ran `bench` skips its finished checkpoints
        _exec(s["session"], "import subprocess; subprocess.run(['pkill','-f','vibes_continuum.py vm'])", timeout=60)
        print(s["session"], s["rows"], _start_runner(s["session"], s["rows"]), flush=True)
    (OUT / "launch/sessions.json").write_text(json.dumps(sessions, indent=1))


def _sync(name, dest):
    """Mirror new checkpoints from the VM into dest/run/checkpoints (8 MiB parts, sha-checked)."""
    code = """
import hashlib, pathlib, subprocess, tarfile, time
vc = pathlib.Path('/content/vc'); ck = vc / 'run/checkpoints'; done = vc / 'synced.txt'
seen = set(done.read_text().split()) if done.exists() else set()
new = sorted(p.name for p in ck.glob('batch_*.npz')) if ck.exists() else []
new = [n for n in new if n not in seen]
for old in vc.glob('sync_part_*'): old.unlink()
if new:
    with tarfile.open(vc / 'sync.tar', 'w') as tf:
        for n in new: tf.add(ck / n, arcname=n)
    subprocess.run(['split', '-b', '8m', str(vc / 'sync.tar'), str(vc / 'sync_part_')], check=True)
    print('SHA', hashlib.sha256((vc / 'sync.tar').read_bytes()).hexdigest())
    print('PARTS', ' '.join(sorted(p.name for p in vc.glob('sync_part_*'))))
    print('NAMES', ' '.join(new))
else:
    print('NONE')
"""
    out = _exec(name, code, timeout=180).stdout
    if "NONE" in out or "SHA" not in out:
        return 0
    get = lambda tag: next(l.split()[1:] for l in out.splitlines() if l.startswith(tag))
    sha, parts, names = get("SHA")[0], get("PARTS"), get("NAMES")
    tmp = dest / "sync_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in parts:
        _retry(lambda p=p: _colab("download", "-s", name, f"/content/vc/{p}", str(tmp / p), timeout=300))
    tar = tmp / "sync.tar"
    with open(tar, "wb") as f:
        for p in parts:
            f.write((tmp / p).read_bytes())
    if _sha(tar) != sha:
        raise RuntimeError("sync sha mismatch")
    (dest / "run/checkpoints").mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        tf.extractall(dest / "run/checkpoints")
    shutil.rmtree(tmp)
    _exec(name, "open('/content/vc/synced.txt','a').write(' '.join(%r) + ' ')" % names, timeout=60)
    return len(names)


def watch(args):
    sessions = json.loads((OUT / "launch/sessions.json").read_text())
    probe = ("import json, pathlib\n"
             "p = pathlib.Path('/content/vc/run/execution_status.json')\n"
             "print('STATUS', p.read_text().replace(chr(10), ' ') if p.exists() else '{}')\n")
    pending = [s for s in sessions if not (OUT / "results" / s["session"] / "DONE").exists()]
    fails = {s["session"]: 0 for s in sessions}
    while pending:
        for s in list(pending):
            name, dest = s["session"], OUT / "results" / s["session"]
            try:
                out = _exec(name, probe, timeout=60).stdout
                st = json.loads(next((l for l in out.splitlines() if l.startswith("STATUS")), "STATUS {}")[7:])
                got = _sync(name, dest)
                fails[name] = 0
            except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError, StopIteration) as exc:
                fails[name] += 1
                print(f"{time.strftime('%H:%M')} {name} probe/sync failed ({type(exc).__name__}), {fails[name]} in a row", flush=True)
                if fails[name] >= 4:
                    have = len(list((dest / "run/checkpoints").glob("batch_*.npz")))
                    raise SystemExit(f"{name} unreachable; {have} batches mirrored locally. Relaunch with --skip set to the "
                                     f"first missing batch start.")
                continue
            have = len(list((dest / "run/checkpoints").glob("batch_*.npz")))
            print(f"{time.strftime('%H:%M')} {name} {st.get('state', '?')} {st.get('done', '-')}/{st.get('total', '-')} "
                  f"{st.get('per_min', '-')}/min; +{got} batches mirrored ({have} local)", flush=True)
            if st.get("state") == "complete" and st.get("total") == int(s["rows"].split(":")[1]):
                (dest / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ"))
                _colab("stop", "-s", name, check=False)
                print(f"{name}: all batches mirrored; VM stopped", flush=True)
                pending.remove(s)
        if pending:
            time.sleep(args.poll)


def _probe_file(code):
    f = OUT / "launch/probe.py"
    f.write_text(code)
    return f


# ---------------------------------------------------------------- collect (local)
def collect(args):
    bundle = OUT / "bundle"
    manifest = _verify_manifest(bundle)
    manifest_hash = _sha(bundle / "MANIFEST.json")
    identity = _run_identity(bundle, manifest)
    rows = pd.read_csv(bundle / "rows.csv", dtype={"filter_id": str})
    wn = np.load(bundle / "wn.npy", allow_pickle=False)
    if (len(rows) != manifest["n_rows"] or not {"group", "group_row", "row", "filter_id"}.issubset(rows)
            or not np.array_equal(rows.row.to_numpy(), np.arange(len(rows)))):
        raise ValueError("collection row identity differs from the bundle manifest")
    if rows.groupby("group").size().to_dict() != manifest["by_group"]:
        raise ValueError("collection groups differ from the bundle manifest")
    for group, sub in rows.groupby("group", sort=False):
        if (not np.array_equal(sub.group_row.to_numpy(), np.arange(len(sub))) or sub.filter_id.isna().any()
                or sub.filter_id.duplicated().any()):
            raise ValueError(f"{group}: collection physical-filter identity/order is invalid")
    output_names = [f"corrected_{variant}_{group}.npy" for variant in VARIANTS for group in rows.group.unique()]
    output_names += ["fit_diagnostics.csv", "collection_receipt.json"]
    if any((OUT / name).exists() for name in output_names):
        raise FileExistsError("refusing to overwrite collected spectra or receipts; preserve the existing collection")
    supplied = getattr(args, "results", None)
    files = sorted({p for root in supplied for p in Path(root).rglob("batch_*.npz")}) if supplied else sorted((OUT / "results").glob("*/run/checkpoints/batch_*.npz"))
    if not files:
        raise ValueError("no checkpoints found; no corrected spectra were written")
    base = {v: np.full((len(rows), len(wn)), np.nan, np.float32) for v in VARIANTS}
    diagnostics, checkpoint_records, seen = [], [], set()
    expected = set(_jobs(len(rows)))
    for path in files:
        before = _sha(path)
        first, variants, indices, values, diag = _read_checkpoint(path, manifest, len(wn), identity)
        for variant, row, baseline in zip(variants, indices, values):
            key = (str(variant), int(row))
            if key in seen:
                raise ValueError(f"duplicate checkpoint job {key}; select one non-overlapping set of --results")
            seen.add(key)
            base[key[0]][key[1]] = baseline
        if _sha(path) != before:
            raise ValueError(f"checkpoint changed during collection: {path.name}")
        diagnostics.extend(diag)
        checkpoint_records.append({"sha256": before, "first": first, "jobs": len(indices)})
    missing = expected - seen
    if missing or seen - expected:
        raise ValueError(f"checkpoint collection is incomplete: {len(missing)} missing of {len(expected)} jobs; no spectra written")
    d = pd.DataFrame(diagnostics).sort_values(["row", "variant"]).reset_index(drop=True)
    failed = int((~d.success).sum())
    counts = {variant: {"expected": len(rows), "collected": int((d.variant == variant).sum()),
                        "failed": int(((d.variant == variant) & ~d.success).sum())} for variant in VARIANTS}
    if failed and not getattr(args, "allow_failures", False):
        _atomic_json(OUT / "collection_failure.json", {"protocol": PROTOCOL, "bundle_manifest_sha256": manifest_hash,
                     "expected_fits": len(expected), "collected_fits": len(seen), "failed_fits": failed,
                     "by_variant": counts, "failed_jobs": d.loc[~d.success, ["variant", "row"]].to_dict("records")})
        raise ValueError(f"{failed} fits failed; inspect collection_failure.json or explicitly use --allow-failures to retain NaNs")
    raw = np.load(bundle / "raw.npy", mmap_mode="r", allow_pickle=False)
    anc = np.load(bundle / "anchored.npy", mmap_mode="r", allow_pickle=False)
    output_hashes = {}
    with tempfile.TemporaryDirectory(prefix=".continuum-collect-", dir=OUT) as tmp:
        staged = Path(tmp)
        for variant in VARIANTS:
            src = raw if variant == "deflate" else anc
            corrected = (np.asarray(src, np.float32) - base[variant]).astype(np.float32)
            for group, sub in rows.groupby("group", sort=False):
                name = f"corrected_{variant}_{group}.npy"
                values = corrected[sub.row.to_numpy()]
                np.save(staged / name, values)
                output_hashes[name] = {"sha256": _sha(staged / name), "shape": list(values.shape)}
        d.to_csv(staged / "fit_diagnostics.csv", index=False)
        output_hashes["fit_diagnostics.csv"] = {"sha256": _sha(staged / "fit_diagnostics.csv")}
        _verify_manifest(bundle)
        if _sha(bundle / "MANIFEST.json") != manifest_hash:
            raise ValueError("bundle manifest changed during collection")
        if any(_sha(path) != record["sha256"] for path, record in zip(files, checkpoint_records)):
            raise ValueError("a checkpoint changed before the collection was committed")
        _atomic_json(staged / "collection_receipt.json", {
            "schema_version": 2, "protocol": PROTOCOL, "status": "complete_with_failures" if failed else "complete",
            "bundle_manifest_sha256": manifest_hash, "bundle_content_hash": manifest["content_hash"],
            "expected_fits": len(expected), "collected_fits": len(seen), "failed_fits": failed, "by_variant": counts,
            "solver": SOLVER, "run_identity": identity, "checkpoints": checkpoint_records, "outputs": output_hashes,
            "failure_policy": "failed baselines remain NaN; no replacement or imputation",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        # The receipt is the final commit marker. A partial filesystem write can
        # never be mistaken by downstream readers for a certified collection.
        for name in output_names:
            os.link(staged / name, OUT / name)  # exclusive publication; never replace an existing file
            (staged / name).unlink()
    print(d.groupby("variant").agg(n=("row", "size"), failed=("success", lambda s: (~s).sum()), median_s=("seconds", "median")))
    print("corrected spectra and verified collection receipt written per variant and group")


def main():
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    def parser(name):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                             help="versioned run directory; defaults to regeneration_v2, preserving the historical run")
        return command

    a = parser("prepare"); a.add_argument("--source", type=Path, required=True)
    a.add_argument("--background-model", type=Path, default=DEFAULT_MODEL)
    a = parser("run"); a.add_argument("--rows", default=""); a.add_argument("--workers", type=int, default=0)
    a = parser("vm"); a.add_argument("--bundle", type=Path, required=True); a.add_argument("--root", type=Path, required=True)
    a.add_argument("--rows", default=""); a.add_argument("--workers", type=int, default=0)
    a = parser("bench"); a.add_argument("--gpu", default="L4")
    a = parser("launch"); a.add_argument("--gpu", default="L4"); a.add_argument("--vms", type=int, default=2)
    a.add_argument("--skip", type=int, default=0, help="first job index (multiple of BATCH) to start from")
    a = parser("watch"); a.add_argument("--poll", type=int, default=300)
    a = parser("collect")
    a.add_argument("--results", type=Path, nargs="+", help="optional non-overlapping checkpoint directories")
    a.add_argument("--allow-failures", action="store_true", help="explicitly retain failed fits as NaN and record their counts")
    args = p.parse_args()
    OUT = args.output
    {"prepare": prepare, "run": local, "vm": vm, "bench": bench, "launch": launch, "watch": watch, "collect": collect}[args.stage](args)


if __name__ == "__main__":
    main()
