"""VIBES with a continuum-protected background, on Colab (2 Nov 2026).

``run_transfer_diagnosis_20261102.py`` showed that the production VIBES background (87 blanks, rank 70) deletes the broad
black-carbon continuum: an injected nu^1 continuum comes back at 0.7 %. This reruns real VIBES (tau 0.1, PB loss,
maxiter 10000) on every IMPROVE, Addis and SPARTAN-network spectrum of the 22 Oct decision bundle with two backgrounds
that cannot absorb a smooth continuum:

* ``deflate``  the production mean, and the 70 components with the smooth directions 1, nu, nu^2 projected out (rank 67).
* ``anchorhp`` VIBES on the PTFE-anchored spectrum (scripts/continuum_anchor.py) with a zero mean and the 70 components
               high-passed (Savitzky-Golay, 41 channels), so VIBES only removes residual *sharp* blank structure.

Stages: ``prepare`` (local bundle) / ``launch`` (Colab VMs, detached runner) / ``watch`` / ``collect`` / ``vm`` (runs on
the VM) / ``bench`` (timing on one VM). VIBES is a cvxpy convex solve per spectrum, CPU-only: the GPU on the runtime
is idle, the VM is chosen for its vCPUs.

    ../../.venv/bin/python workflows/vibes_continuum_20261102.py prepare --source /tmp/xfer1102/bundle
    ../../.venv/bin/python workflows/vibes_continuum_20261102.py launch --gpu L4 --vms 2
    ../../.venv/bin/python workflows/vibes_continuum_20261102.py collect
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
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


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


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
    src = args.source
    bundle = OUT / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "code/scripts").mkdir(parents=True)
    wn = np.load(src / "wn_full.npy")
    z = np.load(AREA / "output/tables/network_baselines_20261019/vibes_bundle/background_model.npz")
    if not np.allclose(z["wn"], wn):
        raise ValueError("background grid differs from bundle grid")
    raw, rows = [], []
    for g in ("improve", "addis", "network"):
        X = np.load(src / f"{g}_raw.npy")
        meta = pd.read_csv(src / f"{g}_meta.csv")
        raw.append(X.astype(np.float32))
        rows.append(pd.DataFrame({"group": g, "group_row": np.arange(len(X)), "filter_id": meta.filter_id.astype(str)}))
    raw = np.vstack(raw)
    rows = pd.concat(rows, ignore_index=True)
    rows["row"] = np.arange(len(rows))
    anchored = ca.AnchoredBackground(wn, z["mean"], z["components"]).correct(raw)
    np.save(bundle / "raw.npy", raw)
    np.save(bundle / "anchored.npy", anchored)
    np.save(bundle / "wn.npy", wn)
    np.savez(bundle / "background_model.npz", wn=z["wn"], mean=z["mean"], components=z["components"])
    rows.to_csv(bundle / "rows.csv", index=False)
    shutil.copy(AREA / "scripts/vibes_baseline.py", bundle / "code/scripts")
    shutil.copytree(AREA / "vendor/pyvibes", bundle / "code/vendor/pyvibes",
                    ignore=shutil.ignore_patterns("__pycache__", "*.ipynb", "data", "docs", ".git"))
    shutil.copy(HERE, bundle / "code/vibes_continuum.py")
    shutil.copy(AREA / "output/tables/vibes_lot241_20261026/bundle/requirements.txt", bundle / "requirements.txt")
    files = {p.name: _sha(p) for p in sorted(bundle.glob("*.np*")) if p.is_file()}
    files["rows.csv"] = _sha(bundle / "rows.csv")
    (bundle / "MANIFEST.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_rows": len(rows),
        "by_group": rows.group.value_counts().to_dict(), "variants": VARIANTS, "batch": BATCH,
        "source": "decision bundle 2026-10-22 (raw spectra), production VIBES background (87 blanks, rank 70)",
        "files": files, "content_hash": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}, indent=1))
    print(f"bundle: {len(rows)} spectra x {len(VARIANTS)} variants = {len(rows) * len(VARIANTS)} fits")


# ---------------------------------------------------------------- vm (runs on Colab)
def _jobs(n_rows):
    """Interleave variants so every VM row range does both."""
    return [(v, r) for r in range(n_rows) for v in VARIANTS]


def vm(args):
    bundle, root = args.bundle, args.root
    sys.path.insert(0, str(bundle / "code/scripts"))
    from joblib import Parallel, delayed, parallel_config
    from threadpoolctl import threadpool_limits
    from vibes_baseline import VibesBackground, vibes_baseline_matrix

    root.mkdir(parents=True, exist_ok=True)
    _status(root, state="running", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cpus=os.cpu_count())
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for name, sha in manifest["files"].items():
        if _sha(bundle / name) != sha:
            raise ValueError(f"bundle file changed: {name}")
    wn = np.load(bundle / "wn.npy")
    z = np.load(bundle / "background_model.npz")
    raw = np.load(bundle / "raw.npy", mmap_mode="r")
    anc = np.load(bundle / "anchored.npy", mmap_mode="r")
    bgs = {v: VibesBackground(wn, m, W, pd.DataFrame(), tuple(), 0.0)
           for v, (m, W) in backgrounds(wn, z["mean"], z["components"]).items()}
    jobs = _jobs(manifest["n_rows"])
    lo, hi = (int(v) for v in args.rows.split(":")) if args.rows else (0, len(jobs))
    hi = min(hi, len(jobs))
    if lo % BATCH:
        raise ValueError(f"--rows start must be a multiple of {BATCH}")

    def one(v, r):
        y = np.asarray((raw if v == "deflate" else anc)[r], float)
        t = time.perf_counter()
        with threadpool_limits(limits=1):
            try:
                base, _, diag = vibes_baseline_matrix(wn, y[None], bgs[v], sample_ids=[f"{v}:{r}"], tau=0.1, loss="PB",
                                                      maxiter=10000)
                d = diag.iloc[0].to_dict()
                b = base[0].astype(np.float32)
            except Exception as exc:  # keep going; record the failure
                d, b = {"success": False, "error": repr(exc)[:300]}, np.full(len(wn), np.nan, np.float32)
        d.update(variant=v, row=int(r), seconds=round(time.perf_counter() - t, 2))
        return b, d

    ck = root / "checkpoints"
    ck.mkdir(exist_ok=True)
    start = time.perf_counter()
    workers = args.workers or os.cpu_count()
    for first in range(lo, hi, BATCH):
        part = jobs[first:min(first + BATCH, hi)]
        path = ck / f"batch_{first:06d}.npz"
        if path.exists():
            continue
        with parallel_config(backend="loky", inner_max_num_threads=1):
            res = Parallel(n_jobs=workers)(delayed(one)(v, r) for v, r in part)
        tmp = path.with_suffix(".partial")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, signature=np.array(manifest["content_hash"][:16]), first=first,
                                variant=np.array([v for v, _ in part]), row=np.array([r for _, r in part]),
                                baseline=np.stack([x[0] for x in res]),
                                diagnostics=np.array(json.dumps([x[1] for x in res], default=str)))
        os.replace(tmp, path)
        done = first + len(part)
        mins = (time.perf_counter() - start) / 60
        _status(root, state="running", done=done, first=lo, total=hi, elapsed_min=round(mins, 1),
                per_min=round((done - lo) / max(mins, 1e-6), 1))
        print(f"{done}/{hi} {mins:.1f} min", flush=True)
    _status(root, state="complete", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_seconds=round(time.perf_counter() - start), workers=workers)


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
    rows = pd.read_csv(OUT / "bundle/rows.csv")
    wn = np.load(OUT / "bundle/wn.npy")
    base = {v: np.full((len(rows), len(wn)), np.nan, np.float32) for v in VARIANTS}
    diag = []
    for f in sorted((OUT / "results").glob("*/run/checkpoints/batch_*.npz")):
        with np.load(f) as z:
            for v, r, b in zip(z["variant"], z["row"], z["baseline"]):
                base[str(v)][int(r)] = b
            diag.extend(json.loads(str(z["diagnostics"])))
    d = pd.DataFrame(diag).drop_duplicates(["variant", "row"])
    print(d.groupby("variant").agg(n=("row", "size"), failed=("success", lambda s: (~s.astype(bool)).sum()),
                                   median_s=("seconds", "median")))
    raw = np.load(OUT / "bundle/raw.npy", mmap_mode="r")
    anc = np.load(OUT / "bundle/anchored.npy", mmap_mode="r")
    for v in VARIANTS:
        src = raw if v == "deflate" else anc
        corrected = (np.asarray(src, np.float32) - base[v]).astype(np.float32)
        for g, sub in rows.groupby("group"):
            np.save(OUT / f"corrected_{v}_{g}.npy", corrected[sub.row.to_numpy()])
    d.to_csv(OUT / "fit_diagnostics.csv", index=False)
    print("corrected spectra written per variant and group")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--source", type=Path, required=True)
    a = sub.add_parser("vm"); a.add_argument("--bundle", type=Path, required=True); a.add_argument("--root", type=Path, required=True)
    a.add_argument("--rows", default=""); a.add_argument("--workers", type=int, default=0)
    a = sub.add_parser("bench"); a.add_argument("--gpu", default="L4")
    a = sub.add_parser("launch"); a.add_argument("--gpu", default="L4"); a.add_argument("--vms", type=int, default=2)
    a.add_argument("--skip", type=int, default=0, help="first job index (multiple of BATCH) to start from")
    a = sub.add_parser("watch"); a.add_argument("--poll", type=int, default=300)
    sub.add_parser("collect")
    args = p.parse_args()
    {"prepare": prepare, "vm": vm, "bench": bench, "launch": launch, "watch": watch, "collect": collect}[args.stage](args)


if __name__ == "__main__":
    main()
