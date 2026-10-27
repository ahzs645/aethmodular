"""Lot-aware VIBES blank test: SPARTAN lot 241/241a field blanks in the blank bank (26 Oct 2026).

The gallery's VIBES background is the 87-blank bank (59 IMPROVE lots 248/251 + 28 Addis, rank 70,
``vibes_colab_full500_rank85_resume_20261001``). On the 19 Oct network run it corrected held-out
SPARTAN field blanks of lots 248/250/251/253 to ~3e-5 RMS, but lots 241/241a 10-15x worse. This
asks whether adding those lots' own blanks fixes that, without hurting other lots or eating signal.

Banks (each refit with the same blank-only leave-one-out one-SE rule, cap 85):

* ``b87``             the gallery bank (the refit reproduces the saved model exactly).
* ``lot241_spartan``  b87 + every SPARTAN lot 241/241a physical field blank (the candidate policy).
* ``lobo_<filter>``   lot241_spartan minus one lot-241/241a blank: that blank is the held-out test.
* ``ctrl251``         b87 + 9 lot-251 SPARTAN blanks from other sites: does *any* 9 more blanks help?
* ``*_r70``           each bank above with the rank fixed at 70 instead of chosen by the one-SE rule:
                      AUMN-0007-7 alone drops lot241_spartan's one-SE rank to 33 (74 without it).

Evaluation, all on SPARTAN field blanks never in the bank being tested (repeat analyses of one
filter are averaged into one physical blank; Addis blanks in b87 are left out everywhere):

* blank residue: a corrected blank should be ~0 (RMS over 4000-500, PTFE 1050-1250, C-H 2800-3000).
* spike recovery: the four-peak shape of ``vibes_large_run`` (1610, 1720, 2920, 3400 cm-1) added at
  0.01 / 0.05 / 0.15 to each blank; recovered = corrected(spiked) - corrected(blank), per peak and lot.
* the 58 lot-241/241a PM2.5 analyses under b87 and lot241_spartan: how much the correction moves.

Stages (from the repo root):

    uv run python research/ftir_hips_chem/workflows/vibes_lot241_20261026.py prepare --hips-bundle DIR
    uv run python research/ftir_hips_chem/workflows/vibes_lot241_20261026.py launch --vms 4   # Colab CLI, waits
    uv run python research/ftir_hips_chem/workflows/vibes_lot241_20261026.py collect
    uv run python research/ftir_hips_chem/workflows/vibes_lot241_20261026.py score
    uv run python research/ftir_hips_chem/workflows/vibes_lot241_20261026.py apply --hips-bundle DIR   # gallery traces
    # on a VM (launch does this):  python vibes_lot241_20261026.py vm --bundle DIR --root DIR --rows a:b

The account allows two Colab runtimes; an L4 does ~11-16 fits/min, this Mac ~27. The 26 Oct run
finished in ~2 h by also running ``vm`` locally on the tail of each VM's row range and ``colab
upload``-ing each finished ``checkpoints/batch_*.npz`` into that VM's checkpoints folder: the runner
skips batches that exist, so the VM's own results then hold every row.
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
ROOT = HERE.parents[3]
TABLES = ROOT / "research/ftir_hips_chem/output/tables"
OUT = TABLES / "vibes_lot241_20261026"
GALLERY_RUN = TABLES / "vibes_colab_full500_rank85_resume_20261001/persistent_results/full500-rank85-9345e818c61929fe"
STAGE = TABLES / "vibes_colab_bundle_full500/stage"
NETWORK = TABLES / "network_baselines_20261019/vibes_bundle"
NETWORK_RUN = TABLES / "network_baselines_20261019"
LOT241 = ("241", "241a")
AMPLITUDES = (0.0, 0.01, 0.05, 0.15)
PEAKS = ((0.7, 1610.0, 30.0), (0.5, 1720.0, 22.0), (1.0, 2920.0, 35.0), (0.8, 3400.0, 130.0))  # vibes_large_run
BANDS = {"all": (0, 1e9), "ptfe": (1050, 1250), "ch": (2800, 3000)}
MAX_COMPONENTS = 85
FIXED_RANK = 70   # b87's one-SE rank
BATCH = 32
SEED = 20261026


def spike_shape(wn):
    return sum(a * np.exp(-0.5 * ((wn - c) / w) ** 2) for a, c, w in PEAKS)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- prepare (local)

def _bank87():
    meta = pd.read_csv(GALLERY_RUN / "background_training_blanks.csv")
    pool = pd.read_csv(STAGE / "data/pool_metadata.csv")
    etad = pd.read_csv(STAGE / "data/etad_metadata.csv")
    px = np.load(STAGE / "data/pool_raw.npy", mmap_mode="r")
    ex = np.load(STAGE / "data/etad_raw.npy", mmap_mode="r")
    rows = []
    for r in meta.itertuples():
        src, frame = (pool, px) if r.source == "IMPROVE" else (etad, ex)
        key = "FilterId" if r.source == "IMPROVE" else "MediaId"
        hit = src.index[src[key].eq(r.FilterId)]
        if len(hit) != 1:
            raise ValueError(f"bank blank {r.sample_id}: {len(hit)} matches")
        rows.append(np.asarray(frame[hit[0]], float))
    return np.load(STAGE / "data/wn.npy"), np.stack(rows), list(meta.sample_id)


def prepare(args):
    sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
    from vibes_baseline import fit_vibes_background

    wn, bank87, ids87 = _bank87()
    saved = np.load(GALLERY_RUN / "background_model.npz")
    if list(map(str, saved["blank_ids"])) != ids87 or not np.array_equal(wn, saved["wn"]):
        raise ValueError("bank ids or grid differ from the gallery model")
    bg = fit_vibes_background(wn, bank87, blank_ids=ids87, max_components=MAX_COMPONENTS)
    if bg.components.shape != saved["components"].shape or np.abs(bg.components - saved["components"]).max() > 1e-10:
        raise ValueError("refit of the 87-blank bank does not reproduce the gallery model")
    print(f"87-blank refit reproduces the gallery model (rank {bg.components.shape[1]})")

    filters = pd.read_csv(Path(args.hips_bundle).expanduser() / "spartan_Filters.csv", low_memory=False)
    lots = filters.set_index("MediaId").ExternalLotId.astype(str)
    cases = pd.read_csv(NETWORK / "cases.csv", dtype=str)
    X = np.load(NETWORK / "X_model_grid.npy", mmap_mode="r")
    if not np.allclose(np.load(NETWORK / "background_model.npz")["wn"], wn):
        raise ValueError("network grid differs from the bank grid")
    cases["lot"] = cases.media_id.astype(int).map(lots)
    if cases.lot.isna().any():
        raise ValueError(f"{cases.lot.isna().sum()} analyses without a lot")

    # physical field blanks: repeat analyses of one MediaId averaged
    fb = cases[cases.filter_type.eq("FB")]
    in87 = {s.split(":")[1] for s in ids87 if s.startswith("etad:")}
    blanks, spectra = [], []
    for mid, g in fb.groupby("media_id", sort=True):
        rows = g.case_row.astype(int).to_numpy()
        blanks.append({"blank_row": len(blanks), "media_id": mid, "blank_id": f"spartan:{mid}", "code": g.code.iloc[0],
                       "filter_id": g.filter_id.iloc[0], "lot": g.lot.iloc[0], "n_analyses": len(rows),
                       "analysis_ids": ";".join(g.analysis_id), "case_rows": ";".join(map(str, rows)),
                       "qc": ";".join(sorted(set(g.qc))), "in_b87": g.code.iloc[0] == "ETAD" and mid in in87})
        spectra.append(np.asarray(X[rows], float).mean(axis=0))
    blanks = pd.DataFrame(blanks)
    S = np.stack(spectra)
    if blanks.in_b87.sum() != len(in87):
        raise ValueError(f"only {blanks.in_b87.sum()} of {len(in87)} Addis bank blanks found by MediaId")
    test = blanks[~blanks.in_b87]
    l241 = test[test.lot.isin(LOT241)]
    pool251 = test[test.lot.eq("251") & ~test.code.isin(l241.code)]
    ctrl = pool251.groupby("code").sample(1, random_state=SEED).sample(len(l241), random_state=SEED)

    # Each added-blank bank twice: the one-SE rank (as the gallery bank was chosen) and rank fixed at
    # b87's 70. One lot-241 blank (AUMN-0007-7, LOO error ~1,700x the median) drops the one-SE rank
    # to 33, so the fixed-rank twin separates "covers lot 241" from "simpler background".
    banks = {"b87": [], "lot241_spartan": list(l241.blank_row), "ctrl251": list(ctrl.blank_row)}
    for r in l241.itertuples():
        banks[f"lobo_{r.filter_id}"] = [b for b in l241.blank_row if b != r.blank_row]
    fixed = {}
    for name in [b for b in banks if b != "b87"]:
        twin = name.replace("lobo_", "lobo_r70_") if name.startswith("lobo_") else f"{name}_r70"
        banks[twin], fixed[twin] = banks[name], FIXED_RANK

    pm = cases[cases.filter_type.eq("PM2.5") & cases.lot.isin(LOT241)]
    jobs = []
    def add(bank, kind, ref, amps):
        jobs.extend({"bank": bank, "kind": kind, "ref": int(ref), "amplitude": a} for a in amps)
    for r in test.itertuples():
        add("b87", "blank", r.blank_row, AMPLITUDES)
        if r.lot not in LOT241:
            add("lot241_spartan", "blank", r.blank_row, AMPLITUDES)
            add("lot241_spartan_r70", "blank", r.blank_row, AMPLITUDES)
    for r in l241.itertuples():
        for bank in (f"lobo_{r.filter_id}", f"lobo_r70_{r.filter_id}", "ctrl251", "ctrl251_r70"):
            add(bank, "blank", r.blank_row, AMPLITUDES)
    for r in pm.itertuples():
        for bank in ("b87", "lot241_spartan", "lot241_spartan_r70"):
            add(bank, "pm25", r.case_row, (0.0,))
    jobs = pd.DataFrame(jobs).sample(frac=1, random_state=SEED).reset_index(drop=True)   # shards balance
    jobs.insert(0, "job", np.arange(len(jobs)))

    bundle = OUT / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "code/scripts").mkdir(parents=True)
    shutil.copy(ROOT / "research/ftir_hips_chem/scripts/vibes_baseline.py", bundle / "code/scripts")
    shutil.copytree(ROOT / "research/ftir_hips_chem/vendor/pyvibes", bundle / "code/vendor/pyvibes")
    shutil.copy(HERE, bundle / "code/vibes_lot241.py")
    shutil.copy(STAGE / "requirements.txt", bundle / "requirements.txt")
    np.save(bundle / "wn.npy", wn)
    np.save(bundle / "bank87.npy", bank87)
    np.save(bundle / "spartan_blanks.npy", S)
    np.save(bundle / "pm25.npy", np.asarray(X[pm.case_row.astype(int).to_numpy()], float))
    pm.assign(pm_row=np.arange(len(pm))).to_csv(bundle / "pm25.csv", index=False)
    jobs["ref"] = [int(np.flatnonzero(pm.case_row.astype(int).to_numpy() == r)[0]) if k == "pm25" else r
                   for k, r in zip(jobs.kind, jobs.ref)]
    blanks.to_csv(bundle / "spartan_blanks.csv", index=False)
    jobs.to_csv(bundle / "jobs.csv", index=False)
    (bundle / "banks.json").write_text(json.dumps({"ids87": ids87, "added_blank_rows": banks, "fixed_rank": fixed}, indent=1))
    files = {str(p.relative_to(bundle)): _sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    (bundle / "MANIFEST.json").write_text(json.dumps({
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_jobs": len(jobs),
        "jobs_by_bank": jobs.bank.value_counts().to_dict(), "n_physical_blanks": len(blanks),
        "n_test_blanks": len(test), "lot241_blanks": list(l241.filter_id), "ctrl251_blanks": list(ctrl.filter_id),
        "pm25_lot241_analyses": len(pm), "max_components": MAX_COMPONENTS, "amplitudes": AMPLITUDES, "peaks": PEAKS,
        "files": files, "content_hash": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()},
        indent=1) + "\n")
    print(f"{len(blanks)} physical field blanks ({len(test)} test, {blanks.in_b87.sum()} in b87); "
          f"lot 241/241a: {len(l241)} blanks, {len(pm)} PM2.5 analyses")
    print(jobs.bank.str.replace(r"lobo_.*", "lobo_*", regex=True).value_counts().to_string())
    print(f"{len(jobs)} jobs -> {bundle}")


# ---------------------------------------------------------------- vm (Colab)

def _status(root, **kw):
    path = root / "execution_status.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(kw)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(cur, indent=2))
    os.replace(tmp, path)


def vm(args):
    bundle, root = args.bundle, args.root
    sys.path.insert(0, str(bundle / "code/scripts"))
    from joblib import Parallel, delayed, parallel_config
    from threadpoolctl import threadpool_limits
    from vibes_baseline import VibesBackground, fit_vibes_background, vibes_baseline_matrix
    from vibes.interference_models.pca import pca

    root.mkdir(parents=True, exist_ok=True)
    _status(root, state="running", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for name, sha in manifest["files"].items():
        if _sha(bundle / name) != sha:
            raise ValueError(f"bundle file changed: {name}")
    wn = np.load(bundle / "wn.npy")
    bank87, S, P = (np.load(bundle / f) for f in ("bank87.npy", "spartan_blanks.npy", "pm25.npy"))
    blanks = pd.read_csv(bundle / "spartan_blanks.csv", dtype={"media_id": str})
    spec = json.loads((bundle / "banks.json").read_text())
    jobs = pd.read_csv(bundle / "jobs.csv")
    lo, hi = (int(v) for v in args.rows.split(":")) if args.rows else (0, len(jobs))
    hi = min(hi, len(jobs))
    if lo % BATCH:
        raise ValueError(f"--rows start must be a multiple of {BATCH}")
    mine = jobs.iloc[lo:hi]
    shape = spike_shape(wn)

    (root / "backgrounds").mkdir(exist_ok=True)
    bgs = {}
    with threadpool_limits(limits=args.workers):
        for bank in sorted(mine.bank.unique()):
            rows = spec["added_blank_rows"][bank]
            B = np.concatenate([bank87, S[rows]]) if rows else bank87
            ids = spec["ids87"] + list(blanks.blank_id.iloc[rows])
            if bank in spec["fixed_rank"]:   # same PCA as fit_vibes_background, rank not chosen by LOO
                mean, _, _, W = pca(B)
                bgs[bank] = VibesBackground(wn.copy(), mean, W[:, :spec["fixed_rank"][bank]], pd.DataFrame(), tuple(ids), 0.0)
            else:
                bgs[bank] = fit_vibes_background(wn, B, blank_ids=ids, max_components=MAX_COMPONENTS)
            np.savez_compressed(root / "backgrounds" / f"{bank}.npz", mean=bgs[bank].mean,
                                components=bgs[bank].components, blank_ids=np.asarray(ids, dtype=str))
            bgs[bank].cv_errors.to_csv(root / "backgrounds" / f"{bank}_rank_cv.csv", index=False)
            print(f"{bank}: {len(ids)} blanks, rank {bgs[bank].components.shape[1]}", flush=True)

    def one(j):
        y = (S[j.ref] if j.kind == "blank" else P[j.ref]) + j.amplitude * shape
        with threadpool_limits(limits=1):
            base, _, diag = vibes_baseline_matrix(wn, y[None], bgs[j.bank], sample_ids=[str(j.job)],
                                                  tau=0.1, loss="PB", maxiter=10000)
        r = diag.iloc[0].to_dict()
        r["job"] = int(j.job)
        return base[0].astype(np.float32), r

    ck = root / "checkpoints"
    ck.mkdir(exist_ok=True)
    start = time.perf_counter()
    for first in range(lo, hi, BATCH):
        part = jobs.iloc[first:min(first + BATCH, hi)]
        path = ck / f"batch_{first:06d}.npz"
        if path.exists():
            continue
        with parallel_config(backend="loky", inner_max_num_threads=1):
            res = Parallel(n_jobs=args.workers)(delayed(one)(j) for j in part.itertuples())
        tmp = path.with_suffix(".partial")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, signature=np.array(manifest["content_hash"][:16]), jobs=part.job.to_numpy(),
                                baseline=np.stack([r[0] for r in res]),
                                diagnostics=np.array(json.dumps([r[1] for r in res], default=str)))
        os.replace(tmp, path)
        done = first + len(part)
        _status(root, state="running", done=done, first=lo, total=hi, elapsed_min=round((time.perf_counter() - start) / 60, 1))
        print(f"{done}/{hi} {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    _status(root, state="complete", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_seconds=round(time.perf_counter() - start), workers=args.workers)


# ---------------------------------------------------------------- launch (local, Colab CLI)

def _colab(*argv, timeout=180, check=True):
    p = subprocess.run(["colab", *argv], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"colab {' '.join(argv[:2])}: {p.stdout[-800:]} {p.stderr[-800:]}")
    return p


def _exec(session, code, timeout=120):
    f = OUT / "launch" / f"exec_{session}.py"
    f.write_text(code)
    return _colab("exec", "-s", session, "-f", str(f), "--timeout", str(timeout), timeout=timeout + 60, check=False)


def _retry(fn, tries=3):
    for k in range(tries):
        try:
            return fn()
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            if k == tries - 1:
                raise
            print(f"  retry after {type(exc).__name__}", flush=True)


def launch(args):
    bundle = OUT / "bundle"
    (OUT / "launch").mkdir(parents=True, exist_ok=True)
    tarball = OUT / "launch/bundle.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(bundle, arcname="bundle")
    print(f"bundle {tarball.stat().st_size / 1e6:.1f} MB", flush=True)
    n_jobs = len(pd.read_csv(bundle / "jobs.csv"))

    sessions = []
    for k in range(args.vms):
        name = f"vl241-{k}"
        p = _colab("new", "-s", name, "--gpu", args.gpu, timeout=600, check=False)
        if p.returncode:
            print(f"{name}: not allocated ({(p.stdout + p.stderr).strip().splitlines()[-1:]})", flush=True)
            break
        sessions.append({"session": name, "t0": time.time()})
        print(f"{name}: allocated", flush=True)
    if not sessions:
        raise SystemExit("no Colab runtime allocated")
    per = -(-n_jobs // len(sessions) // BATCH) * BATCH
    for k, s in enumerate(sessions):
        s["rows"] = f"{k * per}:{min((k + 1) * per, n_jobs)}"
    (OUT / "launch/sessions.json").write_text(json.dumps(sessions, indent=1))

    def start(s):
        # pip and the runner go in one detached shell, so the exec call returns at once
        name = s["session"]
        _retry(lambda: _colab("upload", "-s", name, str(tarball), "/content/vl_bundle.tar.gz", timeout=150))
        setup = f"""
import os, subprocess, sys, tarfile
os.makedirs('/content/vl', exist_ok=True)
tarfile.open('/content/vl_bundle.tar.gz').extractall('/content/vl')
cmd = (f"{{sys.executable}} -m pip install -q -r /content/vl/bundle/requirements.txt && "
       f"{{sys.executable}} /content/vl/bundle/code/vibes_lot241.py vm --bundle /content/vl/bundle "
       f"--root /content/vl/run --rows {s['rows']} --workers {{os.cpu_count()}}")
p = subprocess.Popen(['bash', '-c', cmd], stdout=open('/content/vl/run.log', 'w'), stderr=subprocess.STDOUT,
                     start_new_session=True)
print('started', p.pid, os.cpu_count())
"""
        p = _exec(name, setup, timeout=120)
        return f"{name} rows {s['rows']}: {p.stdout.strip()[-200:]}"

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(sessions)) as pool:
        for line in pool.map(start, sessions):
            print(line, flush=True)
    watch(args)


def watch(args):
    sessions = json.loads((OUT / "launch/sessions.json").read_text())
    probe = ("import json, pathlib, subprocess\n"
             "p = pathlib.Path('/content/vl/run/execution_status.json')\n"
             "print('STATUS', p.read_text().replace(chr(10), ' ') if p.exists() else '{}')\n"
             "print('LOG', subprocess.run(['tail', '-n', '3', '/content/vl/run.log'], capture_output=True, text=True).stdout.replace(chr(10), ' | '))\n")
    pending = [s for s in sessions if not (OUT / "results" / s["session"] / "run/execution_status.json").exists()]
    while pending:
        for s in list(pending):
            name = s["session"]
            try:
                out = _exec(name, probe, timeout=60).stdout
            except subprocess.TimeoutExpired:   # a hung probe is not a failed run; ask again next round
                print(f"{time.strftime('%H:%M')} {name} probe timed out", flush=True)
                continue
            line = next((l for l in out.splitlines() if l.startswith("STATUS")), "STATUS {}")
            try:
                st = json.loads(line[7:])
            except json.JSONDecodeError:
                st = {}
            age = (time.time() - s["t0"]) / 60
            print(f"{time.strftime('%H:%M')} {name} {st.get('state', '?')} {st.get('done', '-')}/{st.get('total', '-')} "
                  f"({age:.0f} min since allocation)", flush=True)
            if st.get("state") == "failed" or (not st and "Traceback" in out):
                print(out[-1500:], flush=True)
                raise SystemExit(f"{name} failed; VM kept for inspection (colab exec -s {name})")
            if age > 50:
                print(f"  warning: {name} is near the 1 h runtime-token limit", flush=True)
            if st.get("state") == "complete":
                # detached tar (checkpoints are already compressed); a blocking exec timed out here on 26 Oct
                _exec(name, "import subprocess; subprocess.Popen(['bash', '-c', 'cd /content/vl && tar cf result.partial "
                            "run run.log && mv result.partial result.tar'], start_new_session=True)", timeout=60)
                while "result.tar\n" not in _colab("ls", "-s", name, "/content/vl", check=False, timeout=120).stdout + "\n":
                    time.sleep(10)
                dest = OUT / "launch" / f"{name}_result.tar"
                _retry(lambda: _colab("download", "-s", name, "/content/vl/result.tar", str(dest), timeout=300))
                with tarfile.open(dest) as tf:
                    tf.extractall(OUT / "results" / name)
                _colab("stop", "-s", name, check=False)
                print(f"{name}: downloaded and stopped", flush=True)
                pending.remove(s)
        if pending:
            time.sleep(args.poll)


# ---------------------------------------------------------------- collect / score (local)

def collect(args):
    bundle = OUT / "bundle"
    jobs = pd.read_csv(bundle / "jobs.csv")
    wn = np.load(bundle / "wn.npy")
    base = np.full((len(jobs), len(wn)), np.nan, np.float32)
    diag = []
    for f in sorted((OUT / "results").glob("*/run/checkpoints/batch_*.npz")):
        with np.load(f) as z:
            base[z["jobs"]] = z["baseline"]
            diag.extend(json.loads(str(z["diagnostics"])))
    d = pd.DataFrame(diag)
    if d.job.duplicated().any() or len(d) != len(jobs):
        raise ValueError(f"{len(d)} of {len(jobs)} jobs collected")
    ranks = []
    for f in sorted((OUT / "results").glob("*/run/backgrounds/*.npz")):
        z = np.load(f)
        ranks.append({"bank": f.stem, "vm": f.parts[-4], "n_blanks": len(z["blank_ids"]), "rank": z["components"].shape[1],
                      "mean_sha": hashlib.sha256(z["mean"].tobytes()).hexdigest()[:12]})
    ranks = pd.DataFrame(ranks)
    if ranks.groupby("bank")[["rank", "mean_sha"]].nunique().gt(1).any().any():
        raise ValueError("a bank was fitted differently on two VMs")
    np.save(OUT / "baselines.npy", base)
    d.sort_values("job").to_csv(OUT / "fit_diagnostics.csv", index=False)
    ranks.drop_duplicates("bank").drop(columns="vm").to_csv(OUT / "backgrounds.csv", index=False)
    print(f"{len(d)} jobs, {(~d.success.astype(bool)).sum()} failed fits")
    print(ranks.drop_duplicates("bank").to_string(index=False))


def _rms(v, wn, band):
    lo, hi = BANDS[band]
    m = (wn >= lo) & (wn <= hi)
    return np.sqrt(np.nanmean(v[..., m] ** 2, axis=-1))


def score(args):
    bundle = OUT / "bundle"
    jobs = pd.read_csv(bundle / "jobs.csv")
    blanks = pd.read_csv(bundle / "spartan_blanks.csv", dtype={"lot": str, "media_id": str})
    wn = np.load(bundle / "wn.npy")
    S, P = np.load(bundle / "spartan_blanks.npy"), np.load(bundle / "pm25.npy")
    base = np.load(OUT / "baselines.npy").astype(float)
    ok = pd.read_csv(OUT / "fit_diagnostics.csv").set_index("job").success.astype(bool)
    shape = spike_shape(wn)
    jobs["ok"] = jobs.job.map(ok)
    y = np.stack([(S[r] if k == "blank" else P[r]) + a * shape for k, r, a in zip(jobs.kind, jobs.ref, jobs.amplitude)])
    corr = y - base
    corr[~jobs.ok.to_numpy()] = np.nan
    jobs["family"] = jobs.bank.str.replace(r"^lobo_r70_.*", "lobo_r70", regex=True).str.replace(r"^lobo_(?!r70$).*", "lobo", regex=True)

    # blank residue
    bj = jobs[jobs.kind.eq("blank") & jobs.amplitude.eq(0)].copy()
    for band in BANDS:
        bj[f"rms_{band}"] = _rms(corr[bj.job], wn, band)
    bj = bj.merge(blanks[["blank_row", "code", "filter_id", "lot", "qc", "n_analyses"]], left_on="ref", right_on="blank_row")
    bj.to_csv(OUT / "blank_residue.csv", index=False)
    by_lot = (bj.groupby(["family", "lot"]).agg(n=("job", "size"), failed=("ok", lambda s: int((~s).sum())),
                                                rms_all=("rms_all", "median"), rms_ptfe=("rms_ptfe", "median"),
                                                rms_ch=("rms_ch", "median"), p90_all=("rms_all", lambda s: s.quantile(.9)))
              .reset_index())
    by_lot.to_csv(OUT / "blank_residue_by_lot.csv", index=False)

    # spike recovery: corrected(spiked) - corrected(blank), same bank
    parent = bj.set_index(["bank", "ref"]).job
    sj = jobs[jobs.kind.eq("blank") & jobs.amplitude.gt(0)].copy()
    sj["parent_job"] = [parent.get((b, r), -1) for b, r in zip(sj.bank, sj.ref)]
    rec = []
    for j in sj.itertuples():
        delta = corr[j.job] - corr[j.parent_job]
        inj = j.amplitude * shape
        r = {"job": j.job, "bank": j.bank, "family": j.family, "ref": j.ref, "amplitude": j.amplitude,
             "rel_rmse": np.sqrt(np.mean((delta - inj) ** 2)) / np.sqrt(np.mean(inj ** 2))}
        for a, c, w in PEAKS:
            m = np.abs(wn - c) <= 2 * w
            r[f"recovered_{int(c)}"] = delta[m].sum() / inj[m].sum()
        rec.append(r)
    rec = pd.DataFrame(rec).merge(blanks[["blank_row", "code", "filter_id", "lot"]], left_on="ref", right_on="blank_row")
    rec.to_csv(OUT / "spike_recovery.csv", index=False)
    cols = ["rel_rmse"] + [f"recovered_{int(c)}" for _, c, _ in PEAKS]
    rec.groupby(["family", "lot", "amplitude"])[cols].median().reset_index().to_csv(OUT / "spike_recovery_by_lot.csv", index=False)

    # lot 241/241a: held-out blank under each bank, paired by blank
    l241 = bj[bj.lot.isin(LOT241)].pivot_table(index=["filter_id", "code", "lot"], columns="family",
                                               values=["rms_all", "rms_ptfe"]).round(6)
    l241.to_csv(OUT / "lot241_heldout_paired.csv")

    # PM2.5 lot 241/241a: how far each lot-241 bank moves the correction from b87
    pj = jobs[jobs.kind.eq("pm25")]
    a = pj[pj.bank.eq("b87")].set_index("ref").job
    pm = pd.read_csv(bundle / "pm25.csv", dtype=str)
    moves = []
    for bank in ("lot241_spartan", "lot241_spartan_r70"):
        b = pj[pj.bank.eq(bank)].set_index("ref").job
        m = pd.DataFrame({"bank": bank, "ref": a.index, "filter_id": pm.filter_id.iloc[a.index].to_numpy(),
                          "lot": pm.lot.iloc[a.index].to_numpy()})
        for band in ("all", "ptfe"):
            m[f"change_{band}"] = _rms(corr[b.loc[a.index]] - corr[a.to_numpy()], wn, band)
            m[f"signal_{band}"] = _rms(corr[a.to_numpy()], wn, band)
        m["change_rel_all"] = m.change_all / m.signal_all
        moves.append(m)
    move = pd.concat(moves, ignore_index=True)
    move.to_csv(OUT / "pm25_lot241_change.csv", index=False)

    # reproducibility: b87 here vs the 19 Oct network run, single-analysis blanks only
    net = NETWORK_RUN / "vibes_baselines_model_grid.npy"
    repro = None
    if net.exists():
        X = np.load(NETWORK / "X_model_grid.npy", mmap_mode="r")
        nb = np.load(net, mmap_mode="r")
        single = bj[bj.bank.eq("b87") & bj.n_analyses.eq(1) & bj.ok]
        rows = [int(blanks.case_rows.iloc[r]) for r in single.ref]
        diff = corr[single.job.to_numpy()] - (np.asarray(X[rows], float) - np.asarray(nb[rows], float))
        repro = {"n": len(single), "median_rms_diff": float(np.nanmedian(_rms(diff, wn, "all"))),
                 "max_rms_diff": float(np.nanmax(_rms(diff, wn, "all")))}

    pd.set_option("display.width", 200)
    summary = {"backgrounds": pd.read_csv(OUT / "backgrounds.csv").to_dict("records"),
               "blank_residue_by_lot": by_lot.round(6).to_dict("records"),
               "spike_recovery_lot_amp": rec.groupby(["family", "lot", "amplitude"])[cols].median().round(3).reset_index().to_dict("records"),
               "pm25_lot241_change": move.groupby("bank")[["change_all", "change_ptfe", "change_rel_all"]].median().round(6).to_dict("index"),
               "reproduces_network_run": repro}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(pd.read_csv(OUT / "backgrounds.csv").to_string(index=False))
    print("\nblank residue (median RMS) by bank family and lot\n", by_lot.round(6).to_string(index=False))
    print("\nlot 241/241a held-out blanks, paired\n", l241.to_string())
    print("\nspike recovery (median) by family, lot, amplitude\n",
          rec.groupby(["family", "lot", "amplitude"])[cols].median().round(3).to_string())
    print("\nPM2.5 lot 241/241a vs b87 (median)\n", move.groupby("bank")[["change_all", "change_ptfe", "signal_all", "change_rel_all"]].median().round(5).to_string())
    print("\nreproduces 19 Oct network run (b87, single-analysis blanks):", repro)


def apply(args):
    """Re-correct every lot 241/241a analysis for the gallery's network VIBES traces.

    PM2.5 and PM10 analyses take ``lot241_spartan_r70``; a field blank takes its own leave-one-out
    bank (``lobo_r70_<filter>``) so its trace stays a held-out correction. Writes ``apply/`` with the
    corrected spectra on the model grid, per-analysis provenance, and a per-lot summary of the
    held-out blank test for the drawer (gallery/data/export_spartan_network.py reads it).
    """
    sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
    from joblib import Parallel, delayed, parallel_config
    from threadpoolctl import threadpool_limits
    from vibes_baseline import VibesBackground, vibes_baseline_matrix

    filters = pd.read_csv(Path(args.hips_bundle).expanduser() / "spartan_Filters.csv", low_memory=False)
    lots = filters.set_index("MediaId").ExternalLotId.astype(str)
    cases = pd.read_csv(NETWORK / "cases.csv", dtype=str)
    cases["lot"] = cases.media_id.astype(int).map(lots)
    X = np.load(NETWORK / "X_model_grid.npy", mmap_mode="r")
    wn = np.load(OUT / "bundle/wn.npy")
    blanks = pd.read_csv(OUT / "bundle/spartan_blanks.csv", dtype={"media_id": str, "lot": str})
    lobo = blanks.set_index("media_id").filter_id
    todo = cases[cases.lot.isin(LOT241)].copy()
    todo["bank"] = [f"lobo_r70_{lobo[m]}" if t == "FB" else "lot241_spartan_r70" for m, t in zip(todo.media_id, todo.filter_type)]
    bgs = {}
    for bank in todo.bank.unique():
        f = next((OUT / "results").glob(f"*/run/backgrounds/{bank}.npz"))
        z = np.load(f)
        bgs[bank] = VibesBackground(wn.copy(), z["mean"], z["components"], pd.DataFrame(), tuple(map(str, z["blank_ids"])), 0.0)

    def one(row, bank):
        with threadpool_limits(limits=1):
            base, _, diag = vibes_baseline_matrix(wn, np.asarray(X[row], float)[None], bgs[bank], sample_ids=[str(row)],
                                                  tau=0.1, loss="PB", maxiter=10000)
        return base[0], bool(diag.iloc[0].success)

    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=os.cpu_count())(delayed(one)(int(r), b) for r, b in zip(todo.case_row, todo.bank))
    rows = todo.case_row.astype(int).to_numpy()
    corrected = (np.asarray(X[rows], float) - np.stack([r[0] for r in res])).astype(np.float32)
    todo["success"] = [r[1] for r in res]
    todo["fit"] = "local"
    # a local non-convergence takes the Colab run's fit of the same input where it has one (the
    # optimizer path differs by platform): PM2.5 under the bank itself, a single-analysis blank under its LOBO bank
    jobs = pd.read_csv(OUT / "bundle/jobs.csv")
    pm = pd.read_csv(OUT / "bundle/pm25.csv", dtype=str).reset_index(drop=True)
    vm_base = np.load(OUT / "baselines.npy", mmap_mode="r")
    ok = pd.read_csv(OUT / "fit_diagnostics.csv").set_index("job").success.astype(bool)
    for i in np.flatnonzero(~todo.success.to_numpy()):
        r = todo.iloc[i]
        if r.filter_type == "PM2.5":
            ref, kind = int(pm.index[pm.analysis_id.eq(r.analysis_id)][0]), "pm25"
        else:
            b = blanks[blanks.media_id.eq(r.media_id)]
            if r.filter_type != "FB" or int(b.n_analyses.iloc[0]) != 1:
                continue
            ref, kind = int(b.blank_row.iloc[0]), "blank"
        job = jobs[jobs.bank.eq(r.bank) & jobs.kind.eq(kind) & jobs.ref.eq(ref) & jobs.amplitude.eq(0)].job
        if len(job) == 1 and ok[int(job.iloc[0])]:
            corrected[i] = np.asarray(X[int(r.case_row)], float) - vm_base[int(job.iloc[0])]
            todo.iloc[i, todo.columns.get_loc("success")] = True
            todo.iloc[i, todo.columns.get_loc("fit")] = "colab"
    todo["rank"] = [bgs[b].components.shape[1] for b in todo.bank]
    todo["n_blanks"] = [len(bgs[b].blank_ids) for b in todo.bank]

    # lots behind each background: IMPROVE lots from the bank table, Addis bank blanks by MediaId
    b87 = pd.read_csv(GALLERY_RUN / "background_training_blanks.csv")
    imp = b87[b87.source.eq("IMPROVE")].LotNumber.astype(int).astype(str)
    etad = b87[b87.source.eq("ETAD")].FilterId.astype(int).map(lots)
    lots87 = sorted(set(imp) | set(etad.dropna()))
    added = blanks[blanks.lot.isin(LOT241) & ~blanks.in_b87]
    models = {
        "b87": {"label": "87-blank background", "rank": 70, "n_blanks": 87, "lots": lots87,
                "blanks": f"{len(imp)} IMPROVE (lots {', '.join(sorted(set(imp)))}) + {len(etad)} Addis field blanks "
                          f"(lots {', '.join(sorted(set(etad.dropna())))})",
                "source": str(GALLERY_RUN.relative_to(ROOT))},
        "lot241_r70": {"label": "87 + lot 241/241a blanks", "rank": FIXED_RANK, "n_blanks": 87 + len(added),
                       "lots": sorted(set(lots87) | set(LOT241)),
                       "blanks": f"the 87-blank background's blanks + {len(added)} SPARTAN lot 241/241a field blanks "
                                 f"({', '.join(added.filter_id)}); rank fixed at {FIXED_RANK}",
                       "note": "Field blanks of lot 241/241a are corrected with the bank that leaves their own blank out.",
                       "source": f"{OUT.relative_to(ROOT)} (26 Oct 2026 lot test)"},
    }
    res_ = pd.read_csv(OUT / "blank_residue.csv", dtype={"lot": str})
    res_ = res_[res_.ok]
    lot_check = {}
    for lot, g in res_.groupby("lot"):
        used = "lobo_r70" if lot in LOT241 else "b87"
        cur, new = g[g.family.eq("b87")], g[g.family.eq(used)]
        lot_check[lot] = {"heldout_blanks": int(len(new)), "model": "lot241_r70" if lot in LOT241 else "b87",
                          "ptfe_rms_b87": float(cur.rms_ptfe.median()), "ptfe_rms": float(new.rms_ptfe.median()),
                          "rms_b87": float(cur.rms_all.median()), "rms": float(new.rms_all.median())}
    ap = OUT / "apply"
    ap.mkdir(exist_ok=True)
    np.save(ap / "corrected_model_grid.npy", corrected)
    todo[["case_row", "analysis_id", "media_id", "code", "filter_id", "filter_type", "lot", "bank", "rank", "n_blanks",
          "success", "fit"]].to_csv(ap / "cases.csv", index=False)
    (ap / "MANIFEST.json").write_text(json.dumps({
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "default_model": "b87",
        "override_model": "lot241_r70", "models": models, "lot_check": lot_check,
        "n_analyses": len(todo), "by_type": todo.filter_type.value_counts().to_dict(), "n_failed": int((~todo.success).sum()),
        "residue": "median RMS absorbance of held-out SPARTAN field blanks after correction; ptfe = 1050-1250 cm-1",
    }, indent=1) + "\n")
    print(todo.groupby(["filter_type", "bank"]).size().to_string())
    print(f"{len(todo)} lot 241/241a analyses re-corrected, {(~todo.success).sum()} failed -> {ap}")
    print("lots in the 87-blank background:", lots87)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--hips-bundle", default=os.getenv("AETHMODULAR_SPARTAN_HIPS_BUNDLE"),
                   help="native HIPS network export folder (has spartan_Filters.csv with ExternalLotId)")
    p = sub.add_parser("vm")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--rows", default="")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    for name in ("launch", "watch"):
        p = sub.add_parser(name)
        p.add_argument("--vms", type=int, default=4)
        p.add_argument("--gpu", default="L4", help="L4 VMs have the most vCPUs per runtime")
        p.add_argument("--poll", type=int, default=60)
    sub.add_parser("collect")
    sub.add_parser("score")
    p = sub.add_parser("apply")
    p.add_argument("--hips-bundle", default=os.getenv("AETHMODULAR_SPARTAN_HIPS_BUNDLE"))
    args = ap.parse_args()
    if args.stage == "vm":
        _status(args.root.mkdir(parents=True, exist_ok=True) or args.root, state="starting")
        try:
            vm(args)
        except BaseException as exc:
            _status(args.root, state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
    else:
        {"prepare": prepare, "launch": launch, "watch": watch, "collect": collect, "score": score, "apply": apply}[args.stage](args)


if __name__ == "__main__":
    main()
