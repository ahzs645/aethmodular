"""Audited driver for the UNCHANGED official EPA PMF5 ME-2 distribution.

No executable, license key or sealed script is bundled or altered. Install/extract
EPA's distribution locally. This module can also be loaded directly for the
public-data Windows CI test without the rest of the scientific package.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

INSTALLER_URL = 'https://www.epa.gov/sites/default/files/2015-03/epa_pmf_5.0_setup.exe'
INSTALLER_SHA256 = '49e020d69e5b4656e4cd540508872c7fae2bfa69a48d2eef0b7abde7cbe8debe'
RUNTIME_HASHES = {
    'me2gfP4_1345c4.exe': 'bcf14aa6db0e8f8d33999006289c6cf01a97631ea497f9aa084f0b2d4f1fdf0a',
    'PMF_bs_6f8xx_sealed_GUI.ini': '0f22afa4a25ff96506077f258a1743206f1370da0abeb510af97fb43bcca2129',
    'me2key.key': 'bf5f671d893dbe3fb7b7d58bab706350127e5f96cb706d6dec1ab17e2b835bbe',
}
ENGINE, SCRIPT, KEY = tuple(RUNTIME_HASHES)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime(runtime):
    runtime = Path(runtime).resolve()
    for name, expected in RUNTIME_HASHES.items():
        if not (runtime/name).is_file() or sha256(runtime/name) != expected:
            raise ValueError('Missing or changed official EPA runtime file: ' + name)
    return runtime


def parameters(n, m, k, seed, robust=False, threshold=4., precision=20):
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (n,m,k,seed,precision)):
        raise ValueError('Dimensions, seed and precision must be integers')
    if not 1 <= k <= min(n,m) or not 1 <= seed < 2**31 or precision != 20:
        raise ValueError('Unsupported dimensions, seed or precision setting')
    if not isinstance(robust, bool) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Invalid robust setting')
    # Original iniparamv 161110 contract. c1=0 reads supplied U; c3=0 adds no
    # proportional uncertainty. contrun=0 starts anew. alowlim=0 enforces G>=0.
    return (f"161110\n0\n{int(robust)} {threshold:.17g} {threshold:.17g} 20 1 0\n"
            "1 11 0\n0 0 1 1.5 0 1\n0 0\n" + ' '.join([str(seed)]*5) +
            f"\n{n} {m} {k} 0 0 -14\n0 0 20 1.2 .35 1.1 .50\n"
            "'pmfdata.txt' 'unused.dat' 'native_result'\n0 1 0 -1 0\n161110\n")


def write_input(path, x, u):
    """ME-2 list-directed numeric stream, X then U; wrap below its 20k limit."""
    with Path(path).open('w', encoding='ascii', newline='\n') as f:
        for matrix in (x,u):
            flat = matrix.ravel(order='C')
            for start in range(0, len(flat), 8):
                f.write(' '.join(format(float(v), '.17g') for v in flat[start:start+8]) + '\n')
    # A streaming token read verifies arbitrary matrix sizes/line wrapping.
    values = np.fromiter((float(v) for line in Path(path).open(encoding='ascii') for v in line.split()), dtype=float)
    if not np.array_equal(values, np.concatenate([x.ravel(),u.ravel()])):
        raise ValueError('Native data serialization did not round-trip exactly')


def read_factors(path, n, m, k):
    tokens = Path(path).read_text(encoding='ascii').split()
    values = np.asarray([float(v.replace('D','E')) for v in tokens])
    if values.size != (n+m+1)*k or not np.isfinite(values).all():
        raise ValueError('Native factor output has wrong size or invalid numbers')
    rows = values.reshape(n+m+1,k)
    g,h,cc = rows[:n],rows[n:n+m].T,rows[-1]
    if np.any(g<0) or np.any(h<0) or not np.array_equal(cc,np.ones(k)):
        raise ValueError('Native factors violate the configured nonnegative AB model')
    return g,h


def verification(x,u,g,h,report):
    """Recompute Q(true), allowing only bounded native text-output rounding."""
    row = np.asarray([float(v.replace('D','E')) for v in report.split()])
    if len(row)!=10 or row[0]!=1 or not np.isfinite(row).all():
        raise ValueError('Unexpected native PMFreport format')
    residual = (x-g@h)/u
    q = float(np.sum(residual**2))
    # Original .dat uses five significant decimal digits. Each printed factor's
    # rounding error is <= 5e-5 * abs(value), conservatively. Propagate that
    # bound, rather than silently accepting an arbitrary percentage Q error.
    dg,dh = 5e-5*abs(g),5e-5*abs(h)
    prediction_error = (dg@abs(h) + abs(g)@dh + dg@dh)/u
    bound = float(np.sum(2*abs(residual)*prediction_error + prediction_error**2))
    native_q = float(row[5])
    bound += 1e-5*max(1.,abs(native_q))
    if abs(q-native_q)>bound:
        raise ValueError(f'Native reconstruction Q mismatch: {q} vs {native_q}; rounding bound {bound}')
    return {'Q_true':q,'native_Q_true':native_q,'native_Q_main':float(row[3]),
            'native_Q_aux':float(row[4]),'native_report':row.tolist(),
            'Q_rounding_bound':bound,'native_iterations':int(row[1]),
            'weighted_rmse_true':float(np.sqrt(np.mean(residual**2))),
            'fraction_abs_scaled_residual_gt3':float(np.mean(abs(residual)>3)),
            'Q_robust_excluded_4':float(np.sum(residual[abs(residual)<=4]**2))}


def run_one(x,u,runtime,output,*,factors=5,seed=1,robust=False,threshold=4.,timeout=600.):
    x,u=np.asarray(x,dtype=float),np.asarray(u,dtype=float)
    if x.ndim!=2 or x.shape!=u.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(u).all() or np.any(u<=0):
        raise ValueError('Native inputs must be finite aligned X/U with U>0')
    text=parameters(*x.shape,factors,seed,robust,threshold)
    if not np.isfinite(timeout) or timeout<=0:raise ValueError('Invalid native timeout')
    if os.name!='nt':raise RuntimeError('Native EPA ME-2 execution requires Windows')
    runtime=verify_runtime(runtime)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    receipt={'engine':'epa-pmf5-me2','version':'ME-2 1.345 / EPA PMF5 sealed 6f8xx',
             'runtime_sha256':dict(RUNTIME_HASHES),'status':'failed','native_engine_executed':False,
             'factors':factors,'seed':seed,'robust':robust,'threshold':threshold,
             'concentrations_shape':list(x.shape),'script_modified':False}
    try:
        with tempfile.TemporaryDirectory(prefix='aeth-me2-') as tmp:
            work=Path(tmp)
            # Runtime files never enter the result bundle or source repository.
            shutil.copy2(runtime/SCRIPT,work/SCRIPT);shutil.copy2(runtime/KEY,work/KEY)
            (work/'iniparams.txt').write_text(text,encoding='ascii')
            write_input(work/'pmfdata.txt',x,u)
            receipt['input_sha256']=sha256(work/'pmfdata.txt')
            receipt['parameters_sha256']=sha256(work/'iniparams.txt')
            try:
                with (work/'engine.stdout').open('w') as so,(work/'engine.stderr').open('w') as se:
                    proc=subprocess.Popen([str(runtime/ENGINE),SCRIPT],cwd=work,stdout=so,stderr=se)
                    receipt['native_engine_executed']=True
                    try:receipt['returncode']=proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        proc.kill();proc.wait();raise TimeoutError('Native fit exceeded configured timeout')
            finally:
                for name in ('iniparams.txt','pmfdata.txt','engine.stdout','engine.stderr',
                             'pmfstatus.txt','PMFreport.txt','native_result.dat','native_result.txt',
                             'native_result.rsd','ME2.LOG'):
                    if (work/name).is_file():shutil.copy2(work/name,out/name)
            stdout=(work/'engine.stdout').read_text(errors='replace')
            stderr=(work/'engine.stderr').read_text(errors='replace')
            status=(work/'pmfstatus.txt').read_text(errors='replace') if (work/'pmfstatus.txt').exists() else ''
            if receipt['returncode']!=0 or 'ERROR in using ME-2' in stdout or 'Fortran runtime error' in stderr or 'PMF=ready' not in status:
                raise RuntimeError('ME-2 did not complete a valid task; inspect retained native logs')
            g,h=read_factors(work/'native_result.dat',*x.shape,factors)
            metrics=verification(x,u,g,h,(work/'PMFreport.txt').read_text())
            converged='Iteration finished, convergence achieved.' in stdout
            receipt.update(status='passed' if converged else 'capped',metrics=metrics,
                           native_convergence_observed=converged,elapsed_seconds=time.monotonic()-started)
            np.savez_compressed(out/'native_factors.npz',G=g,H=h)
            receipt['output_sha256']={p.name:sha256(p) for p in out.iterdir() if p.is_file()}
            return g,h,receipt
    except (OSError,ValueError,RuntimeError,TimeoutError) as exc:
        receipt['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        (out/'native_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
