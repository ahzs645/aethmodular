"""The VIBES blank background for the Experimental tab's "VIBES on one spectrum" page.

uv run python gallery/data/export_vibes_background.py [--model background_model.npz] [--fits vibes_fit_diagnostics.csv]

Reads the 87-blank rank-70 background the 19 Oct network run used (its vibes_bundle) and writes
public/data/experimental/vibes_background.json and vibes_background.bin: the mean spectrum, then the
components row by row, as little-endian float64. The page solves the baseline in the browser with these,
so they are stored at full precision.

Also writes vibes_saved_fits.json: for each scan, the noise level the 19 Oct run stopped at, how many
optimiser steps it took, and the noise level of the 3 Nov converged run (from that run's diagnostics
table, which records both). With the saved noise level the page repeats the saved trace. Nothing is
fitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'research/ftir_hips_chem/output/tables/network_baselines_20261019/vibes_bundle/background_model.npz'
FITS = ROOT / 'research/ftir_hips_chem/output/tables/network_baselines_v2_20261103/vibes_fit_diagnostics.csv'
OUT = ROOT / 'gallery/app/public/data/experimental'
BANK = 'b87'      # the background exported here, as the diagnostics name it
EARLY = 30        # optimiser steps under which the 3 Nov tests call a saved fit an early stop
TAU = 0.1   # the network run's fixed tau


def saved_fits(path: Path) -> dict:
    d = pd.read_csv(path, dtype={'analysis_id': str})
    if d.analysis_id.duplicated().any():
        raise ValueError('the diagnostics table lists a scan twice')
    ok = d.saved_success.astype(bool) & np.isfinite(d.saved_sigma) & (d.saved_sigma > 0)
    early = d.premature_stop.astype(bool)
    if not (early == (ok & (d.saved_iterations < EARLY))).all():
        raise ValueError(f'premature_stop is no longer "saved fit under {EARLY} steps"')
    # the converged run changed background for the lot-241 scans, so its noise level there is another calculation
    same_bank = d.vibes_model == BANK

    def num(values, keep):
        return [float(f'{x:.9g}') if k else None for x, k in zip(values, keep)]

    return dict(
        schema_version=1,
        generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        source='research/ftir_hips_chem/output/tables/network_baselines_v2_20261103/vibes_fit_diagnostics.csv',
        run='19 Oct network run (original optimiser), with the 3 Nov converged run beside it',
        early_stop_below=EARLY, background=BANK,
        columns=dict(analysis_id=d.analysis_id.tolist(), model=d.vibes_model.tolist(), sigma=num(d.saved_sigma, ok),
                     iterations=[int(v) for v in d.saved_iterations], early_stop=[bool(v) for v in early],
                     sigma_converged=num(d.sigma, same_bank & d.success.astype(bool))),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, default=MODEL, help='background_model.npz of the network bundle')
    ap.add_argument('--fits', type=Path, default=FITS, help='vibes_fit_diagnostics.csv of the 3 Nov converged run')
    a = ap.parse_args(argv)
    z = np.load(a.model, allow_pickle=False)
    wn, mean, components = (np.asarray(z[k], dtype='<f8') for k in ('wn', 'mean', 'components'))
    p, c = components.shape
    if wn.shape != (p,) or mean.shape != (p,):
        raise ValueError('background mean, grid and components disagree on the number of channels')
    # the page rebuilds the grid from its two ends, so it must be evenly spaced
    even = np.linspace(wn[0], wn[-1], p)
    if np.abs(wn - even).max() > 1e-6 * abs(wn[0] - wn[-1]) / (p - 1):
        raise ValueError('background grid is not evenly spaced')
    if not (np.isfinite(mean).all() and np.isfinite(components).all()):
        raise ValueError('background has values that are not numbers')
    body = mean.tobytes() + np.ascontiguousarray(components).tobytes()
    sources = {}
    for name in z['blank_ids']:
        sources[str(name).split(':')[0]] = sources.get(str(name).split(':')[0], 0) + 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'vibes_background.bin').write_bytes(body)
    meta = dict(
        schema_version=1,
        generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        source='research/ftir_hips_chem/workflows/network_baselines_20261019.py (vibes_bundle/background_model.npz)',
        file='experimental/vibes_background.bin', bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
        layout='float64 little-endian: mean (n), then components row by row (n x rank)',
        grid=dict(first=float(wn[0]), last=float(wn[-1]), n=int(p)),
        rank=int(c), tau=TAU, blanks=sources,
        # RMS absorbance each component adds at unit weight: how small the later ones are
        component_rms=[float(f'{v:.4g}') for v in np.linalg.norm(components, axis=0) / np.sqrt(p)],
    )
    (OUT / 'vibes_background.json').write_text(json.dumps(meta, ensure_ascii=False, separators=(',', ':')))
    (OUT / 'vibes_saved_fits.json').write_text(json.dumps(saved_fits(a.fits), ensure_ascii=False, separators=(',', ':')))
    for name in ('vibes_background.json', 'vibes_background.bin', 'vibes_saved_fits.json'):
        print(f'  {(OUT / name).relative_to(ROOT)} ({(OUT / name).stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()
