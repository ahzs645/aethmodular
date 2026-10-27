"""The 26 Oct lot-aware VIBES blank test for the gallery's Experimental tab.

uv run python gallery/data/export_vibes_lots.py

Reads the scored run of research/ftir_hips_chem/workflows/vibes_lot241_20261026.py (stages through
`score` and `apply`) and writes public/data/experimental/vibes_lots.json: held-out field-blank residue
by lot and bank, the nine lot 241/241a blanks one by one with their corrected spectra, spike recovery,
the blank-PCA rank curves, the lot 241/241a PM2.5 changes, and the backgrounds. No refit here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research/ftir_hips_chem/output/tables/vibes_lot241_20261026'
OUT = ROOT / 'gallery/app/public/data/experimental/vibes_lots.json'
LOT241 = ('241', '241a')
# the banks a reader compares; the lot 241/241a rows read their leave-one-blank-out versions
FAMILIES = {'b87': 'Current (87 blanks)', 'ctrl251_r70': '+9 lot-251 blanks (control)',
            'lot241_spartan': '+9 lot-241/241a blanks, one-SE rank', 'lot241_spartan_r70': '+9 lot-241/241a blanks, rank 70',
            'lobo': '+9 lot-241/241a blanks, one-SE rank', 'lobo_r70': '+9 lot-241/241a blanks, rank 70',
            'ctrl251': '+9 lot-251 blanks, one-SE rank'}
SAME = {'lobo': 'lot241_spartan', 'lobo_r70': 'lot241_spartan_r70'}   # one series across the two lot groups


def cols(df, digits=None):
    return {c: [None if (isinstance(v, float) and not np.isfinite(v)) else (round(v, digits) if digits and isinstance(v, float) else v)
                for v in df[c].tolist()] for c in df.columns}


def main():
    manifest = json.loads((RUN / 'bundle/MANIFEST.json').read_text())
    jobs = pd.read_csv(RUN / 'bundle/jobs.csv')
    blanks = pd.read_csv(RUN / 'bundle/spartan_blanks.csv', dtype={'lot': str, 'media_id': str})
    wn = np.load(RUN / 'bundle/wn.npy')
    S = np.load(RUN / 'bundle/spartan_blanks.npy')
    base = np.load(RUN / 'baselines.npy', mmap_mode='r')
    ok = pd.read_csv(RUN / 'fit_diagnostics.csv').set_index('job').success.astype(bool)

    res = pd.read_csv(RUN / 'blank_residue.csv', dtype={'lot': str})
    res = res[res.ok]
    res['series'] = res.family.map(lambda f: SAME.get(f, f))
    by_lot = (res.groupby(['series', 'lot']).agg(n=('job', 'size'), rms_all=('rms_all', 'median'), rms_ptfe=('rms_ptfe', 'median'),
                                                 rms_ch=('rms_ch', 'median'), q25_ptfe=('rms_ptfe', lambda s: s.quantile(.25)),
                                                 q75_ptfe=('rms_ptfe', lambda s: s.quantile(.75))).reset_index())
    per241 = res[res.lot.isin(LOT241)][['series', 'filter_id', 'code', 'lot', 'rms_all', 'rms_ptfe', 'rms_ch']]

    spikes = pd.read_csv(RUN / 'spike_recovery.csv', dtype={'lot': str})
    spikes['series'] = spikes.family.map(lambda f: SAME.get(f, f))
    rec_cols = [c for c in spikes.columns if c.startswith('recovered_')] + ['rel_rmse']
    spike = spikes.groupby(['series', 'lot', 'amplitude'])[rec_cols].median().reset_index()
    spike['n'] = spikes.groupby(['series', 'lot', 'amplitude']).size().to_numpy()

    # the nine lot 241/241a blanks, corrected (amplitude 0) under the current, control and own-lot banks
    keep = {'b87': 'b87', 'ctrl251_r70': 'ctrl251_r70', 'lobo_r70': 'lot241_spartan_r70'}
    spectra = []
    for b in blanks[blanks.lot.isin(LOT241)].itertuples():
        entry = {'filter_id': b.filter_id, 'code': b.code, 'lot': b.lot, 'raw': np.round(S[b.blank_row] - np.median(S[b.blank_row]), 6).tolist(), 'corrected': {}}
        for bank in ('b87', 'ctrl251_r70', f'lobo_r70_{b.filter_id}'):
            j = jobs[jobs.bank.eq(bank) & jobs.kind.eq('blank') & jobs.ref.eq(b.blank_row) & jobs.amplitude.eq(0)].job
            if len(j) == 1 and ok[int(j.iloc[0])]:
                key = keep['lobo_r70' if bank.startswith('lobo_r70_') else bank]
                entry['corrected'][key] = np.round(S[b.blank_row] - base[int(j.iloc[0])], 6).tolist()
        spectra.append(entry)

    # one-SE rank: leave-one-blank-out error by number of components, from the VM's fitted backgrounds
    curves = []
    for bank, label in [('b87', 'Current 87 blanks'), ('lot241_spartan', '+9 lot-241/241a blanks'),
                        ('lobo_AUMN-0007-7', '+8, without AUMN-0007-7')]:
        f = next((RUN / 'results').glob(f'*/run/backgrounds/{bank}_rank_cv.csv'))
        cv = pd.read_csv(f)
        rank = np.load(f.with_name(f'{bank}.npz'))['components'].shape[1]
        curves.append({'bank': bank, 'label': label, 'rank': int(rank), 'components': cv.components.tolist(),
                       'mean_mse': cv.mean_mse.tolist(), 'se_mse': cv.se_mse.tolist()})

    pm = pd.read_csv(RUN / 'pm25_lot241_change.csv', dtype={'lot': str})
    pm = pm[pm.bank.eq('lot241_spartan_r70')]
    pm25 = pd.read_csv(RUN / 'bundle/pm25.csv', dtype=str)
    pm['code'] = pm25.code.iloc[pm.ref].to_numpy()
    pm['analysis_id'] = pm25.analysis_id.iloc[pm.ref].to_numpy()
    pm['date'] = pm25.date.iloc[pm.ref].to_numpy()

    backgrounds = pd.read_csv(RUN / 'backgrounds.csv')
    backgrounds = backgrounds[~backgrounds.bank.str.startswith('lobo_')].drop(columns='mean_sha')
    lobo = pd.read_csv(RUN / 'backgrounds.csv')
    lobo = lobo[lobo.bank.str.match(r'^lobo_(?!r70)')]
    summary = json.loads((RUN / 'summary.json').read_text())
    apply = json.loads((RUN / 'apply/MANIFEST.json').read_text())

    out = dict(schema_version=1, generated=datetime.now(timezone.utc).isoformat(), source=str(RUN.relative_to(ROOT)),
               settings=dict(bundle=manifest['content_hash'][:16], n_jobs=manifest['n_jobs'], amplitudes=manifest['amplitudes'],
                             peaks=manifest['peaks'], lot241_blanks=manifest['lot241_blanks'], ctrl251_blanks=manifest['ctrl251_blanks'],
                             n_failed=int((~ok).sum()), reproduces_network_run=summary['reproduces_network_run'],
                             lobo_ranks=dict(zip(lobo.bank.str.replace('lobo_', ''), lobo['rank'].astype(int))),
                             applied=dict(n=apply['n_analyses'], by_type=apply['by_type'], lots87=apply['models']['b87']['lots'])),
               series=FAMILIES, by_lot=cols(by_lot, 8), per241=cols(per241, 8), spike=cols(spike, 4), wn=np.round(wn, 2).tolist(),
               spectra=spectra, rank_curves=curves, pm25=cols(pm[['filter_id', 'code', 'lot', 'date', 'analysis_id', 'change_all', 'change_ptfe',
                                                                  'signal_all', 'signal_ptfe', 'change_rel_all']], 7),
               backgrounds=cols(backgrounds))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(',', ':'), allow_nan=False))
    print(f'{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {len(spectra)} blank spectra, {len(pm)} PM2.5 filters')


if __name__ == '__main__':
    main()
