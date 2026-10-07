"""Reapply the fixed external pilot on rank-70 full-range corrections.

Sampling pairs, exclusions, matching and model selection remain fixed. EC
models use shared channels, after VIBES correction on 4000–500 cm⁻¹.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / 'scripts'))
from vibes_current import MIGRATION, load_completed, prepare_shared_run, receipt, sha  # noqa: E402
from vibes_baseline import VibesBackground, vibes_baseline_matrix  # noqa: E402


def main():
    spec = importlib.util.spec_from_file_location('pilot', AREA / 'workflows/run_adama_bishoftu_frozen_pilot.py')
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    data = load_completed()
    shared_run = prepare_shared_run()
    out = MIGRATION / 'external'
    sources = dict(data['sources'])
    for path in [pilot.ADAMA_SPECTRA, pilot.ETBI_SPECTRA, pilot.ADAMA_CARBON, pilot.ADAMA_LEDGER, pilot.ETBI_REFERENCE]:
        # External data paths are displayed with the central resolver.
        from aethmodular_cli.env import display_path
        sources[display_path(path)] = sha(path)
    receipt(out, sources)
    audit = json.loads((AREA / 'output/tables/vibes_fullrange/failure_check_20261002/audit.json').read_text())
    background = ROOT / audit['original_result'] / 'background_model.npz'
    if sha(background) != audit['background_sha256']:
        raise ValueError('Frozen rank-70 background changed')
    wn = data['grids']['VIBES-full-r70']
    with np.load(background) as z:
        if not np.array_equal(z['wn'], wn) or z['components'].shape[1] != 70:
            raise ValueError('External correction background/grid mismatch')
        bg = VibesBackground(wn, z['mean'], z['components'], pd.DataFrame(), tuple(z['blank_ids'].astype(str)), 0.)
    adama = pd.read_csv(pilot.ADAMA_SPECTRA).rename(columns={'Wavelength': 'SampleAnalysisId'})
    adama_ids, adama_x = pilot.on_grid(adama, 'SampleAnalysisId', wn)
    etbi = pd.read_csv(pilot.ETBI_SPECTRA)
    etbi_ids, etbi_x = pilot.on_grid(etbi, 'MediaId', wn)
    if (len(adama_ids), len(etbi_ids)) != (5, 26):
        raise ValueError('External pilot population changed')
    ids = [f'adama:{int(v)}' for v in adama_ids] + [f'etbi:{int(v)}' for v in etbi_ids]
    raw = np.vstack([adama_x, etbi_x])
    path = out / 'corrected_full.npy'
    values = np.load(path) if path.exists() else np.full(raw.shape, np.nan)
    if values.shape != raw.shape:
        raise ValueError('External resume shape changed')
    diagnostics = []
    diag_path = out / 'external_fit_diagnostics.csv'
    if diag_path.exists():
        diagnostics = pd.read_csv(diag_path).to_dict('records')
    with threadpool_limits(limits=1):
        for i, sid in enumerate(ids):
            if np.isfinite(values[i]).all():
                continue
            _, corrected, diag = vibes_baseline_matrix(wn, raw[i:i+1], bg, sample_ids=[sid], tau=.1, loss='PB', maxiter=20000)
            diagnostics.append(diag.iloc[0].to_dict())
            pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
            if not diag.success.all() or not np.isfinite(corrected).all():
                raise RuntimeError(f'External rank-70 correction did not converge: {sid}')
            values[i] = corrected[0]
            np.save(path, values)
            print(f'Completed external rank-70 correction {i+1}/{len(ids)}: {sid}', flush=True)
    shared = wn >= 1425
    original = AREA / 'output/tables/adama_bishoftu_frozen_pilot/external_corrected.npz'
    with np.load(original) as z:
        if z['ids'].tolist() != ids:
            raise ValueError('Spline external case identities differ')
        corrected = {'AIRSpec': z['AIRSpec'].copy(), 'VIBES': values[:, shared]}
    grid = wn[shared]
    if corrected['AIRSpec'].shape != corrected['VIBES'].shape:
        raise ValueError('External shared grid differs')
    pilot.RUN, pilot.OUT = shared_run, out
    models = {m: shared_run / f'pls_full_pool_{m}.npz' for m in corrected}
    pred_adama = {m: pilot.predict_loading(x[:5], models[m]) for m, x in corrected.items()}
    pred_etbi = {m: pilot.predict_loading(x[5:], models[m]) for m, x in corrected.items()}
    adama_rows, _ = pilot.adama_table(adama_ids, pred_adama)
    bish, matched = pilot.matched_sites(grid, {m: x[5:] for m, x in corrected.items()}, etbi_ids, pred_etbi)
    for path in [background, original, path, *models.values(), Path(__file__)]:
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    signature = json.loads((shared_run / 'RUN_MANIFEST.json').read_text())['signature']
    (out / 'manifest.json').write_text(json.dumps(dict(frozen_run_signature=signature, background_rank=70,
        correction_window=[4000,500], model_window=[4000,1425], source_sha256=sources,
        counts={'adama':len(adama_rows),'bishoftu':len(bish),'matched':len(matched)}), indent=2))
    spec = importlib.util.spec_from_file_location('pilot_export', ROOT / 'gallery/data/export_baseline_external_pilot.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.SOURCE = out
    exporter.DEST = ROOT / 'gallery/app/public/data/rank70'
    exporter.main()


if __name__ == '__main__':
    main()
