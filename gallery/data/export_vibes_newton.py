"""The 3 Nov VIBES adoption tests for the gallery's Experimental tab.

uv run python gallery/data/export_vibes_newton.py

Reads the tables of research/ftir_hips_chem/workflows/vibes_newton_refit_20261103.py and the v2 run
(output/tables/network_baselines_v2_20261103), and writes public/data/experimental/vibes_newton.json: the L-BFGS-B
port against the saved run, the calibration rebuilt on Newton baselines, band-area changes, EC per changed filter
and the two independent checks on every Addis early stop. No refit here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'research/ftir_hips_chem/output/tables'
RUN = TABLES / 'vibes_newton_20261103'
V2 = TABLES / 'network_baselines_v2_20261103'
OUT = ROOT / 'gallery/app/public/data/experimental/vibes_newton.json'


def cols(df: pd.DataFrame, digits: int = 5) -> dict[str, list]:
    def clean(v):
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else round(float(v), digits)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    return {c: [clean(v) for v in df[c].tolist()] for c in df.columns}


def main():
    manifest = json.loads((V2 / 'MANIFEST.json').read_text())
    addis = pd.read_csv(RUN / 'addis_premature_checks.csv')
    n = len(addis)
    closer = int((addis.closer == 'newton').sum())
    flatter = int((addis.flat_newton < addis.flat_saved).sum())
    # b87 only: the 76 lot-241 analyses changed bank in v2, so their change is not the optimiser's
    d = pd.read_csv(V2 / 'changed_or_premature.csv')
    d = d[(d.vibes_model == 'b87') & (d.filter_type != 'FB')]
    ec = d[['filter_id', 'filter_key', 'code', 'saved_iterations', 'premature_stop', 'baseline_change_frac_rms',
            'ec_saved_baseline_frozen_model_ug', 'ec_newton_baseline_rebuilt_model_ug']].rename(columns={
                'ec_saved_baseline_frozen_model_ug': 'ec_saved', 'ec_newton_baseline_rebuilt_model_ug': 'ec_newton'})
    payload = dict(
        schema_version=1,
        generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        source='research/ftir_hips_chem/workflows/vibes_newton_refit_20261103.py',
        settings=dict(analyses=manifest['analyses'], certified=manifest['map_certified'],
                      premature=manifest['premature_stop_flagged'], lot241=manifest['lot241_overrides']),
        checks=dict(n=n, closer=closer, closer_p=round(binomtest(closer, n).pvalue, 3), flatter=flatter,
                    flatter_p=round(binomtest(flatter, n).pvalue, 3),
                    res_saved=round(float(addis.res_saved.abs().mean()), 3),
                    res_newton=round(float(addis.res_newton.abs().mean()), 3),
                    flat_saved=round(float(addis.flat_saved.median()), 5),
                    flat_newton=round(float(addis.flat_newton.median()), 5)),
        addis=cols(addis),
        port=cols(pd.read_csv(RUN / 'port/port_vs_saved.csv')),
        calibration=cols(pd.read_csv(RUN / 'calibration_scores.csv')),
        bands=cols(pd.read_csv(RUN / 'band_areas.csv')),
        featureless=cols(pd.read_csv(RUN / 'featureless_window.csv')),
        ec=cols(ec, 4),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    print(f'  {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()
