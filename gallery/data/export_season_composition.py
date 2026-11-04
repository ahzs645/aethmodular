"""The 3 Nov season-or-composition study for the gallery's Experimental tab.

uv run python gallery/data/export_season_composition.py

Reads the tables of research/ftir_hips_chem/workflows/run_season_composition_20261103.py and writes
public/data/experimental/season_composition.json: Addis FTIR EC against the IMPROVE TOR yardstick by season and model,
the Kiremt/Belg effects under each covariate set, IMPROVE's organic sensitivity with independent TOR OC, the per-filter
Addis points, leave-one-season-out, Bishoftu and the spectral-window split. No refit here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research/ftir_hips_chem/output/tables/season_composition_20261103'
OUT = ROOT / 'gallery/app/public/data/experimental/season_composition.json'
FILTER_COLS = ['filter_id', 'date', 'season', 'lot', 'hips_ug', 'exp_tor_hips', 'oc_over_bc', 'rm_over_bc', 'pm25_ugm3',
               'so4_ugm3', 'nsk_ugm3']


def cols(df: pd.DataFrame, digits: int = 4) -> dict[str, list]:
    def clean(v):
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else round(float(v), digits)
        if isinstance(v, np.integer):
            return int(v)
        return v
    return {c: [clean(v) for v in df[c].tolist()] for c in df.columns}


def main():
    receipt = json.loads((RUN / 'receipt.json').read_text())
    models = receipt['models']
    f = pd.read_csv(RUN / 'addis_filters.csv', dtype={'lot': str})
    ratio = pd.DataFrame({m: f[m] / f.exp_tor_hips for m in models})
    filters = pd.concat([f[FILTER_COLS], ratio.add_prefix('r|')], axis=1)
    payload = dict(
        schema_version=1, generated=datetime.now(timezone.utc).isoformat(), source=str(RUN.relative_to(ROOT)),
        settings=receipt,
        medians=cols(pd.read_csv(RUN / 'season_medians.csv')),
        tertiles=cols(pd.read_csv(RUN / 'loading_tertiles.csv')),
        effects=cols(pd.read_csv(RUN / 'season_effects.csv')),
        composition=cols(pd.read_csv(RUN / 'composition.csv')),
        improve_slope=cols(pd.read_csv(RUN / 'improve_organic_slope.csv')),
        improve_bins=cols(pd.read_csv(RUN / 'improve_ocec_bins.csv')),
        leave_out=cols(pd.read_csv(RUN / 'leave_season_out.csv')),
        bishoftu=cols(pd.read_csv(RUN / 'bishoftu.csv')),
        windows=cols(pd.read_csv(RUN / 'season_windows.csv')),
        filters=cols(filters),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    print(f'  {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()
