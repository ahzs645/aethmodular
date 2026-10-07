"""Export physical-filter public SPARTAN chemistry for drawer reference populations.

Run: uv run python gallery/data/export_spartan_reference.py
Keeps public-source rows and known exclusion flags; never substitutes public BC
for a HIPS measurement. Carbon is the public FTIR product, not thermal EC/OC.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from common.spartan_io import RAW_DIR, build_datetime, normalize_fid, read_spartan_csv, site_from_path  # noqa: E402
sys.path.insert(0, str(ROOT / 'research/ftir_hips_chem/scripts'))
from config import SITES  # noqa: E402
from outliers import apply_exclusion_flags  # noqa: E402
from aethmodular_cli.env import display_path  # noqa: E402

PARAMETERS = {
    'EC (FTIR)': 'EC PM2.5', 'EC (ChemSpec FTIR)': 'EC PM2.5',
    'OC (FTIR)': 'OC PM2.5', 'OC (ChemSpec FTIR)': 'OC PM2.5', 'OM (FTIR)': 'OM PM2.5',
    'PM2.5 mass': 'Filter PM2.5 mass', 'Sulfate': 'Sulfate Ion PM2.5',
    'Nitrate': 'Nitrate Ion PM2.5', 'Ammonium': 'Ammonium Ion PM2.5',
    'Potassium': 'Potassium Ion PM2.5', 'Sodium': 'Sodium Ion PM2.5',
    'Calcium': 'Calcium Ion PM2.5', 'Magnesium': 'Magnesium Ion PM2.5',
    **{name: f'{name} PM2.5' for name in ['Iron', 'Aluminum', 'Titanium', 'Silicon', 'Zinc', 'Copper', 'Lead']},
}


def build_site(path: Path, site_name: str) -> pd.DataFrame:
    raw = read_spartan_csv(path)
    raw['id'] = normalize_fid(raw.Filter_ID)
    raw['date'] = build_datetime(raw).reindex(raw.index)
    # Metadata and measurements retain one row per physical filter. Repeat
    # reports of one parameter use the same mean convention as the network map.
    info = raw.groupby('id', sort=True).date.first().to_frame()
    for field, parameter in PARAMETERS.items():
        values = raw.loc[raw.Parameter_Name.eq(parameter), ['id', 'Value', 'Units', 'Analysis_Description']].copy()
        if parameter in {'EC PM2.5', 'OC PM2.5', 'OM PM2.5'}:
            if not values.Analysis_Description.fillna('').str.upper().eq('FTIR').all():
                raise ValueError(f'Non-FTIR carbon in {path.name}: {parameter}')
        units = values.Units.fillna('').str.lower()
        nano = units.str.contains(r'ng/m3|ng/m\^3|ng m-3|nanograms', regex=True)
        micro = units.str.contains(r'ug/m3|µg/m3|ug/m\^3|ug m-3|micrograms', regex=True)
        if (~(nano | micro)).any():
            raise ValueError(f'Unrecognized concentration unit in {path.name}: {field}')
        values['Value'] = pd.to_numeric(values.Value, errors='coerce')
        values.loc[nano, 'Value'] /= 1000
        info[field] = values.groupby('id').Value.mean().replace([np.inf, -np.inf], np.nan)
    info = info.reset_index()
    # Exclusions in the study registry are calendar dates; public starts carry
    # hours. Normalize before the canonical ±1-day comparison, as the gallery
    # filter pivot does, so an afternoon start at the boundary is not missed.
    info['date'] = info.date.dt.normalize()
    # Use the canonical flag API, matching the gallery export. Preserve flags
    # and reasons in the payload; reference selection removes flagged samples.
    info = apply_exclusion_flags(info, site_name)
    return info


def main():
    paths = sorted((RAW_DIR / 'FilterBased/ChemSpecPM25').glob('*.csv'))
    if not paths:
        raise FileNotFoundError('No local public SPARTAN PM2.5 chemistry files')
    focus = {cfg['code']: name for name, cfg in SITES.items()}
    rows, sources = [], {}
    for path in paths:
        code = site_from_path(path)
        site = focus.get(code, code).replace('_', ' ')
        with path.open('rb') as stream:
            sources[display_path(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
        data = build_site(path, focus.get(code, code))
        for r in data.to_dict('records'):
            rows.append(dict(id=r['id'], code=code, site=site,
                date=str(r['date'])[:10] if pd.notna(r['date']) else '',
                excluded=bool(r['is_excluded']), exclusion_reason=r['exclusion_reason'],
                **{field: float(r[field]) if pd.notna(r[field]) else None for field in PARAMETERS}))
    out = ROOT / 'gallery/app/public/data/spartan_reference.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema_version=1, generated=datetime.now(timezone.utc).isoformat(),
        source='Local mirror of public SPARTAN ChemSpecPM25', source_sha256=sources,
        note='Public PM2.5 chemistry, one row per physical filter over all available dates. EC/OC/OM are FTIR-derived public reports; the two EC and OC labels refer to the same product. HIPS and FTIR functional-group measurements are unavailable here. Known study exclusions are flagged; other sites retain public reporting quality decisions. No sidebar season/date subset is applied.',
        fields=list(PARAMETERS), field_units={f: 'µg/m³' for f in PARAMETERS},
        field_sources=PARAMETERS, n=len(rows), n_sites=len(paths), rows=rows),
        separators=(',', ':'), allow_nan=False))
    print(f'Wrote {out.relative_to(ROOT)}: {len(rows):,} physical filters at {len(paths)} sites')


if __name__ == '__main__':
    main()
