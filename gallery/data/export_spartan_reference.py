"""Export physical-filter public SPARTAN chemistry for drawer reference populations.

Run: uv run python gallery/data/export_spartan_reference.py
Keeps public-source rows and known exclusion flags. Public 'BC PM2.5' measured by HIPS
(fAbs / 10; SSR-curve estimates are dropped) is exported as its own field, 'BC (public HIPS)', never
written into the in-house 'HIPS BC'. Carbon is the public FTIR product, not thermal EC/OC.

HIPS (Fabs, its uncertainty and MDL, and BC = Fabs / MAC) comes from SPARTAN_HIPS_Batch1-51.v2.csv,
the same table behind the four focus sites' HIPS in filters.json (identical values there), for every
site it covers. PM2.5 filters only: field and lab blanks and PM10 filters are not paired with PM2.5
chemistry. Filters with HIPS but no public chemistry row are kept when they have a Fabs value and a
sample date, so each site's HIPS is complete.
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
from config import MAC_VALUE, SITES  # noqa: E402
from data_matching import add_base_filter_id  # noqa: E402
from data_paths import maia_data_root  # noqa: E402
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
    'BC (public HIPS)': 'BC PM2.5',
}
# in-house HIPS, with the field names filters.json uses, so one axis spans every site
HIPS_COLUMNS = {'HIPS Fabs': 'Fabs', 'HIPS Fabs uncertainty': 'Uncertainty', 'HIPS MDL': 'MDL'}
HIPS_FIELDS = [*HIPS_COLUMNS, 'HIPS BC']
HIPS_UNITS = {'HIPS Fabs': 'Mm⁻¹', 'HIPS Fabs uncertainty': 'Mm⁻¹', 'HIPS MDL': 'Mm⁻¹', 'HIPS BC': 'µg/m³'}
HIPS_CSV = ('Spartan', 'SPARTAN_HIPS_Batch1-51.v2.csv')


def load_hips() -> tuple[Path, pd.DataFrame]:
    """PM2.5 HIPS per physical filter (base id), with its sample date and the four HIPS fields."""
    path = maia_data_root().joinpath(*HIPS_CSV)
    raw = pd.read_csv(path)
    raw = add_base_filter_id(raw[raw.FilterType.eq('PM2.5')])
    if raw.base_filter_id.duplicated().any():
        raise ValueError(f'Repeated PM2.5 filters in {path.name}')
    out = raw.rename(columns={'base_filter_id': 'id', 'Site': 'code'}).set_index('id')
    out['date'] = pd.to_datetime(out.SampleDate, errors='coerce')
    for field, column in HIPS_COLUMNS.items():
        out[field] = pd.to_numeric(out[column], errors='coerce')
    out['HIPS BC'] = out['HIPS Fabs'] / MAC_VALUE
    return path, out[['code', 'date', *HIPS_FIELDS]].replace([np.inf, -np.inf], np.nan)


def build_site(path: Path | None, site_name: str, hips: pd.DataFrame) -> pd.DataFrame:
    """One row per physical filter: the public chemistry (when the site has a file) and its HIPS."""
    info = public_chemistry(path) if path is not None else pd.DataFrame(columns=['date', *PARAMETERS]).rename_axis('id')
    # HIPS joins on the physical filter; a filter with HIPS only takes the HIPS sample date
    info = info.join(hips[HIPS_FIELDS], how='outer')
    info['date'] = info.date.astype('datetime64[ns]').fillna(hips.date.reindex(info.index))
    # a HIPS-only filter needs a value (288 PM2.5 rows have no Fabs) and a date to be drawn
    info = info[info[[*PARAMETERS, *HIPS_FIELDS]].notna().any(axis=1) & (info.index.isin(hips.index) <= info.date.notna())]
    info = info.reset_index()
    # Exclusions in the study registry are calendar dates; public starts carry
    # hours. Normalize before the canonical ±1-day comparison, as the gallery
    # filter pivot does, so an afternoon start at the boundary is not missed.
    info['date'] = info.date.dt.normalize()
    # Use the canonical flag API, matching the gallery export. Preserve flags
    # and reasons in the payload; reference selection removes flagged samples.
    info = apply_exclusion_flags(info, site_name)
    return info


def public_chemistry(path: Path) -> pd.DataFrame:
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
        if parameter == 'BC PM2.5':
            # keep only HIPS measurements; drop values "Estimated from the HIPS-SSR BC curve" (method 220)
            values = values[values.Analysis_Description.fillna('').str.upper().eq('HIPS')]
        units = values.Units.fillna('').str.lower()
        nano = units.str.contains(r'ng/m3|ng/m\^3|ng m-3|nanograms', regex=True)
        micro = units.str.contains(r'ug/m3|µg/m3|ug/m\^3|ug m-3|micrograms', regex=True)
        if (~(nano | micro)).any():
            raise ValueError(f'Unrecognized concentration unit in {path.name}: {field}')
        values['Value'] = pd.to_numeric(values.Value, errors='coerce')
        values.loc[nano, 'Value'] /= 1000
        info[field] = values.groupby('id').Value.mean().replace([np.inf, -np.inf], np.nan)
    return info


def main():
    paths = {site_from_path(p): p for p in sorted((RAW_DIR / 'FilterBased/ChemSpecPM25').glob('*.csv'))}
    if not paths:
        raise FileNotFoundError('No local public SPARTAN PM2.5 chemistry files')
    hips_path, hips = load_hips()
    focus = {cfg['code']: name for name, cfg in SITES.items()}
    fields = [*PARAMETERS, *HIPS_FIELDS]
    rows, sources = [], {}
    for path in [*paths.values(), hips_path]:
        with path.open('rb') as stream:
            sources[display_path(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    # every site with public chemistry, and any HIPS site without it
    codes = sorted(set(paths) | set(hips.code))
    for code in codes:
        site = focus.get(code, code).replace('_', ' ')
        data = build_site(paths.get(code), focus.get(code, code), hips[hips.code.eq(code)])
        for r in data.to_dict('records'):
            rows.append(dict(id=r['id'], code=code, site=site,
                date=str(r['date'])[:10] if pd.notna(r['date']) else '',
                excluded=bool(r['is_excluded']), exclusion_reason=r['exclusion_reason'],
                **{field: float(r[field]) if pd.notna(r[field]) else None for field in PARAMETERS},
                **{field: round(float(r[field]), 6) if pd.notna(r[field]) else None for field in HIPS_FIELDS}))
    out = ROOT / 'gallery/app/public/data/spartan_reference.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema_version=1, generated=datetime.now(timezone.utc).isoformat(),
        source='Local mirror of public SPARTAN ChemSpecPM25; SPARTAN_HIPS_Batch1-51.v2.csv (HIPS)', source_sha256=sources,
        note='Public PM2.5 chemistry, one row per physical filter over all available dates. EC/OC/OM are FTIR-derived public reports; the two EC and OC labels refer to the same product. BC (public HIPS) is the public BC where HIPS measured it (fAbs / 10; values estimated from the HIPS-SSR curve are dropped), kept separate from the in-house HIPS BC. HIPS Fabs, its uncertainty and MDL, and HIPS BC (Fabs / MAC 10) come from HIPS batches 1–51 (PM2.5 filters; blanks and PM10 left out) wherever that table covers a site, including filters without public chemistry. FTIR functional groups are unavailable here. Known study exclusions are flagged; other sites retain public reporting quality decisions. No sidebar season/date subset is applied.',
        fields=fields, field_units={**{f: 'µg/m³' for f in PARAMETERS}, **HIPS_UNITS},
        field_sources={**PARAMETERS, **{f: f'{HIPS_CSV[1]}: {HIPS_COLUMNS.get(f, "Fabs / MAC")}' for f in HIPS_FIELDS}},
        n=len(rows), n_sites=len(codes), rows=rows),
        separators=(',', ':'), allow_nan=False))
    with_hips = sum(r['HIPS Fabs'] is not None for r in rows)
    print(f'Wrote {out.relative_to(ROOT)}: {len(rows):,} physical filters at {len(codes)} sites; '
          f'HIPS on {with_hips:,} at {hips.code.nunique()} sites')


if __name__ == '__main__':
    main()
