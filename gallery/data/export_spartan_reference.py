"""Export physical-filter SPARTAN chemistry for the network views and drawer reference populations.

Run: uv run python gallery/data/export_spartan_reference.py (after `uv run aeth spartan pull`)
Keeps public-source rows and known exclusion flags. Every ChemSpecPM25 parameter is exported; an
unmapped parameter in a new pull stops the export rather than being dropped. Public 'BC PM2.5'
measured by HIPS (fAbs / 10) is its own field, 'BC (public HIPS)', never written into the in-house
'HIPS BC'; values estimated from the HIPS-SSR curve are a separate field. Carbon is the public FTIR
product, not thermal EC/OC. Elements measured by XRF/ICP-MS that share a name with an ion field
(Ca, K, Na, Mg) are labelled '(element)'. Filters dated after the export day are dropped (a
public date typo, e.g. ILHA-0216 in the October 2026 pull).

'Sampled volume' (m³, for the gallery's mass-per-filter units) is HIPS's measured volume where the
filter has HIPS, else the public hours sampled × 0.3 m³/h (5 L/min) for the SS5i samplers, which
matches HIPS's volume to a median ratio of 1.00 (5-95 %: 0.95-1.03) on 2,524 shared filters. Older SS4i
filters have no validated flow and get no volume.

FTIR functional groups (alkane CH, alcohol COH, carboxylic COOH, non-acid CO) come from the lab's
batch 2-3 deliverable. Its OM equals the public OM; the batch 4 deliverable's organic groups run
about 2x the public OM and the four-site file (its OC and EC agree), so batch 4 is not used.

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
    # the rest of the public release
    'Nitrite': 'Nitrite Ion PM2.5', 'Phosphate': 'Phosphate Ion PM2.5', 'Bromide': 'Bromide Ion PM2.5',
    **{name: f'{name} PM2.5' for name in ['Chromium', 'Manganese', 'Vanadium', 'Nickel', 'Cobalt', 'Cerium',
        'Arsenic', 'Cadmium', 'Selenium', 'Antimony', 'Barium', 'Lithium', 'Phosphorus', 'Rubidium',
        'Strontium', 'Tin', 'Sulfur', 'Chlorine']},
    **{f'{name} (element)': f'{name} PM2.5' for name in ['Calcium', 'Potassium', 'Sodium', 'Magnesium']},
    'BC (public SSR estimate)': 'BC PM2.5',
    'Equivalent BC (SSR)': 'Equivalent BC PM2.5',
}
# one public parameter, two fields: split by the method that produced the value
METHOD = {'BC (public HIPS)': 'HIPS', 'BC (public SSR estimate)': 'ESTIMATED FROM THE HIPS-SSR BC CURVE'}
# in-house HIPS, with the field names filters.json uses, so one axis spans every site
HIPS_COLUMNS = {'HIPS Fabs': 'Fabs', 'HIPS Fabs uncertainty': 'Uncertainty', 'HIPS MDL': 'MDL'}
HIPS_FIELDS = [*HIPS_COLUMNS, 'HIPS BC']
VOLUME = 'Sampled volume'
SS5I_M3_PER_HOUR = 0.3  # 5 L/min
HIPS_UNITS = {'HIPS Fabs': 'Mm⁻¹', 'HIPS Fabs uncertainty': 'Mm⁻¹', 'HIPS MDL': 'Mm⁻¹', 'HIPS BC': 'µg/m³'}
HIPS_CSV = ('Spartan', 'SPARTAN_HIPS_Batch1-51.v2.csv')
# FTIR functional groups, lab batch 2-3 (gallery field names, as in filters.json)
FG_COLUMNS = {'Alkane CH': 'aCH', 'Alcohol COH': 'aCOH', 'Carboxylic COOH': 'COOH', 'Non-acid CO': 'naCO'}
FG_FIELDS = list(FG_COLUMNS)
FG_CSV = ('EC-HIPS-Aeth Comparison', 'Data', 'Lab deliverables', 'FTIR EC',
          'SPARTAN_FTIR data batch 2 and 3 resubmitted with MDLs.csv')
LAB_FIELDS = [*HIPS_FIELDS, *FG_FIELDS]


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
    out['hips_volume'] = pd.to_numeric(out.Volume, errors='coerce')
    return path, out[['code', 'date', *HIPS_FIELDS, 'hips_volume']].replace([np.inf, -np.inf], np.nan)


def load_functional_groups() -> tuple[Path, pd.DataFrame]:
    """PM2.5 functional groups per physical filter; the sample name carries site, number and type."""
    path = maia_data_root().joinpath(*FG_CSV)
    raw = pd.read_csv(path)
    # e.g. 'Tensor II_SN151_S_AEAZ_0089_1_PM25_10_08_2020_0_csv'
    parts = raw['sample'].astype(str).str.extract(r'_([A-Z]{4})_(\d{4})_\d+_(PM2_?5)_')
    raw = raw[parts[2].notna()].assign(code=parts[0], id=parts[0] + '-' + parts[1])
    if raw.id.duplicated().any():
        raise ValueError(f'Repeated PM2.5 filters in {path.name}')
    out = raw.set_index('id')
    out['date'] = pd.to_datetime(out.date, errors='coerce')
    for field, column in FG_COLUMNS.items():
        out[field] = pd.to_numeric(out[column], errors='coerce')
    return path, out[['code', 'date', *FG_FIELDS]].replace([np.inf, -np.inf], np.nan)


def build_site(path: Path | None, site_name: str, labs: list[tuple[pd.DataFrame, list[str]]]) -> pd.DataFrame:
    """One row per physical filter: the public chemistry (when the site has a file) and the lab tables."""
    info = public_chemistry(path) if path is not None else pd.DataFrame(columns=['date', 'ss5i_volume', *PARAMETERS]).rename_axis('id')
    info['date'] = info.date.astype('datetime64[ns]')
    for lab, fields in labs:
        # a lab table joins on the physical filter; a filter only it has takes its sample date
        info = info.join(lab[fields], how='outer')
        info['date'] = info.date.fillna(lab.date.reindex(info.index))
    # HIPS's measured volume first, else the SS5i hours estimate (public_chemistry)
    hips_volume = labs[0][0].hips_volume.reindex(info.index)
    info[VOLUME] = hips_volume.fillna(info.get('ss5i_volume', pd.Series(np.nan, index=info.index)))
    # a filter needs a value (288 PM2.5 HIPS rows have no Fabs) and a date to be drawn
    info = info[info[[*PARAMETERS, *LAB_FIELDS]].notna().any(axis=1) & info.date.notna()]
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
    unmapped = set(raw.Parameter_Name.dropna()) - set(PARAMETERS.values())
    if unmapped:
        raise ValueError(f'Unmapped public parameters in {path.name}: {sorted(unmapped)}; add them to PARAMETERS')
    info = raw.groupby('id', sort=True).date.first().to_frame()
    first = raw.groupby('id', sort=True)
    hours = first.Hours_sampled.apply(lambda v: pd.to_numeric(v, errors='coerce').median())
    ss5i = first.Collection_Description.first().fillna('').str.contains('SS5i', case=False)
    info['ss5i_volume'] = (hours * SS5I_M3_PER_HOUR).where(ss5i & hours.gt(0))
    for field, parameter in PARAMETERS.items():
        values = raw.loc[raw.Parameter_Name.eq(parameter), ['id', 'Value', 'Units', 'Analysis_Description']].copy()
        if parameter in {'EC PM2.5', 'OC PM2.5', 'OM PM2.5'}:
            if not values.Analysis_Description.fillna('').str.upper().eq('FTIR').all():
                raise ValueError(f'Non-FTIR carbon in {path.name}: {parameter}')
        if field in METHOD:
            values = values[values.Analysis_Description.fillna('').str.strip().str.upper().str.startswith(METHOD[field])]
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
    fg_path, fg = load_functional_groups()
    focus = {cfg['code']: name for name, cfg in SITES.items()}
    fields = [*PARAMETERS, *LAB_FIELDS]
    rows, sources = [], {}
    for path in [*paths.values(), hips_path, fg_path]:
        with path.open('rb') as stream:
            sources[display_path(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    # every site with public chemistry, and any HIPS site without it
    codes = sorted(set(paths) | set(hips.code))
    stray = fg[~fg.code.isin(codes)]
    if len(stray):
        print(f'  ! {len(stray)} functional-group filters under unknown site codes left out: {sorted(stray.index)}')
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    for code in codes:
        site = focus.get(code, code).replace('_', ' ')
        data = build_site(paths.get(code), focus.get(code, code),
                          [(hips[hips.code.eq(code)], HIPS_FIELDS), (fg[fg.code.eq(code)], FG_FIELDS)])
        future = data[data.date > today]
        if len(future):
            print(f'  ! {code}: {len(future)} filters dated after today left out: {future.id.tolist()} {future.date.dt.date.tolist()}')
            data = data[data.date <= today]
        for r in data.to_dict('records'):
            # a measurement the filter lacks is left out of its row (most rows carry a fraction of the fields)
            rows.append(dict(id=r['id'], code=code, site=site,
                date=str(r['date'])[:10] if pd.notna(r['date']) else '',
                excluded=bool(r['is_excluded']), exclusion_reason=r['exclusion_reason'],
                **{field: float(r[field]) for field in PARAMETERS if pd.notna(r[field])},
                **{field: round(float(r[field]), 6) for field in LAB_FIELDS if pd.notna(r[field])},
                **({VOLUME: round(float(r[VOLUME]), 4)} if pd.notna(r[VOLUME]) else {})))
    out = ROOT / 'gallery/app/public/data/spartan_reference.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema_version=1, generated=datetime.now(timezone.utc).isoformat(),
        source='Local mirror of public SPARTAN ChemSpecPM25; SPARTAN_HIPS_Batch1-51.v2.csv (HIPS); SPARTAN FTIR batch 2-3 lab deliverable (functional groups)', source_sha256=sources,
        note='Public PM2.5 chemistry, one row per physical filter over all available dates. EC/OC/OM are FTIR-derived public reports; the two EC and OC labels refer to the same product. BC (public HIPS) is the public BC where HIPS measured it (fAbs / 10), kept separate from the in-house HIPS BC; BC (public SSR estimate) is the public BC estimated from the HIPS-SSR curve, and Equivalent BC (SSR) the smoke-stain reflectometer value. Elements sharing a name with an ion field are labelled (element). HIPS Fabs, its uncertainty and MDL, and HIPS BC (Fabs / MAC 10) come from HIPS batches 1–51 (PM2.5 filters; blanks and PM10 left out) wherever that table covers a site, including filters without public chemistry. FTIR functional groups come from the lab batch 2-3 deliverable (2020-01 to 2023-03; the batch 4 groups are about 2x the public OM and are not used). Known study exclusions are flagged; other sites retain public reporting quality decisions. No sidebar season/date subset is applied.',
        fields=[*fields, VOLUME], field_units={**{f: 'µg/m³' for f in [*PARAMETERS, *FG_FIELDS]}, **HIPS_UNITS, VOLUME: 'm³'},
        field_sources={**{f: p + (f' ({METHOD[f].lower()})' if f in METHOD else '') for f, p in PARAMETERS.items()},
                       **{f: f'{HIPS_CSV[1]}: {HIPS_COLUMNS.get(f, "Fabs / MAC")}' for f in HIPS_FIELDS},
                       **{f: f'{FG_CSV[-1]}: {c}' for f, c in FG_COLUMNS.items()},
                       VOLUME: f'{HIPS_CSV[1]}: Volume, else public Hours_sampled × {SS5I_M3_PER_HOUR} m³/h (SS5i)'},
        n=len(rows), n_sites=len(codes), rows=rows),
        separators=(',', ':'), allow_nan=False))
    with_hips = sum('HIPS Fabs' in r for r in rows)
    print(f'Wrote {out.relative_to(ROOT)}: {len(rows):,} physical filters at {len(codes)} sites; '
          f'HIPS on {with_hips:,} at {hips.code.nunique()} sites; functional groups on '
          f'{sum("Alkane CH" in r for r in rows):,} at {len({r["code"] for r in rows if "Alkane CH" in r})} sites')


if __name__ == '__main__':
    main()
