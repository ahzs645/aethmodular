"""Export SPARTAN's own PM2.5 mass reconstruction, filter by filter, for the composition donut map.

Run: uv run python gallery/data/export_spartan_composition.py (after `uv run aeth spartan pull`)

Source: the public FilterBased ReconstrPM25 files (SPARTAN data version 3.1.x), which carry SPARTAN's
reconstructed components per filter (method codes 615-620), with ChemSpecPM25 for potassium. The
gallery does not recompute the chemistry. It splits PM2.5 into the twelve slices of SPARTAN's network
composition figure, and they sum to the filter's PM2.5 mass exactly:

  Black carbon   BC PM2.5 (HIPS, or estimated from the HIPS-SSR curve; methods 219-221)
  TEO            Trace Element Oxides (ng/m3 in the file, converted to ug/m3)
  Dust           Fine Soil (Al, Si, Ca, Fe, Ti with the regional MAL and CF in the file)
  Sodium         Sea Salt / 2.54, the sea-salt sodium (Sea Salt = 2.54 [Na+]ss)
  Chloride       Sea Salt x 1.54 / 2.54, the rest of the sea salt (Malm et al. 1994)
  Potassium      Potassium Ion PM2.5 (ChemSpecPM25)
  Nitrate, Ammonium, Sulfate   the ion chromatography values (newer filters report ammonium as 'Ammonium',
                 Teflon NH4 plus 0.29 nylon NO3, method 621, instead of 'Ammonium Ion PM2.5'; never both)
  Water          particle-bound water at the weighing RH (35%), the remainder of SPARTAN's
                 'Residual Matter with OC' mass balance once the slices above and OC are taken out
  Org. carbon    OC PM2.5 (FTIR)
  Residual       Residual Matter with OC (mainly the non-carbon mass of the organics)

SPARTAN publishes no water parameter. 'Residual Matter with OC' is PM2.5 less water, OC, the ions,
sea salt, potassium, BC, TEO and soil, so water is what is left. Over the October 2026 pull it is
never meaningfully negative, with a median of 2% of PM2.5 and a maximum of 9%, matching SPARTAN's
filter-water measurements at 35% RH.

A filter is kept when SPARTAN computed its balance (PM2.5 mass and 'Residual Matter with OC' both
present). SPARTAN computes that balance with a missing OC, BC or potassium counted as zero (older
filters have no FTIR OC: Kelowna, Manila, Buenos Aires), so the gallery does the same and the residual
holds that mass; each site records how many filters lacked each. These filters reproduce the PM2.5
means printed in SPARTAN's network figure (Delhi 92, Jaipur 48, Addis Ababa 23). Filters dated after
the export day are dropped (a public date typo, e.g. ILHA-0216 in the October 2026 pull).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from common.spartan_io import RAW_DIR, build_datetime, read_spartan_csv, site_from_path  # noqa: E402
from aethmodular_cli.env import display_path  # noqa: E402

RECON = RAW_DIR / 'FilterBased' / 'ReconstrPM25'
CHEM = RAW_DIR / 'FilterBased' / 'ChemSpecPM25'
OUT = ROOT / 'gallery/app/public/data/spartan_composition.json'

# the public parameters each slice reads
SOURCE = {
    'pm': 'Filter PM2.5 mass', 'bc': 'BC PM2.5', 'teo': 'Trace Element Oxides', 'dust': 'Fine Soil',
    'ss': 'Sea Salt', 'no3': 'Nitrate Ion PM2.5', 'nh4': 'Ammonium Ion PM2.5', 'so4': 'Sulfate Ion PM2.5',
    'oc': 'OC PM2.5', 'rm': 'Residual Matter with OC',
}
AMMONIUM_621 = 'Ammonium'
# the balance itself; any other slice SPARTAN left out of it counts as zero, as in its balance
REQUIRED = ('pm', 'rm')
POTASSIUM = 'Potassium Ion PM2.5'
# the slices, in the figure's legend order, with its labels
SLICES = [('bc', 'Black Carbon'), ('teo', 'TEO'), ('dust', 'Dust'), ('na', 'Sodium'), ('cl', 'Chloride'),
          ('k', 'Potassium'), ('no3', 'Nitrate'), ('nh4', 'Ammonium'), ('so4', 'Sulfate'), ('water', 'Water'),
          ('oc', 'Org. Carbon'), ('rm', 'Residual')]
SEA_SALT_NA = 2.54  # Sea Salt = 2.54 [Na+]ss


def _long(path: Path) -> pd.DataFrame:
    raw = read_spartan_csv(path)
    raw['date'] = build_datetime(raw).reindex(raw.index)
    value = pd.to_numeric(raw['Value'], errors='coerce')
    # TEO is reported in ng/m3
    value = value.where(~raw['Units'].astype(str).str.startswith('Nano'), value / 1000)
    return raw.assign(value=value)[['Site_Code', 'Filter_ID', 'date', 'Latitude', 'Longitude', 'Parameter_Name', 'value']]


def _site_name(path: Path) -> str | None:
    with open(path, errors='replace') as f:
        for _ in range(5):
            line = f.readline()
            if line.startswith('# Site:'):
                return line.split(':', 1)[1].strip()
    return None


def main():
    recon_files = sorted(RECON.glob('FilterBased_ReconstrPM25_*.csv'))
    if not recon_files:
        sys.exit(f'No ReconstrPM25 files under {display_path(RECON)}: run `uv run aeth spartan pull` first')
    sources, sites, filters = {}, [], []
    dropped = {'no balance': 0, 'future date': 0}
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    for path in recon_files:
        code = site_from_path(path)
        chem_path = CHEM / f'FilterBased_ChemSpecPM25_{code}.csv'
        for p in (path, chem_path):
            if p.exists():
                with open(p, 'rb') as stream:
                    sources[display_path(p)] = hashlib.file_digest(stream, 'sha256').hexdigest()
        long = _long(path)
        wide = long.pivot_table(index='Filter_ID', columns='Parameter_Name', values='value', aggfunc='first')
        dates = long.groupby('Filter_ID')['date'].first()
        k = pd.Series(dtype=float)
        if chem_path.exists():
            chem = _long(chem_path)
            k = chem[chem['Parameter_Name'] == POTASSIUM].groupby('Filter_ID')['value'].first()
        lat, lon = long['Latitude'].dropna(), long['Longitude'].dropna()
        site = dict(code=code, name=_site_name(path) or code,
                    lat=round(float(lat.iloc[0]), 4) if len(lat) else None,
                    lon=round(float(lon.iloc[0]), 4) if len(lon) else None,
                    first=None, last=None, n_pm=int(wide.get(SOURCE['pm'], pd.Series(dtype=float)).notna().sum()), n=0,
                    missing={'oc': 0, 'bc': 0, 'k': 0, 'other': 0})
        sampled = dates.dropna()
        sampled = sampled[sampled <= today]
        if len(sampled):
            site['first'], site['last'] = sampled.min().date().isoformat(), sampled.max().date().isoformat()
        for fid, row in wide.iterrows():
            v = {key: row.get(name) for key, name in SOURCE.items()}
            if pd.isna(v['nh4']):
                v['nh4'] = row.get(AMMONIUM_621)
            v['k'] = k.get(fid)
            if any(pd.isna(v[key]) for key in REQUIRED) or pd.isna(dates.get(fid)):
                dropped['no balance'] += 1
                continue
            if dates[fid] > today:
                dropped['future date'] += 1
                continue
            for key in v:
                if pd.isna(v[key]):
                    site['missing'][key if key in ('oc', 'bc', 'k') else 'other'] += 1
                    v[key] = 0.0
            s = dict(bc=v['bc'], teo=v['teo'], dust=v['dust'], na=v['ss'] / SEA_SALT_NA,
                     cl=v['ss'] * (SEA_SALT_NA - 1) / SEA_SALT_NA, k=float(v['k']), no3=v['no3'], nh4=v['nh4'],
                     so4=v['so4'], oc=v['oc'], rm=v['rm'])
            s['water'] = v['pm'] - sum(s.values())
            filters.append([code, dates[fid].date().isoformat(), round(v['pm'], 3), *(round(s[key], 4) for key, _ in SLICES)])
            site['n'] += 1
        sites.append(site)
    newest = max(s['last'] for s in sites if s['last'])
    OUT.write_text(json.dumps(dict(
        schema_version=1, generated=datetime.now(timezone.utc).isoformat(),
        source='SPARTAN public FilterBased ReconstrPM25 (reconstructed components) and ChemSpecPM25 (potassium)',
        source_sha256=sources,
        note=(f"One row per filter with SPARTAN's mass balance; the slices sum to PM2.5 mass (ug/m3). Water is the "
              f"remainder of SPARTAN's 'Residual Matter with OC' balance; sodium and chloride split SPARTAN's sea salt "
              f'by its 2.54 sodium factor; an OC, BC or potassium value SPARTAN lacked counts as zero (its mass stays in '
              f'the residual), as in its balance. Filters left out: {dropped["no balance"]} without the balance, '
              f'{dropped["future date"]} dated after the export. Newest sample: {newest}.'),
        slices=[dict(key=key, label=label) for key, label in SLICES],
        columns=['site', 'date', 'pm', *(key for key, _ in SLICES)],
        sites=sites, filters=filters,
    ), separators=(',', ':')))
    print(f'wrote {display_path(OUT)}: {len(filters)} filters at {sum(1 for s in sites if s["n"])} sites ({dropped})')


if __name__ == '__main__':
    main()
