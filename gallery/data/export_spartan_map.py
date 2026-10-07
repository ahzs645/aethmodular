"""Export all available SPARTAN site locations and public PM2.5 filter summaries.

Run: uv run python gallery/data/export_spartan_map.py
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
from aethmodular_cli.env import display_path  # noqa: E402

FIELDS = {
    'PM2.5 mass': 'Filter PM2.5 mass',
    'EC (FTIR)': 'EC PM2.5',
    'OC (FTIR)': 'OC PM2.5',
    'BC (public optical product)': 'BC PM2.5',
    'Iron': 'Iron PM2.5',
    'Sulfate': 'Sulfate Ion PM2.5',
    'Nitrate': 'Nitrate Ion PM2.5',
}


def main():
    registry_path = ROOT / 'research/spartan/inventory/master_connections.csv'
    registry = pd.read_csv(registry_path).set_index('site')
    locations = pd.read_csv(ROOT / 'research/spartan/inventory/site_locations.csv').set_index('site')
    codes = set(registry.index) | {site_from_path(p) for p in RAW_DIR.rglob('*.csv')}
    files = {site_from_path(p): p for p in (RAW_DIR / 'FilterBased/ChemSpecPM25').glob('*.csv')}
    if not files:
        raise FileNotFoundError('No local SPARTAN ChemSpecPM25 files available')
    sites, sources = [], {}
    for path in [registry_path, ROOT / 'research/spartan/inventory/site_locations.csv']:
        with path.open('rb') as stream:
            sources[display_path(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    for code in sorted(codes):
        reg = registry.loc[code] if code in registry.index else pd.Series(dtype=object)
        name = reg.get('lookup_name', code)
        name = str(name) if pd.notna(name) else code
        lat, lon = reg.get('lookup_lat'), reg.get('lookup_lon')
        if code in locations.index:
            lat, lon = locations.loc[code, ['lat', 'lon']]
        stats = {label: dict(n=0, median=None, mean=None) for label in FIELDS}
        n, date_min, date_max = 0, '—', '—'
        if code in files:
            path = files[code]
            with path.open() as stream:
                for _ in range(5):
                    line = stream.readline().strip()
                    if line.startswith('# Site:'):
                        name = line.split(':', 1)[1].strip() or name
            with path.open('rb') as stream:
                sources[display_path(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
            data = read_spartan_csv(path)
            data['id'] = normalize_fid(data.Filter_ID)
            n = int(data.id.nunique())
            dates = build_datetime(data)
            if len(dates):
                date_min, date_max = str(dates.min())[:10], str(dates.max())[:10]
            lat_values = pd.to_numeric(data.Latitude, errors='coerce').dropna()
            lon_values = pd.to_numeric(data.Longitude, errors='coerce').dropna()
            if len(lat_values) and len(lon_values):
                lat, lon = float(lat_values.iloc[0]), float(lon_values.iloc[0])
            for label, parameter in FIELDS.items():
                values = data.loc[data.Parameter_Name.eq(parameter), ['id', 'Value', 'Units']].copy()
                values['Value'] = pd.to_numeric(values.Value, errors='coerce')
                ng = values.Units.str.contains('ng/m3', regex=False, na=False)
                values.loc[ng, 'Value'] /= 1000
                values = values.groupby('id').Value.mean()
                values = values[np.isfinite(values)]
                stats[label] = dict(n=len(values), median=float(values.median()) if len(values) else None,
                                    mean=float(values.mean()) if len(values) else None)
        sites.append(dict(code=code, name=name, location=name, lat=float(lat) if pd.notna(lat) else None,
                          lon=float(lon) if pd.notna(lon) else None, n_filters=n,
                          date_min=date_min, date_max=date_max, stats=stats))
    out = ROOT / 'gallery/app/public/data/spartan_map.json'
    out.write_text(json.dumps(dict(generated=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        note='All sites in the local SPARTAN inventory, including sites without public PM2.5 filter chemistry. Statistics use the available public ChemSpecPM25 data over its full date range, one value per physical filter. Hollow markers have no selected measurement. EC and OC are FTIR-derived; public BC is an optical product. Sidebar season, date and exclusion choices apply to the four-focus-site views.',
        field_units={label: 'µg/m³' for label in FIELDS}, sites=sites), separators=(',', ':'), allow_nan=False))
    print(f'Wrote {out.relative_to(ROOT)}: {len(sites)} sites, {sum(s["n_filters"] for s in sites):,} public PM2.5 filters')


if __name__ == '__main__':
    main()
