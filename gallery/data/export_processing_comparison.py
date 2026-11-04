"""ONA/CMA/DEMA on aligned MA350 minutes, without fitting a calibration.

uv run python gallery/data/export_processing_comparison.py --minute PATH
uv run python gallery/data/export_processing_comparison.py --sample

The sample is the older analysis repo's short, real Jacros record, explicitly
labelled as a sample. Without an override, read the canonical Jacros minute file.
Never substitute daily aggregates for minute-resolution processing.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from export_aeth_quality_minute import read_minutes
from export_research_spectral import Page
from src.data.qc.smoothing import dema, ona

OUT = ROOT / 'gallery/app/public/data/aethalometer/quality'
METHODS = ('Raw', 'ONA', 'CMA', 'DEMA')


def compare(df: pd.DataFrame, *, cma_window=3, dema_span=15, delta_atn=0.05):
    """Matched comparisons; no padding missing minutes with zero, or crossing gaps."""
    if cma_window < 1 or cma_window % 2 != 1:
        raise ValueError('CMA window must be a positive odd number')
    tables = {k: [] for k in ('trace', 'windows', 'lag', 'daily', 'agreement', 'summary', 'coverage')}
    for channel in ('Blue', 'IR'):
        raw = pd.to_numeric(df[f'{channel} BCc'], errors='coerce') / 1000
        atn = pd.to_numeric(df[f'{channel} ATN1'], errors='coerce')
        boundary = df['tu'].diff().dt.total_seconds().ne(60) | atn.diff().abs().gt(30) | atn.isna() | atn.shift().isna()
        for col in ('Session ID', 'Tape position'):
            if col in df:
                boundary |= df[col].ne(df[col].shift())
        segment = boundary.cumsum()
        finite_raw = raw[np.isfinite(raw)]
        tables['coverage'].append({'channel': channel, 'source_minutes': len(df),
            'finite_BCc': len(finite_raw), 'negative_BCc': int(finite_raw.lt(0).sum()),
            'minimum': finite_raw.min(), 'p01': finite_raw.quantile(.01),
            'median': finite_raw.median(), 'p99': finite_raw.quantile(.99), 'maximum': finite_raw.max()})
        ona_values, windows = ona(raw.to_numpy(), atn.to_numpy(), segment.to_numpy(), delta_atn)
        values = pd.DataFrame({'Raw': raw, 'ONA': ona_values,
            'CMA': raw.groupby(segment).transform(lambda s: s.rolling(cma_window, center=True, min_periods=1).mean()),
            'DEMA': dema(raw, segment, dema_span)}, index=df.index)
        # Every method is scored on the same finite observations.
        common = np.isfinite(values.to_numpy()).all(axis=1)
        values.loc[~common] = np.nan
        # A bounded, contiguous example; not a thinned trace that conceals spikes.
        for i in range(min(1440, len(df))):
            for method in METHODS:
                tables['trace'].append({'channel': channel, 'time': df['t'].iloc[i].isoformat(),
                                        'method': method, 'bc': values[method].iloc[i]})
        # Exact frequency counts preserve the full window distribution without
        # sending hundreds of thousands of repeated window lengths to a browser.
        for (points, reached), count in windows.groupby(['points', 'threshold_reached']).size().items():
            tables['windows'].append({'channel': channel, 'minutes': int(points),
                                      'threshold_reached': bool(reached), 'count': int(count)})
        for method in METHODS[1:]:
            for lag in range(-20, 21):
                # Positive lag: processed(t + lag) agrees with raw(t), i.e. delay.
                shifted = values[method].groupby(segment).shift(-lag)
                pairs = pd.DataFrame({'x': values.Raw, 'y': shifted}).dropna()
                corr = pairs.x.corr(pairs.y) if len(pairs) >= 3 and pairs.x.std() > 0 and pairs.y.std() > 0 else np.nan
                tables['lag'].append({'channel': channel, 'method': method, 'lag_minutes': lag,
                                      'correlation': corr, 'n': len(pairs)})
            difference = values[method] - values.Raw
            tables['summary'].append({'channel': channel, 'method': method, 'n': int(common.sum()),
                                      'bias': difference.mean(), 'RMSE': np.sqrt((difference ** 2).mean()),
                                      'raw_SD': values.Raw.std(), 'processed_SD': values[method].std()})
        # 09:00 local day labels, one common mask. Require >=720 matched minutes;
        # an incomplete demonstration sample must not invent a daily comparison.
        dates = (df['t'] - pd.Timedelta(hours=9)).dt.normalize()
        daily = values.groupby(dates).mean()
        counts = values.Raw.groupby(dates).count()
        daily = daily[counts >= 720]
        for date, row in daily.iterrows():
            for method in METHODS:
                tables['daily'].append({'channel': channel, 'date': date.isoformat(),
                                        'method': method, 'bc': row[method], 'n_minutes': int(counts.loc[date])})
            for a, b in combinations(METHODS, 2):
                tables['agreement'].append({'channel': channel, 'date': date.isoformat(),
                    'pair': f'{b} − {a}', 'mean': (row[a] + row[b]) / 2, 'difference': row[b] - row[a]})
    return {k: pd.DataFrame(v) for k, v in tables.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group()
    source.add_argument('--minute', type=Path)
    source.add_argument('--sample', action='store_true')
    args = ap.parse_args()
    if args.sample:
        path = ROOT.parent / 'aethdata-analysis/Sample Data/Jacros simplified.csv'
    elif args.minute:
        path = args.minute
    else:
        from data_paths import aethalometry_dir
        path = aethalometry_dir() / 'Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv'
    if not path.exists():
        raise SystemExit('Minute source unavailable; existing export left unchanged. Supply --minute PATH or --sample.')
    df = read_minutes(path, 'Africa/Addis_Ababa')
    p = Page('processing', 'ONA, CMA and DEMA processing comparison', out=OUT, exporter=Path(__file__))
    for key, frame in compare(df).items():
        p.add(key, frame, notebook='CMA DEMA ONA Comparison Polars', read='computed from minute CSV',
              path=path.name, scope='sample' if args.sample else 'supplied minute record')
    p.add('settings', pd.DataFrame([{'scope': 'Short Jacros sample' if args.sample else 'Jacros minute record',
        'first_utc': df.tu.min().isoformat(), 'last_utc': df.tu.max().isoformat(), 'minutes': len(df),
        'CMA_points': 3, 'DEMA_span_points': 15, 'ONA_delta_ATN': 0.05}]),
        notebook='CMA DEMA ONA Comparison Polars', read='processing settings', path=path.name)
    p.text['scope'] = ('This is the short Jacros sample, not the full Addis record. ' if args.sample else '') + (
        'UTC timestamps converted to Addis local time. BC in µg/m³; signed finite observations retained. '
        'All methods restart at tape/session changes, missing attenuation and minute gaps. '
        'CMA: centred 3-point mean; DEMA: span 15, adjust=False; ONA: ΔATN 0.05. '
        'Lag uses Pearson r on finite pairs at each lag; positive lag means processing delay. '
        'Daily plots require at least 720 common minutes per 09:00 local day.')
    p.text['scope'] += (' These are processing differences on the supplied cleaned CSV, '
        'before additional QC exclusions. Signed values and extremes are retained; '
        'inspect the input ranges before interpreting RMSE as a method ranking.')
    p.write()


if __name__ == '__main__':
    main()
