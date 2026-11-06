"""Per-filter screening with explicit retained scans and historical membership."""
from __future__ import annotations

from datetime import datetime
import math
from .io import read_records, write_table


def parse_date(value):
    text = str(value or '').strip()
    if text.lower() in ('', 'nan', 'none', 'nat'):
        return None
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).date().isoformat()
    except ValueError:
        pass
    for fmt in ('%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def valid_volume(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def evaluate(records, historical_ids=(), minimum_blanks=2, check_blanks=True):
    if isinstance(minimum_blanks, bool) or int(minimum_blanks) != minimum_blanks or minimum_blanks < 2:
        raise ValueError('Minimum blanks must be an integer >= 2')
    historical = set(historical_ids)
    groups = {}
    for row in records:
        key = (row.get('site', '').strip(), row.get('filter_id', '').strip())
        if not all(key):
            raise ValueError('site and physical filter_id are required')
        groups.setdefault(key, []).append(row)
    result = []
    for (site, fid), candidates in sorted(groups.items()):
        record = {'site': site, 'filter_id': fid, 'historical_102': fid in historical,
                  'analysis_id': '', 'status': 'held', 'reason': '',
                  'candidate_analysis_ids': ';'.join(r.get('analysis_id', '') for r in candidates)}
        retained = [r for r in candidates if r.get('analysis_status', '').strip().lower() == 'retained']
        if len(retained) != 1 or not retained[0].get('analysis_id', '').strip():
            record['reason'] = 'expected exactly one identified retained analysis'
        else:
            r = retained[0]
            record.update(analysis_id=r['analysis_id'], status='eligible', date=parse_date(r.get('date')) or '',
                          volume_m3=r.get('volume_m3', ''), lot=r.get('lot', ''),
                          instrument=r.get('instrument', ''), batch=r.get('batch', ''))
            fraction = r.get('particle_fraction', '').upper().strip()
            exclusion = r.get('exclusion_reason', '').strip()
            if exclusion:
                record.update(status='excluded', reason=exclusion)
            elif fraction in ('FB', 'FIELD BLANK'):
                record.update(status='blank', reason='field blank, not ambient observation')
            elif fraction != 'PM2.5':
                record.update(status='excluded', reason='not PM2.5')
            elif not record['date'] or not valid_volume(r.get('volume_m3')):
                record.update(status='held', reason='invalid/missing sampling date or measured volume')
            if record['status'] in ('eligible', 'blank') and str(r.get('lot', '')).strip().lower() in ('', 'nan', 'none'):
                record.update(status='held', reason='unknown filter lot')
            if record['status'] == 'eligible' and check_blanks:
                try:
                    count = int(r.get('blank_count', ''))
                except (ValueError, TypeError):
                    count = 0
                if count < minimum_blanks:
                    record.update(status='held', reason='insufficient matching blanks')
        result.append(record)
    return result


def audit_csv(metadata, output, historical_ids=(), minimum_blanks=2):
    from pathlib import Path
    if Path(output).exists():
        raise FileExistsError('Audit already exists')
    records = read_records(metadata, required=('filter_id', 'site', 'particle_fraction', 'analysis_id',
        'analysis_status', 'date', 'volume_m3', 'lot'))
    result = evaluate(records, historical_ids, minimum_blanks)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_table(output, result)
    return result
