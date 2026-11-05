"""Strict numerical/identity contracts and self-verifying immutable snapshots."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .solver import array_sha256, validate_arrays

SCHEMA = "aethmodular-pmf/2"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def labels(values, name):
    a = np.asarray(values, dtype=str)
    if a.ndim != 1 or len(a) == 0 or any(not v.strip() for v in a) or len(set(a)) != len(a):
        raise ValueError(f'{name} must be a nonempty, unique, one-dimensional list of nonblank labels')
    return a


def read_matrix(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError('Matrix needs a header, sample IDs and at least one numeric column')
    header = labels(rows[0], 'CSV headers')
    ids, values = [], []
    for row in rows[1:]:
        if len(row) != len(header):
            raise ValueError('Ragged matrix row')
        ids.append(row[0])
        try:
            values.append([float(s) for s in row[1:]])
        except ValueError as exc:
            raise ValueError(f'Nonnumeric matrix entry for sample {row[0]!r}') from exc
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite matrix entry')
    return labels(ids, 'Sample IDs'), header[1:], array


def write_matrix(path, ids, features, values):
    ids, features, values = labels(ids, 'IDs'), labels(features, 'Features'), np.asarray(values)
    if values.shape != (len(ids), len(features)) or not np.isfinite(values).all():
        raise ValueError('Cannot export nonfinite or dimensionally inconsistent matrix')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', *features])
        writer.writerows([s, *[format(float(v), '.17g') for v in row]] for s, row in zip(ids, values))


def write_table(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def read_records(path, required=(), unique=None):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        labels(names, 'Metadata headers')
        if not set(required).issubset(names):
            raise ValueError(f'Missing columns: {sorted(set(required) - set(names))}')
        records = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in records):
        raise ValueError('Ragged metadata row')
    if unique is not None:
        labels([r[unique] for r in records], unique)
    return records


def fresh_directory(path):
    path = Path(path)
    # Even empty existing paths are rejected: callers cannot accidentally repurpose a run.
    path.mkdir(parents=True, exist_ok=False)
    return path


@dataclass
class Dataset:
    X: np.ndarray
    U: np.ndarray
    sample_ids: np.ndarray
    features: np.ndarray
    protocol: dict = field(default_factory=lambda: {'status': 'unknown'})
    metadata: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    def __post_init__(self):
        self.X, self.U = validate_arrays(self.X, self.U)
        self.sample_ids, self.features = labels(self.sample_ids, 'Sample IDs'), labels(self.features, 'Features')
        if self.X.shape != (len(self.sample_ids), len(self.features)):
            raise ValueError('Matrix dimensions and identities differ')
        if not isinstance(self.protocol, dict):
            raise ValueError('Protocol must be a JSON object')
        canonical_hash(self.protocol)
        for key, values in self.metadata.items():
            a = np.asarray(values, dtype=str)
            if a.shape != (len(self.X),):
                raise ValueError(f'Metadata {key} has wrong dimensions')
            self.metadata[key] = a

    @property
    def identity(self):
        return canonical_hash({'X': array_sha256(self.X), 'U': array_sha256(self.U),
            'sample_ids': self.sample_ids.tolist(), 'features': self.features.tolist(),
            'protocol': self.protocol, 'metadata': {k: v.tolist() for k, v in self.metadata.items()}, 'sources': self.sources})

    @property
    def compatibility(self):
        return canonical_hash({'features': self.features.tolist(), 'protocol': self.protocol})

    def subset(self, indices):
        return Dataset(self.X[indices], self.U[indices], self.sample_ids[indices], self.features,
            self.protocol.copy(), {k: v[indices] for k, v in self.metadata.items()}, self.sources.copy())

    def save(self, path):
        path = Path(path)
        meta = path.with_suffix('.json')
        if path.exists() or meta.exists():
            raise FileExistsError('Snapshot already exists')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, U=self.U, sample_ids=self.sample_ids, features=self.features,
            **{'meta_' + k: v for k, v in self.metadata.items()})
        write_json(meta, {'schema': SCHEMA, 'identity': self.identity, 'compatibility': self.compatibility,
            'array_file_sha256': sha256(path), 'protocol': self.protocol, 'sources': self.sources,
            'samples': len(self.X), 'features': self.X.shape[1]})


def from_csv(x_path, u_path, protocol=None):
    ids, features, x = read_matrix(x_path)
    uids, ufeatures, u = read_matrix(u_path)
    if not np.array_equal(ids, uids) or not np.array_equal(features, ufeatures):
        raise ValueError('X and U identities/order differ')
    return Dataset(x, u, ids, features, protocol or {'status': 'unknown'},
        sources={'X_csv_sha256': sha256(x_path), 'U_csv_sha256': sha256(u_path)})


def load_dataset(path, protocol=None):
    path = Path(path)
    sidecar = path.with_suffix('.json')
    info = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if not isinstance(info, dict):
        raise ValueError('Dataset sidecar must be a JSON object')
    with np.load(path, allow_pickle=False) as z:
        if 'features' in z:
            features = z['features']
        elif 'wavenumbers' in z:
            features = np.asarray([format(float(w), '.17g') for w in z['wavenumbers']])
        else:
            raise ValueError('Explicit feature identities are required')
        metadata = {k[5:]: z[k] for k in z.files if k.startswith('meta_')}
        for key in ('dates', 'volumes', 'lot_ids', 'analysis_ids', 'instrument_ids'):
            if key in z:
                metadata[key] = z[key]
        data = Dataset(z['X'], z['U'], z['sample_ids'], features,
            info.get('protocol', protocol or {'status': 'unknown'}), metadata,
            info.get('sources', {'legacy_npz_sha256': sha256(path)}))
    if info.get('schema') == SCHEMA:
        if info['array_file_sha256'] != sha256(path) or info['identity'] != data.identity:
            raise ValueError('Snapshot checksum/identity mismatch')
        if info['compatibility'] != data.compatibility:
            raise ValueError('Snapshot compatibility mismatch')
        if protocol is not None and protocol != data.protocol:
            raise ValueError('Cannot override a versioned snapshot protocol')
    elif info:
        raise ValueError('Unrecognized dataset sidecar schema')
    return data


def source_snapshot():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def archive_source(path):
    """Retain the exact executable source, vendor licenses and pin receipts."""
    import zipfile
    root = Path(__file__).parent
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                archive.write(p, arcname='src/pmf/' + str(p.relative_to(root)))
    return sha256(path)
