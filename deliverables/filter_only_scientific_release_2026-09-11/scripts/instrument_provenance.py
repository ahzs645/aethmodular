"""Stage timestamped instrument exports while preserving unresolved provenance.

Source row numbers are zero-based CSV data records (header excluded). Keeping a
datum identifier does not prove that its channel value was never interpolated.
"""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from config import WAVELENGTHS_NM
    from interval_evidence import file_hash
    from active_interval_matching import StreamContract
except ImportError:
    from .config import WAVELENGTHS_NM
    from .interval_evidence import file_hash
    from .active_interval_matching import StreamContract


METADATA_COLUMNS = [
    "Serial number",
    "Time (UTC)",
    "Datum ID",
    "Session ID",
    "Data format version",
    "Firmware version",
    "App version",
    "Timezone offset (mins)",
    "Date local (yyyy/MM/dd)",
    "Time local (hh:mm:ss)",
    "Timebase (s)",
    "Status",
    "Readable status",
    "Optical config",
    "Tape position",
    "Flow setpoint (mL/min)",
    "Flow total (mL/min)",
    "Flow1 (mL/min)",
    "Flow2 (mL/min)",
]
CHANNEL_COLUMNS = [f"{c} {v}" for c in WAVELENGTHS_NM for v in ["BC1", "BC2", "BCc"]]
STAGE_VERSION = 2


def stage_instrument_csv(path, output_dir, site, chunksize=100_000):
    """Losslessly retain selected source fields plus parsed UTC/native channels.

    No gaps are filled, no rows are discarded and no correction or unit conversion
    is applied. Nullable observation flags remain unknown until evidence supplies
    them. The cache is accepted only after rehashing the full source and output.
    """
    path = Path(path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{site}_timestamped_inputs.parquet"
    sidecar = target.with_suffix(".json")
    source_hash = file_hash(path)
    if sidecar.exists() and target.exists():
        cached = json.loads(sidecar.read_text())
        if (
            cached.get("stage_version") == STAGE_VERSION
            and cached.get("source_file_hash") == source_hash
            and cached.get("staged_file_hash") == file_hash(target)
        ):
            return cached
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"Time (UTC)", *CHANNEL_COLUMNS}
    if not required <= set(header):
        raise ValueError(f"Instrument source lacks required channels/time: {path}")
    columns = [c for c in METADATA_COLUMNS + CHANNEL_COLUMNS if c in header]
    writer = None
    offset = 0
    missing_time = 0
    time_min = None
    time_max = None
    counts = {c: 0 for c in WAVELENGTHS_NM}
    unique = {
        c: set()
        for c in [
            "Serial number",
            "Firmware version",
            "App version",
            "Data format version",
            "Timebase (s)",
            "Optical config",
        ]
    }
    timestamp_parts = []
    ids = set()
    duplicate_ids = 0
    temporary = target.with_suffix(".partial.parquet")
    try:
        for raw in pd.read_csv(
            path, usecols=columns, dtype="string", keep_default_na=False, chunksize=chunksize
        ):
            staged = raw.copy()
            staged["timestamp_utc"] = pd.to_datetime(
                raw["Time (UTC)"], utc=True, errors="coerce", format="mixed"
            )
            staged["source_row"] = np.arange(offset, offset + len(raw), dtype=np.int64)
            staged["source_file_hash"] = source_hash
            staged["is_observed"] = pd.Series(pd.NA, index=raw.index, dtype="boolean")
            for channel in WAVELENGTHS_NM:
                values = pd.to_numeric(raw[f"{channel} BCc"], errors="coerce").astype("float64")
                staged[f"{channel}_bcc_native"] = values
                staged[f"{channel}_numeric_finite"] = np.isfinite(values)
                counts[channel] += int(np.isfinite(values).sum())
            missing_time += int(staged.timestamp_utc.isna().sum())
            valid_times = staged.timestamp_utc.dropna()
            if len(valid_times):
                lo, hi = valid_times.min(), valid_times.max()
                time_min = lo if time_min is None else min(lo, time_min)
                time_max = hi if time_max is None else max(hi, time_max)
                timestamp_parts.append(pd.DatetimeIndex(valid_times))
            for field in unique:
                if field in raw:
                    unique[field].update(raw[field].unique().tolist())
            if {"Serial number", "Session ID", "Datum ID"} <= set(raw):
                for key in raw[["Serial number", "Session ID", "Datum ID"]].itertuples(
                    index=False, name=None
                ):
                    if key in ids:
                        duplicate_ids += 1
                    else:
                        ids.add(key)
            table = pa.Table.from_pandas(staged, preserve_index=False)
            if writer is None:
                schema = table.schema.with_metadata(
                    {
                        **(table.schema.metadata or {}),
                        b"source_file": str(path).encode(),
                        b"source_sha256": source_hash.encode(),
                        b"observation_status": b"unresolved",
                        b"source_row_convention": b"zero-based CSV data row; header excluded",
                    }
                )
                writer = pq.ParquetWriter(temporary, schema, compression="zstd")
            writer.write_table(table)
            offset += len(raw)
        if writer is None:
            raise ValueError(f"Empty instrument export: {path}")
    finally:
        if writer is not None:
            writer.close()
    if file_hash(path) != source_hash:
        raise ValueError("Instrument source changed during staging")
    temporary.replace(target)
    timestamps = (
        timestamp_parts[0].append(timestamp_parts[1:]) if timestamp_parts else pd.DatetimeIndex([])
    )
    meta = dict(
        stage_version=STAGE_VERSION,
        site=site,
        source_file=str(path),
        source_file_hash=source_hash,
        source_bytes=path.stat().st_size,
        staged_file=str(target.resolve()),
        staged_file_hash=file_hash(target),
        source_rows=offset,
        invalid_timestamp_rows=missing_time,
        duplicate_timestamp_rows=int(timestamps.duplicated().sum()),
        duplicate_acquisition_ids=duplicate_ids,
        timestamp_min_utc=str(time_min),
        timestamp_max_utc=str(time_max),
        distinct_utc_minute_slots=int(timestamps.floor("min").nunique()) if len(timestamps) else 0,
        metadata_values={k: sorted(v) for k, v in unique.items()},
        finite_native_rows=counts,
        observation_provenance_status="unresolved",
        processing_history_status="cleaned_export_found; upstream_operations_unrecovered",
        source_row_convention="zero-based CSV data row; header excluded",
    )
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def processing_records(staged_sources, sites):
    """One explicit processing contract per site and channel, including gaps."""
    by_site = {s["site"]: s for s in staged_sources}
    rows = []
    for site in sites:
        source = by_site.get(site)
        for channel, wavelength in WAVELENGTHS_NM.items():
            contract = StreamContract(
                stream_id=f"{site}:{channel}:timestamped_csv", wavelength_nm=wavelength
            )
            rows.append(
                dict(
                    site=site,
                    channel=channel,
                    **asdict(contract),
                    original_channel=f"{channel} BCc",
                    staged_channel=f"{channel}_bcc_native",
                    source_file=source["source_file"] if source else None,
                    source_file_hash=source["source_file_hash"] if source else None,
                    staged_file=source["staged_file"] if source else None,
                    staged_file_hash=source["staged_file_hash"] if source else None,
                    source_rows=source["source_rows"] if source else 0,
                    source_metadata=json.dumps(source["metadata_values"], sort_keys=True)
                    if source
                    else "{}",
                    timestamp_parse_verified=source is not None
                    and source["invalid_timestamp_rows"] == 0,
                    duplicate_timestamp_rows=source["duplicate_timestamp_rows"] if source else None,
                    native_finite_rows=source["finite_native_rows"][channel] if source else 0,
                    configured_unit="ng/m3",
                    unit_evidence="repository configuration only; source header lacks unit",
                    wavelength_evidence="config.WAVELENGTHS_NM; channel naming consistent, export lacks wavelength",
                    cadence_evidence="Timebase field retained; nominal grid origin and acquisition timestamp role unresolved",
                    correction_history="unrecovered; BC1, BC2 and BCc source fields preserved",
                    interpolation_history="unrecovered; is_observed remains nullable unknown",
                    optical_conversion_status="unresolved; no instrument coefficient assumed from HIPS MAC_VALUE",
                    stream_status="timestamped_input_recovered_provenance_unresolved"
                    if source
                    else "original_observations_not_staged",
                    provenance_evidence_file=None,
                    provenance_evidence_hash=None,
                    observed_column="is_observed",
                    valid_column=f"{channel}_numeric_finite",
                )
            )
    return pd.DataFrame(rows)


def apply_processing_evidence(records, evidence_path):
    """Apply explicit reviewed stream contracts; a source hash binds every update.

    JSON is a list of records keyed by stream_id. Each requires source_file_hash,
    provenance_evidence_file/hash and a human-readable resolution_reason. It may
    point at a new staged parquet carrying documented boolean observation flags.
    Evidence hashes attest content identity; they do not automate scientific review.
    """
    out = records.copy()
    for item in json.loads(Path(evidence_path).read_text()):
        required = {
            "stream_id",
            "source_file_hash",
            "provenance_evidence_file",
            "provenance_evidence_hash",
            "resolution_reason",
            "evidence_scope",
            "scope_justification",
        }
        if not required <= set(item) or any(not item[k] for k in required):
            raise ValueError(
                "Processing evidence requires source identity and a documented resolution"
            )
        select = out.stream_id.eq(item["stream_id"])
        if select.sum() != 1:
            raise ValueError("Processing evidence stream must occur exactly once")
        original = out.loc[select].iloc[0]
        if item["evidence_scope"] not in {"full_export", "bounded_period"}:
            raise ValueError(
                "Reviewed stream evidence requires an explicit full-export or bounded-period scope"
            )
        contract_fields = StreamContract.__dataclass_fields__
        StreamContract(**{key: item.get(key, original[key]) for key in contract_fields})
        if original.source_file_hash != item["source_file_hash"]:
            raise ValueError("Processing evidence does not match staged source hash")
        if file_hash(item["provenance_evidence_file"]) != item["provenance_evidence_hash"]:
            raise ValueError("Processing evidence hash changed")
        stage = Path(item.get("staged_file", original.staged_file))
        stage_hash = item.get("staged_file_hash", original.staged_file_hash)
        if file_hash(stage) != stage_hash:
            raise ValueError("Evidence-linked observation table hash changed")
        for key, value in item.items():
            if key not in out:
                out[key] = None
            out.loc[select, key] = value
    return out
