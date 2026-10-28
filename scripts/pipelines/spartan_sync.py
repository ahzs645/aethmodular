"""Reconcile a downloaded SPARTAN public file with the copy already on disk.

The public bucket is SPARTAN's live release, refreshed daily, and it is not always whole: on
2026-10-28 FilterBased_ChemSpecPM25_TWKA.csv held its header lines and no rows (175 Kaohsiung
filters) for part of the day, and ILHA-0216 starts on 2026-11-17, after the file was updated
(2026-10-13). Installing what the bucket serves would have dropped the one and published the
other. So a pull keeps what it already has and takes what SPARTAN adds or revises:

- a row is identified by its key (filter, parameter, method; or date, hour, parameter, method
  for the time-resolved products; or the date for the wide nephelometer files);
- SPARTAN's row wins wherever both copies have the key (a revised value is taken);
- a key only the local copy has is retained, appended after SPARTAN's rows, and reported;
- a file with no rows never replaces one with rows ("held");
- a row that starts after the file was updated (a date typo, the start in the future) is set
  aside (quarantined), not installed;
- a row that ends before it starts, with a plausible start, is installed and flagged: the
  exporters key a filter on its start (ETAD-0262 ends in 2024 for 2025, ZAPR-0224 in 2021
  for 2022, both with good data), so only the log carries it, for SPARTAN to correct.

Nothing here touches the network. `reconcile(remote, local)` returns the text to install
(None to keep the local file) with what it retained, held or set aside; the pull writes that
to a sync log (research/spartan/inventory/sync_log.csv) so every row kept against the bucket
or set aside is listed. When nothing is retained or set aside, the installed text is the
remote text byte for byte.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

# columns that identify a row, in key order, whichever a product has
KEY_COLUMNS = ("Filter_ID", "Year_local", "Month_local", "Day_local", "hour_local",
               "Year", "Month", "Day", "Hour", "Parameter_Code", "Method_Code")
START = ("Start_Year_local", "Start_Month_local", "Start_Day_local", "Start_hour_local")
END = ("End_Year_local", "End_Month_local", "End_Day_local", "End_hour_local")
# a row's name in the log: its filter, else its date
LABEL_COLUMNS = ("Filter_ID", "Year_local", "Year")


def header_index(lines: list[str], max_scan: int = 5) -> int:
    """Index of the header row: SPARTAN files open with 0-3 free-text or '#' lines.

    The same rule as common.spartan_io.find_header_line, on lines already in memory.
    """
    for i, line in enumerate(lines[:max_scan]):
        s = line.strip()
        if s and "," in s and not s.startswith("#"):
            low = s.lower()
            if any(tok in low for tok in ("site_code", "year", "year_local")):
                return i
    return 0


@dataclass
class Parsed:
    preamble: list[str]
    header: str
    columns: list[str]
    lines: list[str]            # data lines, as written (no line ending)
    rows: list[list[str]]       # the same, split into fields


def parse(text: str) -> Parsed:
    lines = text.splitlines()
    h = header_index(lines)
    header = lines[h] if lines else ""
    columns = [c.strip() for c in next(csv.reader([header]))] if header else []
    data = [ln for ln in lines[h + 1:] if ln.strip()]
    rows = list(csv.reader(data))
    return Parsed(lines[:h], header, columns, data, rows)


def _key_index(columns: list[str]) -> list[int]:
    return [columns.index(c) for c in KEY_COLUMNS if c in columns]


def _when(row: list[str], idx: list[int]) -> tuple[int, ...] | None:
    try:
        return tuple(int(float(row[i])) for i in idx)
    except (ValueError, IndexError):
        return None


MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def updated(p: Parsed) -> tuple[int, int, int]:
    """The day the file says it was updated ('# File Updated: 2026-10-13', 'File Updated: 13-Aug-2024'), else today."""
    for line in p.preamble + [p.header]:
        if m := re.search(r"(\d{4})-(\d{2})-(\d{2})", line):
            return int(m[1]), int(m[2]), int(m[3])
        if (m := re.search(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", line)) and m[2].lower() in MONTHS:
            return int(m[3]), MONTHS[m[2].lower()], int(m[1])
    today = datetime.now(timezone.utc)
    return today.year, today.month, today.day


def _dates(p: Parsed) -> tuple[list[int], list[int]]:
    """(rows that start after the file was updated, rows that end before they start but start in time)"""
    if not all(c in p.columns for c in START + END):
        return [], []
    s_idx = [p.columns.index(c) for c in START]
    e_idx = [p.columns.index(c) for c in END]
    day = updated(p)
    future, backwards = [], []
    for i, row in enumerate(p.rows):
        s, e = _when(row, s_idx), _when(row, e_idx)
        if s is not None and s[:3] > day:
            future.append(i)
        elif s is not None and e is not None and s > e:
            backwards.append(i)
    return future, backwards


def _label(p: Parsed, row: list[str]) -> str:
    for c in LABEL_COLUMNS:
        if c in p.columns:
            i = p.columns.index(c)
            if c == "Filter_ID":
                return row[i].strip()
            date = [c, c.replace("Year", "Month"), c.replace("Year", "Day")]
            if all(d in p.columns for d in date):
                return "-".join(row[p.columns.index(d)].strip().zfill(2) for d in date)
    return ""


def _labels(p: Parsed, idx: list[int]) -> list[str]:
    return sorted({_label(p, p.rows[i]) for i in idx} - {""})


@dataclass
class Result:
    text: str | None                  # what to install; None keeps the local file
    status: str                       # installed | held: <why>
    remote_rows: int = 0
    local_rows: int = 0
    installed_rows: int = 0
    retained_rows: int = 0
    retained: list[str] = field(default_factory=list)      # filters (or dates) kept from the local copy
    quarantined_rows: int = 0
    quarantined: list[str] = field(default_factory=list)   # filters (or dates) set aside: start after the update
    quarantine_text: str = ""         # header plus the rows set aside, for <file>.quarantined
    flagged_rows: int = 0
    flagged: list[str] = field(default_factory=list)       # installed, but end before a plausible start
    note: str = ""


def reconcile(remote: str, local: str | None) -> Result:
    """What to install for a downloaded file, given the copy on disk (None if there is none)."""
    r = parse(remote)
    lo = parse(local) if local else None
    n_local = len(lo.rows) if lo else 0
    if not r.rows:
        if n_local:
            return Result(None, "held: remote has no rows", 0, n_local, n_local,
                          note=f"kept the local file's {n_local} rows")
        return Result(remote, "installed", 0, 0, 0)

    future, backwards = _dates(r)
    bad = set(future)
    keep = [i for i in range(len(r.rows)) if i not in bad]
    flags = dict(flagged_rows=len(backwards), flagged=_labels(r, backwards))
    retained: list[int] = []
    note = ""
    key_idx = _key_index(r.columns)
    if lo and lo.rows:
        if lo.columns != r.columns:
            # SPARTAN changed the file's columns: its rows cannot be completed with ours
            note = f"columns changed; the local file's rows were not carried over ({n_local} before, {len(keep)} now)"
        elif key_idx:
            ours = {tuple(row[i].strip() for i in key_idx) for row in (r.rows[i] for i in keep)}
            ours |= {tuple(r.rows[i][j].strip() for j in key_idx) for i in bad}
            retained = [i for i, row in enumerate(lo.rows)
                        if tuple(row[j].strip() for j in key_idx if j < len(row)) not in ours]
            lo_bad = set(_dates(lo)[0])
            retained = [i for i in retained if i not in lo_bad]
        elif len(keep) < n_local:
            return Result(None, "held: remote has fewer rows and no row key", len(r.rows), n_local, n_local,
                          note=f"{len(keep)} rows in the remote file against {n_local} here")

    if not retained and not bad:
        return Result(remote, "installed", len(r.rows), n_local, len(r.rows), note=note, **flags)
    lines = r.preamble + [r.header] + [r.lines[i] for i in keep] + [lo.lines[i] for i in retained]
    text = "\n".join(lines) + "\n"
    quarantine = "\n".join([r.header] + [r.lines[i] for i in sorted(bad)]) + "\n" if bad else ""
    return Result(text, "installed", len(r.rows), n_local, len(keep) + len(retained),
                  retained_rows=len(retained), retained=_labels(lo, retained) if lo else [],
                  quarantined_rows=len(bad), quarantined=_labels(r, sorted(bad)),
                  quarantine_text=quarantine, note=note, **flags)


def read_text(raw: bytes) -> str:
    """SPARTAN files are UTF-8 (some with a byte-order mark); undecodable bytes are replaced."""
    return raw.decode("utf-8-sig", errors="replace")


def as_bytes(text: str) -> bytes:
    return text.encode("utf-8")

