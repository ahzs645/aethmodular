"""Read the results an executed notebook saved in its own outputs.

Most phase-3 notebooks write their tables to gitignored ``output/`` folders, but
the executed ``.ipynb`` is committed with every ``display(df)`` rendered as a
pandas HTML table and every ``print`` as stream text. Those outputs are the
committed record of what the notebook found, so an exporter can publish them
where the CSV is absent. This module turns them back into data:

    nb = ExecutedNotebook(path)
    stats = nb.table("center_stats.to_csv")          # the cell whose code contains the marker
    text = nb.stream("print(f'IMPROVE implied MAC")  # that cell's printed output

A cell is found by a piece of its source code, never by its index, so inserting
a cell above it does not silently move the read. pandas renders a wide frame
with a ``...`` column when it truncates; such a table is refused rather than
read short. Parsing uses the standard library only (``html.parser``).
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


class _Tables(HTMLParser):
    """Collect every <table> as header rows and body rows of (tag, text) cells."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict[str, list[list[tuple[str, str]]]]] = []
        self._section: str | None = None
        self._row: list[tuple[str, str]] | None = None
        self._cell: tuple[str, list[str]] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append({"head": [], "body": []})
        elif tag in ("thead", "tbody"):
            self._section = "head" if tag == "thead" else "body"
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("th", "td") and self._row is not None:
            self._cell = (tag, [])

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._cell is not None and self._row is not None:
            self._row.append((self._cell[0], "".join(self._cell[1]).strip()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1][self._section or "body"].append(self._row)
            self._row = None
        elif tag in ("thead", "tbody"):
            self._section = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell[1].append(data)


def _value(text: str):
    if text in ("", "NaN", "nan", "None", "<NA>", "NaT"):
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


def parse_html_tables(html: str) -> list[pd.DataFrame]:
    """Every pandas-rendered table in an HTML fragment, as a DataFrame.

    A named index (pandas writes its name on a second header row) becomes a
    column of that name; an unnamed 0..n-1 index is dropped. Multi-row column
    headers (a pivot) are joined with " · ". A truncated table raises.
    """
    parser = _Tables()
    parser.feed(html)
    frames = []
    for table in parser.tables:
        head, body = table["head"], table["body"]
        if not body:
            continue
        n_index = sum(1 for tag, _ in body[0] if tag == "th")
        index_names: list[str] = []
        header_rows = head
        # pandas puts the index name(s) on their own last header row: names, then empty cells
        if len(head) >= 2 and all(text == "" for _, text in head[-1][n_index:]):
            index_names = [text for _, text in head[-1][:n_index]]
            header_rows = head[:-1]
        columns: list[str] = []
        for i in range(n_index, max((len(r) for r in header_rows), default=n_index)):
            parts = [r[i][1] for r in header_rows if i < len(r) and r[i][1]]
            columns.append(" · ".join(parts))
        if not index_names and header_rows:
            index_names = [text for _, text in header_rows[0][:n_index]]
        if any(c == "..." for c in columns) or any(text == "..." for row in body for _, text in row):
            raise ValueError("the notebook displayed a truncated table; re-run it with the full frame or read its CSV")
        records = []
        for row in body:
            index = [text for tag, text in row if tag == "th"]
            cells = [text for tag, text in row if tag == "td"]
            # a MultiIndex row omits repeated outer labels; carry them down
            if len(index) < n_index and records:
                index = [records[-1]["__index__"][j] for j in range(n_index - len(index))] + index
            records.append({"__index__": index, **{c: _value(v) for c, v in zip(columns, cells)}})
        frame = pd.DataFrame([{k: v for k, v in r.items() if k != "__index__"} for r in records], columns=columns)
        index_values = [r["__index__"] for r in records]
        default_range = n_index == 1 and not any(index_names) and [v[0] for v in index_values] == [str(i) for i in range(len(records))]
        if n_index and not default_range:
            for j in reversed(range(n_index)):
                name = index_names[j] if j < len(index_names) and index_names[j] else ("index" if n_index == 1 else f"index_{j}")
                frame.insert(0, name, [_value(v[j]) if j < len(v) else None for v in index_values])
        frames.append(frame)
    return frames


class ExecutedNotebook:
    """An executed notebook's saved outputs, addressed by a marker in the cell's source."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.cells = json.loads(self.path.read_text(encoding="utf-8"))["cells"]

    def _cell(self, marker: str) -> dict:
        hits = [c for c in self.cells if c["cell_type"] == "code" and marker in "".join(c["source"])]
        if len(hits) != 1:
            raise LookupError(f"{self.path.name}: {len(hits)} code cells contain {marker!r} (need exactly one)")
        return hits[0]

    def tables(self, marker: str) -> list[pd.DataFrame | None]:
        """Every HTML table the matching cell displayed, in output order; None for a truncated one."""
        out: list[pd.DataFrame | None] = []
        for output in self._cell(marker).get("outputs", []):
            html = (output.get("data") or {}).get("text/html")
            if html:
                try:
                    out.extend(parse_html_tables("".join(html)))
                except ValueError:
                    out.append(None)
        return out

    def table(self, marker: str, nth: int = 0) -> pd.DataFrame:
        found = self.tables(marker)
        if len(found) <= nth:
            raise LookupError(f"{self.path.name}: the cell with {marker!r} displayed {len(found)} tables, not {nth + 1}")
        if found[nth] is None:
            raise ValueError(f"{self.path.name}: table {nth} of the cell with {marker!r} was displayed truncated")
        return found[nth]

    def stream(self, marker: str) -> str:
        """The matching cell's printed (stdout) text."""
        return "".join("".join(o.get("text", "")) for o in self._cell(marker).get("outputs", [])
                       if o.get("output_type") == "stream" and o.get("name", "stdout") == "stdout")

    def plain(self, marker: str) -> list[str]:
        """The matching cell's text/plain results (e.g. a displayed string)."""
        return ["".join((o.get("data") or {})["text/plain"]) for o in self._cell(marker).get("outputs", [])
                if "text/plain" in (o.get("data") or {})]
