"""Merge a cache zip written by remote_setup.pack_results() into the local explorer cache.

    uv run python calibration_explorer/colab/cli/merge_results.py explorer_cache_app.zip [...]

Fit and curve caches are content-keyed, so a file that already exists locally
is the same result and is left alone. `batch_results.jsonl` is appended, not
overwritten, skipping rows whose batch key is already present. Afterwards,
re-export the gallery grid with `python gallery/data/export_calibration.py`.
"""

import json
import sys
import zipfile
from pathlib import Path

CACHE = Path(__file__).resolve().parents[2] / "cache"
RESULTS = CACHE / "batch_results.jsonl"


def row_key(r: dict) -> str:
    """app._batch_row_key: rows without the evaluation-view fields mean "all",
    and the group scheme only counts when a group is actually selected."""
    base = "|".join(str(r.get(f)) for f in ("cohort", "cutoff", "selection_space", "spectra",
                                             "mode", "lot", "target", "eval_lot"))
    view = "|".join(str(r.get(f) or "all") for f in ("eval_group", "eval_split"))
    scheme = str(r.get("group_scheme") or "default") if r.get("eval_group") not in (None, "all") else "default"
    return f"{base}|{view}|{scheme}|k{r.get('k')}"


def main(zips: list[str]) -> None:
    seen = set()
    if RESULTS.is_file():
        with RESULTS.open() as f:
            for line in f:
                try:
                    seen.add(row_key(json.loads(line)))
                except json.JSONDecodeError:
                    continue
    added_files = added_rows = dup_rows = 0
    for z in zips:
        with zipfile.ZipFile(z) as archive:
            for info in archive.infolist():
                name = Path(info.filename)
                if info.is_dir() or name.parts[:1] != ("cache",):
                    continue
                rel = Path(*name.parts[1:])
                if rel.name == "batch_results.jsonl":
                    with archive.open(info) as src, RESULTS.open("a") as dst:
                        for raw in src:
                            line = raw.decode().strip()
                            if not line:
                                continue
                            k = row_key(json.loads(line))
                            if k in seen:
                                dup_rows += 1
                                continue
                            seen.add(k)
                            dst.write(line + "\n")
                            added_rows += 1
                    continue
                dest = CACHE / rel
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(info))
                added_files += 1
        print(f"{z}: merged")
    print(f"cache files added {added_files:,} · batch rows added {added_rows:,} · duplicate rows skipped {dup_rows:,}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
