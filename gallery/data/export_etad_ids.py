"""Map Addis external filter ids to the Spectral similarity sample ids.

Run with uv run --no-sync python gallery/data/export_etad_ids.py.

The gallery's SPARTAN drawer knows an Addis filter by its external base id
("ETAD-0017"); the similarity traces are keyed "etad:<MediaId>". This reads the
target rows of the frozen full-profile run's cases.csv (read-only) and writes
gallery/app/public/data/similarity/etad_ids.json:

    {"source": ..., "n": ..., "ids": {"ETAD-0017": ["etad:401"], ...}}

The base id is the external id with a trailing "-<digits>" removed, matching
baseFilterId in gallery/app/src/lib/highlight.tsx. A base id that several
MediaIds share keeps all of them, in cases.csv order.
"""

from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
CASES = RUN / "cases.csv"
OUT = ROOT / "gallery/app/public/data/similarity/etad_ids.json"
SUFFIX = re.compile(r"-\d+$")


def base_id(external: str) -> str:
    return SUFFIX.sub("", external.strip())


def main() -> None:
    cases = pd.read_csv(CASES, usecols=["sample_id", "kind", "filter_id"], dtype=str)
    targets = cases[(cases["kind"] == "target") & cases["filter_id"].notna() & cases["sample_id"].str.startswith("etad:")]
    ids: dict[str, list[str]] = {}
    for sid, ext in zip(targets["sample_id"], targets["filter_id"]):
        lst = ids.setdefault(base_id(ext), [])
        if sid not in lst:
            lst.append(sid)
    payload = {
        "source": str(CASES.relative_to(ROOT)),
        "n": len(ids),
        "n_shared": sum(len(v) > 1 for v in ids.values()),
        "ids": dict(sorted(ids.items())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {payload['n']} base ids from {len(targets)} target rows, {payload['n_shared']} shared")


if __name__ == "__main__":
    main()
