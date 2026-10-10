"""Map Addis external filter ids to the Spectral similarity sample ids.

Run with uv run --no-sync python gallery/data/export_etad_ids.py.

The gallery's SPARTAN drawer knows an Addis filter by its external base id
("ETAD-0017"); the similarity traces are keyed "etad:<MediaId>". This reads the
target rows of the frozen full-profile run's cases.csv (read-only) and writes
gallery/app/public/data/similarity/etad_ids.json:

    {"source": ..., "n": ..., "ids": {"ETAD-0017": ["etad:401"], ...},
     "status": {"ETAD-0017": {"in_run": true, "in_test_set": true, "note": "..."}, ...}}

``status`` (1 Oct 2026 meeting: "label the excluded samples more clearly") covers
every Addis base id in the run's ledger or the Addis test-set list, so a drawer
or tooltip can say why a filter has no spectrum or no test-set prediction.

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
LEDGER = ROOT / "gallery/app/public/data/baseline_addis_ledger.csv"
EVALUATION = ROOT / "research/ftir_hips_chem/output/tables/ann_weekly_20260910/addis_evaluation.csv"


def addis_status(ids: dict[str, list[str]]) -> dict[str, dict]:
    ledger = pd.read_csv(LEDGER)
    ev = pd.read_csv(EVALUATION)
    ev_base = {base_id(x): x for x in ev.ExternalFilterId}
    out: dict[str, dict] = {}
    for r in ledger.itertuples():
        b = r.filter_id  # the ledger already holds base ids (ETAD-0017)
        in_run, in_test = b in ids, b in ev_base
        if r.role != "sample":
            note = (("PM10 filter" + (" (it has HIPS Fabs)" if in_test else "")
                     + ", outside the PM2.5 comparison: no spectrum in the frozen run and not in the Addis test set")
                    if r.role == "outside_scope"
                    else f"field blank ({'trains the VIBES background' if r.role == 'blank_train' else 'held out to check the VIBES background'})")
        elif in_test and in_run:
            note = "in the Addis test set (233 filters with HIPS Fabs and a spectrum in the frozen run)"
        elif in_run:
            note = ("in the spectra run but not in the Addis test set: no HIPS Fabs"
                    + ("" if isinstance(r.date, str) else " and no sample date"))
        else:
            note = "not in the frozen spectra run"
        if bool(r.is_excluded):
            note = f"excluded from SPARTAN data: {r.exclusion_reason}; " + note
        out[b] = {"in_run": in_run and r.role == "sample", "in_test_set": in_test and in_run and r.role == "sample", "note": note}
    for b, ext in ev_base.items():
        if b not in out:
            d = ev.loc[ev.ExternalFilterId.eq(ext), "date"].iloc[0]
            out[b] = {"in_run": b in ids, "in_test_set": False,
                      "note": f"has HIPS Fabs but its spectrum is not in the frozen run (sampled {d}, after the run was built)"}
    return dict(sorted(out.items()))
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
        "status": addis_status(ids),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {payload['n']} base ids from {len(targets)} target rows, {payload['n_shared']} shared")
    notes = pd.Series([v["note"].split(" (")[0] for v in payload["status"].values()]).value_counts()
    print(notes.to_string())


if __name__ == "__main__":
    main()
