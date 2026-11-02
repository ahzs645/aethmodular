"""The Ethiopia pilot filters as per-filter records, for the gallery's filter drawer.

The five Adama PTFE filters (CSU AMOD pilot, July 2024, each with a date-paired quartz TOR filter)
and the 26 Bishoftu filters the frozen external pilot scored are spread over page exports
(baseline_external_pilot.json, experimental/tor.json, research/spectral/ethiopia_calibrations.json)
that each carry the columns their page draws. This export is the filter-level view of the same
sources: one record per filter, every measurement with its unit, and the checks that join them.

    gallery/app/public/data/pilot_filters.json   records, fields, units, groups, the Addis reference
    gallery/app/public/data/pilot_spectra.json   AIRSpec and completed rank-70 VIBES corrected spectra

Sources (all local research outputs; nothing is read from Drive):
  - the frozen pilot reapplied on the completed rank-70 VIBES correction
    (vibes_fullrange/gallery_rank70_20261002/external: migrate_vibes_external_rank70.py): the
    Adama pairs, the Bishoftu and Addis frozen predictions, the Bishoftu-Addis matches, and the
    full-range VIBES corrected spectra
  - the AIRSpec corrected spectra of the same 31 filters (adama_bishoftu_frozen_pilot/external_corrected.npz)
  - the quartz filters' TOR thermal fractions and char/soot split (spartan_ec_2026_06_16, µg per filter)
  - June 02's general-calibration FTIR EC (ftir_ec_calibration_2026_06_25)

The Adama spectrum-to-filter map (analysis 4744-4748 to J-numbers) is the pilot's provisional,
date-based one; it is checked against June 08's barcodes and carried as a status, not hidden.

Run after workflows/migrate_vibes_external_rank70.py:
    uv run --locked --no-sync python gallery/data/export_pilot_filters.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "research/ftir_hips_chem/output/tables"
PILOT = TABLES / "vibes_fullrange/gallery_rank70_20261002/external"
AIRSPEC = TABLES / "adama_bishoftu_frozen_pilot/external_corrected.npz"
GRID = TABLES / "vibes_fullrange/rank30/wn.npy"
TOR_FRACTIONS = ROOT / "research/spartan_ec_2026_06_16/tables/adama_char_soot_classification.csv"
JUNE02 = ROOT / "research/ftir_ec_calibration_2026_06_25/tables/adama_tor_vs_ftir.csv"
JUNE08 = ROOT / "research/ftir_ec_calibration_2026_06_25/tables/adama_new_ec_by_variant.csv"
DEST = ROOT / "gallery/app/public/data"
TRACES_R70 = DEST / "similarity/traces_VIBES-full-r70.json"
BIN = 8  # channels per display bin, as the similarity traces (export_similarity.py, export_vibes_fullrange_traces.py)
AIRSPEC_FLOOR = 1425  # the spline baseline's window ends here; the models use 4000-1425

ADAMA_SITE = "Adama (CSU AMOD pilot)"
BISHOFTU_SITE = "Bishoftu, Ethiopia"

# display name -> unit, in drawer order within each group
UNITS = {
    "PTFE sampled volume": "m3", "Quartz sampled volume": "m3", "PTFE ÷ quartz volume": "ratio", "Start offset": "min",
    "Sampled volume": "m3", "PTFE lot": "", "Season": "",
    "EC (TOR)": "ug/m3", "EC (TOT)": "ug/m3", "OC (TOR)": "ug/m3", "OC/EC (TOR)": "ratio",
    "OC1": "ug/m3", "OC2": "ug/m3", "OC3": "ug/m3", "OC4": "ug/m3", "OP": "ug/m3",
    "EC1": "ug/m3", "EC2": "ug/m3", "EC3": "ug/m3",
    "Char EC": "ug/m3", "Soot EC": "ug/m3", "Char/soot": "ratio", "Char share of EC": "ratio", "TOR source class": "",
    "HIPS Fabs": "Mm-1",
    "FTIR EC, spline baseline": "ug/m3", "FTIR EC, VIBES baseline": "ug/m3",
    "FTIR EC loading, spline baseline": "ug", "FTIR EC loading, VIBES baseline": "ug",
    "Spline FTIR ÷ TOR EC": "ratio", "VIBES FTIR ÷ TOR EC": "ratio", "Spline FTIR ÷ TOT EC": "ratio", "VIBES FTIR ÷ TOT EC": "ratio",
    "Band 1617, spline baseline": "absorbance", "Band 1617, VIBES baseline": "absorbance",
    "Band 2920, spline baseline": "absorbance", "Band 2920, VIBES baseline": "absorbance",
    "FTIR EC, June 2026 general calibration": "ug/m3",
    "Matched Addis filter": "", "Fabs ratio to the matched Addis filter": "ratio",
}
GROUPS = [
    ("Sampling", ["PTFE sampled volume", "Quartz sampled volume", "PTFE ÷ quartz volume", "Start offset", "Sampled volume", "PTFE lot", "Season"]),
    ("Carbon (TOR, quartz)", ["EC (TOR)", "EC (TOT)", "OC (TOR)", "OC/EC (TOR)"]),
    ("TOR thermal fractions (quartz)", ["OC1", "OC2", "OC3", "OC4", "OP", "EC1", "EC2", "EC3"]),
    ("Char and soot (TOR)", ["Char EC", "Soot EC", "Char/soot", "Char share of EC", "TOR source class"]),
    ("Optical (HIPS)", ["HIPS Fabs"]),
    ("FTIR EC, frozen calibration", ["FTIR EC, spline baseline", "FTIR EC, VIBES baseline", "FTIR EC loading, spline baseline",
                                     "FTIR EC loading, VIBES baseline", "Spline FTIR ÷ TOR EC", "VIBES FTIR ÷ TOR EC",
                                     "Spline FTIR ÷ TOT EC", "VIBES FTIR ÷ TOT EC"]),
    ("Organic bands (corrected spectrum)", ["Band 1617, spline baseline", "Band 1617, VIBES baseline", "Band 2920, spline baseline", "Band 2920, VIBES baseline"]),
    ("Earlier calibrations", ["FTIR EC, June 2026 general calibration"]),
    ("Matched Addis filter", ["Matched Addis filter", "Fabs ratio to the matched Addis filter"]),
]
# the measurements an Addis filter has under the same frozen model, so a pilot filter can be placed against them
REFERENCE_FIELDS = ["FTIR EC, spline baseline", "FTIR EC, VIBES baseline", "HIPS Fabs",
                    "Band 1617, spline baseline", "Band 1617, VIBES baseline", "Band 2920, spline baseline", "Band 2920, VIBES baseline"]
FROZEN = {"FTIR EC, spline baseline": "AIRSpec_EC_ugm3", "FTIR EC, VIBES baseline": "VIBES_EC_ugm3", "HIPS Fabs": "Fabs",
          "Band 1617, spline baseline": "AIRSpec_band1617", "Band 1617, VIBES baseline": "VIBES_band1617",
          "Band 2920, spline baseline": "AIRSpec_band2920", "Band 2920, VIBES baseline": "VIBES_band2920"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(v):
    """A JSON value: finite floats rounded, NaN as null, numpy scalars as Python ones."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if not np.isfinite(v) else float(f"{v:.6g}")
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"pilot filters: {what}")


def adama_records(pairs: pd.DataFrame, fractions: pd.DataFrame, june02: pd.DataFrame, june08: pd.DataFrame) -> list[dict]:
    check(len(pairs) == 5 and pairs.FilterId_ptfe.is_unique, "expected the five Adama pairs")
    # the spectrum-to-filter map: the pilot's provisional one must be June 08's barcodes, filter for filter
    barcode = june08.set_index("FilterId")["barcode"].astype(int)
    provisional = pairs.set_index("FilterId_ptfe")["SampleAnalysisId_provisional"].astype(int)
    check(barcode.reindex(provisional.index).equals(provisional), "the pilot's spectrum map differs from June 08's barcodes")
    # June 02 pairs the PTFE and quartz filters by date, as the pilot does
    j02 = june02.set_index("ptfe_FilterId")
    check(j02.quartz_FilterId.reindex(pairs.FilterId_ptfe).tolist() == pairs.FilterId_quartz.tolist(), "June 02's quartz pairing differs")
    frac = fractions.set_index("sample")
    out = []
    for r in pairs.itertuples():
        quartz_m3 = r.Volume_liters / 1000
        f = frac.loc[r.FilterId_quartz]
        # the TOR fractions are µg on the quartz filter; their EC over the quartz volume is the pair's TOR EC
        check(abs(f.EC_TOR / quartz_m3 - r.ECTR) < 2e-3, f"{r.FilterId_quartz}: TOR fractions ÷ quartz volume ≠ ECTR")
        per_m3 = lambda col: f[col] / quartz_m3  # noqa: E731
        flags = [x for x in [r.flag if isinstance(r.flag, str) else None, r.exclusion_reason if r.is_excluded else None] if x]
        values = {
            "PTFE sampled volume": r.Volume_m3, "Quartz sampled volume": quartz_m3, "PTFE ÷ quartz volume": r.volume_ratio,
            "Start offset": r.start_offset_min,
            "EC (TOR)": r.ECTR, "EC (TOT)": r.ECTT, "OC (TOR)": per_m3("OC_TOR"), "OC/EC (TOR)": f.OC_EC_ratio,
            **{k: per_m3(k) for k in ["OC1", "OC2", "OC3", "OC4", "OP", "EC1", "EC2", "EC3"]},
            "Char EC": per_m3("char_EC"), "Soot EC": per_m3("soot_EC"), "Char/soot": f.char_soot_ratio,
            "Char share of EC": f.char_frac_of_EC, "TOR source class": f.source_class,
            "FTIR EC, spline baseline": r.AIRSpec_EC_ugm3, "FTIR EC, VIBES baseline": r.VIBES_EC_ugm3,
            "FTIR EC loading, spline baseline": r.AIRSpec_loading_ug, "FTIR EC loading, VIBES baseline": r.VIBES_loading_ug,
            "Spline FTIR ÷ TOR EC": r.AIRSpec_over_ECTR, "VIBES FTIR ÷ TOR EC": r.VIBES_over_ECTR,
            "Spline FTIR ÷ TOT EC": r.AIRSpec_over_ECTT, "VIBES FTIR ÷ TOT EC": r.VIBES_over_ECTT,
            "FTIR EC, June 2026 general calibration": j02.loc[r.FilterId_ptfe, "FTIR_EC_general_ug_m3"],
        }
        out.append({
            "id": r.FilterId_ptfe, "campaign": "Adama", "site": ADAMA_SITE, "code": "ADAMA", "date": r.date,
            "medium": "PTFE", "partner": {"label": "co-sampled quartz filter (TOR)", "id": r.FilterId_quartz},
            "spectrum": f"adama:{int(r.SampleAnalysisId_provisional)}",
            "spectrum_map": "provisional",
            "flags": flags,
            "notes": [f"Pairing: {r.pairing_status}." if isinstance(r.pairing_status, str) else None,
                      "TOR is measured on the date-paired quartz filter; its fractions are divided by the quartz volume."],
            "values": {k: value(v) for k, v in values.items()},
        })
    for rec in out:
        rec["notes"] = [n for n in rec["notes"] if n]
    return out


def bishoftu_records(bish: pd.DataFrame, matched: pd.DataFrame) -> list[dict]:
    check(len(bish) == 26 and bish.ExternalFilterId.is_unique, "expected the 26 Bishoftu filters")
    match = matched.set_index("bishoftu_filter_id")
    out = []
    for r in bish.itertuples():
        m = match.loc[r.ExternalFilterId] if r.ExternalFilterId in match.index else None
        values = {
            "Sampled volume": r.Volume_m3, "PTFE lot": int(r.LotId), "Season": r.Group, "HIPS Fabs": r.Fabs,
            "FTIR EC, spline baseline": r.AIRSpec_EC_ugm3, "FTIR EC, VIBES baseline": r.VIBES_EC_ugm3,
            "Band 1617, spline baseline": r.AIRSpec_band1617, "Band 1617, VIBES baseline": r.VIBES_band1617,
            "Band 2920, spline baseline": r.AIRSpec_band2920, "Band 2920, VIBES baseline": r.VIBES_band2920,
            "Matched Addis filter": None if m is None else f"{m.addis_filter_id} ({m.addis_date})",
            "Fabs ratio to the matched Addis filter": None if m is None else m.Fabs_ratio,
        }
        out.append({
            "id": r.ExternalFilterId, "campaign": "Bishoftu", "site": BISHOFTU_SITE, "code": "ETBI", "date": r.date,
            "medium": "PTFE", "partner": None, "spectrum": f"etbi:{int(r.MediaId)}", "spectrum_map": "confirmed",
            "flags": [r.exclusion_reason] if r.is_excluded and isinstance(r.exclusion_reason, str) else [],
            "notes": ["Scored by the frozen external pilot; the SPARTAN network export carries this filter's HIPS and volume."],
            "values": {k: value(v) for k, v in values.items() if v is not None or k in UNITS},
        })
    return out


def binned(X: np.ndarray, wn: np.ndarray) -> tuple[list[float], list[list[float]]]:
    bins = [np.arange(i, min(i + BIN, len(wn))) for i in range(0, len(wn), BIN)]
    bin_wn = [round(float(wn[b].mean()), 3) for b in bins]
    y = np.stack([X[:, b].mean(axis=1) for b in bins], axis=1)
    return bin_wn, [[float(f"{v:.5g}") for v in row] for row in y]


def spectra() -> dict:
    """Both corrected sets, rows in the pilot's order (external_corrected.npz ids; the rank-70 run kept them)."""
    wn = np.load(GRID)
    full = np.load(PILOT / "corrected_full.npy")
    with np.load(AIRSPEC) as z:
        ids = z["ids"].astype(str).tolist()
        air = z["AIRSpec"]
    check(full.shape == (len(ids), len(wn)) and np.isfinite(full).all(), "rank-70 corrected spectra are incomplete")
    shared = wn >= AIRSPEC_FLOOR
    check(air.shape == (len(ids), int(shared.sum())) and np.isfinite(air).all(), "AIRSpec corrected spectra do not fit the shared grid")
    vibes_wn, vibes = binned(full, wn)
    air_wn, airspec = binned(air, wn[shared])
    # the drawer draws these with the similarity traces' axes: the same bins
    if TRACES_R70.exists():
        check(json.loads(TRACES_R70.read_text())["bin_wn"] == vibes_wn, "VIBES bins differ from traces_VIBES-full-r70.json")
    return {"ids": ids, "bin": BIN, "units": "absorbance (baseline-corrected)",
            "AIRSpec": {"bin_wn": air_wn, "y": airspec, "window": [4000, AIRSPEC_FLOOR]},
            "VIBES": {"bin_wn": vibes_wn, "y": vibes, "window": [4000, 500], "background_rank": 70,
                      "presentation_status": "completed_uncapped"}}


def main() -> None:
    manifest = json.loads((PILOT / "manifest.json").read_text())
    check(manifest.get("background_rank") == 70, "the external pilot is not the completed rank-70 run")
    pairs = pd.read_csv(PILOT / "adama_candidate_pairs.csv")
    bish = pd.read_csv(PILOT / "bishoftu_frozen_predictions.csv")
    addis = pd.read_csv(PILOT / "addis_frozen_predictions.csv")
    matched = pd.read_csv(PILOT / "addis_bishoftu_loading_matched.csv")
    check(len(addis) == 253, "expected the 253 Addis frozen targets")
    records = adama_records(pairs, pd.read_csv(TOR_FRACTIONS), pd.read_csv(JUNE02), pd.read_csv(JUNE08)) + bishoftu_records(bish, matched)

    # the Addis frozen targets under the same model: what "where it sits" places a pilot filter against
    reference = [{"id": value(r.filter_id), "site": "Addis Ababa", "code": "ETAD", "date": value(r.date), "excluded": bool(r.is_excluded),
                  **{k: value(getattr(r, col)) for k, col in FROZEN.items()}} for r in addis.itertuples()]
    spec = spectra()
    check(sorted(r["spectrum"] for r in records) == sorted(spec["ids"]), "the records and the corrected spectra are different filters")

    sources = [PILOT / n for n in ("manifest.json", "adama_candidate_pairs.csv", "bishoftu_frozen_predictions.csv",
                                   "addis_frozen_predictions.csv", "addis_bishoftu_loading_matched.csv", "corrected_full.npy")]
    sources += [AIRSPEC, GRID, TOR_FRACTIONS, JUNE02, JUNE08]
    payload = {
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "frozen_run_signature": manifest["frozen_run_signature"], "background_rank": 70,
        "source_sha256": {rel(p): sha(p) for p in sources},
        "units": UNITS,
        "groups": [{"label": g, "fields": f} for g, f in GROUPS],
        "campaigns": {
            "Adama": {"site": ADAMA_SITE, "n": 5, "note": "CSU AMOD pilot, July 2024: PTFE filters for FTIR, each with a date-paired quartz filter for TOR. "
                      "The spectrum-to-filter map is the pilot's provisional, date-based one."},
            "Bishoftu": {"site": BISHOFTU_SITE, "n": 26, "note": "SPARTAN Bishoftu filters scored by the frozen external pilot."},
        },
        "reference": {"label": "SPARTAN Addis, same frozen model", "n": len(reference), "fields": REFERENCE_FIELDS, "rows": reference},
        "filters": records,
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "pilot_filters.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    (DEST / "pilot_spectra.json").write_text(json.dumps({"schema_version": 1, "frozen_run_signature": manifest["frozen_run_signature"], **spec},
                                                        separators=(",", ":"), allow_nan=False))
    print(f"wrote pilot_filters.json ({len(records)} filters, {len(reference)} Addis reference rows) and pilot_spectra.json "
          f"({len(spec['ids'])} spectra, {len(spec['AIRSpec']['bin_wn'])} + {len(spec['VIBES']['bin_wn'])} bins)")


if __name__ == "__main__":
    main()
