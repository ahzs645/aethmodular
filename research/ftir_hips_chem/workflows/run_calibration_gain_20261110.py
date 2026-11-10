"""Why the FTIR EC calibration reads low on heavy filters, and whether anything fixes it (10 Nov 2026).

docs/addis-reference-closure-2026-11-10.md found that the raw-spectra calibration responds to thermal EC with a
gain of about 0.78 and makes up the difference with a fixed offset. This asks, with whole IMPROVE sites held out
and without choosing filters by their own TOR values:

1. does removing the offset and dividing by the gain (both fitted on other sites) bring heavy filters to 1 without
   harming light ones;
2. does the gain move with loading weights, with blanks added at zero, or with the number of components;
3. does the baseline (spline, VIBES, PTFE-anchored VIBES) change the gain.

Three sources, one or more per run:

    # raw spectra from the IMPROVE tables on the Drive (blanks included). About 15 minutes on 4 cores.
    uv run python research/ftir_hips_chem/workflows/run_calibration_gain_20261110.py --drive --summary

    # the calibrations the 7 Nov closeout saved (predictions only, 1,821 heavier filters: indicative)
    uv run python research/ftir_hips_chem/workflows/run_calibration_gain_20261110.py --closeout

    # baseline-corrected spectra: the 22 Oct decision bundle (improve_meta.csv, improve_<prep>.npy) and, for the
    # anchored variants, the continuum output (corrected_<variant>_improve.npy). Run this where the bundle lives
    # (Colab or a laptop: PLS only, no baselining; about 15 minutes per preprocessing on 4 cores).
    uv run python research/ftir_hips_chem/workflows/run_calibration_gain_20261110.py \\
        --bundle "<Drive>/Aethmodular VIBES/decision_workbook_20261022/bundle" \\
        --continuum-output "<dir with corrected_anchorhp_improve.npy>" \\
        --models raw spline vibes vibes_anchorhp --output <new dir>

On Colab: mount Drive, ``pip install uv`` (or use the notebook's own Python with numpy, pandas, scikit-learn),
clone the repository, and run the ``--bundle`` command above. Send back ``calibration_gain.json`` from the output.

Writes ``output/tables/calibration_gain_20261110/`` (git-ignored) and, with ``--summary``, the tables quoted in the
note as docs/calibration-gain-2026-11-10.json. Nothing here uses an Addis measurement to fit or choose anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path[:0] = [str(AREA / "scripts"), str(AREA / "workflows")]
import calibration_gain as cg  # noqa: E402

CLOSEOUT = ROOT / "gallery/review/closeout_20261107/continuum_baseline.json"
OUT = AREA / "output/tables/calibration_gain_20261110"
SUMMARY = ROOT / "docs/calibration-gain-2026-11-10.json"
URBAN = ("ATLA1", "BIRM1", "FRES1", "PHOE1", "PITT1", "PUSO1")
KMAX, K = 25, 18
K_SWEEP = (6, 12, 18, 25)
BLANK_WEIGHT = 5.0
PREP_NAMES = {"raw": "Raw spectra", "spline": "Spline baseline", "vibes": "Production VIBES", "vibes_deflate": "Deflated VIBES",
              "vibes_anchorhp": "PTFE-anchored VIBES", "anchored": "PTFE-anchored (linear)"}
CLOSEOUT_NAMES = {"raw k18": "Raw spectra", "spline k18": "Spline baseline", "vibes k18": "Production VIBES", "vibes_deflate k18": "Deflated VIBES",
                  "vibes_anchorhp k18": "PTFE-anchored VIBES", "anchored_noCH k18": "Anchored, C–H masked",
                  "anchored_balanced k12": "Anchored, OC/EC-balanced weights"}
CONTINUUM = {"vibes_deflate": "deflate", "vibes_anchorhp": "anchorhp"}


# --------------------------------------------------------------------------- one preprocessing, every variant
def evaluate(name, x, ec, oc, sites, dates, *, blanks=None, n_boot=500):
    """Variants of one calibration on one set of spectra: response, fair scores, and the recalibration test."""
    folds = cg.site_folds(sites)
    weights = cg.loading_weights(ec)
    zero = None if blanks is None else np.zeros(len(blanks))
    variants = {"As fitted": {}, "Weighted toward heavy filters": {"weights": weights}}
    if blanks is not None:
        variants |= {"Blanks added at zero": {"extra_x": blanks, "extra_y": zero, "extra_weight": BLANK_WEIGHT},
                     "Weighted, and blanks at zero": {"weights": weights, "extra_x": blanks, "extra_y": zero, "extra_weight": BLANK_WEIGHT}}
    rows, sweep = [], []
    for variant, options in variants.items():
        held_out, extra = cg.held_out_predictions(x, ec, folds, KMAX, predict=None if blanks is None else {"blanks": blanks}, **options)
        for k in K_SWEEP:
            sweep.append({"preprocessing": name, "variant": variant, "k": k, **cg.response(held_out[:, k - 1], ec, oc, sites, n_boot=0)})
        prediction = held_out[:, K - 1]
        fitted = cg.response(prediction, ec, oc, sites, n_boot=n_boot)
        blank = float(np.median(extra["blanks"][:, K - 1])) if blanks is not None else float("nan")
        row = {"preprocessing": name, "variant": variant, "recalibrated": "no", "blank_reading": blank, **fitted,
               **cg.fair_scores(prediction, ec, sites, dates, URBAN)}
        rows.append(row)
        # remove the offset and divide by the gain, fitted on the other folds' sites
        corrected = cg.recalibrate_by_site(prediction, ec, sites, folds)
        simple = cg.response(prediction, ec, ec, sites, with_oc=False, n_boot=0)
        rows.append({"preprocessing": name, "variant": variant, "recalibrated": "offset removed, divided by the gain",
                     "blank_reading": (blank - simple["offset"]) / simple["gain"], **cg.response(corrected, ec, oc, sites, n_boot=n_boot),
                     **cg.fair_scores(corrected, ec, sites, dates, URBAN)})
    return pd.DataFrame(rows), pd.DataFrame(sweep)


# --------------------------------------------------------------------------- sources
def from_drive(n_boot):
    import run_addis_reference_closure_20261110 as closure

    loaded = closure.improve_pool()
    if loaded is None:
        raise FileNotFoundError("IMPROVE spectra not found: set AETHMODULAR_FTIR_LOCAL_DB and AETHMODULAR_ETAD_DIR (uv run aeth drive mirror)")
    pool, x, spectra, channels = loaded["pool"], loaded["x"], loaded["spectra"], loaded["channels"]
    catalogue = pd.read_csv(loaded["local"] / "tables/ftir_catalog.csv", encoding="utf-8-sig", dtype=str)
    blank_ids = set(catalogue.loc[catalogue["FilterPurposeId"].eq("2"), "AnalysisId"])
    blanks = spectra.loc[spectra["AnalysisId"].astype(str).isin(blank_ids), channels].to_numpy(float)
    blanks = blanks[np.isfinite(blanks).all(axis=1)]
    print(f"drive: {len(pool)} filters, {pool['Site'].nunique()} sites, {len(blanks)} blanks")
    return evaluate("Raw spectra", x, pool["ec_ug"].to_numpy(), pool["oc_ug"].to_numpy(), pool["Site"].to_numpy(), pool["day"].to_numpy(),
                    blanks=blanks, n_boot=n_boot)


def _column(meta, names, what):
    for name in names:
        if name in meta.columns:
            return meta[name]
    raise KeyError(f"bundle improve_meta.csv has no {what} column (looked for {', '.join(names)}); it has {', '.join(meta.columns)}")


def from_bundle(bundle: Path, models, continuum: Path | None, n_boot):
    meta = pd.read_csv(bundle / "improve_meta.csv")
    sites, dates = _column(meta, ("site", "Site"), "site").to_numpy(), _column(meta, ("date", "SampleDate", "day"), "date").to_numpy()
    ec = _column(meta, ("ec_ug", "tor_ug"), "TOR EC per filter").to_numpy(float)
    oc = (meta["oc_ug"] if "oc_ug" in meta.columns else _column(meta, ("tor_oc_ugm3",), "TOR OC") * _column(meta, ("volume_m3",), "volume")).to_numpy(float)
    tables, sweeps = [], []
    for model in models:
        if model in CONTINUUM:
            if continuum is None:
                raise SystemExit(f"{model} needs --continuum-output (the directory holding corrected_{CONTINUUM[model]}_improve.npy)")
            path = continuum / f"corrected_{CONTINUUM[model]}_improve.npy"
        else:
            path = bundle / f"improve_{model}.npy"
        x = np.load(path, mmap_mode="r")
        if len(x) != len(meta):
            raise ValueError(f"{path.name} has {len(x)} rows, improve_meta.csv has {len(meta)}: they must be aligned")
        keep = np.isfinite(ec) & (ec > 0) & (ec < 150) & np.isfinite(oc) & np.isfinite(np.asarray(x[:, 0], float))
        if "raw_qc_ok" in meta.columns:                      # the closeout's one recorded raw-spectrum QC exclusion
            keep &= meta["raw_qc_ok"].astype(bool).to_numpy()
        print(f"bundle {model}: {int(keep.sum())} of {len(meta)} filters, {x.shape[1]} channels")
        table, sweep = evaluate(PREP_NAMES.get(model, model), np.asarray(x[keep], float), ec[keep], oc[keep], sites[keep], dates[keep], n_boot=n_boot)
        tables.append(table)
        sweeps.append(sweep)
    return pd.concat(tables, ignore_index=True), pd.concat(sweeps, ignore_index=True)


def from_closeout(n_boot):
    """The saved held-out predictions of each closeout calibration. Only 1,821 heavier filters were exported, so the
    response here is indicative: compare its Raw spectra row with the full refit before reading the others."""
    improve = pd.DataFrame(json.loads(CLOSEOUT.read_text())["improve_filters"])
    ec, sites, dates = improve["tor_ug"].to_numpy(float), improve["site"].to_numpy(), improve["date"].to_numpy()
    oc = ec * improve["ocec"].to_numpy(float)
    rows = []
    for key, label in CLOSEOUT_NAMES.items():
        prediction = improve[key].to_numpy(float)
        rows.append({"calibration": label, "filters": len(improve), **cg.response(prediction, ec, oc, sites, min_filters=8, n_boot=n_boot),
                     **{k: v for k, v in cg.fair_scores(prediction, ec, sites, dates, URBAN).items() if k.startswith(("urban", "heavy"))}})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive", action="store_true", help="raw spectra from the IMPROVE tables on the Drive")
    parser.add_argument("--closeout", action="store_true", help="the saved predictions of the 7 Nov closeout calibrations")
    parser.add_argument("--bundle", type=Path, help="decision bundle directory with improve_meta.csv and improve_<prep>.npy")
    parser.add_argument("--continuum-output", type=Path, help="directory with corrected_<variant>_improve.npy")
    parser.add_argument("--models", nargs="+", default=["raw", "spline", "vibes"], help="preprocessings to read from the bundle")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--summary", nargs="?", const=SUMMARY, type=Path, help="also write the tables as one JSON file")
    args = parser.parse_args(argv)
    if not (args.drive or args.closeout or args.bundle):
        parser.error("choose at least one of --drive, --closeout, --bundle")

    args.output.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    variants, sweeps = [], []
    for source in ([from_drive(args.bootstrap)] if args.drive else []) + ([from_bundle(args.bundle, args.models, args.continuum_output, args.bootstrap)] if args.bundle else []):
        variants.append(source[0])
        sweeps.append(source[1])
    if variants:
        tables |= {"variants": pd.concat(variants, ignore_index=True), "components": pd.concat(sweeps, ignore_index=True)}
    if args.closeout:
        tables["closeout_calibrations"] = from_closeout(args.bootstrap)
    for name, frame in tables.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows")
    payload = {"generated_from": "research/ftir_hips_chem/workflows/run_calibration_gain_20261110.py",
               "settings": {"components": K, "folds": "5, whole IMPROVE sites held out", "blank_weight": BLANK_WEIGHT,
                            "light_ug": cg.LIGHT_UG, "heavy_ug": cg.HEAVY_UG},
               **{name: json.loads(frame.to_json(orient="records", double_precision=4)) for name, frame in tables.items()}}
    (args.output / "calibration_gain.json").write_text(json.dumps(payload, indent=1) + "\n")
    if args.summary:
        args.summary.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
