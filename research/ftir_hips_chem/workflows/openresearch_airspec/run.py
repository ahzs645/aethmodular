"""Run the unchanged AIRSpec calibration from an OpenResearch source archive.

Data are copied from a content-addressed local bundle, verified before and after
execution, and resolved exclusively within this run. No live Drive data are used.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[4]
CONTRACT = ROOT / "docs/openresearch-execution/airspec"
OUTPUT = ROOT / "research/ftir_hips_chem/output/tables/openresearch_airspec_run"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_file(path, expected):
    if not Path(path).is_file() or sha(path) != expected:
        raise ValueError(f"Missing or changed pinned file: {path}")


def safe_relative(path):
    value = Path(path)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Unsafe manifest path: {path}")
    return value


def compare_table(expected, actual, tolerance, name):
    if list(expected.columns) != list(actual.columns) or expected.shape != actual.shape:
        return [{"table": name, "column": "<structure>", "passed": False,
                 "max_abs_difference": None}]
    checks = []
    for column in expected.columns:
        a, b = expected[column], actual[column]
        numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        if numeric:
            av, bv = a.to_numpy(float), b.to_numpy(float)
            passed = bool(np.allclose(av, bv, rtol=0, atol=tolerance, equal_nan=True))
            finite = np.isfinite(av) & np.isfinite(bv)
            delta = float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else 0.0
        else:
            passed = a.equals(b)
            delta = None
        checks.append({"table": name, "column": column, "passed": passed,
                       "max_abs_difference": delta})
    return checks


def export_calibration(ns, out):
    ids = ns["ocec"]["AnalysisId"].to_numpy()
    y, sites = ns["ocec_y"], ns["ocec_sites"]
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                      random_state=ns["SPLIT_SEED"]).split(ids, groups=sites))
    if not set(sites[train]).isdisjoint(sites[test]):
        raise ValueError("Training and test sites overlap")
    membership = []
    for partition, positions in [("train", train), ("test", test)]:
        membership.extend({"cohort": "lowest-OCEC 800", "AnalysisId": int(ids[i]),
                           "Site": str(sites[i]), "partition": partition, "y": float(y[i])}
                          for i in positions)
    pd.DataFrame(membership).to_csv(out / "split_membership.csv", index=False)
    metrics, curves = [], []
    for df1, (model, k, curve, heldout) in ns["ocec_fits"].items():
        metrics.append({"cohort": "lowest-OCEC 800", "df1": df1, "k": int(k), **heldout})
        curves.append(curve.assign(df1=df1))
        predictions = model.predict(ns["corrected_pool_rows"](ids, df1)).ravel()
        pd.DataFrame({"AnalysisId": ids, "Site": sites, "y": y, "prediction": predictions,
                      "partition": np.where(np.isin(np.arange(len(ids)), test), "test", "train")}
                     ).to_csv(out / f"ocec_df{df1}_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "heldout_metrics.csv", index=False)
    pd.concat(curves).to_csv(out / "component_curves.csv", index=False)
    pd.DataFrame([{"df1": d, "k": int(f[1])} for d, f in ns["smoke_fits"].items()]
                 ).to_csv(out / "smoke_components.csv", index=False)
    return {"train": len(train), "test": len(test)}, metrics


def main():
    if OUTPUT.exists():
        raise ValueError("Use a fresh OpenResearch run directory; output already exists")
    OUTPUT.mkdir(parents=True)
    config = json.loads((CONTRACT / "config.json").read_text())
    summary = {"status": "failed", "scope": config["scope"],
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "bundle_id": config["bundle_id"], "run_root": str(ROOT),
               "environment": {"python": sys.version, **{n: importlib.metadata.version(n)
                               for n in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]}}}
    staged, reads = {}, set()
    try:
        verify_file(CONTRACT / "inputs.json", config["inputs_manifest_sha256"])
        bundle = Path.home() / ".local/share/aethmodular/input-bundles" / config["bundle_id"]
        verify_file(bundle / "manifest.json", config["inputs_manifest_sha256"])
        entries = json.loads((CONTRACT / "inputs.json").read_text())
        for entry in entries:
            relative = safe_relative(entry["path"])
            source, target = bundle / relative, ROOT / relative
            verify_file(source, entry["sha256"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                verify_file(target, entry["sha256"])
            else:
                shutil.copyfile(source, target)
            verify_file(target, entry["sha256"])
            target.chmod(0o444)
            staged[target.resolve()] = entry["sha256"]
        for relative, expected in config["source_hashes"].items():
            verify_file(ROOT / safe_relative(relative), expected)
        for group, folder in [("expected_tables", "expected"),
                              ("previous_local_tables", "previous-local")]:
            for relative, expected in config[group].items():
                verify_file(CONTRACT / folder / safe_relative(relative), expected)
        # Derive location overrides from the committed recipe, never the caller's
        # environment. Existing loaders retain their original scientific behavior.
        os.environ["AETHMODULAR_DRIVE_ROOT"] = str(ROOT / ".drive_cache")
        os.environ["AETHMODULAR_DATA_ROOT"] = str(ROOT / "research/ftir_hips_chem")
        phase3 = ROOT / "research/ftir_ec_phase3"
        allowed = set(staged)
        allowed.update(p.resolve() for p in CONTRACT.rglob("*.csv"))
        generated = [OUTPUT.resolve(), (phase3 / "output/tables/ftir13").resolve()]
        def audit(event, args):
            if event != "open":
                return
            name, mode, flags = args
            if not isinstance(name, (str, os.PathLike)):
                return
            if (mode and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR):
                return
            path = Path(name).resolve()
            if path.suffix.lower() not in {".csv", ".npz", ".npy", ".pkl"}:
                return
            if path not in allowed and not any(path.is_relative_to(p) for p in generated):
                raise ValueError(f"Undeclared scientific input read: {path}")
            reads.add(str(path.relative_to(ROOT)))
        sys.addaudithook(audit)
        print("ORX_EFFECTIVE_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
        print(f"ORX_INPUTS_VERIFIED files={len(staged)} bundle={config['bundle_id']}", flush=True)
        os.chdir(phase3)
        ns = runpy.run_path(str(phase3 / "scripts/run_ftir_13.py"), run_name="__reproduction__")
        splits, metrics = export_calibration(ns, OUTPUT)
        checks = []
        reproduced = OUTPUT / "reproduced_tables"
        reproduced.mkdir()
        for name in config["expected_tables"]:
            actual = phase3 / "output/tables/ftir13" / name
            checks.extend(compare_table(pd.read_csv(CONTRACT / "expected" / name),
                          pd.read_csv(actual), config["absolute_tolerance"], "historical/" + name))
            shutil.copy2(actual, reproduced / name)
        for name in config["previous_local_tables"]:
            checks.extend(compare_table(pd.read_csv(CONTRACT / "previous-local" / name),
                          pd.read_csv(OUTPUT / name), config["absolute_tolerance"], "previous-local/" + name))
        pd.DataFrame(checks).to_csv(OUTPUT / "table_comparison.csv", index=False)
        components = {str(d): int(f[1]) for d, f in ns["ocec_fits"].items()}
        for path, expected in staged.items():
            verify_file(path, expected)
        for relative, expected in config["source_hashes"].items():
            verify_file(ROOT / relative, expected)
        if not all(c["passed"] for c in checks):
            raise ValueError("One or more result comparisons failed")
        if components != config["expected_components"] or splits != config["expected_split"]:
            raise ValueError("Components or split sizes changed")
        summary.update(status="passed", inputs_verified=len(staged), source_files_verified=len(config["source_hashes"]),
                       inputs_unchanged=True, split=splits, sites_disjoint=True,
                       components=components, heldout_metrics=metrics,
                       historical_tables=len(config["expected_tables"]),
                       historical_column_checks=sum(c["table"].startswith("historical/") for c in checks),
                       previous_local_tables=len(config["previous_local_tables"]),
                       total_column_checks=len(checks),
                       max_abs_difference=max(c["max_abs_difference"] or 0 for c in checks))
    except Exception as error:
        summary["error"] = str(error)
        traceback.print_exc()
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["scientific_reads"] = sorted(reads)
        (OUTPUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        artifacts = {str(p.relative_to(OUTPUT)): sha(p) for p in OUTPUT.rglob("*") if p.is_file()}
        (OUTPUT / "artifact_hashes.json").write_text(json.dumps(artifacts, indent=2) + "\n")
        print("ORX_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
