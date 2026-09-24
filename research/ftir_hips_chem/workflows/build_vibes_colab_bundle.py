"""Stage portable raw FTIR arrays + audited metadata, then package Colab code/data.

Run with uv run --extra vibes python <this file>. No cloud writes are performed.
"""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research/ftir_hips_chem"
sys.path.insert(0, str(AREA / "scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
from phase3_common import PATHS, load_tor_loadings, load_addis_evaluation
from vibes_comparison import load_comparison_data
from calibration_modes import protocol_train_mask

# --full-range stages the whole exported grid (4000-500 cm-1) for a VIBES-only
# run; the default reproduces the AIRSpec-window bundle (4000-1425 cm-1).
FULL_RANGE = "--full-range" in sys.argv
OUT = AREA / ("output/tables/vibes_colab_bundle_full500" if FULL_RANGE else "output/tables/vibes_colab_bundle")
STAGE = OUT / "stage"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage_data():
    data = STAGE / "data"
    data.mkdir(parents=True, exist_ok=True)
    wn, etad, ledger, source_hashes = load_comparison_data(full_range=FULL_RANGE)
    np.save(data / "wn.npy", wn)
    np.save(data / "etad_raw.npy", etad)
    eval_meta, _, _ = load_addis_evaluation()
    ledger = ledger.join(eval_meta.set_index("MediaId")[["Fabs", "EC_deployed_ugm3"]], how="left")
    ledger.to_csv(data / "etad_metadata.csv")
    raw_path = PATHS.ftir_dir / "local_db/spectra_248_251.csv"
    catalog_path = PATHS.ftir_dir / "local_db/tables/ftir_catalog.csv"
    tor_path = PATHS.ftir_dir / "local_db/tables/results_tor.csv"
    cohort_path = ROOT / "research/ftir_ec_phase3/output/tables/ftir11/lowest_ocec_800_cohort.csv"
    header = pd.read_csv(raw_path, nrows=0)
    wcols = [c for c in header if c not in ("AnalysisId", "FilterId", "SampleDate", "Site")]
    wcols = [c for c in wcols if (FULL_RANGE or 1425 < float(c)) and float(c) < 4000]
    wcols.sort(key=lambda c: -float(c))
    np.testing.assert_allclose(np.array(wcols, float), wn, rtol=0, atol=1e-8)
    raw = pd.read_csv(raw_path, usecols=["AnalysisId", "FilterId", "SampleDate", "Site"] + wcols)
    catalog = pd.read_csv(catalog_path, usecols=["AnalysisId", "FilterPurposeId", "LotNumber"])
    raw = raw.merge(catalog, how="left", on="AnalysisId", validate="one_to_one")
    raw["finite_scan"] = np.isfinite(raw[wcols].to_numpy()).all(axis=1)
    raw["date"] = pd.to_datetime(raw.SampleDate, format="mixed", errors="coerce").dt.normalize()
    grouped = raw.groupby("FilterId", sort=True)
    for col in ["Site", "date", "FilterPurposeId", "LotNumber"]:
        if (grouped[col].nunique() > 1).any():
            raise ValueError(f"Conflicting {col} across scans of a physical filter")
    # Keep every physical filter in the ledger. Missing catalog is not silently
    # interpreted as "sample", and missing scan channels are never mean-imputed.
    meta = (
        grouped[["AnalysisId", "Site", "date", "FilterPurposeId", "LotNumber"]]
        .first()
        .reset_index()
    )
    meta["n_scans"] = grouped.size().to_numpy()
    meta["finite_spectrum"] = grouped.finite_scan.all().to_numpy()
    meta["all_scans_have_purpose"] = (
        grouped.FilterPurposeId.count().to_numpy() == meta.n_scans.to_numpy()
    )
    X = grouped[wcols].mean().to_numpy(float)
    np.save(data / "pool_raw.npy", X)
    tor = load_tor_loadings()
    meta = meta.merge(tor, on=["Site", "date"], how="left", validate="many_to_one")
    locked = pd.read_csv(cohort_path)
    meta["locked800"] = meta.FilterId.isin(locked.FilterId)
    meta["eligible"] = (
        meta.finite_spectrum
        & meta.all_scans_have_purpose
        & meta.FilterPurposeId.eq(1)
        & meta.Site.notna()
        & np.isfinite(meta.TOR_EC_loading_ug)
        & meta.TOR_EC_loading_ug.gt(0)
    )
    meta["reason"] = np.select(
        [
            ~meta.finite_spectrum,
            ~meta.all_scans_have_purpose,
            meta.FilterPurposeId.eq(2),
            ~meta.FilterPurposeId.eq(1),
            ~meta.eligible,
        ],
        [
            "Nonfinite contributing scan",
            "Unverified filter purpose",
            "Field blank: background diagnostics only",
            "Not a verified sample",
            "Missing/nonpositive TOR EC or missing site",
        ],
        default="",
    )
    e = meta.eligible.to_numpy()
    train = protocol_train_mask(
        "site_heldout", X[e], meta.loc[e, "TOR_EC_loading_ug"], meta.loc[e, "Site"]
    )
    train_sites = set(meta.loc[e, "Site"].to_numpy()[train])
    test_sites = set(meta.loc[e, "Site"].to_numpy()[~train])
    assert train_sites.isdisjoint(test_sites)
    meta["split"] = np.where(
        meta.Site.isin(train_sites),
        "train",
        np.where(meta.Site.isin(test_sites), "test", "unused_site"),
    )
    meta["row"] = np.arange(len(meta))
    meta.to_csv(data / "pool_metadata.csv", index=False)
    for p in [raw_path, catalog_path, tor_path, cohort_path]:
        source_hashes[str(p)] = sha(p)
    info = {
        "source_sha256": source_hashes,
        "raw_pool_scans": len(raw),
        "physical_pool_filters": len(meta),
        "eligible_pool_filters": int(e.sum()),
        "locked800_eligible": int((meta.locked800 & meta.eligible).sum()),
        "pool_field_blanks": int(meta.FilterPurposeId.eq(2).sum()),
        "etad_pm25_filters": int(ledger.role.eq("sample").sum()),
        "channels": len(wn),
        "split_seed": 20260717,
        "train_sites": sorted(train_sites),
        "test_sites": sorted(test_sites),
        "purpose_mapping_reference": "research/spartan_ec_2026_06_16/recreation_app/app.py: 1=Sample, 2=Field Blank, 3=Lab Blank",
        "replicates": "Average raw scans by physical FilterId/MediaId before both baselines",
        "protocol": "One full-eligible-pool site split reused for both methods and both cohorts; not the historical cohort-specific split",
    }
    (data / "DATA_PROVENANCE.json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in info.items()
                if k not in ("source_sha256", "train_sites", "test_sites")
            },
            indent=2,
        ),
        flush=True,
    )


def package():
    targets = [
        (AREA / "scripts", STAGE / "research/ftir_hips_chem/scripts"),
        (AREA / "vendor/pyvibes", STAGE / "research/ftir_hips_chem/vendor/pyvibes"),
    ]
    for source, dest in targets:
        shutil.copytree(
            source,
            dest,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    phase = STAGE / "research/ftir_ec_phase3/scripts"
    phase.mkdir(parents=True, exist_ok=True)
    for name in ["airspec_baseline.py", "calibration_modes.py"]:
        shutil.copy2(ROOT / "research/ftir_ec_phase3/scripts" / name, phase / name)
    requirements = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "cvxpy",
        "clarabel",
        "joblib",
        "threadpoolctl",
        "tqdm",
    ]
    (STAGE / "requirements.txt").write_text(
        "\n".join(f"{p}=={importlib.metadata.version(p)}" for p in requirements) + "\n"
    )
    files = {
        str(p.relative_to(STAGE)): sha(p)
        for p in sorted(STAGE.rglob("*"))
        if p.is_file()
        and p.name != "BUNDLE_MANIFEST.json"
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }
    content_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (STAGE / "BUNDLE_MANIFEST.json").write_text(
        json.dumps({"content_hash": content_hash, "files_sha256": files}, indent=2) + "\n"
    )
    target = OUT / "vibes_large_run_bundle.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in [*files, "BUNDLE_MANIFEST.json"]:
            archive.write(STAGE / name, name)
    (OUT / (target.name + ".sha256")).write_text(sha(target) + "  " + target.name + "\n")
    print(f"Bundle: {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    return target


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--code-only" not in sys.argv:
        stage_data()
    package()
