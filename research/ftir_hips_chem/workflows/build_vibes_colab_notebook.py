"""Generate the portable long-run notebook; build the data/code bundle first."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research/ftir_hips_chem"
BUNDLE = AREA / "output/tables/vibes_colab_bundle/vibes_large_run_bundle.zip"
checksum = (BUNDLE.parent / (BUNDLE.name + ".sha256")).read_text().split()[0]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.metadata = {
    "colab": {"name": "VIBES_AIRSpec_Large_Run_Colab.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nb.cells = [
    md("""# Large FTIR comparison: VIBES versus AIRSpec

This notebook compares **separately refitted PLS calibrations**, not just corrected
spectra. The staged data contain 13,634 IMPROVE scans (13,632 physical filters),
**12,393 eligible calibration filters**, 338 labeled IMPROVE field blanks, and all
253 Addis PM2.5 filters plus the 37 Addis field blanks.

### Before running

1. Upload `vibes_large_run_bundle.zip` to **My Drive → Aethmodular VIBES**.
2. Open this notebook in Google Colab and choose a **CPU runtime**. The solvers use
   CPUs; allocating a GPU will not accelerate this implementation.
3. Run the cells in order. Authorize your own Drive mount when Colab asks.
4. Leave `PROFILE="full"` for the large experiment, or choose `"pilot"` first.

The bundle includes raw numeric arrays, metadata, exact analysis code, the supplied
MIT-licensed pyvibes source, and a hash manifest. No GitHub checkout or local Mac
paths are needed. Python scientific packages are installed into an **isolated
virtual environment**; the notebook itself only displays the saved tables/figures.

Results and atomic batch checkpoints go to your Drive. After a runtime reset,
run the same cells with the same settings: completed batches are reused. Colab
runtime duration and resources vary; see the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).
"""),
    code(f'''from pathlib import Path
import os, sys, json, subprocess, hashlib, shutil, zipfile, csv, html
from IPython.display import display, Markdown, Image, HTML

PROFILE = os.environ.get("AETH_VIBES_PROFILE", "full")  # "smoke", "pilot", "full"
WORKERS = 2  # Modest CPU parallelism; worker BLAS thread count is limited to one.
BATCH_SIZE = 64
MAX_PLS_COMPONENTS = 30
TAU = 0.1
BACKGROUND_SOURCE = "combined"  # "etad_only" reproduces the smaller background bank.
RUN_SMOKE_FIRST = True
BUNDLE_NAME = "vibes_large_run_bundle.zip"
EXPECTED_BUNDLE_SHA256 = "{checksum}"

# Local override is only for testing this exact notebook outside Colab.
LOCAL_BUNDLE = os.environ.get("AETH_VIBES_LOCAL_BUNDLE")
if LOCAL_BUNDLE:
    ARCHIVE = Path(LOCAL_BUNDLE).resolve()
    SESSION_ROOT = Path(os.environ["AETH_VIBES_LOCAL_WORK"]).resolve()
    RESULTS_ROOT = SESSION_ROOT / "persistent_results"
else:
    from google.colab import drive
    drive.mount("/content/drive")
    ARCHIVE = Path("/content/drive/MyDrive/Aethmodular VIBES") / BUNDLE_NAME
    SESSION_ROOT = Path("/content/aeth_vibes")
    RESULTS_ROOT = Path("/content/drive/MyDrive/Aethmodular VIBES/results")
SESSION_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
if not ARCHIVE.is_file():
    raise FileNotFoundError(f"Upload the bundle to {{ARCHIVE}} or edit ARCHIVE above.")
print("Bundle:", ARCHIVE)
print("Persistent results:", RESULTS_ROOT)
'''),
    md("""## 1. Copy locally and verify the portable inputs

One archive is copied from Drive and extracted on the runtime's local disk. This
reduces repeated Drive reads. The archive checksum and every bundled file are
verified; a changed dataset or implementation receives a different checkpoint ID.
"""),
    code("""def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

LOCAL_ARCHIVE = SESSION_ROOT / BUNDLE_NAME
if not LOCAL_ARCHIVE.exists() or file_sha(LOCAL_ARCHIVE) != EXPECTED_BUNDLE_SHA256:
    shutil.copy2(ARCHIVE, LOCAL_ARCHIVE)
if file_sha(LOCAL_ARCHIVE) != EXPECTED_BUNDLE_SHA256:
    raise ValueError("Bundle checksum mismatch. Use the notebook and bundle from the same build.")
WORK = SESSION_ROOT / ("bundle-" + EXPECTED_BUNDLE_SHA256[:12])
WORK.mkdir(exist_ok=True)
with zipfile.ZipFile(LOCAL_ARCHIVE) as archive:
    for member in archive.infolist():
        if not (WORK / member.filename).resolve().is_relative_to(WORK.resolve()):
            raise ValueError("Unsafe archive member")
    archive.extractall(WORK)
manifest = json.loads((WORK / "BUNDLE_MANIFEST.json").read_text())
for name, expected in manifest["files_sha256"].items():
    if file_sha(WORK / name) != expected:
        raise ValueError("Corrupted bundle member: " + name)
provenance = json.loads((WORK / "data/DATA_PROVENANCE.json").read_text())
display(Markdown("**Input counts**"))
display({k: v for k, v in provenance.items()
         if k not in ("source_sha256", "train_sites", "test_sites")})
print("Disjoint sites:", len(provenance["train_sites"]), "train;", len(provenance["test_sites"]), "test")
assert set(provenance["train_sites"]).isdisjoint(provenance["test_sites"])
"""),
    md("""## 2. Install the isolated analysis environment

No scientific packages in the notebook kernel are upgraded. This avoids conflicts
with Colab's preinstalled TensorFlow/NumPy stack. The runtime must use Python 3.11
or later; all package versions used for the local validation are pinned in the bundle.
The first installation may take a few minutes; rerunning this cell reuses it.
"""),
    code("""if sys.version_info < (3, 11):
    raise RuntimeError("Use a Colab Python runtime >=3.11 for the pinned analysis environment.")
ENV = SESSION_ROOT / "analysis_env"
PYTHON = ENV / "bin/python"
if not PYTHON.exists():
    creation = subprocess.run([sys.executable, "-m", "venv", str(ENV)])
    if creation.returncode:
        # Some hosted distributions omit ensurepip, which stdlib venv requires.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"], check=True)
        subprocess.run([sys.executable, "-m", "virtualenv", str(ENV)], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "install", "-q", "-r", str(WORK / "requirements.txt")], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "check"], check=True)
RUNNER = WORK / "research/ftir_hips_chem/scripts/vibes_large_run.py"
print("Isolated interpreter:", PYTHON)
"""),
    md("""## 3. Experimental design and audit

- Both methods get identical **raw replicate-averaged physical filters** and the
  same 2,002-channel grid (approximately 1426–3998 cm⁻¹). No negative-value clipping.
- Eligible IMPROVE calibration rows have verified sample purpose, complete spectra,
  a site label, and finite positive TOR EC loading. The full inclusion ledger is
  `data/pool_metadata.csv`; unknown purpose is not guessed from low loading.
- One full-pool **site-disjoint 80/20 split**, seed 20260717, is frozen before any
  fit. The same site assignment applies to both baselines and both cohorts.
- The historical lowest-OC/EC 800 list is carried forward as fixed membership:
  **762 remain eligible** in this audited export. This run uses a common full-pool
  split, so it is **not numerically interchangeable** with the old cohort-specific
  split. The cohort was historically selected with OC/EC labels; it is a fixed
  restricted-domain study, not an unbiased sample of the whole network.
- The VIBES bank contains up to 50 IMPROVE blanks per lot from **training sites
  only**, with a quarter reserved first, plus the original 28 Addis training
  blanks. Outer-test-site blanks and all nine Addis held-out blanks are excluded.
  This includes independent target-site blank adaptation, never target aerosols.
- Blank-only LOO chooses PCA rank, capped at 30. The cap and whether selection hits
  it are reported. VIBES uses PB loss, τ=0.1, and logged optimizer continuation;
  AIRSpec uses DF1=6/DF2=4. Neither is tuned to the held-out TOR results.
- Fit PLS separately for each baseline. Select component count using five-fold
  **site-grouped CV on training rows only** and the existing first-major-minimum
  rule. Final comparison uses identical valid train/test rows for both methods.
- Report predictive R², RMSE, MAE and bias, plus a **paired site-bootstrap** interval
  for ΔRMSE. A confidence interval spanning zero is inconclusive.
- Addis transfer is shown against HIPS EC-equivalent as an optical comparison,
  **not independent thermal EC ground truth**. A change in intercept alone does
  not establish improved EC accuracy.

Blank diagnostics and known-peak addition tests supplement the calibration result.
Multiple amplitudes on one blank are repeated measures, not independent filters.
"""),
    code("""def show_table(path, limit=25):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []
    def td(value):
        return "<td>" + html.escape(str(value)) + "</td>"
    body = "<tr>" + "".join("<th>" + html.escape(c) + "</th>" for c in columns) + "</tr>"
    body += "".join("<tr>" + "".join(td(row[c]) for c in columns) + "</tr>" for row in rows[:limit])
    display(HTML("<div style='overflow:auto'><table>" + body + "</table></div>"))
    if len(rows) > limit:
        print(f"Showing {limit}/{len(rows)} rows; full table: {path}")

show_table(WORK / "data/pool_metadata.csv", limit=5)

def run_profile(profile):
    config = {"profile": profile, "workers": WORKERS, "batch_size": BATCH_SIZE,
              "max_pls_components": 5 if profile == "smoke" else MAX_PLS_COMPONENTS,
              "tau": TAU, "background_source": BACKGROUND_SOURCE}
    config_path = SESSION_ROOT / (profile + "_config.json")
    pointer = SESSION_ROOT / (profile + "_result.json")
    config_path.write_text(json.dumps(config, indent=2))
    env = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(PYTHON), "-u", str(RUNNER), "--bundle", str(WORK),
               "--results", str(RESULTS_ROOT), "--config", str(config_path), "--pointer", str(pointer)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Run failed; completed checkpoints remain on Drive. See the error above.")
    return Path(json.loads(pointer.read_text())["result_path"])
"""),
    md("""## 4. Small end-to-end check

The smoke profile uses 64 calibration filters (32 historical-cohort members and
32 others), 12 Addis targets, held-out blanks and known additions. This checks
installation, parallel workers, checkpoint writes, CV, predictions and plotting.
Its small held-out set is **not a scientific result to generalize**.
"""),
    code("""if RUN_SMOKE_FIRST:
    SMOKE_RESULTS = run_profile("smoke")
    show_table(SMOKE_RESULTS / "calibration_scores.csv")
"""),
    md("""## 5. Large run (resumable)

`full` uses all 12,393 eligible calibration filters, all 253 Addis PM2.5 filters,
all reserved blank filters, and 27 ETAD peak-injection cases. `pilot` limits the
calibration set to the available historical cohort plus 400 other filters.

This is the long cell. Each batch writes one compressed checkpoint directly to
Drive using a temporary file followed by replacement. Progress reports completed
and resumed rows. Failed VIBES fits are saved explicitly; paired model fits use
only rows valid under both methods. The outer site assignment is never redrawn.

**Resume:** after disconnecting/restarting Colab, rerun the notebook with the same
bundle and settings. Batch identity includes input/code hashes, scientific settings
and package versions. Changing those starts a separate run. You may change the
worker count without invalidating completed numerical results; batch size is fixed
within a run. One-time PCA and final PLS/report generation are rerun on resume.
"""),
    code("""RESULTS = SMOKE_RESULTS if PROFILE == "smoke" and RUN_SMOKE_FIRST else run_profile(PROFILE)
run_manifest = json.loads((RESULTS / "RUN_MANIFEST.json").read_text())
display(run_manifest)
if run_manifest["background_rank_hits_cap"]:
    display(Markdown("**PCA rank hit the configured cap. Treat the blank model as capped; test a larger cap as a separately labeled sensitivity run.**"))
print("Saved results:", RESULTS)
"""),
    md("""## 6. Held-out calibration performance

Read predictive R² and error metrics from this table. The scatterplot statistics
box reports squared correlation separately. Deming slopes on 1:1 TOR panels use
λ=1 descriptively because a full TOR/prediction error-variance model is unavailable.
"""),
    code("""show_table(RESULTS / "calibration_scores.csv")
show_table(RESULTS / "paired_site_bootstrap.csv")
for name in ["01_training_only_cv", "02_heldout_tor", "03_paired_model_comparison"]:
    display(Image(filename=str(RESULTS / "plots" / (name + ".png"))))
"""),
    md("""## 7. Addis transfer, spectral features, blanks and recovery

The same target filters are predicted by each newly fitted calibration.
The HIPS association is descriptive and is not used to choose a winning model.
Band comparisons use the existing local-continuum feature definitions. Blank and
injection plots use paired successful rows under both methods.
"""),
    code("""for name in ["04_addis_transfer", "05_addis_band_features", "06_blanks_and_recovery", "07_runtime_and_coverage", "08_matched_spectral_overlays"]:
    display(Image(filename=str(RESULTS / "plots" / (name + ".png"))))
show_table(RESULTS / "addis_predictions.csv", limit=8)
"""),
    md("""## 8. Audit numerical failures and save the report

A promising baseline should improve held-out TOR performance consistently, preserve
known additions, and behave reasonably on independent blanks. Better agreement
with AIRSpec alone is not a success criterion. Do not tune settings against the
held-out results and then describe the same test set as untouched validation.

The ZIP below contains tables, eight figure families, preprocessing provenance,
and portable PLS coefficients. Large corrected arrays and resumable checkpoints
remain separately in the run folder on Drive. `pls_*.npz` stores coefficients,
training X mean, and training y mean; for these unscaled models predict loading as
`(X - x_mean) @ coefficient.T + y_mean`, then divide by sample volume for µg/m³.
"""),
    code("""with (RESULTS / "fit_diagnostics.csv").open(newline="") as handle:
    diagnostics = list(csv.DictReader(handle))
failures = [row for row in diagnostics if row["success"].lower() != "true"]
retries = [row for row in diagnostics if int(row["retry_count"]) > 0]
print(f"{len(diagnostics)} attempts, {len(failures)} failures, {len(retries)} logged retries")
if failures:
    display(failures[:10])
report = RESULTS / "comparison_report.zip"
with zipfile.ZipFile(report, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RESULTS.iterdir()):
        if path.suffix in (".csv", ".json") or path.name.startswith("pls_"):
            archive.write(path, path.name)
    for path in sorted((RESULTS / "plots").glob("*.png")):
        archive.write(path, "plots/" + path.name)
print("Report:", report)
print("Notebook outputs contain your research data; save/share deliberately.")
"""),
]
path = AREA / "colab/VIBES_AIRSpec_Large_Run_Colab.ipynb"
nbf.write(nb, path)
print(path)
