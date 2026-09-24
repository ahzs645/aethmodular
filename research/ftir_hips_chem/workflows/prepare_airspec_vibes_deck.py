"""Freeze source-backed values for the September 2026 comparison presentation.

Reads completed results only. Does not refit models or alter evaluation membership.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "deliverables/airspec_vibes_2026-09-22"
TABLES = ROOT / "research/ftir_hips_chem/output/tables"
CLOUD = TABLES / "vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
AUDIT = TABLES / "vibes_subgroup_audit"
SOURCES = {}


def record(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def csv(path):
    return pd.read_csv(record(path))


scores = csv(CLOUD / "calibration_scores.csv")
groups = csv(AUDIT / "subgroup_metrics.csv")
paired = csv(AUDIT / "paired_residuals.csv")
# Independent cross-check against the saved held-out predictions.
for row in scores.itertuples():
    group = groups[(groups.cohort == row.cohort) & (groups.dimension == "overall")].iloc[0]
    observations = paired[paired.cohort == row.cohort]
    assert len(observations) == row.n_test
    assert observations.sample_id.is_unique
    truth = observations.y.to_numpy()
    error = observations[row.method].to_numpy() - truth
    recovered = {
        "RMSE": np.sqrt(np.mean(error**2)),
        "MAE": np.mean(np.abs(error)),
        "bias": np.mean(error),
        "predictive_R2": 1 - np.sum(error**2) / np.sum((truth-truth.mean())**2),
    }
    for metric in ("RMSE", "MAE", "bias", "predictive_R2"):
        assert np.isclose(getattr(row, metric), group[f"{row.method}_{metric}"], atol=1e-10, rtol=0)
        assert np.isclose(getattr(row, metric), recovered[metric], atol=1e-10, rtol=0)
    assert row.n_test == group.n

z = np.load(record(TABLES / "vibes_case_investigation/inspection_spectra.npz"))
idx = list(z["sample_ids"]).index("improve:1970697")
# Full spectral resolution, sorted only for chart x coordinates. Never averaged.
order = np.argsort(z["wn"])
spectrum = {key: z[key][idx, order].tolist() for key in ("raw", "airspec", "vibes")}
spectrum["wn"] = z["wn"][order].tolist()
spectrum["sample_id"] = str(z["sample_ids"][idx])
cases = csv(TABLES / "vibes_case_investigation/case_evidence.csv")
cases = cases[cases.sample_id.isin(["improve:1970697", "improve:1973864"])]

for rel in (
    "docs/openresearch-retrospective/claim_updates.json",
    "docs/openresearch-retrospective/followthrough-2026-09-21.md",
    "research/ftir_hips_chem/output/tables/vibes_case_investigation/report.md",
    "research/ftir_hips_chem/output/tables/airspec_locked_reproduction/report.md",
    "research/ftir_hips_chem/output/tables/addis_validation_readiness/report.md",
    "research/ftir_hips_chem/output/tables/historical_gap_followthrough/report.md",
    "research/ftir_ec_phase3/scripts/airspec_baseline.py",
    "research/ftir_hips_chem/scripts/vibes_baseline.py",
    "research/ftir_hips_chem/scripts/vibes_large_run.py",
    "research/ftir_hips_chem/vendor/pyvibes/README.md",
    "research/ftir_hips_chem/vendor/pyvibes/CITATION.cff",
):
    record(ROOT / rel)

data = {
    "scores": scores.to_dict("records"),
    "overall": groups[groups.dimension == "overall"].to_dict("records"),
    "loading": groups[(groups.cohort == "full_pool") & (groups.dimension == "loading_band")].to_dict("records"),
    "blanks": csv(AUDIT / "blank_summary.csv").to_dict("records"),
    "injections": csv(AUDIT / "injection_summary.csv").to_dict("records"),
    "features": csv(AUDIT / "addis_feature_summary.csv").to_dict("records"),
    "cases": cases.to_dict("records"),
    "spectrum": spectrum,
    "run": json.loads(record(CLOUD / "RUN_MANIFEST.json").read_text()),
}
# Dimension spelling is checked instead of silently making an empty chart.
if len(data["loading"]) != 4:
    print("Available dimensions:", groups.dimension.unique())
    raise ValueError("Expected four loading quartiles")
for folder in (OUT / ".build", OUT / "output"):
    folder.mkdir(parents=True, exist_ok=True)
def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


(OUT / ".build/data.json").write_text(json.dumps(finite_json(data), indent=2, allow_nan=False))
(OUT / "output/source_manifest.json").write_text(json.dumps({
    "date": "2026-09-22", "source_sha256": SOURCES,
    "run_signature": data["run"]["signature"],
    "scope": "Completed frozen comparison and documented historical interpretations. No new model fitting.",
    "units": "EC prediction errors: micrograms/filter. Spectra: absorbance. No mass-to-concentration conversion.",
    "external_method_reference": "https://amt.copernicus.org/articles/12/2313/2019/",
    "pyvibes_reference": "Vendored pyvibes 1.0.0 README and CITATION.cff, Kamper, Krymova and Takahama.",
}, indent=2))
print(f"Prepared {len(SOURCES)} hashed sources and four reconciled calibration score rows")
