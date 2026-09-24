"""Audit the completed Colab comparison without refitting or altering its files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research/ftir_hips_chem"
sys.path.insert(0, str(AREA / "scripts"))
from vibes_error_audit import audit_groups, pair_predictions


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(frame, columns):
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3e}" if value != 0 and abs(value) < .001 else f"{value:.4f}"
        return str(value)
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(cell(v) for v in row) + " |" for row in frame[columns].itertuples(index=False, name=None)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=AREA / "output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09")
    parser.add_argument("--output", type=Path, default=AREA / "output/tables/vibes_subgroup_audit")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if source == out or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Keep the frozen source and audit output separate")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "RUN_MANIFEST.json").read_text())
    input_paths = sorted(p for p in source.iterdir() if p.is_file())
    notebook = AREA / "notebooks/archive/executed/VIBES_AIRSpec_Colab_full_executed.ipynb"
    bundle = AREA / "output/tables/vibes_colab_bundle/vibes_large_run_bundle.zip"
    completion_path = source.parents[1] / "COMPLETION.json"
    completion = json.loads(completion_path.read_text())
    input_paths.extend([notebook, bundle, completion_path])
    input_paths.extend(sorted((source / "plots").glob("*.png")))
    before = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    with zipfile.ZipFile(bundle) as archive:
        bundle_manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        files = bundle_manifest["files_sha256"]
        content_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        if content_hash != manifest["bundle_hash"] or content_hash != bundle_manifest["content_hash"]:
            raise ValueError("Local input bundle differs from the completed run")
        for name, expected in files.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Bundle member mismatch: {name}")
    nb = json.loads(notebook.read_text())
    nb_errors = [o for c in nb["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if nb_errors:
        raise ValueError("Executed notebook has saved errors")
    cases = pd.read_csv(source / "case_audit.csv")
    pd.testing.assert_frame_equal(cases.drop(columns="paired_valid"), pd.read_csv(source / "cases.csv"))
    fits = pd.read_csv(source / "fit_diagnostics.csv")
    if len(cases) != manifest["n_cases"] or len(fits) != len(cases):
        raise ValueError("Case or fit count mismatch")
    if not fits.sample_id.equals(cases.sample_id) or not fits.success.all() or manifest["n_failed"] != 0:
        raise ValueError("Incomplete or misaligned completed fits")
    if (completion["state"] != "complete" or completion["completed"] != len(cases)
            or completion["total"] != len(cases) or completion["failed_fits"] != 0
            or completion["figures"] != len(list((source / "plots").glob("*.png")))
            or int(fits.retry_count.gt(0).sum()) != manifest["n_retried"]):
        raise ValueError("Completion receipt or retry count mismatch")
    paired = pair_predictions(pd.read_csv(source / "heldout_predictions.csv"), cases)
    groups, bands, residuals, influence = audit_groups(paired, cases, repeats=args.bootstrap_repeats)
    loading_influence = []
    for (cohort, band), frame in residuals.groupby(["cohort", "loading_band"]):
        for site, part in frame.groupby("Site"):
            remaining = frame.loc[~frame.Site.eq(site)]
            delta = np.nan if remaining.empty else (
                np.sqrt(np.mean(remaining.VIBES_error**2)) - np.sqrt(np.mean(remaining.AIRSpec_error**2))
            )
            loading_influence.append(dict(cohort=cohort, loading_band=band, site=site, n=len(part),
                                          contribution_to_band_delta_MSE=part.delta_squared_error.sum()/len(frame),
                                          delta_RMSE_without_site=delta))
    loading_influence = pd.DataFrame(loading_influence)
    scores = pd.read_csv(source / "calibration_scores.csv")
    checks = []
    for score in scores.itertuples():
        g = groups.loc[groups.cohort.eq(score.cohort) & groups.dimension.eq("overall")].iloc[0]
        for metric in ("RMSE", "MAE", "bias", "predictive_R2"):
            actual, expected = g[f"{score.method}_{metric}"], getattr(score, metric)
            if not np.isclose(actual, expected, rtol=0, atol=1e-10):
                raise ValueError(f"Saved {metric} mismatch: {score.cohort} {score.method}")
            checks.append(dict(cohort=score.cohort, method=score.method, metric=metric, absolute_error=abs(actual - expected)))
        if g.n != score.n_test or int(bands.loc[bands.cohort.eq(score.cohort), "n_train"].iloc[0]) != score.n_train:
            raise ValueError("Saved population count mismatch")
    # Every partition must recover the exact full test population.
    for cohort, frame in groups.groupby("cohort"):
        n = int(frame.loc[frame.dimension.eq("overall"), "n"].iloc[0])
        for dimension in ("Site", "lot", "loading_band"):
            if frame.loc[frame.dimension.eq(dimension), "n"].sum() != n:
                raise ValueError(f"Partition count mismatch: {cohort} {dimension}")
    blank = pd.read_csv(source / "blank_metrics.csv")
    blank_summary = blank.groupby(["source", "method"]).agg(n=("sample_id", "size"), median_rms=("rms_from_zero", "median")).reset_index()
    injections = pd.read_csv(source / "injection_metrics.csv")
    injection_summary = injections.groupby(["amplitude", "method"]).agg(n=("sample_id", "size"), independent_parents=("parent_id", "nunique"), median_recovery_rmse=("recovery_rmse", "median")).reset_index()
    features = pd.read_csv(source / "spectral_features.csv")
    feature_rows = []
    for feature in ("CH_peak", "carbonyl_peak", "shoulder_1600_peak"):
        wide = features.loc[features.kind.eq("target")].pivot(index="sample_id", columns="method", values=feature)
        if len(wide) != int(cases.kind.eq("target").sum()) or not np.isfinite(wide).all().all():
            raise ValueError("Incomplete external feature comparison")
        feature_rows.append(dict(feature=feature, n=len(wide), association_R2=wide.corr().loc["AIRSpec", "VIBES"]**2,
                                 median_AIRSpec=wide.AIRSpec.median(), median_VIBES=wide.VIBES.median(),
                                 median_paired_difference=(wide.VIBES-wide.AIRSpec).median()))
    feature_summary = pd.DataFrame(feature_rows)
    outputs = {"subgroup_metrics.csv": groups, "training_loading_bands.csv": bands,
               "paired_residuals.csv": residuals, "site_influence.csv": influence,
               "loading_site_sensitivity.csv": loading_influence,
               "blank_summary.csv": blank_summary, "injection_summary.csv": injection_summary,
               "addis_feature_summary.csv": feature_summary}
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    columns = ["cohort", "group", "n", "n_sites", "AIRSpec_RMSE", "VIBES_RMSE", "delta_RMSE", "delta_RMSE_ci_low", "delta_RMSE_ci_high"]
    q3 = groups.loc[groups.cohort.eq("full_pool") & groups.dimension.eq("loading_band") & groups.group.eq("Q3")].iloc[0]
    q3_sites = loading_influence.loc[loading_influence.cohort.eq("full_pool") & loading_influence.loading_band.eq("Q3")]
    full_bands = bands.loc[bands.cohort.eq("full_pool")].iloc[0]
    full_loo = influence.loc[influence.cohort.eq("full_pool"), "delta_RMSE_without_site"]
    report = ["# VIBES/AIRSpec audit of saved predictions", "",
              "The full Colab run completed 12,808 cases with zero failed fits. This audit reconciles every held-out prediction and reuses the frozen fitted models. AIRSpec remains the current EC default; the pooled evidence does not establish an accuracy gain from VIBES.", "",
              "All EC errors below are µg/filter. Delta means VIBES minus AIRSpec; negative RMSE/MAE deltas favor VIBES. Bias is prediction minus reference, so a more negative bias is not automatically better. Predictive R² in the CSV means 1−SSE/SST, not squared correlation.", "",
              "## Main findings", "",
              f"The clearest exploratory loading pattern is full-pool Q3 ({full_bands.q50:.3f}–{full_bands.q75:.3f} µg/filter, lower boundary excluded): {int(q3.n)} filters across {int(q3.n_sites)} sites. VIBES increases RMSE by {q3.delta_RMSE:.3f} (95% pointwise interval {q3.delta_RMSE_ci_low:.3f} to {q3.delta_RMSE_ci_high:.3f}) and MAE by {q3.delta_MAE:.3f}. Mean biases are almost the same, so a common offset alone does not explain this gap.", "",
              f"Within that band, {(q3_sites.contribution_to_band_delta_MSE > 0).sum()} of {len(q3_sites)} sites have higher VIBES MSE. The RMSE delta stays between {q3_sites.delta_RMSE_without_site.min():.3f} and {q3_sites.delta_RMSE_without_site.max():.3f} when each site is omitted in turn. BRIS1 and CACR1 warrant source-spectrum inspection; this is a post-hoc diagnostic priority, not a reason to exclude their filters.", "",
              f"For the complete full-pool population, the leave-one-site-out RMSE delta ranges from {full_loo.min():.3f} to {full_loo.max():.3f}. Both lot intervals include zero. The restricted cohort's apparent RMSE benefit accompanies higher MAE, and its pooled uncertainty includes zero. None of these results establishes a subgroup switching rule.", "",
              "## Pooled comparison", "", table(groups.loc[groups.dimension.eq("overall")], columns), "",
              "## Filter lot", "", table(groups.loc[groups.dimension.eq("lot")], columns), "",
              "## EC loading", "", table(groups.loc[groups.dimension.eq("loading_band")], columns), "",
              "Training-only quartile thresholds (µg/filter):", "", table(bands, list(bands.columns)), "",
              "Q1 ≤ q25; Q2 (q25,q50]; Q3 (q50,q75]; Q4 > q75. Each cohort uses its own training thresholds. These define exploratory strata; they are not a new exclusion rule or a prediction-time method selector.", "",
              "## Site details and sensitivity", "",
              "Single-site RMSE, MAE, bias, predictive R² and paired deltas are in [subgroup_metrics.csv](subgroup_metrics.csv). Individual-site intervals are unavailable because each such group contains only one independent site cluster.", "",
              "[site_influence.csv](site_influence.csv) reports each site's additive contribution to the pooled MSE difference and the RMSE difference if that site is omitted for sensitivity only. The primary scores retain every filter. Positive contributions favor AIRSpec; negative contributions favor VIBES.", "",
              "Full-pool site contributions, sorted by absolute magnitude:", "",
              table(influence.loc[influence.cohort.eq("full_pool")].assign(magnitude=lambda d: abs(d.contribution_to_overall_delta_MSE)).sort_values("magnitude", ascending=False).head(10),
                    ["site", "n", "contribution_to_overall_delta_MSE", "delta_RMSE_without_site"]), "",
              "All paired residuals and prediction disagreements are retained in [paired_residuals.csv](paired_residuals.csv). No new exclusions were applied; the original frozen eligibility and paired-valid flags define membership.", "",
              "[loading_site_sensitivity.csv](loading_site_sensitivity.csv) repeats the site contribution and omission diagnostic within every loading band in both cohorts.", "",
              "## Spectroscopy is a separate outcome", "",
              "Blank RMS close to zero measures background removal. Synthetic recovery measures recovery of the injected shape. Neither establishes accuracy of ambient chemical concentrations or independent Addis thermal EC.", "",
              table(blank_summary, list(blank_summary.columns)), "",
              table(injection_summary, list(injection_summary.columns)), "",
              "The same nine independent parent blanks recur across amplitudes; the 27 injections are not 27 independent parent blanks. The recovery statistic here is the median per amplitude and method.", "",
              table(feature_summary, list(feature_summary.columns)), "",
              "Feature association R² is squared Pearson correlation between methods, not predictive validation. Feature magnitudes are in the saved spectral absorbance scale; paired shifts do not establish chemical truth. External HIPS association is not an independent thermal reference.", "",
              "## Interpretation limits and next experiment", "",
              f"- {args.bootstrap_repeats:,} paired bootstrap draws resample held-out sites with replacement, then calculate sample-weighted metrics. Both methods use the identical draw. The 95% percentile intervals condition on these fitted models; they omit training/refitting uncertainty.",
              "- These intervals use a different seed and more draws than the original 2,000-draw Colab intervals, so endpoints differ slightly. Predictions and point estimates are unchanged.",
              "- This is exploratory analysis after inspection of the test results. Intervals are pointwise, without multiplicity correction; no subgroup is a confirmed method-selection rule. Few-site intervals are labelled in the CSV.",
              "- Lots, sites and loadings are associated. Marginal subgroup differences do not identify causal lot effects. Cohorts have different test populations and cannot be ranked as competing training sets using these scores.",
              "- Use the audit to specify training-only grouped validation of a targeted hypothesis. Freeze a new external evaluation before further tuning. Independent Addis thermal EC is required to answer Addis accuracy.",
              "- Keep a separate spectroscopy follow-up: expand independent blank/standard validation across lots and weak-signal amplitudes before claiming improved chemical feature preservation.", "",
              "## Provenance and verification", "",
              f"External Colab signature: `{manifest['signature']}`. Input bundle content-manifest SHA-256: `{manifest['bundle_hash']}`. The full run is external evidence, not a newly executed OpenResearch run.", "",
              "[audit_manifest.json](audit_manifest.json) records input/output hashes, saved-score reconciliation and runtime versions. The local bundle hash matches the run manifest; the archived executed notebook has no saved errors. Input hashes were checked again after this audit.", ""]
    (out / "report.md").write_text("\n".join(report))
    after = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    if before != after:
        raise ValueError("Frozen input changed during audit")
    receipt = dict(
        status="passed", created_at_utc=datetime.now(timezone.utc).isoformat(),
        origin="local post-hoc audit of completed external Colab predictions; no refit",
        source_signature=manifest["signature"], source_bundle_hash=manifest["bundle_hash"],
        source_hashes=before, source_unchanged=True, input_bundle_matches=True,
        archived_notebook_errors=0, n_cases=len(cases), n_predictions_paired=len(paired),
        bootstrap=dict(repeats=args.bootstrap_repeats, seed=20260921, cluster="Site", confidence=0.95),
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
        score_reconciliation=checks,
        workflow_sha256=sha(Path(__file__)), analysis_sha256=sha(AREA / "scripts/vibes_error_audit.py"),
        output_hashes={name: sha(out / name) for name in [*outputs, "report.md"]},
    )
    (out / "audit_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(groups.loc[groups.dimension.isin(["overall", "lot", "loading_band"]), columns].to_string(index=False))
    print(f"Verified audit: {out / 'report.md'}")


if __name__ == "__main__":
    main()
