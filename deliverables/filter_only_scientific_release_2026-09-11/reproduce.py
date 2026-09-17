"""Reproduce the completed filter-only analyses from portable frozen inputs."""

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from filter_diagnostics import summarize_points
from filter_relationship_stability import (
    prepare_points,
    population_memberships,
    evaluate,
    summarize,
    paired_influence,
    processing_summary,
)
from filter_proportionality import benchmark, summaries, id11_sensitivity, KEYS


def verify_inputs():
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    for r in manifest["immutable_files"]:
        path = ROOT / r["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("Release source/input changed: " + r["path"])


def same(got, expected, keys, columns):
    a = got.set_index(keys).sort_index()[columns]
    b = expected.set_index(keys).sort_index()[columns]
    pd.testing.assert_frame_equal(
        a, b, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10
    )


def run(figures=True):
    verify_inputs()
    data = ROOT / "data"
    output = ROOT / "reproduced"
    output.mkdir(exist_ok=True)
    raw = pd.read_parquet(data / "diagnostic/analysis_points.parquet")
    assert (raw.eligible_filter_diagnostic.sum(), raw.eligible_ec_ratio_analysis.sum()) == (
        545,
        480,
    )
    descriptive = summarize_points(raw)
    for name, d in descriptive.items():
        expected = pd.read_parquet(data / "diagnostic" / (name + ".parquet"))
        pd.testing.assert_frame_equal(
            d, expected, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10
        )
        d.to_parquet(output / ("diagnostic_" + name + ".parquet"), index=False)
    fields = [
        "point_id",
        "site",
        "base_filter_id",
        "filter_ids",
        "date",
        "is_excluded",
        "exclusion_reason",
        "eligible_filter_diagnostic",
        "eligible_ec_ratio_analysis",
        "ftir_ec_ugm3",
        "hips_fabs_Mm1",
        "ftir_ec_mdl_ugm3",
        "ec_mdl_multiple",
        "ftir_ec_nonpositive",
        "ftir_ec_below_mdl",
        "ftir_ec_ugm3_source_rows",
        "hips_fabs_Mm1_source_rows",
    ]
    pts = prepare_points(
        raw[fields], pd.read_parquet(data / "diagnostic/original_measurements.parquet")
    )
    members = population_memberships(
        pts, pd.read_parquet(data / "diagnostic/sensitivity_point_links.parquet")
    )
    b, p, c, cohorts = evaluate(pts, members)
    expected_b = pd.read_parquet(data / "stability/block_performance_and_influence.parquet")
    expected_p = pd.read_parquet(data / "stability/heldout_filter_predictions.parquet")
    fold = ["site", "population", "variant", "scheme", "block"]
    point = fold + ["point_id"]
    same(
        b,
        expected_b,
        fold,
        [
            "train_ids",
            "test_ids",
            "split_sha256",
            "status",
            "train_n",
            "test_n",
            "ols_mae",
            "median_mae",
            "ols_mean_signed_error",
            "ols_slope",
            "ols_intercept",
        ],
    )
    same(
        p,
        expected_p,
        point,
        [
            "ols_prediction",
            "median_prediction",
            "ols_error",
            "median_error",
            "split_sha256",
            "ftir_ec_ugm3_source_rows",
            "hips_fabs_Mm1_source_rows",
        ],
    )
    stable_summary = summarize(b, p)
    influence, _ = paired_influence(p, c, b)
    # The extension takes the independently regenerated stability predictions.
    pb, pp = benchmark(b, p, members)
    ps = summaries(pb, pp)
    ib, ip, pairs, changes = id11_sensitivity(b, members)
    si = summaries(ib, ip, KEYS + ["training_choice"])
    same(
        pp,
        pd.read_parquet(data / "proportionality/proportionality_predictions.parquet"),
        point,
        [
            "ols_prediction",
            "median_prediction",
            "proportional_prediction",
            "proportional_error",
            "paired_abs_error_difference",
            "split_sha256",
        ],
    )
    same(
        ps,
        pd.read_parquet(data / "proportionality/proportionality_summary.parquet"),
        KEYS + ["weighting"],
        [
            "evaluated_filters",
            "evaluated_blocks",
            "intercept_better_blocks",
            "median_mae",
            "proportional_mae",
            "ols_mae",
            "ols_mean_signed_error",
            "delta_mae_proportional_minus_ols",
        ],
    )
    same(
        ip,
        pd.read_parquet(data / "proportionality/id11_training_predictions.parquet"),
        KEYS + ["block", "training_choice", "point_id"],
        [
            "status",
            "split_sha256",
            "ols_prediction",
            "median_prediction",
            "proportional_prediction",
        ],
    )
    same(
        ib,
        pd.read_parquet(data / "proportionality/id11_training_blocks.parquet"),
        KEYS + ["block", "training_choice"],
        ["status", "train_ids", "test_ids", "train_n", "test_n", "common_support"],
    )
    tables = {
        "stability_blocks": b,
        "stability_predictions": p,
        "stability_summary": stable_summary,
        "proportionality_blocks": pb,
        "proportionality_predictions": pp,
        "proportionality_summary": ps,
        "id11_training_blocks": ib,
        "id11_training_predictions": ip,
        "id11_paired_changes": changes,
        "id11_training_summary": si,
    }
    for name, d in tables.items():
        d.to_parquet(output / (name + ".parquet"), index=False)
    if figures:
        import matplotlib.pyplot as plt
        from plotting.filter_diagnostics import relationship
        from plotting.filter_proportionality import make_figures

        fig = relationship(raw)
        fig.savefig(output / "point_relationships.png", bbox_inches="tight")
        plt.close(fig)
        make_figures(tables, cohorts, output / "proportionality_figures")
    result = {
        "status": "passed",
        "scope": "frozen filter-only diagnostic, stability, proportionality and ID-11 analyses",
        "diagnostic_filters": 545,
        "ratio_filters": 480,
        "stability_prediction_rows": len(p),
        "proportionality_prediction_rows": len(pp),
        "supported_proportional_fits": int(pb.status.eq("evaluated").sum()),
        "previous_predictions_and_split_ids": "unchanged within 1e-10 numeric tolerance; ID strings exact",
        "common_ID11_test_membership": "exact match",
        "external_data_access": "not required",
        "python": sys.version.split()[0],
        "release_root": str(ROOT),
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return tables


if __name__ == "__main__":
    run(figures="--no-figures" not in sys.argv)
