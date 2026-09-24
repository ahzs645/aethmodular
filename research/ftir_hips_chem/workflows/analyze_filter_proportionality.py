#!/usr/bin/env python
"""Reproduce specification v2 without altering the completed stability analysis."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
from filter_proportionality import benchmark, summaries, id11_sensitivity, KEYS, MODELS
from interval_evidence import file_hash

OUT = AREA / "output/tables/filter_proportionality"
FROZEN = OUT / "frozen_inputs"
PLOTS = AREA / "output/plots/filter_proportionality"
MANUAL_URL = "https://aethlabs.com/sites/all/content/microaeth/maX/MA200%20MA300%20MA350%20Operating%20Manual%20Rev%2005%20May%202023.pdf"


def verify_freeze():
    record = json.loads((OUT / "prefit_freeze.json").read_text())
    for r in record["inputs"]:
        for key in ["original_path", "frozen_path"]:
            if file_hash(Path(r[key])) != r["sha256"]:
                raise ValueError("Frozen input changed: " + r[key])
    for r in record["previous_outputs"]:
        if file_hash(Path(r["path"])) != r["sha256"]:
            raise ValueError("Completed stability output changed: " + r["path"])
    for p in [
        ROOT / "docs/filter-proportionality-spec-v2-2026-09-10.md",
        FROZEN / "analysis_specification_v2.md",
    ]:
        if file_hash(p) != record["specification_sha256"]:
            raise ValueError("Extension specification changed")
    return record


def table(df):
    def cell(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, (float, np.floating)):
            return f"{x:.3f}"
        return str(x).replace("|", " / ")

    return "\n".join(
        ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
        + ["| " + " | ".join(map(cell, r)) + " |" for r in df.itertuples(index=False, name=None)]
    )


def evidence_packages(members):
    # Preserve original packet in frozen_inputs; extend a separate copy.
    original = (FROZEN / "upstream_ChemSpec_question_packet.md").read_text()
    for name in [
        "CHTS-0658_original_source_rows.csv",
        "CHTS-0658_source_metadata.csv",
        "CHTS-0658_unified_row_links.csv",
    ]:
        original = original.replace("](" + name + ")", "](frozen_inputs/" + name + ")")
    ids = members.loc[
        members.population.eq("diagnostic"),
        [
            "point_id",
            "site",
            "base_filter_id",
            "filter_ids",
            "date",
            "ftir_CalibrationSetId",
            "ftir_AnalysisDate",
            "ftir_AnalysisTime",
            "ftir_LotId",
            "ftir_ec_mdl_ugm3",
            "ftir_ec_ugm3_source_rows",
            "unified_source_file_hash",
        ],
    ].copy()
    ids.to_csv(OUT / "FTIR_calibration_identifier_filter_links.csv", index=False)
    definition = (
        ids.groupby(["site", "ftir_CalibrationSetId"], sort=False)
        .agg(
            n=("point_id", "size"),
            first_reported_date=("date", "min"),
            last_reported_date=("date", "max"),
        )
        .reset_index()
    )
    original += (
        """\n## Additional FTIR prediction provenance questions — draft, not sent

For the source EC_ftir predictions associated with CalibrationSetId 11 and 17,
please provide:

1. The definition of each identifier and the dates/filters to which it applies.
   Are these calibration sets, model versions, reporting branches or another concept?
2. The prediction model/version mapping for each ID, including any preprocessing,
   analytical method changes and the exact upstream file/version containing these predictions.
3. The reference EC target and its measurement method; the training population,
   training dates, model fitting/tuning/selection procedure and evaluation design.
4. The original training, tuning and independent evaluation membership for each
   attached physical filter, including whether replicate spectra or related filter
   records cross those roles. Please distinguish downstream use here from original
   FTIR model evaluation membership.
5. Whether LotId denotes a physical filter lot, analytical batch or another field;
   if a true analytical-batch key exists, provide its definition and row linkage.

[Exact filter and original EC source-row links](FTIR_calibration_identifier_filter_links.csv)
identify the 545 diagnostic filters. The IDs below are reported identifiers; their
observed date ranges do not establish official applicability periods.

"""
        + table(definition)
        + """

We have not treated IDs 11/17 as documented model versions, mapped ChemSpec
217/218 to them, subtracted group residual means or relabeled any predictions
corrected. The downstream proportionality and ID-11 training comparisons do not
establish original FTIR training independence. The CHTS-0658 result-role question
above remains open and both source values remain preserved.
"""
    )
    (OUT / "upstream_questions_v2_draft.md").write_text(original)

    raw = pd.read_parquet(FROZEN / "analysis_points.parquet")
    candidate = raw.loc[raw.point_id.eq("JPL:USPA-0257")]
    fields = [
        "point_id",
        "base_filter_id",
        "filter_ids",
        "date",
        "hips_fabs_Mm1",
        "hips_fabs_Mm1_source_rows",
        "portal_interval_start_utc",
        "portal_interval_end_utc",
        "portal_hours_sampled",
        "portal_collection_descriptions",
        "portal_source_links",
        "schedule_status",
        "sampling_schedule_verified",
        "instrument_source_file_hash",
    ]
    (OUT / "USPA-0257_filter_evidence.json").write_text(
        candidate[fields].to_json(orient="records", date_format="iso", indent=2) + "\n"
    )
    records = pd.read_parquet(FROZEN / "USPA-0257_record_crosswalk.parquet")
    screen = records[
        [
            "timestamp_utc",
            "datum_key",
            "source_row",
            "status",
            "Readable status",
            "IR BCc",
            "ir_bcc",
        ]
    ].copy()
    # Flag only. Existing local rules are evidence of a candidate rule, not proof
    # that it was approved/applied to this session or that missing rows were observed.
    flags = {
        2: "startup",
        4: "tape_advance",
        16: "optical_saturation",
        32: "sample_timing_error",
        128: "flow_unstable",
    }
    for bit, name in flags.items():
        screen[name] = (pd.to_numeric(screen.status).astype("int64") & bit) != 0
    screen["any_existing_local_status_flag"] = screen[list(flags.values())].any(axis=1)
    screen["screen_role"] = "candidate_status_screen_only; not a reviewed quality decision"
    screen.to_parquet(OUT / "USPA-0257_candidate_status_screen.parquet", index=False)
    screen.groupby(["status", "Readable status"], dropna=False).agg(
        n=("datum_key", "size"), flagged_n=("any_existing_local_status_flag", "sum")
    ).reset_index().to_csv(OUT / "USPA-0257_status_summary.csv", index=False)
    local = ROOT / "src/external/calibration.py"
    register = [
        dict(
            requirement="Physical filter and reported collection envelope",
            status="source_linked_reported_bounds",
            evidence="USPA-0257; 23 June 2023 16:00 to 24 June 2023 16:00 UTC; 24 reported hours; SS5i sampler",
            remaining="Confirm continuous operation or obtain filter-linked active segments and clock basis",
        ),
        dict(
            requirement="Contemporaneous instrument identities",
            status="matched_records",
            evidence="1,440 timestamp/datum matches and exact IR BCc values; MA350-0229 session 46, format 3, firmware 1.12, app 1.6",
            remaining="Record agreement alone does not certify observed-data provenance",
        ),
        dict(
            requirement="Observation/correction history",
            status="unresolved_for_session",
            evidence="DualSpot on is reported; cleaned CSV and SQLite agree",
            remaining="Producer trace from original download to cleaned CSV and database; transformations, removed/inserted records, time corrections and applicable dates/sessions",
        ),
        dict(
            requirement="Applicable quality decision",
            status="candidate_screen_not_approved",
            evidence="Status 131648 on all rows; zero triggers in existing local startup/tape/optical/timing/flow-status screen",
            remaining="Confirm session-specific applicability; check original flow/optics and exclusions; absence of status flags does not prove overall validity",
        ),
        dict(
            requirement="Timestamp alignment",
            status="manual_time_source_reported",
            evidence="Status decomposes to 64+512+131072, matching saved labels",
            remaining="Clock setting, drift/offset corrections and alignment with sampler timestamps",
        ),
        dict(
            requirement="Exported IR BCc units",
            status="nominal_manufacturer_definition_identified",
            evidence="Indexed official May 2023 manual defines format-3 IR BCc as DualSpot-compensated mass concentration in ng/m3",
            remaining="Confirm that the specific cleaned export retained native scale; full manual fetch returned 404, so the PDF was not archived",
        ),
        dict(
            requirement="Absorption conversion and wavelength treatment",
            status="not_verified",
            evidence="No reviewed conversion applied",
            remaining="Document optical conversion and wavelength treatment after the eBC evidence chain is supported",
        ),
    ]
    pd.DataFrame(register).to_csv(OUT / "USPA-0257_evidence_register.csv", index=False)
    provenance = {
        "local_quality_source": str(local),
        "local_quality_source_sha256": file_hash(local),
        "manual_url": MANUAL_URL,
        "manual_revision": "May 2023 rev05",
        "manual_access": "Official search-index excerpts retrieved; direct full-PDF fetch returned 404",
        "screen_flag_bits": flags,
        "scope": "USPA-0257 reported envelope / MA350-0229 session 46 only",
        "reviewed_interval_ebc_eligible": False,
        "absorption_eligible": False,
        "active_sampling_intervals_verified": 0,
        "candidate_records_source_sha256": file_hash(FROZEN / "USPA-0257_record_crosswalk.parquet"),
    }
    (OUT / "USPA-0257_evidence_sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    text = f"""# USPA-0257 / session 46: candidate-specific evidence package

No verified interval eBC or absorption comparison is produced. The existing SQLite
retrieval is retained; no new database query or broad inventory was run.

{table(pd.DataFrame(register))}

[Filter identity, reported bounds and original portal row links](USPA-0257_filter_evidence.json)
connect the physical filter to its source records. [The retained crosswalk](frozen_inputs/USPA-0257_record_crosswalk.parquet)
and [original targeted retrieval](frozen_inputs/USPA-0257_candidate_retrieval.json)
contain source hashes and database row IDs.

## Status and quantity definitions

The source labels on every row are DualSpot on, Time source manual and Ext. power.
Their summed code is 131648 = 64 + 512 + 131072. The manual's indexed status table
identifies those as active second-spot operation, manual/computer time source and
external power. It identifies format-3 IR BCc as mass concentration with DualSpot
loading compensation in ng/m³. [AethLabs operating manual, May 2023, sections 6.1/6.3]({MANUAL_URL}).
The full PDF returned 404 during direct retrieval; these are retrieved official
index excerpts, corroborated for status by the local code, not an archived PDF.
The vendor also documents DualSpot compensation on its [MA350 product page](https://aethlabs.com/products/ma350).

The existing local `remove_tape_advance` and `remove_concerning_statuses` functions
screen startup, tape advance, flow instability, optical saturation and sampling
timing errors. [Local source]({local}:104); SHA-256 `{file_hash(local)}`.
All 1,440 retained rows have zero triggers in this **candidate screen**. No rows
were removed. Its session-specific applicability and the treatment of flow/optics
outside status flags are not established. The local routine's treatment of missing
status as manual time is also not evidence of original measurement validity.

[Per-record flags](USPA-0257_candidate_status_screen.parquet) retain timestamps,
datum identities and raw values. Passing this screen does not prove that rows
were observed, that clocks agree, or that upstream removals/interpolation are known.

## Exact remaining request — not sent

For physical filter USPA-0257 (alias USPA-0257-1), provide sampler continuous-mode
confirmation or active on/off records for its reported envelope and the clock/timezone
basis. For MA350-0229 session 46, provide the original download and the producer's
processing log linking it to the cleaned CSV and database: time corrections,
calibration/flow changes, smoothing, deleted/inserted records and quantity scaling.
Identify the approved quality rule and its applicability to this session; supply
supporting flow/optical checks and any previous exclusions. Confirm whether IR BCc
retained the manufacturer's native ng/m³ scale.

Valid observations can include documented, appropriate corrections. The target is
supported correction history, not an assumption that every valid value is uncorrected.
The evidence is scoped to this candidate; no whole-export certification follows.
"""
    (OUT / "USPA-0257_evidence_package.md").write_text(text)


def write_report(tables, figures):
    s = tables["proportionality_summary"]
    b = tables["proportionality_blocks"]
    si = tables["id11_training_summary"]
    ib = tables["id11_training_blocks"]
    changes = tables["id11_paired_changes"]
    diag = s.loc[s.population.eq("diagnostic")]
    primary = diag.loc[diag.scheme.eq("leave_quarter_out")]
    forward = diag.loc[diag.scheme.eq("later_period")]
    cols = [
        "site",
        "weighting",
        "evaluated_filters",
        "evaluated_blocks",
        "intercept_better_blocks",
        "median_mae",
        "proportional_mae",
        "ols_mae",
        "delta_mae_proportional_minus_ols",
    ]
    bias = [
        "site",
        "scheme",
        "weighting",
        "median_mean_signed_error",
        "proportional_mean_signed_error",
        "ols_mean_signed_error",
        "largest_test_block_fraction",
    ]
    text = [
        "# Proportionality and limits of temporal transfer — specification v2",
        "**Addis benefits consistently from an intercept under this evaluation; the other sites do not show the same pattern.** "
        "For Addis, withheld-quarter MAE is 11.576 Mm⁻¹ for proportional prediction versus 3.938 Mm⁻¹ with an intercept "
        "(paired improvement 7.637 Mm⁻¹, 66.0%). The intercept wins all eight quarters. In later-period prediction, "
        "MAE is 13.110 versus 4.274 Mm⁻¹ and the intercept wins all six supported quarters. "
        "This supports an intercept-bearing empirical prediction over the tested proportional benchmark in this record. "
        "It does not establish a physical offset, a time-invariant conversion or independently accurate EC.",
        "**The conclusion is site-specific.** Beijing’s primary intercept advantage is 0.429 Mm⁻¹ with five of nine "
        "quarters improved; its later-period filter-weighted advantage is −0.029 Mm⁻¹. Delhi’s primary proportional "
        "MAE is 11.844 versus 14.534 Mm⁻¹ with an intercept, despite three of five quarters favoring the intercept. "
        "JPL differences are small: the intercept has primary MAE 1.328 versus proportional 1.286 Mm⁻¹. "
        "No practical-equivalence margin was specified, so small differences are described rather than declared equivalent.",
        "**Temporal bias and unequal weights matter.** Delhi’s primary intercept-model bias changes sign from "
        "−1.779 Mm⁻¹ with equal filter weights to +2.151 Mm⁻¹ with equal quarter weights. "
        "Its later-period bias is strongly negative under both weightings (−15.298 and −13.626 Mm⁻¹); "
        "26 filters in 2024Q2 contribute 68.4% of its 38 evaluated filters. This is an estimand difference, not a discrepancy. "
        "Beijing 2024Q3 remains in both evaluations; zero test EC values lie outside the training range, "
        "so range extrapolation alone does not explain that quarter’s failure.",
        "**Restricting Addis training to ID 11 gives a modest improvement on common later ID-11 filters.** "
        "Both choices support 127 test filters across five quarters. Intercept-model MAE falls from 4.009 to 3.884 Mm⁻¹ "
        "(−0.125), and mean signed error falls from +1.621 to +0.619 Mm⁻¹. The equal-quarter MAE change is −0.112 Mm⁻¹. "
        "This aggregate uses the same test filters under both training choices; it is not compared to the earlier 155-filter aggregate "
        "as though the evaluation set were unchanged. ID and date remain confounded; no causal processing claim or residual correction is made.",
        "## Three predictions on the same frozen holdouts",
        "The completed stability analysis and its specification remain unchanged. This extension was chosen after its results "
        "were inspected and [specification v2](frozen_inputs/analysis_specification_v2.md) was hashed before the new fits. "
        "The original 545 diagnostic / 480 ratio pairs, sensitivity memberships, calendar blocks, training IDs and support rules "
        "are unchanged. ETAD-0037 remains in the baseline; its completed omission result is retained by reference.",
        "Training median predicts a constant HIPS value. Proportional least squares predicts kE with "
        "k = sum(EH)/sum(E²), calculated from training filters only. OLS with intercept predicts a+bE. "
        "Existing median/OLS predictions were copied, not changed. No positivity filter or prediction clipping was introduced; "
        "the five nonpositive diagnostic EC values remain and can yield nonpositive proportional predictions.",
        "All errors use reported HIPS units (Mm⁻¹). Signed error = prediction − reported HIPS. "
        "The paired MAE difference is proportional minus intercept; positive favors the intercept. "
        "Both models use identical supported test filters, with at least ten training filters and two distinct EC values.",
        "## Primary: withheld-quarter stability",
        table(primary[cols]),
        "Training includes earlier and later quarters. This evaluates stability across the observed record, not prospective prediction.",
        "## Secondary: later-period prediction",
        table(forward[cols]),
        "Training includes strictly earlier quarters. Unsupported early folds remain visible. The two evaluation schemes "
        "have different test populations and are not a controlled algorithm comparison.",
        "## Directional error under both weighting choices",
        table(diag[bias]),
        "Equal-filter errors average over observed filters; equal-quarter errors average the supported quarterly errors. "
        "Equal-quarter RMSE is the square root of mean quarterly MSE. No weighting was chosen because it gave a preferred sign. "
        "The largest_test_block_fraction is unchanged across weighting rows because it describes sample composition.",
        "## Fixed denominator sensitivities",
        table(
            s.loc[
                s.scheme.eq("leave_quarter_out") & s.weighting.eq("equal_filter"),
                [
                    "site",
                    "population",
                    "evaluated_filters",
                    "evaluated_blocks",
                    "intercept_better_blocks",
                    "proportional_mae",
                    "ols_mae",
                    "delta_mae_proportional_minus_ols",
                ],
            ]
        ),
        "Full population n and EC/date ranges remain in the [frozen membership summary](frozen_inputs/population_membership_summary.parquet). "
        "In particular, JPL has seven filters at 2× MDL and zero supported model evaluations; these counts are not interchangeable. "
        "The predefined sensitivity populations are not threshold candidates from which the best score is selected.",
        "## Bounded ID-11 training sensitivity",
        table(
            si.loc[
                si.population.eq("diagnostic"),
                [
                    "training_choice",
                    "weighting",
                    "evaluated_filters",
                    "evaluated_blocks",
                    "median_mae",
                    "proportional_mae",
                    "ols_mae",
                    "ols_mean_signed_error",
                ],
            ]
        ),
        "All models use exactly the same ID-11 test filters under both training choices. The first ID-11 test quarter contains "
        "one filter; the next contains 28, but the restricted model has only one earlier ID-11 training filter. Both early "
        "quarters therefore remain unavailable for a paired comparison. Five subsequent quarters satisfy the shared rule.",
        table(changes.loc[changes.population.eq("diagnostic")]),
        "Negative changes mean lower MAE with ID-11-only training. Quarter-specific losses remain visible. "
        "The [full common-support ledger](id11_training_blocks.csv) records training/test IDs, counts, ranges and reasons. "
        "The [paired predictions](id11_paired_predictions.parquet) preserve original measurement links. "
        "All-earlier median/OLS predictions were reconciled to the previous predictions on the common filters.",
        "## Evidence packages",
        "[The extended upstream packet](upstream_questions_v2_draft.md) is **drafted, not sent**. It preserves the original "
        "CHTS-0658 result-role question and exact source rows, and adds definitions/applicability of IDs 11/17, "
        "model/reference-target mappings, true analytical-batch metadata and original training/evaluation membership. "
        "No result is labeled corrected; ChemSpec 217/218 are not treated as prediction versions. "
        "Holding filters out downstream does not establish upstream FTIR independence.",
        "[USPA-0257’s candidate evidence package](USPA-0257_evidence_package.md) now includes source-linked collection bounds, "
        "the retained session-46 record crosswalk, per-record non-destructive status flags, source definitions and exact "
        "remaining questions. All 1,440 rows report status 131648 and trigger none of the limited existing local status checks. "
        "This is a candidate screen, not an approved quality decision. Manual-time alignment, active collection and the "
        "session-specific observation/correction and quantity-scaling trace remain unresolved. The nominal manufacturer IR BCc "
        "unit is identified, but no reviewed eBC mean or absorption comparison is produced. No database search was repeated.",
        "## Reproduction",
        "Run `uv run aeth doctor`, then `uv run python research/ftir_hips_chem/workflows/analyze_filter_proportionality.py`. "
        "Add `--notebook` to generate the active notebook and execute its archived copy. Frozen source hashes, specification "
        "hash and output/code hashes are in [prefit freeze](prefit_freeze.json) and [manifest](manifest.json).",
        "[All held-out predictions](proportionality_predictions.parquet) include the paired absolute-error differences. "
        "[Block comparisons](proportionality_blocks.csv) include training/test IDs, date/EC ranges, nonpositive counts "
        "and original split fingerprints. [Both weighting schemes](proportionality_summary.csv) and "
        "[all ID-11 sensitivity results](id11_training_summary.csv) are machine-readable.",
        "## Figures",
    ]
    text += [f"![{p.stem}]({p})" for p in figures]
    text += [
        "## Retained diagnostic block failures",
        table(
            b.loc[
                b.population.eq("diagnostic")
                & b.site.isin(["Beijing", "Delhi"])
                & b.status.eq("evaluated"),
                [
                    "site",
                    "scheme",
                    "block",
                    "test_n",
                    "outside_training_ec_n",
                    "median_mae",
                    "proportional_mae",
                    "ols_mae",
                    "ols_mean_signed_error",
                    "delta_mae_proportional_minus_ols",
                ],
            ]
        ),
    ]
    (OUT / "proportionality_temporal_transfer_report.md").write_text("\n\n".join(text) + "\n")


def run():
    freeze = verify_freeze()
    b0 = pd.read_parquet(FROZEN / "block_performance_and_influence.parquet")
    p0 = pd.read_parquet(FROZEN / "heldout_filter_predictions.parquet")
    members = pd.read_parquet(FROZEN / "population_point_links.parquet")
    b, p = benchmark(b0, p0, members)
    s = summaries(b, p)
    ib, ip, pairs, changes = id11_sensitivity(b0, members)
    si = summaries(ib, ip, KEYS + ["training_choice"])
    tables = {
        "proportionality_blocks": b,
        "proportionality_predictions": p,
        "proportionality_summary": s,
        "id11_training_blocks": ib,
        "id11_training_predictions": ip,
        "id11_paired_predictions": pairs,
        "id11_paired_changes": changes,
        "id11_training_summary": si,
    }
    # Preserve exactly the previous downstream predictions on the common all-earlier filters.
    keys = KEYS + ["block", "point_id"]
    old = p0.loc[p0.variant.eq("baseline")]
    check = ip.loc[ip.training_choice.eq("all_earlier") & ip.ols_prediction.notna()].merge(
        old[keys + ["ols_prediction", "median_prediction"]],
        on=keys,
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    assert (
        len(check)
        == ip.loc[ip.training_choice.eq("all_earlier") & ip.ols_prediction.notna()].shape[0]
    )
    for model in ["ols", "median"]:
        assert np.allclose(
            check[model + "_prediction_new"], check[model + "_prediction_old"], atol=1e-10
        )
    for name, d in tables.items():
        d.to_parquet(OUT / (name + ".parquet"), index=False)
        if "predictions" not in name:
            d.to_csv(OUT / (name + ".csv"), index=False)
    evidence_packages(members)
    from plotting.filter_proportionality import make_figures

    figures = make_figures(
        tables, pd.read_parquet(FROZEN / "population_membership_summary.parquet"), PLOTS
    )
    write_report(tables, figures)
    code = [
        Path(__file__),
        AREA / "scripts/filter_proportionality.py",
        AREA / "scripts/filter_relationship_stability.py",
        AREA / "scripts/plotting/filter_proportionality.py",
        AREA / "scripts/plotting/filter_relationship_stability.py",
        AREA / "scripts/plotting/utils.py",
        AREA / "scripts/config.py",
        ROOT / "src/external/calibration.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
    ]
    outputs = [x for x in OUT.iterdir() if x.is_file() and x.name != "manifest.json"] + list(
        PLOTS.glob("*")
    )
    manifest = {
        "specification_version": 2,
        "specification_sha256": freeze["specification_sha256"],
        "diagnostic_n": 545,
        "ratio_n": 480,
        "supported_proportional_folds": int(b.status.eq("evaluated").sum()),
        "common_id11_diagnostic_n": int(
            si.loc[si.population.eq("diagnostic"), "evaluated_filters"].iloc[0]
        ),
        "code": [{"path": str(x), "sha256": file_hash(x)} for x in code],
        "outputs": [{"path": str(x), "sha256": file_hash(x)} for x in sorted(outputs)],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_freeze()
    print(
        s.loc[
            s.population.eq("diagnostic") & s.weighting.eq("equal_filter"),
            [
                "site",
                "scheme",
                "proportional_mae",
                "ols_mae",
                "delta_mae_proportional_minus_ols",
                "intercept_better_blocks",
            ],
        ].to_string(index=False)
    )
    return tables, figures


def build_notebook():
    import nbformat as nbf
    from nbclient import NotebookClient

    prior = nbf.read(AREA / "filter_relationship_stability.ipynb", as_version=4)
    setup = next(c.source for c in prior.cells if c.cell_type == "code")
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (aethmodular)",
        "language": "python",
        "name": "python3",
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# Proportionality and limits of temporal transfer\n\nEight new figures. Specification v2 was frozen before the extension fits, after the completed stability results were inspected. Prior populations, exclusions and holdouts are unchanged."
        ),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell(
            "The canonical load/flag/clean procedure was completed in the frozen baseline. This notebook reuses its hashed membership and source links; it does not silently apply new exclusions."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport importlib.util\nfrom IPython.display import display\npath=Path('workflows/analyze_filter_proportionality.py').resolve()\nspec=importlib.util.spec_from_file_location('proportional_workflow',path)\nworkflow=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(workflow)\ntables,figure_paths=workflow.run()\nb=tables['proportionality_blocks']\ns=tables['proportionality_summary']\nib=tables['id11_training_blocks']\ncohorts=pd.read_parquet(workflow.FROZEN/'population_membership_summary.parquet')\nfrom plotting import filter_proportionality as charts"
        ),
    ]
    panels = [
        (
            "The proportionality question",
            "charts.paired_blocks(b)",
            "Positive paired MAE differences favor an intercept. Addis improves by 7.64 Mm⁻¹ overall and wins all eight quarters. Other sites show less consistent results; Delhi’s aggregate favors proportional prediction despite three quarter-level intercept wins. Statistical coefficients are not physical MAC estimates.",
        ),
        (
            "Quarter-specific bias",
            "charts.bias_blocks(b)",
            "Signed errors are predicted minus reported HIPS. Bias persists with either model form. Beijing 2024Q3 remains included and has no test EC beyond its training range. The diagnostic cohort retains all five nonpositive EC predictions; no model predictions were clipped.",
        ),
        (
            "Prediction into later periods",
            "charts.forward_errors(b)",
            "All three models use the same earlier-only training folds and common test filters within a scheme. Addis’s intercept model wins all six supported quarters. Delhi retains strong negative signed error with either EC-based model. Early unavailable folds remain shown.",
        ),
        (
            "Fixed denominator populations",
            "charts.denominator_sensitivity(s,cohorts)",
            "Every sensitivity uses saved membership. Counts show full populations; JPL’s seven-filter 2× subset has zero supported folds. Changing populations changes concentration distributions and baseline errors together, so this is not evidence that a method deteriorates at higher EC. No threshold was tuned.",
        ),
        (
            "What receives equal weight",
            "charts.weighting(s)",
            "Delhi’s primary intercept bias is −1.779 Mm⁻¹ over filters and +2.151 Mm⁻¹ over quarters. Both estimates are retained. Later-period bias remains negative under either weighting. Marker annotations report both choices, not uncertainty bounds.",
        ),
        (
            "Training on earlier ID-11 filters",
            "charts.id11_errors(ib)",
            "Identical 127 later ID-11 filters are compared under both training choices. OLS MAE falls from 4.009 to 3.884 Mm⁻¹ and mean error from +1.621 to +0.619 Mm⁻¹. This is a bounded predictive sensitivity; ID/date confounding prevents attributing it to a model change. No residual means were subtracted.",
        ),
        (
            "Which metadata folds are supported",
            "charts.id11_support(ib)",
            "All-earlier and restricted training must each meet the same minimum. The first two ID-11 test quarters are unavailable for a paired comparison. Common support begins in 2023Q3 and covers five quarters; compare these same filters rather than the previous 155-filter forward aggregate.",
        ),
        (
            "Why Delhi’s final quarter matters",
            "charts.test_weights(b)",
            "Delhi’s 2024Q2 quarter supplies 26 of 38 later-period test filters (68.4%). Filter-weighted errors emphasize that quarter; equal-quarter errors assign each supported quarter the same weight. Neither is chosen for a preferred sign.",
        ),
    ]
    for title, call, note in panels:
        cells += [
            nbf.v4.new_markdown_cell("## " + title),
            nbf.v4.new_code_cell("fig = " + call + "\ndisplay(fig)\nplt.close(fig)"),
            nbf.v4.new_markdown_cell("**Notes.** " + note),
        ]
    cells.append(
        nbf.v4.new_markdown_cell(
            f"## Evidence and limits\n\nThe [report]({OUT.relative_to(AREA).as_posix()}/proportionality_temporal_transfer_report.md), [extended unsent questions]({OUT.relative_to(AREA).as_posix()}/upstream_questions_v2_draft.md) and [USPA-0257 evidence package]({OUT.relative_to(AREA).as_posix()}/USPA-0257_evidence_package.md) retain source links and unresolved requirements. Corrected instrument observations can be valid when their history is documented. No verified interval mean, physical offset or upstream FTIR independence is claimed."
        )
    )
    nb.cells = cells
    nbf.write(nb, AREA / "filter_proportionality.ipynb")
    NotebookClient(
        nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(AREA)}}
    ).execute()
    target = AREA / "notebooks/archive/executed/filter_proportionality.ipynb"
    nbf.write(nb, target)
    print("Executed notebook:", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", action="store_true")
    args = parser.parse_args()
    if args.notebook:
        build_notebook()
    else:
        run()
