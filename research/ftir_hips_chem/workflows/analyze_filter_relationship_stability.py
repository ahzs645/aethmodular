#!/usr/bin/env python
"""Reproduce the frozen filter relationship analysis and optional evidence retrieval."""

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
from filter_relationship_stability import (
    prepare_points,
    population_memberships,
    evaluate,
    summarize,
    paired_influence,
    processing_summary,
    SITE_ORDER,
)
from interval_evidence import file_hash
from data_paths import aethalometry_dir

OUT = AREA / "output/tables/filter_relationship_stability"
PLOTS = AREA / "output/plots/filter_relationship_stability"


def verify_freeze():
    freeze = json.loads((OUT / "prefit_freeze.json").read_text())
    for r in freeze["inputs"]:
        for key in ["original_path", "frozen_path"]:
            if file_hash(Path(r[key])) != r["sha256"]:
                raise ValueError("Frozen input changed: " + r[key])
    for path in [
        ROOT / "docs/filter-relationship-stability-spec-2026-09-10.md",
        OUT / "frozen_inputs/analysis_specification.md",
    ]:
        if file_hash(path) != freeze["specification_sha256"]:
            raise ValueError("Specification changed after freeze")
    return freeze


def upstream_packet():
    frozen = OUT / "frozen_inputs"
    rows = pd.read_parquet(frozen / "earlier_chemspec_ec_rows.parquet")
    rows = rows.loc[rows.base_filter_id.eq("CHTS-0658")]
    links = pd.read_parquet(frozen / "chemspec_source_to_unified.parquet")
    links = links.loc[links.base_filter_id.eq("CHTS-0658")]
    source = Path(rows.export_file.iloc[0])
    assert file_hash(source) == rows.export_sha256.iloc[0]
    # Preserve exact source header and original two physical CSV lines.
    lines = source.read_text().splitlines(keepends=True)
    selected = list(lines[:4]) + [lines[int(i) + 4] for i in rows.export_source_row]
    (OUT / "CHTS-0658_original_source_rows.csv").write_text("".join(selected))
    rows.to_csv(OUT / "CHTS-0658_source_metadata.csv", index=False)
    links.to_csv(OUT / "CHTS-0658_unified_row_links.csv", index=False)
    text = f"""# Draft upstream question — not sent

In the data-version-3.0 export updated 29 July 2025, CHTS-0658 has two rows for EC
parameter 28203 with the same method and MDL, but Value entries of 0.93 and 0.06.
Both precede our importer. What distinguishes the result roles of these rows,
which field or key identifies that distinction, and what export-generation rule
produces them? Please provide the result-role definition and any key omitted from
this export, without assuming that either row is the preferred concentration.

## Exact evidence attached

- [Original source rows](CHTS-0658_original_source_rows.csv), with original preamble and header.
- [All source metadata](CHTS-0658_source_metadata.csv).
- [Mapping to unified measurement rows](CHTS-0658_unified_row_links.csv).
- [Frozen original measurements](frozen_inputs/original_measurements.parquet), indexed by source_row.

Source: [{source.name}]({source})

SHA-256: `{rows.export_sha256.iloc[0]}`.

| Source data row (zero based) | File line (one based) | Value | MDL | Unified source_row |
|---:|---:|---:|---:|---:|
| 5883 | 5888 | 0.93 | 0.06375 | 5859 |
| 5884 | 5889 | 0.06 | 0.06375 | 5860 |

Both: Method_Code 217; Parameter_Code 28203; EC PM2.5; Analysis_Description FTIR;
units µg/m³; Conditions Ambient local. Reported start/end: 5–13 July 2022 at
09:00 local, Hours_sampled 24. These bounds do not establish an active schedule.
Blank Analytical_MDL, UNC and Flag fields do not resolve result roles.

The recovered importer assigns Value to Concentration once per source row. The
competing values already occur in the upstream Value field; it did not create the
second value by flattening MDL. Historical commit:
`{links.historical_importer_commit.iloc[0]}`; recovered importer SHA-256:
`{links.historical_importer_sha256.iloc[0]}`.
Both values remain preserved; no active parser correction or authoritative-value
selection is proposed. ChemSpec's FTIR description does not establish an independent EC reference.
"""
    (OUT / "upstream_ChemSpec_question_packet.md").write_text(text)


def retrieve_candidate():
    """One indexed, date/session-specific, read-only SQLite query; capped at 45 s."""
    stage = (
        AREA / "output/tables/active_interval_matches/observations/JPL_timestamped_inputs.parquet"
    )
    columns = [
        "timestamp_utc",
        "Serial number",
        "Session ID",
        "Datum ID",
        "source_row",
        "source_file_hash",
        "is_observed",
        "IR BCc",
        "Status",
        "Readable status",
    ]
    data = pd.read_parquet(stage, columns=columns)
    data = data.loc[
        data.timestamp_utc.ge("2023-06-23 16:00Z") & data.timestamp_utc.lt("2023-06-24 16:00Z")
    ]
    data.to_parquet(OUT / "USPA-0257_staged_candidate_rows.parquet", index=False)
    sessions = sorted(data["Session ID"].astype(str).unique().tolist())
    assert sessions == ["46"]
    db = aethalometry_dir() / "aethalometer_combined.db"
    # Parameterized predicate uses the inspected serial/time index, not a global aggregate.
    sql = "SELECT id,serial_number,time_utc,datum_id,session_id,timebase_s,status,readable_status,ir_bcc,firmware_version,app_version,data_format_version FROM aethalometer_data WHERE serial_number=? AND time_utc>=? AND time_utc<? AND session_id=? ORDER BY time_utc LIMIT 2000"
    params = ["MA350-0229", "2023-06-23 16:00:00", "2023-06-24 16:00:00", 46]
    code = """import sqlite3,json,sys\np,sql,params=json.loads(sys.argv[1]); c=sqlite3.connect(p+'?mode=ro',uri=True,timeout=2); c.row_factory=sqlite3.Row\nplan=[dict(x) for x in c.execute('EXPLAIN QUERY PLAN '+sql,params)]\nrows=[dict(x) for x in c.execute(sql,params)]\nprint(json.dumps({'plan':plan,'rows':rows}));c.close()"""
    record = {
        "candidate": "JPL:USPA-0257",
        "retrieved_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "database": str(db),
        "query": sql,
        "parameters": params,
        "query_limit": 2000,
        "staged_n": len(data),
        "staged_sessions": sessions,
        "staged_sha256": file_hash(stage),
        "source_csv_sha256": data.source_file_hash.iloc[0],
        "is_observed_known_n": int(data.is_observed.notna().sum()),
        "verified_active_intervals": 0,
        "interval_ebc_comparison_eligible": False,
        "absorption_comparison_eligible": False,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, json.dumps([db.as_uri(), sql, params])],
            capture_output=True,
            text=True,
            timeout=45,
            check=True,
        )
        payload = json.loads(result.stdout)
        record.update(
            {
                "status": "query_completed",
                "query_plan": payload["plan"],
                "sqlite_n": len(payload["rows"]),
            }
        )
        rows = pd.DataFrame(payload["rows"])
        rows.to_parquet(OUT / "USPA-0257_sqlite_candidate_rows.parquet", index=False)
        record["query_limit_reached"] = len(rows) == 2000
        if len(rows):
            rows["timestamp_utc"] = pd.to_datetime(rows.time_utc, utc=True)
            record["sqlite_timestamp_min"] = str(rows.timestamp_utc.min())
            record["sqlite_timestamp_max"] = str(rows.timestamp_utc.max())
            # Crosswalk raw identity and values, without conferring observed status.
            left = data.copy()
            left["datum_key"] = left["Datum ID"].astype(float).astype(int)
            rows["datum_key"] = pd.to_numeric(rows.datum_id).astype(int)
            match = left.merge(
                rows,
                on=["timestamp_utc", "datum_key"],
                how="outer",
                indicator=True,
                suffixes=("_staged", "_sqlite"),
            )
            record["matched_timestamp_datum_n"] = int(match._merge.eq("both").sum())
            record["only_staged_n"] = int(match._merge.eq("left_only").sum())
            record["only_sqlite_n"] = int(match._merge.eq("right_only").sum())
            x = pd.to_numeric(match["IR BCc"], errors="coerce")
            y = pd.to_numeric(match.ir_bcc, errors="coerce")
            record["paired_finite_ir_n"] = int((x.notna() & y.notna()).sum())
            record["exact_equal_ir_n"] = int(x.eq(y).sum())
            record["maximum_absolute_ir_difference_native"] = float((x - y).abs().max())
            record["reported_status_codes"] = {
                str(k): int(v) for k, v in rows.status.value_counts().items()
            }
            match.to_parquet(OUT / "USPA-0257_record_crosswalk.parquet", index=False)
    except subprocess.TimeoutExpired:
        record["status"] = "candidate_query_timed_out_45_seconds"
    except subprocess.CalledProcessError as exc:
        record.update(status="candidate_query_failed", error=exc.stderr)
    record["remaining_evidence"] = [
        "Filter-linked sampler on/off or verified continuous-operation evidence",
        "Session 46 observation/correction history and quality decisions",
        "Source-backed IR BCc eBC unit definition; absorption additionally needs optical conversion and wavelength treatment",
    ]
    (OUT / "USPA-0257_candidate_retrieval.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def run_analysis():
    freeze = verify_freeze()
    frozen = OUT / "frozen_inputs"
    raw = pd.read_parquet(frozen / "analysis_points.parquet")
    assert (raw.eligible_filter_diagnostic.sum(), raw.eligible_ec_ratio_analysis.sum()) == (
        545,
        480,
    )
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
    points = prepare_points(raw[fields], pd.read_parquet(frozen / "original_measurements.parquet"))
    members = population_memberships(
        points, pd.read_parquet(frozen / "sensitivity_point_links.parquet")
    )
    b, p, c, cohorts = evaluate(points, members)
    s = summarize(b, p)
    inf, paired = paired_influence(p, c, b)
    metadata = processing_summary(p)
    individual = c.loc[
        c.variant.eq("without_ETAD_0037"),
        ["site", "population", "delta_slope_from_original", "delta_intercept_from_original"],
    ].rename(
        columns={
            "delta_slope_from_original": "ETAD_0037_omission_slope_change",
            "delta_intercept_from_original": "ETAD_0037_omission_intercept_change",
        }
    )
    b = b.merge(individual, on=["site", "population"], how="left", validate="many_to_one")
    b = b.merge(
        inf[
            [
                "site",
                "population",
                "scheme",
                "block",
                "common_n",
                "mae_change_without_minus_baseline",
            ]
        ],
        on=["site", "population", "scheme", "block"],
        how="left",
        validate="many_to_one",
    )
    scored = p.loc[p.variant.eq("baseline") & p.ols_error.notna()]
    meta_summary = (
        scored.groupby(["site", "population", "scheme", "ftir_CalibrationSetId"], sort=False)
        .agg(
            n=("point_id", "size"),
            reported_date_min=("date", "min"),
            reported_date_max=("date", "max"),
            ec_min=("ftir_ec_ugm3", "min"),
            ec_max=("ftir_ec_ugm3", "max"),
            mean_signed_error=("ols_error", "mean"),
            mae=("ols_error", lambda x: x.abs().mean()),
            mdl_min=("ftir_ec_mdl_ugm3", "min"),
            mdl_max=("ftir_ec_mdl_ugm3", "max"),
        )
        .reset_index()
    )
    tables = {
        "block_performance_and_influence": b,
        "heldout_filter_predictions": p,
        "full_cohort_coefficients": c,
        "population_membership_summary": cohorts,
        "population_point_links": members,
        "performance_summary": s,
        "ETAD_0037_paired_influence": inf,
        "ETAD_0037_common_predictions": paired,
        "processing_metadata_residuals": metadata,
        "processing_metadata_summary": meta_summary,
    }
    for name, table in tables.items():
        table.to_parquet(OUT / (name + ".parquet"), index=False)
        if name not in [
            "heldout_filter_predictions",
            "ETAD_0037_common_predictions",
            "population_point_links",
        ]:
            table.to_csv(OUT / (name + ".csv"), index=False)
    upstream_packet()
    from plotting.filter_relationship_stability import make_figures

    figures = make_figures(b, p, c, s, inf, cohorts, members, PLOTS)
    write_report(b, p, c, s, inf, cohorts, members, figures)
    code = [
        Path(__file__),
        AREA / "scripts/filter_relationship_stability.py",
        AREA / "scripts/plotting/filter_relationship_stability.py",
        AREA / "scripts/plotting/utils.py",
        AREA / "scripts/plotting/overlays.py",
        AREA / "scripts/config.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
    ]
    outputs = [x for x in OUT.iterdir() if x.is_file() and x.name not in ["manifest.json"]] + list(
        PLOTS.glob("*")
    )
    manifest = {
        "specification_sha256": freeze["specification_sha256"],
        "prefit_freeze_sha256": file_hash(OUT / "prefit_freeze.json"),
        "diagnostic_n": 545,
        "ratio_n": 480,
        "analysis": "frozen within-site reported-date stability; no upstream independence validation",
        "code": [{"path": str(x), "sha256": file_hash(x)} for x in code],
        "outputs": [{"path": str(x), "sha256": file_hash(x)} for x in sorted(outputs)],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_freeze()
    print(
        s.loc[
            s.population.eq("diagnostic") & s.variant.eq("baseline"),
            [
                "site",
                "scheme",
                "evaluated_filters",
                "improved_blocks",
                "evaluated_blocks",
                "median_mae",
                "ols_mae",
                "ols_mean_signed_error",
            ],
        ].to_string(index=False)
    )
    return tables, figures


def md_table(frame):
    def fmt(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", " / ")

    return "\n".join(
        [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + [
            "| " + " | ".join(fmt(x) for x in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def write_report(b, p, c, s, inf, cohorts, members, figures):
    base = s.loc[s.variant.eq("baseline") & s.population.eq("diagnostic")]
    cols = [
        "site",
        "evaluated_filters",
        "evaluated_blocks",
        "improved_blocks",
        "median_mae",
        "ols_mae",
        "ols_mean_signed_error",
        "mae_improvement_pct",
    ]
    text = [
        "# Within-site stability of reported HIPS and FTIR-predicted EC",
        "## Scientific finding",
        "**Addis has a useful within-record relationship under this specification, with remaining temporal bias.** "
        "The linear model improves on the training median in all eight withheld quarters: MAE 3.938 versus 8.888 Mm⁻¹ "
        "(55.7% improvement). Block MAE ranges from 2.542 to 5.857 Mm⁻¹; block mean signed error ranges "
        "from −4.115 to +2.304 Mm⁻¹. The near-zero overall signed error therefore hides opposing block errors. "
        "Later-period MAE is 4.274 versus 9.332 Mm⁻¹ across 155 evaluated filters, with mean overprediction +2.256 Mm⁻¹.",
        "**The Addis conclusion survives the existing denominator sensitivities and the ETAD-0037 check.** "
        "At 2× MDL, 189 filters retain 55.9% MAE improvement and all eight quarters improve. "
        "At 5×, 183 filters retain 55.3% improvement, but seven of eight quarters improve. "
        "Removing ETAD-0037 changes the full-cohort slope by +0.033 and intercept by −0.206 Mm⁻¹. "
        "On 189 common held-out filters its omission changes primary MAE by only −0.0049 Mm⁻¹; "
        "whole-quarter omissions change slope by −0.121 to +0.137 and intercept by −1.113 to +0.992 Mm⁻¹. "
        "The filter itself has a baseline held-out error of −8.309 Mm⁻¹. Its omission worsens later-period "
        "MAE on 155 common filters by +0.172 Mm⁻¹, concentrated in the first evaluated later block. "
        "It does not account for the persistent quarter-level structure.",
        "**The same specification gives heterogeneous results elsewhere.** Beijing improves in seven of nine withheld "
        "quarters; 2024Q3 is worse than the constant baseline in both evaluations (10 test filters). "
        "Delhi improves in all five withheld quarters, but later-period prediction has substantial underprediction "
        "(mean error −15.298 Mm⁻¹), and its five-filter 2023Q2 block is worse than the baseline. "
        "JPL improves in all six diagnostic quarters, but aggregate improvement falls from 23.9% to 9.6% "
        "for ratio eligibility and 0.5% at 1.5× MDL. The seven-point 2× subset cannot support the declared "
        "training minimum. These comparisons describe different populations, not a search for the best threshold.",
        "**Processing metadata and time are confounded in Addis.** CalibrationSetId 17 appears on 34 diagnostic "
        "filters dated 7 December 2022–22 March 2023, with mean held-out error −3.178 Mm⁻¹. "
        "ID 11 appears on the other 156 filters, dated 29 March 2023–21 September 2024, with mean error "
        "+0.723 Mm⁻¹. This is an identifiable processing association, not evidence that a documented "
        "model change caused the difference. The model-version mapping and original FTIR training membership "
        "remain unresolved. No atmospheric attribution follows from this analysis.",
        "The descriptive baseline is unchanged: 545 diagnostic and 480 ratio pairs. "
        "The results below evaluate transfer across reported-date calendar quarters. "
        "They do not validate the original FTIR predictions, determine a physical MAC, or calibrate an aethalometer.",
        "## Primary: withheld-quarter stability",
        "Training uses all other quarters, including earlier and later filters. Positive MAE improvement means lower OLS error. "
        "All errors are in reported HIPS units, Mm⁻¹. Summary MAE weights each evaluated filter equally.",
        md_table(base.loc[base.scheme.eq("leave_quarter_out"), cols]),
        "## Secondary: later-period prediction",
        "Training uses strictly earlier quarters. Early folds without 10 training filters remain unavailable; "
        "the different evaluated populations prevent direct interpretation of the two tables as a controlled algorithm comparison.",
        md_table(base.loc[base.scheme.eq("later_period"), cols]),
        "## Fixed denominator sensitivities",
        "Saved memberships are reused without threshold tuning. JPL at 2× MDL remains a seven-filter sensitivity; "
        "it is not a replacement estimate and has no supported paired model evaluation under the frozen training rule.",
        md_table(
            s.loc[
                s.variant.eq("baseline") & s.scheme.eq("leave_quarter_out"),
                [
                    "site",
                    "population",
                    "evaluated_filters",
                    "evaluated_blocks",
                    "improved_blocks",
                    "ols_mae",
                    "median_mae",
                    "mae_improvement_pct",
                ],
            ]
        ),
        "Full membership counts, IDs, reported-date bounds and EC ranges, including zero-point populations, are in "
        "[population membership summary](population_membership_summary.csv) and "
        "[exact population links](population_point_links.parquet).",
        "## ETAD-0037 influence",
        "ETAD-0037 is retained in the baseline. No exclusion registry was edited. "
        "The named omission sensitivity compares errors on common evaluated filters only. "
        "Negative MAE change means lower error after omission.",
        md_table(
            inf.loc[
                inf.block.eq("ALL_COMMON_FILTERS"),
                [
                    "population",
                    "scheme",
                    "common_n",
                    "baseline_ols_mae",
                    "without_ols_mae",
                    "mae_change_without_minus_baseline",
                    "max_abs_prediction_change",
                ],
            ]
        ),
        "Full-cohort coefficient references below describe influence only; their fitted errors are not held-out performance.",
        md_table(
            c.loc[
                c.site.eq("Addis_Ababa") & c.population.eq("diagnostic"),
                [
                    "variant",
                    "slope",
                    "intercept",
                    "delta_slope_from_original",
                    "delta_intercept_from_original",
                ],
            ]
        ),
        "Whole-quarter coefficient changes are included in [block performance and influence](block_performance_and_influence.csv); "
        "per-quarter changes in common-filter errors are in [paired influence](ETAD_0037_paired_influence.csv).",
        "## Source-linked processing metadata",
        "Each prediction links to actual EC_ftir source rows and their CalibrationSetId, AnalysisDate, AnalysisTime, LotId and MDL. "
        "Reported IDs 11/17 are calibration-set identifiers; their mapping to FTIR models or versions is not documented here. "
        "LotId is not established as an analytical batch. The residual summaries are descriptive and may be confounded by date, "
        "concentration and population membership. No new metadata-based model was fitted.",
        "The recovered importer accepts already computed EC_ftir values from Four_Sites_FTIR_data.v2.csv. "
        "Existing repository notes identify ChemSpec EC as FTIR-derived, which agrees with the traced export description. "
        "Those notes do not establish the reference target, original training population, model/version mapping or whether these "
        "particular filters were held out upstream. Downstream regression holdouts cannot establish that independence. "
        "Earlier repository hypotheses treating ChemSpec as independent TOR are superseded by the source trace; "
        "method codes 217/218 are not used as prediction-model versions.",
        "[Metadata-group dates and errors](processing_metadata_summary.csv) and [residuals by source metadata and quarter](processing_metadata_residuals.csv) include predictor and MDL ranges. "
        "No unresolved uncertainty field or general model RMSE was used to weight the regression.",
        "## Upstream ChemSpec question packet",
        "[The draft packet](upstream_ChemSpec_question_packet.md) includes exact CHTS-0658 export rows, "
        "source SHA-256, metadata and mappings to unified rows 5859/5860. It has not been sent. "
        "Both competing values predate the recovered importer; neither is selected as authoritative and no parser was changed.",
        "## One real instrument candidate",
    ]
    candidate = OUT / "USPA-0257_candidate_retrieval.json"
    if candidate.exists():
        r = json.loads(candidate.read_text())
        text += [
            f"USPA-0257, reported UTC envelope 23 June 2023 16:00 to 24 June 2023 16:00: candidate-specific query status **{r['status']}**. "
            f"Recovered {r.get('sqlite_n', 'unknown')} SQLite rows; {r.get('matched_timestamp_datum_n', 'unknown')} match staged timestamp/datum identities in session 46. "
            "The query used the serial/time index and did not reach its 2,000-row limit. "
            "Record matching does not establish that the input was observed, uncorrected or quality-approved.",
            "No complete interval comparison is supported yet: filter-linked active-operation evidence, scoped observation/correction history, "
            "quality decisions and source-backed eBC units remain missing. Absorption would additionally need optical-conversion and wavelength evidence. "
            "No eBC mean or absorption result is presented as verified.",
            "[Exact query, plan, source hashes and unresolved gates](USPA-0257_candidate_retrieval.json); "
            "[record crosswalk](USPA-0257_record_crosswalk.parquet).",
        ]
    else:
        text += [
            "Candidate retrieval not run. Use --retrieve-candidate to attempt the bounded query."
        ]
    text += [
        "## Reproduction and analysis rules",
        "[Frozen specification](frozen_inputs/analysis_specification.md) was hashed before the first new fit; "
        "[prefit freeze](prefit_freeze.json) records the time and source hashes. The questions were motivated by previously inspected descriptive results. "
        "No post-fit changes to blocks, populations, model family or minimum training rule were made.",
        "Run from the repository root: `uv run aeth doctor`, then "
        "`uv run python research/ftir_hips_chem/workflows/analyze_filter_relationship_stability.py`. "
        "The default analysis reproduces against the frozen inputs and saved candidate retrieval. "
        "Use --retrieve-candidate only to refresh that separate database evidence.",
        "OLS has an intercept and equal filter weights. Its predictor is a reported product, not an error-free EC reference. "
        "Constant baseline = training median HIPS. Minimum training n = 10 and two distinct EC values; "
        "n < 5 test blocks are flagged but retained. Empty calendar quarters are retained. "
        "All training/test identities are disjoint. No extrapolation points are removed.",
        "Explicit calendar blocks avoid treating irregular filter rows as equally spaced time intervals; see "
        "[TimeSeriesSplit assumptions](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). "
        "Training-only fitting follows [cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) "
        "and [leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html). "
        "Quarter boundaries were fixed rather than optimized. Reported dates are not active sampling periods.",
        "[Per-filter held-out predictions, signed residuals, split fingerprints and original source links](heldout_filter_predictions.parquet); "
        "[all block errors, IDs, reported-date/EC ranges, extrapolation and coefficient influence](block_performance_and_influence.csv). "
        "Per-filter residual = predicted HIPS minus reported HIPS. MAE improvement = median-baseline MAE minus OLS MAE. "
        "Both filter-weighted and equal-block results are retained in [performance summary](performance_summary.csv).",
        "## Figures",
    ]
    for path in figures:
        text.append(f"![{path.stem}]({path})")
    text += [
        "## Diagnostic block detail",
        md_table(
            b.loc[
                b.variant.eq("baseline")
                & b.population.eq("diagnostic")
                & b.scheme.eq("leave_quarter_out"),
                [
                    "site",
                    "block",
                    "status",
                    "train_n",
                    "test_n",
                    "test_ec_min",
                    "test_ec_max",
                    "outside_training_ec_n",
                    "ols_mean_signed_error",
                    "ols_mae",
                    "median_mae",
                    "mae_improvement",
                ],
            ]
        ),
    ]
    (OUT / "filter_relationship_stability_report.md").write_text("\n\n".join(text) + "\n")


def build_notebook():
    import nbformat as nbf
    from nbclient import NotebookClient

    prior = nbf.read(AREA / "filter_only_diagnostics.ipynb", as_version=4)
    setup = next(cell.source for cell in prior.cells if cell.cell_type == "code")
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (aethmodular)",
        "language": "python",
        "name": "python3",
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# Reported-date relationship stability\n\nNine reproducible figures from the frozen 545 diagnostic / 480 ratio pairs. Details belong in the notes below each chart. Reported dates do not establish active sampling intervals."
        ),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell(
            "The standard exclusion flow was applied when the baseline was built. This notebook deliberately uses the frozen values, identity-linked flags and saved memberships, verified by hash. Reapplying newly changed rules would alter the estimand."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport importlib.util\nfrom IPython.display import display\npath=Path('workflows/analyze_filter_relationship_stability.py').resolve()\nspec=importlib.util.spec_from_file_location('stability_workflow',path)\nworkflow=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(workflow)\ntables, figure_paths=workflow.run_analysis()\nb=tables['block_performance_and_influence']\np=tables['heldout_filter_predictions']\nc=tables['full_cohort_coefficients']\ns=tables['performance_summary']\ninf=tables['ETAD_0037_paired_influence']\ncohorts=tables['population_membership_summary']\nfrom plotting import filter_relationship_stability as charts"
        ),
    ]
    panels = [
        (
            "Withheld-block errors",
            "charts.mae_blocks(b,'leave_quarter_out')",
            "Addis improves on the training median in all eight quarters, with MAE 3.94 versus 8.89 Mm⁻¹. Training here includes later as well as earlier filters; this is within-record stability. Delhi/JPL also improve in every diagnostic block; Beijing has two exceptions. Each site uses its own vertical scale.",
        ),
        (
            "Bias across reported-date blocks",
            "charts.bias_blocks(b)",
            "Positive error means overprediction. Addis aggregate bias is near zero, but quarter biases range from −4.11 to +2.30 Mm⁻¹. Opposing errors cancel in the overall mean; residual time structure remains.",
        ),
        (
            "Frozen denominator populations",
            "charts.sensitivity(s,cohorts)",
            "Counts are full population membership; MAE uses only supported held-out folds. No threshold was selected from these errors. JPL retains 84 ratio filters and 28 at 1.5× MDL; improvements shrink to 9.6% and 0.5%. Its seven-point 2× population remains visible but cannot supply ten training filters. Addis 1.5×, 2× and 3× share the same 189 filters.",
        ),
        (
            "One filter versus whole blocks",
            "charts.coefficients(b,c)",
            "The full-cohort OLS coefficients are descriptive references, not validation scores. ETAD-0037 omission changes slope by +0.033 and intercept by −0.206 Mm⁻¹. Whole-quarter omissions have wider coefficient changes. The filter stays in the baseline and exclusion registry.",
        ),
        (
            "Held-out predictions",
            "charts.prediction_scatter(p)",
            "Every displayed prediction is generated with that filter’s quarter held out. The 1:1 line compares predicted and reported HIPS in the same units. No regression slope is fitted to this prediction-comparison plot. The FTIR predictor itself is not established as an independent reference.",
        ),
        (
            "Later-period prediction",
            "charts.mae_blocks(b,'later_period')",
            "Only strictly earlier quarters enter training. Early unsupported folds remain unavailable. Addis evaluates 155 filters with MAE 4.27 versus 9.33 Mm⁻¹; mean error is +2.26. Delhi retains substantial negative bias (−15.30 Mm⁻¹), despite average improvement. The changed evaluated populations prevent a direct algorithm comparison with leave-quarter-out results.",
        ),
        (
            "Training and test EC ranges",
            "charts.concentration_ranges(b)",
            "Extrapolation counts refer to test EC below the minimum or above the maximum training EC. Points are retained. Different concentration ranges can change prediction difficulty; a zero extrapolation count does not establish comparable distributions or precise EC measurements.",
        ),
        (
            "Source-linked metadata",
            "charts.metadata_plot(p)",
            "CalibrationSetId values come from actual EC_ftir source rows. Addis ID 17 covers 34 early filters, through 22 March 2023; ID 11 covers 156 later filters, beginning 29 March. Date and identifier effects cannot be separated here. IDs are not documented FTIR model versions, LotId is not a confirmed batch, and ChemSpec method codes 217/218 are not used as substitutes.",
        ),
        (
            "Errors on common filters",
            "charts.paired_plot(inf)",
            "The ETAD-0037 sensitivity compares the same 189 other filters. Primary MAE changes by only −0.0049 Mm⁻¹. Predictions in its own withheld quarter do not change because it was absent from training already. Later-period common-filter MAE worsens by +0.172 Mm⁻¹ after omission, concentrated in the first evaluated later block.",
        ),
    ]
    for title, call, note in panels:
        cells += [
            nbf.v4.new_markdown_cell("## " + title),
            nbf.v4.new_code_cell("fig = " + call + "\ndisplay(fig)\nplt.close(fig)"),
            nbf.v4.new_markdown_cell("**Notes.** " + note),
        ]
    cells += [
        nbf.v4.new_markdown_cell(
            "## Separate evidence actions\n\nThe ChemSpec upstream question packet is drafted, not sent. The candidate-specific SQLite lookup recovered 1,440 session-46 records for USPA-0257 and matched all staged timestamp/datum IDs. Verified active-operation evidence, scoped correction/quality history and source-backed eBC units remain missing; no verified interval mean is reported.\n\nThe [full report](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/filter_relationship_stability/filter_relationship_stability_report.md), prediction parquet, block CSV, original measurement links and manifests are regenerated by the workflow. The specification was frozen before fitting; original FTIR training independence remains unresolved."
        )
    ]
    nb.cells = cells
    active = AREA / "filter_relationship_stability.ipynb"
    nbf.write(nb, active)
    NotebookClient(
        nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(AREA)}}
    ).execute()
    archived = AREA / "notebooks/archive/executed/filter_relationship_stability.ipynb"
    archived.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, archived)
    print("Executed notebook:", archived)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieve-candidate", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument(
        "--notebook",
        action="store_true",
        help="Create active notebook and execute an archived copy",
    )
    args = parser.parse_args()
    if args.retrieve_candidate or args.candidate_only:
        retrieve_candidate()
    if not args.candidate_only:
        if args.notebook:
            build_notebook()
        else:
            run_analysis()
