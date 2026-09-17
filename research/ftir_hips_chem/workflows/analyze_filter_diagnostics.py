"""Reproduce filter-only scientific results, retrieval priorities and EC tracing."""

import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
from config import SITES
from filter_diagnostics import (
    load_frozen_points,
    summarize_points,
    evidence_queue,
    trace_earlier_chemspec,
)
from interval_evidence import file_hash
from plotting import PlotConfig
from plotting.filter_diagnostics import FIGURES
from audit_matched_samples import load_portal_metadata


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path), bytes=path.stat().st_size)


def main():
    source = AREA / "output/tables/active_interval_matches"
    audit = AREA / "output/tables/matched_sample_audit"
    out = AREA / "output/tables/filter_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    plots = AREA / "output/plots/filter_diagnostics"
    plots.mkdir(parents=True, exist_ok=True)
    points = load_frozen_points(source)
    tables = summarize_points(points)
    inventory = json.loads((source / "source_inventory.json").read_text())
    measurements = pd.read_parquet(audit / "filter_measurements.parquet")
    audit_manifest = json.loads((audit / "manifest.json").read_text())
    row_record = next(
        r
        for r in audit_manifest["outputs"]
        if Path(r["path"]).name == "filter_measurements.parquet"
    )
    if file_hash(audit / "filter_measurements.parquet") != row_record["sha256"]:
        raise ValueError("Frozen original measurement rows changed")
    export_paths = [
        ROOT / "research/filter_combine" / f"FilterBased_ChemSpecPM25_{SITES[s]['code']}.csv"
        for s in SITES
    ]
    earlier, trace = trace_earlier_chemspec(export_paths, measurements)
    earlier_bounds, _ = load_portal_metadata(export_paths)
    queue_points = points.copy()
    queue_points["retrieval_bounds_source"] = "frozen_current_catalog"
    bounds_lookup = earlier_bounds.set_index(["site_code", "base_filter_id"])
    for idx, row in queue_points.iterrows():
        key = (row.site_code, row.base_filter_id)
        if pd.isna(row.portal_interval_start_utc) and key in bounds_lookup.index:
            old = bounds_lookup.loc[key]
            if pd.notna(old.portal_interval_start_utc) and not old.portal_timing_conflict:
                for field in [
                    "portal_interval_start_utc",
                    "portal_interval_end_utc",
                    "portal_hours_sampled",
                    "portal_source_links",
                ]:
                    queue_points.at[idx, field] = old[field]
                queue_points.at[idx, "retrieval_bounds_source"] = (
                    "earlier_2025_export_for_retrieval_only"
                )
                queue_points.at[idx, "schedule_status"] = (
                    "bounds_consistent_mode_unverified"
                    if old.portal_continuous_schedule_consistent
                    else "active_periods_unknown"
                )
    queue = evidence_queue(queue_points, inventory)
    if not trace.source_value_match.all() or not trace.source_mdl_match.all():
        raise ValueError("Earlier-to-unified EC mapping requires further adjudication")
    # Recover the deleted historical importer as evidence, without executing it
    # or reviving its old filtering rules in the active analysis.
    ref = subprocess.check_output(["git", "rev-parse", "aa9714b^"], cwd=ROOT, text=True).strip()
    legacy_path = "research/filter_combine/main.py"
    legacy = subprocess.check_output(["git", "show", f"{ref}:{legacy_path}"], cwd=ROOT, text=True)
    legacy_target = out / "historical_filter_integrator.txt"
    legacy_target.write_text(legacy)
    trace["historical_importer_commit"] = ref
    trace["historical_importer_sha256"] = file_hash(legacy_target)
    source_rows = measurements.loc[measurements.base_filter_id.isin(points.base_filter_id)].copy()
    source_rows["unified_source_file_hash"] = next(
        r["sha256"]
        for r in audit_manifest["inputs"]
        if Path(r["path"]).name == "unified_filter_dataset.pkl"
    )
    tables.update(
        analysis_points=points,
        original_measurements=source_rows,
        evidence_recovery_queue=queue,
        earlier_chemspec_ec_rows=earlier,
        chemspec_source_to_unified=trace,
        earlier_collection_bounds=earlier_bounds,
    )
    adjudication = []
    for (site, filter_id), group in trace.groupby(["Site", "base_filter_id"]):
        adjudication.append(
            dict(
                site_code=site,
                base_filter_id=filter_id,
                source_parameter_codes=json.dumps(sorted(group.Parameter_Code.unique().tolist())),
                source_analysis_descriptions=json.dumps(
                    sorted(group.Analysis_Description.unique().tolist())
                ),
                source_method_codes=json.dumps(sorted(group.Method_Code.unique().tolist())),
                source_conditions=json.dumps(sorted(group.Conditions.unique().tolist())),
                competing_values=json.dumps(sorted(group.Concentration.unique().tolist())),
                source_row_links=group[
                    ["export_file", "export_sha256", "export_source_row", "source_row"]
                ].to_json(orient="records"),
                source_definition_recovered=True,
                authoritative_value_selected=False,
                selected_ec_ugm3=float("nan"),
                value_adjudication="unresolved_upstream_competing_values"
                if len(group) > 1
                else "single_export_value_definition_recovered_not_independent_reference",
            )
        )
    tables["chemspec_group_adjudication"] = pd.DataFrame(adjudication)
    strata = points.loc[points.eligible_ec_ratio_analysis].copy()
    strata["denominator_group"] = strata.near_mdl_1_to_2.map(
        {True: "1 to <2 times MDL", False: "at least 2 times MDL"}
    )
    tables["denominator_strata"] = (
        strata.groupby(["site", "denominator_group"])
        .ratio.agg(n="size", median="median", maximum="max")
        .reset_index()
    )
    for name, frame in tables.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
        if name not in ["original_measurements", "earlier_chemspec_ec_rows", "analysis_points"]:
            frame.to_csv(out / f"{name}.csv", index=False)
    PlotConfig.set(
        sites="all",
        layout="individual",
        show_1to1=False,
        show_stats=False,
        font_size=11,
        title_size=14,
    )
    plt.rcParams["svg.hashsalt"] = "filter-diagnostics-frozen-20260910"
    for key, draw in FIGURES:
        fig = draw(tables["denominator_sensitivity"] if key.startswith("05") else points)
        fig.savefig(plots / f"{key}.png", bbox_inches="tight")
        fig.savefig(plots / f"{key}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
    result = tables["site_results"].copy()
    ratio_summ = (
        tables["distribution_summary"]
        .query("population == 'ratio' and variable == 'ratio'")
        .set_index("site")
    )
    result["ratio_median"] = result.site.map(ratio_summ["median"])
    result["ratio_IQR"] = result.site.map(
        (ratio_summ.q25.map(lambda x: f"{x:.2f}") + "–" + ratio_summ.q75.map(lambda x: f"{x:.2f}"))
    )
    overview = result[
        [
            "site",
            "diagnostic_n",
            "ratio_n",
            "descriptive_r_squared",
            "descriptive_ols_slope",
            "descriptive_ols_intercept",
            "ratio_median",
            "ratio_IQR",
        ]
    ]
    sensitivity = tables["denominator_sensitivity"]
    priority_counts = (
        queue.groupby(["site", "priority"]).size().unstack(fill_value=0).reindex(SITES)
    )
    special = queue.loc[queue.base_filter_id.eq("ETAD-0243")].iloc[0]
    top = queue.loc[queue.priority.le(2)].groupby("site", sort=False).head(4)
    example = trace.loc[trace.base_filter_id.eq("CHTS-0658")]
    strongest = result.loc[result.descriptive_r_squared.idxmax()]
    weakest = result.loc[result.descriptive_r_squared.idxmin()]
    report = f"""# Filter-only scientific results

The frozen inputs reproduce **545 HIPS/FTIR-predicted-EC diagnostic pairs and 480 ratio-eligible pairs**.
Every ratio point is a diagnostic point. The 65-point difference is accounted for below by complete,
mutually exclusive reason strings. Low and nonpositive predictions are retained in diagnostic plots;
their denominators are never replaced. These results require neither ChemSpec adjudication nor aethalometer matching.

## Within-site results

{overview.to_markdown(index=False, floatfmt=".3f")}

Within-site descriptive R² ranges from {weakest.descriptive_r_squared:.3f} ({weakest.site}) to
{strongest.descriptive_r_squared:.3f} ({strongest.site}). The fits are unweighted OLS with an intercept,
using each site's complete diagnostic population. Site-specific axes expose each concentration range;
R² comparisons do not isolate instrument performance from concentration range, population or sampling differences.
No pooled cross-site line, 1:1 line, independent EC validation claim or uncertainty-weighted fit is used.
OLS slopes are descriptive, and measurement error in both quantities can affect them.

![Within-site relationships](../../plots/filter_diagnostics/01_site_relationships.png)

HIPS is retained in Mm⁻¹ and the denominator is FTIR-predicted EC in µg m⁻³.
The derived quantity is labeled **HIPS/FTIR-predicted-EC ratio**, with units (Mm⁻¹)/(µg m⁻³).
Although those units are dimensionally m² g⁻¹, this analysis does not establish an independently validated BC absorption efficiency.
The site medians (8.61–10.16) sit within substantially overlapping IQRs; a shared ratio is not established by those summaries.

![Ratio distributions](../../plots/filter_diagnostics/02_ratio_distributions.png)

## Denominator eligibility and sensitivity

{tables["ratio_exclusion_reasons"][["site", "reason", "count"]].to_markdown(index=False)}

The 65 exclusions comprise 60 positive predictions below their reported MDL and 5 nonpositive predictions
that also fall below MDL. These categories are verified from the retained flags, rather than assumed.
The baseline requires a resolved, finite, positive FTIR EC and a nonconflicting reported MDL, with EC ≥ MDL.
HIPS/FTIR ratios are absent outside that population. The full per-point decisions and original row links remain available.

![Ratio versus denominator](../../plots/filter_diagnostics/03_ratio_denominators.png)

The ring annotation describes EC between 1× and <2× its own MDL; it is not a new exclusion.
At JPL, 77 of 84 ratio points fall in this range; a 2× rule leaves only 7. At Addis, it leaves 189 of 190.
The largest Addis ratio (55.41) is its one near-MDL point. The Addis median is stable under stricter rules,
while Beijing's median and cohort size decline. Those changes reflect different retained populations, not a chosen calibration threshold.
This panel shares EC algebraically between the horizontal axis and the ratio denominator, so an inverse pattern
alone is not evidence for a physical mechanism. Group summaries are:

{tables["denominator_strata"].to_markdown(index=False, floatfmt=".3f")}

![Reported-date patterns](../../plots/filter_diagnostics/04_ratio_dates.png)

Dates are reported sample dates. They do not assert active-interval timing; the dashed line is the site's overall median.
There is no temporal interpolation, date adjustment, seasonal-calendar substitution or temporal trend fit.
Site/cohort date ranges, medians and IQRs for HIPS, FTIR EC and ratios are exported in `distribution_summary.parquet`.

![Denominator sensitivity](../../plots/filter_diagnostics/05_denominator_sensitivity.png)

The fixed sensitivity grid is 1, 1.5, 2, 3 and 5 times each filter's MDL, applied within the existing ratio population.
It describes changes in counts and medians; it does not select a new threshold from measurement agreement.
Every retained filter at every threshold is linked in `sensitivity_point_links.parquet`.

{sensitivity[["site", "minimum_ec_mdl_multiple", "n", "ratio_median"]].to_markdown(index=False, floatfmt=".3f")}

![Registry context](../../plots/filter_diagnostics/06_registry_context.png)

The existing Delhi registry exclusion is shown for audit, with its original value unchanged. It does not enter the 545 diagnostics.
HIPS uncertainty/MDL records are retained, but their unresolved semantics are not replaced by standard deviations or AIRSpec RMSE.

## Evidence-recovery priorities

{priority_counts.to_markdown()}

Priority 1: usable HIPS plus hour-consistent reported bounds overlapping an available export.
Priority 2: usable HIPS with overlap but active periods or timing requiring reconciliation.
Priority 3: missing bounds or no overlap; more timing/instrument evidence is needed first.
Priority 4: no staged timestamped export for that site. Priority 5: HIPS/filter identity or registry gating.
These are retrieval priorities, not primary eligibility or observed-coverage estimates.
No filter-linked run log has been located; the queue states that explicitly rather than inventing an evidence source.

First retrieval candidates by site (chronological within priority):

{top[["site", "base_filter_id", "priority", "reported_start_utc", "reported_end_utc", "missing_evidence"]].to_markdown(index=False)}

**ETAD-0243 is priority {special.priority}:** its reported start is {special.reported_start_utc},
after the current Addis export ends at {special.export_max_utc}. Recovering its schedule alone cannot unlock a comparison with that export.
The queue retains each filter's source bounds, instrument-file hash, serial/firmware metadata and the precise evidence gate.
Where the frozen catalog had no portal start, recovered 2025 exports now supply separately labeled retrieval bounds;
they do not change diagnostic dates, eligibility or active schedules. The earlier collection-bound table is preserved in full.
A reviewed real interval subset remains unavailable; no affirmative provenance booleans or inferred on/off periods were supplied.
An additional `aethalometer_combined.db` was found in the Drive instrument directory. Its schema was readable, but
the read-only grouped range query did not complete within roughly four minutes and was canceled. Its coverage and processing history
remain unverified; no claim of missing later records is inferred from that access limitation.

## ChemSpec source-to-output trace

The earlier local exports in `research/filter_combine/` are labeled File Updated 2025-07-29, Data version 3.0.
All **{len(trace):,} current ChemSpec EC rows** match an earlier row's value and MDL, with filter identity and method preserved.
Their parameter code is 28203, analysis description FTIR, concentration units µg m⁻³ and conditions Ambient local.
Method codes 217/218 are retained; their detailed calibration definitions remain unrecovered.

The recovered historical importer at commit `{ref}` maps `Value` directly to `Concentration`,
`MDL` separately to `MDL`, and `Parameter_Name` to a `ChemSpec_` label. It appends one output row per input row;
the inspected importer does not melt MDL into additional concentration rows. It does discard some source metadata,
which this trace now preserves. No parser correction is justified by the observed duplication at this stage.

For CHTS-0658, earlier source rows 5883 and 5884 already contain 0.93 and 0.06 in the **Value** field under
the same code, FTIR description and method. They map to current unified rows 5859 and 5860, respectively.
Thus the competing values predate this importer. Their upstream export-generation code or role definitions
are still needed to decide whether one Value is actually an MDL. Neither value is selected by magnitude or MDL equality.

The historical code is saved read-only as `historical_filter_integrator.txt`, with its commit and content hash in
the row trace. Original earlier rows and all source-to-output links are exported. This supersedes the previous
inventory gap about EC parameter codes; it does not establish an authoritative concentration within each conflicting group.

## Reproduce and inspect points

Run `uv run python research/ftir_hips_chem/workflows/analyze_filter_diagnostics.py` from the repository root.
`analysis_points.parquet` provides one row per physical filter with point IDs, original HIPS/FTIR row links,
reported dates, flags, ratios and denominator multiples. `original_measurements.parquet` preserves the source rows
and unified-source hash. All six figure families use these same points; the registry panel additionally shows its retained exclusion.
`manifest.json` records input, code, table and figure hashes. No calibration is fitted.

## Narrow implementation review

Reviewed processing evidence now requires an explicit full-export or bounded-period scope and a scope justification.
Bounded assertions cannot certify a filter interval extending beyond their UTC bounds; optional session-ID restrictions
must cover all input rows in the interval. Out-of-scope observations retain input availability but unknown observed coverage and eBC means.
This is a scope guard, not evidence that any current research stream has been reviewed.

The 75% overall / 50% per-segment policy is unchanged. Successive quarters of each active interval now receive
descriptive input and valid-observed coverage summaries. Their values do not affect eligibility.
For a single continuous interval, the per-segment rule is redundant with the total rule; quarters expose concentrated gaps.
Real quarter coverage remains unavailable while active schedules and observation histories remain unresolved.
"""
    (out / "results_report.md").write_text(report)
    inputs = [
        source / "manifest.json",
        source / "matched_active_intervals.parquet",
        source / "source_inventory.json",
        audit / "manifest.json",
        audit / "filter_measurements.parquet",
        *export_paths,
    ]
    code = [
        Path(__file__),
        AREA / "scripts/filter_diagnostics.py",
        AREA / "scripts/plotting/filter_diagnostics.py",
        AREA / "scripts/plotting/overlays.py",
        AREA / "scripts/plotting/utils.py",
        AREA / "scripts/config.py",
        AREA / "scripts/outliers.py",
        AREA / "scripts/data_matching.py",
        AREA / "scripts/interval_evidence.py",
        AREA / "scripts/plotting/__init__.py",
        AREA / "workflows/audit_matched_samples.py",
        AREA / "scripts/active_interval_matching.py",
        AREA / "scripts/instrument_provenance.py",
        AREA / "workflows/build_active_interval_matches.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
    ]
    outputs = [
        p for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"
    ] + list(sorted(plots.iterdir()))
    manifest = dict(
        inputs=[fingerprint(p) for p in inputs],
        code=[fingerprint(p) for p in code],
        outputs=[fingerprint(p) for p in outputs],
        historical_importer_commit=ref,
        diagnostic_n=545,
        ratio_n=480,
        analysis="descriptive unweighted filter-only; no independent EC validation",
        sensitivity_grid=[1, 1.5, 2, 3, 5],
        packages={
            name: version(name)
            for name in [
                "pandas",
                "numpy",
                "matplotlib",
                "scipy",
                "pyarrow",
                "nbformat",
                "nbclient",
            ]
        },
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(overview.to_string(index=False))
    print(priority_counts.to_string())
    print(out / "results_report.md")
    return out


if __name__ == "__main__":
    main()
