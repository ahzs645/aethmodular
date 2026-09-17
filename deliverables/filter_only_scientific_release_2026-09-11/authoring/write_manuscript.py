"""Compose a source-linked scientific draft from completed, frozen output tables."""

from pathlib import Path
import csv
import hashlib
import json
import re
import sys

import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
claims = []


def read(name):
    return pd.read_parquet(ROOT / name)


def get(cid, file, filters, column, transform=None):
    d = read(file)
    for key, val in filters.items():
        d = d.loc[d[key].eq(val)]
    if len(d) != 1:
        raise ValueError((cid, len(d)))
    value = d[column].iloc[0]
    if transform:
        value = transform(value)
    claims.append(
        dict(
            claim_id=cid,
            source_table=file,
            source_sha256=hashlib.sha256((ROOT / file).read_bytes()).hexdigest(),
            selector=json.dumps(filters),
            column=column,
            value=float(value) if isinstance(value, (float, int)) else str(value),
            calculation="direct selected cell",
        )
    )
    return value


def derived(cid, file, selector, value, calculation):
    claims.append(
        dict(
            claim_id=cid,
            source_table=file,
            source_sha256=hashlib.sha256((ROOT / file).read_bytes()).hexdigest(),
            selector=json.dumps(selector),
            column="",
            value=value,
            calculation=calculation,
        )
    )
    return value


def mdtable(rows, columns):
    return "\n".join(
        ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def main():
    summ = "data/proportionality/proportionality_summary.parquet"
    idfile = "data/proportionality/id11_training_summary.parquet"
    source = "data/diagnostic/site_results.parquet"
    blocks = "data/proportionality/proportionality_blocks.parquet"
    cohorts = read("data/stability/population_membership_summary.parquet")
    table1 = []
    for site in ["Addis_Ababa", "Beijing", "Delhi", "JPL"]:
        n = get("cohort_" + site, source, {"site": site}, "diagnostic_n")
        nr = get("ratio_" + site, source, {"site": site}, "ratio_n")
        n2 = cohorts.loc[
            cohorts.site.eq(site)
            & cohorts.population.eq("mdl_2x")
            & cohorts.variant.eq("baseline"),
            "n",
        ].item()
        derived(
            "mdl2_" + site,
            "data/stability/population_membership_summary.parquet",
            {"site": site, "population": "mdl_2x", "variant": "baseline"},
            int(n2),
            "direct n",
        )
        low = get("date_min_" + site, source, {"site": site}, "diagnostic_date_min")
        high = get("date_max_" + site, source, {"site": site}, "diagnostic_date_max")
        table1.append(
            [
                site.replace("_", " "),
                str(n),
                str(nr),
                str(n2),
                f"{str(low)[:10]} to {str(high)[:10]}",
            ]
        )
    scores = {}
    table2 = []
    table3 = []
    for site in ["Addis_Ababa", "Beijing", "Delhi", "JPL"]:
        for scheme in ["leave_quarter_out", "later_period"]:
            key = {
                "site": site,
                "population": "diagnostic",
                "scheme": scheme,
                "weighting": "equal_filter",
            }
            row = {
                c: get(f"{site}_{scheme}_{c}", summ, key, c)
                for c in [
                    "evaluated_filters",
                    "evaluated_blocks",
                    "intercept_better_blocks",
                    "median_mae",
                    "proportional_mae",
                    "ols_mae",
                    "delta_mae_proportional_minus_ols",
                    "ols_mean_signed_error",
                ]
            }
            scores[site, scheme] = row
            table2.append(
                [
                    site.replace("_", " "),
                    "Withheld quarter" if scheme == "leave_quarter_out" else "Later period",
                    str(int(row["evaluated_filters"])),
                    f"{row['median_mae']:.3f}",
                    f"{row['proportional_mae']:.3f}",
                    f"{row['ols_mae']:.3f}",
                    f"{row['delta_mae_proportional_minus_ols']:+.3f}",
                ]
            )
            q = get(
                f"{site}_{scheme}_quarter_bias",
                summ,
                {**key, "weighting": "equal_quarter"},
                "ols_mean_signed_error",
            )
            table3.append(
                [
                    site.replace("_", " "),
                    "Withheld quarter" if scheme == "leave_quarter_out" else "Later period",
                    f"{row['ols_mean_signed_error']:+.3f}",
                    f"{q:+.3f}",
                ]
            )
    idvals = {}
    for training in ["all_earlier", "id11_earlier"]:
        for metric in [
            "evaluated_filters",
            "evaluated_blocks",
            "proportional_mae",
            "ols_mae",
            "ols_mean_signed_error",
        ]:
            idvals[training, metric] = get(
                "id11_" + training + "_" + metric,
                idfile,
                {
                    "site": "Addis_Ababa",
                    "population": "diagnostic",
                    "scheme": "later_period",
                    "training_choice": training,
                    "weighting": "equal_filter",
                },
                metric,
            )
    advantage = idvals["id11_earlier", "proportional_mae"] - idvals["id11_earlier", "ols_mae"]
    pct = 100 * advantage / idvals["id11_earlier", "proportional_mae"]
    derived(
        "id11_model_form_advantage",
        idfile,
        {
            "training_choice": "id11_earlier",
            "population": "diagnostic",
            "weighting": "equal_filter",
        },
        advantage,
        "proportional_mae - ols_mae",
    )
    derived(
        "id11_model_form_percent",
        idfile,
        {
            "training_choice": "id11_earlier",
            "population": "diagnostic",
            "weighting": "equal_filter",
        },
        pct,
        "100*(proportional_mae - ols_mae)/proportional_mae",
    )
    gain = idvals["all_earlier", "ols_mae"] - idvals["id11_earlier", "ols_mae"]
    biaschange = (
        idvals["all_earlier", "ols_mean_signed_error"]
        - idvals["id11_earlier", "ols_mean_signed_error"]
    )
    derived(
        "id11_training_mae_gain_pct",
        idfile,
        {"population": "diagnostic", "weighting": "equal_filter"},
        100 * gain / idvals["all_earlier", "ols_mae"],
        "100*(all_earlier OLS MAE - id11_earlier OLS MAE)/all_earlier OLS MAE",
    )
    derived(
        "id11_training_bias_change",
        idfile,
        {"population": "diagnostic", "weighting": "equal_filter"},
        biaschange,
        "all_earlier mean signed error - id11_earlier mean signed error",
    )
    b = read(blocks)
    delhi = b.loc[
        b.site.eq("Delhi")
        & b.population.eq("diagnostic")
        & b.scheme.eq("later_period")
        & b.status.eq("evaluated")
    ]
    last = delhi.loc[delhi.block.eq("2024Q2")].iloc[0]
    errorfraction = (last.test_n * last.ols_mae) / (delhi.test_n * delhi.ols_mae).sum()
    derived(
        "delhi_final_filter_share",
        blocks,
        {"site": "Delhi", "scheme": "later_period", "population": "diagnostic"},
        100 * last.test_n / delhi.test_n.sum(),
        "100*2024Q2 test_n/sum(supported test_n)",
    )
    derived(
        "delhi_final_absolute_error_share",
        blocks,
        {"site": "Delhi", "scheme": "later_period", "population": "diagnostic"},
        100 * errorfraction,
        "100*2024Q2(test_n*ols_mae)/sum(supported test_n*ols_mae)",
    )
    ap = scores["Addis_Ababa", "leave_quarter_out"]
    af = scores["Addis_Ababa", "later_period"]
    sections = []

    def add(kind, text, **kwargs):
        sections.append(dict(kind=kind, text=text, **kwargs))

    add(
        "title",
        "Site dependent proportionality and temporal transfer of reported HIPS and FTIR predicted EC",
    )
    add("subtitle", "Methods and Results")
    add("heading", "Abstract")
    add(
        "paragraph",
        f"The usefulness of proportional prediction differed across four sites in a frozen cohort of 545 physical filters with paired reported HIPS and FTIR-predicted EC. We compared a training-median HIPS baseline, proportional least squares and ordinary least squares with an intercept using fixed reported-date calendar quarters. In Addis Ababa, the intercept model reduced withheld-quarter mean absolute error from {ap['proportional_mae']:.3f} to {ap['ols_mae']:.3f} Mm⁻¹ and later-period error from {af['proportional_mae']:.3f} to {af['ols_mae']:.3f} Mm⁻¹. The aggregate intercept advantage persisted when training and evaluation were restricted to CalibrationSetId 11. Other sites showed smaller or evaluation-dependent differences, and directional errors remained. These findings characterize prediction between reported products; they do not establish a physical conversion coefficient, independently validate EC or calibrate an aethalometer. [P1–P4]",
    )
    add("heading", "Methods")
    add("subheading", "Meaning and coverage of the baseline")
    add(
        "paragraph",
        "The frozen study baseline means the retained physical-filter populations, flags, source-linked values and specified analysis rules. The prediction baseline means the training-median HIPS prediction used to judge predictive value; the proportional fit is a separate comparator. These terms do not mean that upstream FTIR spectral baseline correction has been independently reconstructed or verified. This draft consolidates the diagnostic, stability and proportionality reports, including their denominator sensitivities, influence checks, weighting choices and retained temporal failures. The accompanying coverage map separates these completed results from the pending source and instrument reviews. [P1–P5]",
    )
    add("subheading", "Physical filter cohorts and reported quantities")
    add(
        "paragraph",
        "The analysis combined records from Addis Ababa, Beijing, Delhi and JPL in Pasadena using physical filter identity. Replicate suffixes were normalized with the repository identity helper; original row identifiers, aliases, units, conflict flags and source hashes were retained. The frozen diagnostic cohort required a usable, non-conflicting same-filter pair of reported HIPS and FTIR-predicted EC and the existing registry eligibility decisions. It contained 545 physical filters. HIPS was analyzed in its reported absorption units, Mm⁻¹, and FTIR-predicted EC in µg/m³. The predictor is a reported prediction product, not an independently established, error-free EC reference. [P1]",
    )
    add(
        "table",
        "Table 1 Frozen physical filter populations",
        columns=["Site", "Diagnostic n", "Ratio n", "2× MDL n", "Reported date range"],
        rows=table1,
    )
    add(
        "paragraph",
        "Ratio eligibility retained 480 filters with positive EC meeting the frozen baseline MDL rule. Saved sensitivity memberships at 1.5×, 2×, 3× and 5× MDL were reused without threshold optimization. Below-MDL and nonpositive EC predictions retained their flags and remained in the diagnostic analysis; no values were substituted and no model predictions were clipped. The ratio summarizes reported HIPS divided by reported FTIR-predicted EC and is not interpreted as a physical mass absorption coefficient. [P1, P2]",
    )
    add("subheading", "Reported date blocks and training procedures")
    add(
        "paragraph",
        "Calendar quarters were assigned from the frozen reported dates, using January–March, April–June, July–September and October–December. The primary analysis withheld one quarter and trained on all remaining quarters, including earlier and later filters. The secondary analysis trained only on earlier quarters and tested the next quarter. These are complementary evaluations of the same observed record with overlapping test filters, not independent replications. Reported dates define these statistical blocks; they do not verify active sampling periods. [P2, P3]",
    )
    add(
        "paragraph",
        "For each site and frozen population, the compared predictions were the training median of HIPS, a proportional prediction Ĥ = kE, and an intercept-bearing prediction Ĥ = a + bE. The proportional coefficient was fitted by equal-filter least squares as k = Σ(EH)/Σ(E²), using training filters only. The intercept model used unweighted ordinary least squares. Each paired comparison required at least ten training filters and two distinct finite EC values. Empty or unsupported folds remained in the ledger, and all models were evaluated on identical supported test filters. No additional regression family, positivity rule or error-based exclusion was introduced. [P2, P3]",
    )
    add("subheading", "Errors sensitivities and provenance")
    add(
        "paragraph",
        "Signed error was predicted minus reported HIPS; positive values indicate overprediction. Mean absolute error was the main comparison metric. The paired intercept advantage was proportional MAE minus intercept-model MAE, so positive differences favor the intercept. Equal-filter summaries averaged over evaluated physical filters. Equal-quarter summaries assigned the same weight to each supported quarterly error; equal-quarter RMSE was the square root of mean quarterly MSE. Neither weighting was selected because it gave a preferred sign. No practical-equivalence margin was specified. [P3]",
    )
    add(
        "paragraph",
        "The bounded metadata sensitivity was restricted to Addis. On the same later ID-11 test filters, training on all eligible earlier filters was compared with training on earlier ID-11 filters only. Both training choices had to satisfy the original support rule. Identifiers were taken from actual EC source rows rather than inferred from date. CalibrationSetId definitions, model mappings, original FTIR training membership and the meaning of LotId as an analytical-batch field remain unresolved. The completed ETAD-0037 omission analysis was retained; no further individual exclusion search was performed. [P2–P4]",
    )
    add(
        "paragraph",
        "The descriptive results motivated the frozen stability specification, and the completed stability results motivated the separately frozen proportionality extension. Each specification preceded the fits it governed. This sequence supports held-out comparisons for the specified procedures but is not an entirely untouched confirmatory exercise. Holding a filter out downstream does not establish that it was held out when its upstream FTIR prediction was developed. [P2–P4]",
    )
    add("heading", "Results")
    add("subheading", "Denominator eligibility changes the represented population")
    add(
        "paragraph",
        "The four site ratio distributions overlapped, while denominator eligibility affected them differently. The diagnostic and ratio cohorts contained 190/190 Addis filters, 163/150 Beijing filters, 62/56 Delhi filters and 130/84 JPL filters. At 2× MDL, Addis retained 189 filters and JPL retained seven. The JPL subset therefore remained a small sensitivity population and could not support the declared minimum training size. It was not treated as a replacement estimate of the JPL relationship. The point-level relationships are shown in Figure 1; complete denominator distributions and memberships accompany the release. [P1, P3]",
    )
    add("subheading", "The predictive cost of proportionality is site dependent")
    add(
        "paragraph",
        f"At Addis, the intercept model improved on proportional prediction in all eight withheld quarters and all six supported later-period quarters. Primary MAE was {ap['proportional_mae']:.3f} Mm⁻¹ for the proportional model, {ap['median_mae']:.3f} Mm⁻¹ for the training-median baseline and {ap['ols_mae']:.3f} Mm⁻¹ with an intercept. Later-period MAEs were {af['proportional_mae']:.3f}, {af['median_mae']:.3f} and {af['ols_mae']:.3f} Mm⁻¹, respectively. Thus forcing this least-squares prediction through the origin performed poorly in the evaluated Addis record, even relative to a constant HIPS prediction. The aggregate intercept advantage persisted across the predefined denominator sensitivities. This result concerns prediction form and does not identify physical background absorption. [P3]",
    )
    add(
        "table",
        "Table 2 Equal filter mean absolute errors and paired model differences in Mm⁻¹",
        columns=[
            "Site",
            "Evaluation",
            "Test n",
            "Median",
            "Proportional",
            "Intercept",
            "Difference",
        ],
        rows=table2,
    )
    add(
        "paragraph",
        "Beijing’s primary intercept advantage was +0.429 Mm⁻¹, but its equal-filter later-period advantage was −0.029 Mm⁻¹. Delhi favored proportional prediction in the primary diagnostic analysis by 2.689 Mm⁻¹; the aggregate primary proportional advantage also occurred in the ratio population and every supported MDL sensitivity. It was therefore not confined to nonpositive diagnostic denominators. Delhi’s later-period intercept advantage was +0.884 Mm⁻¹, while both EC-based models retained strong underprediction. JPL’s aggregate differences were small (−0.042 and −0.077 Mm⁻¹), without a specified margin permitting a declaration of equivalence. Figures 2 and 3 retain the quarter-specific differences and failures. [P3]",
    )
    add("subheading", "Directional errors remain in temporal transfer")
    add(
        "paragraph",
        "Addis’s near-zero primary intercept-model bias (+0.025 Mm⁻¹) coexisted with quarter-specific bias and later-period overprediction (+2.256 Mm⁻¹). Delhi’s later-period intercept-model mean error was −15.298 Mm⁻¹, close in magnitude to its MAE of 15.934 Mm⁻¹, indicating strongly directional error. Beijing 2024Q3 remained a failure relative to the median baseline despite all ten test EC values lying inside the training minimum and maximum. Range overlap alone did not explain that failure. [P2, P3]",
    )
    add(
        "table",
        "Table 3 Intercept model signed errors under both weighting choices in Mm⁻¹",
        columns=["Site", "Evaluation", "Equal filters", "Equal quarters"],
        rows=table3,
    )
    add(
        "paragraph",
        f"Delhi’s primary mean signed error changed sign from −1.779 Mm⁻¹ with equal-filter weights to +2.151 Mm⁻¹ with equal-quarter weights. In later-period evaluation, the final quarter contributed 26 of 38 test filters (68.4%) and {100 * errorfraction:.1f}% of total intercept-model absolute error, calculated by summing test count multiplied by block MAE. The first percentage describes sample composition; the second describes error contribution. Both weighting estimands and the complete block ledger are retained. [P3]",
    )
    add("subheading", "The intercept advantage persists within CalibrationSetId 11")
    add(
        "paragraph",
        f"The strongest ID-11 result concerns model form. When training and evaluation were both restricted to ID-11 filters, proportional MAE was {idvals['id11_earlier', 'proportional_mae']:.3f} Mm⁻¹ and intercept-model MAE was {idvals['id11_earlier', 'ols_mae']:.3f} Mm⁻¹, an aggregate advantage of {advantage:.3f} Mm⁻¹ ({pct:.1f}% lower MAE). Pooling IDs 11 and 17 was therefore not required for the aggregate intercept advantage. This statement does not assume that either ID denotes a documented FTIR model version, and it does not assert an identical advantage in every ID-11 quarter. [P4]",
    )
    add(
        "paragraph",
        f"Training composition produced a separate, smaller change. On 127 common test filters in five supported later quarters, restricting training lowered intercept-model MAE from {idvals['all_earlier', 'ols_mae']:.3f} to {idvals['id11_earlier', 'ols_mae']:.3f} Mm⁻¹ (3.1%). Mean signed error decreased from +{idvals['all_earlier', 'ols_mean_signed_error']:.3f} to +{idvals['id11_earlier', 'ols_mean_signed_error']:.3f} Mm⁻¹, a change of {biaschange:.3f} Mm⁻¹. Restricted training improved MAE in three quarters and worsened it in two. Figure 4 shows those common-filter comparisons. Date and identifier were confounded, so these changes do not establish a causal processing effect. [P4]",
    )
    add("heading", "Interpretation and scope")
    add(
        "paragraph",
        "The completed comparison supports an intercept-bearing empirical prediction for Addis under the tested specification and documents site-dependent limits elsewhere. It does not establish a time-invariant conversion, identify the physical or processing origin of an offset, independently validate FTIR EC or provide an aethalometer calibration. Resolving upstream result roles, calibration-set definitions and training membership is the next evidential task. No additional regression model or residual-mean correction is required to complete the stated filter-only comparison.",
    )
    add(
        "paragraph",
        "The instrument comparison remains separate. The USPA-0257 package contains source-linked reported bounds, a session-46 record crosswalk and a limited status screen, but active collection and clock alignment, observation/correction history, export-specific scaling and an applicable quality decision are unresolved. The candidate does not currently qualify for a reviewed interval comparison. Documented and appropriate corrections can be compatible with valid observations; an absence of corrections is not required. One future accepted candidate would demonstrate the processing path, not establish calibration. [P5]",
    )
    add("heading", "Source and reproducibility references")
    refs = [
        (
            "P1",
            "Frozen cohorts and descriptive results",
            "data/diagnostic/analysis_points.parquet; data/diagnostic/site_results.parquet; data/diagnostic/distribution_summary.parquet; data/diagnostic/sensitivity_point_links.parquet",
        ),
        (
            "P2",
            "Reported date stability specification and outputs",
            "specifications/filter-relationship-stability-spec-2026-09-10.md; data/stability/block_performance_and_influence.parquet; data/stability/heldout_filter_predictions.parquet",
        ),
        (
            "P3",
            "Proportionality specification and outputs",
            "specifications/filter-proportionality-spec-v2-2026-09-10.md; data/proportionality/proportionality_summary.parquet; data/proportionality/proportionality_blocks.parquet; data/proportionality/proportionality_predictions.parquet",
        ),
        (
            "P4",
            "Common ID 11 comparisons and source metadata",
            "data/proportionality/id11_training_summary.parquet; data/proportionality/id11_paired_changes.parquet; data/proportionality/id11_training_blocks.parquet; data/proportionality/FTIR_calibration_identifier_filter_links.csv",
        ),
        (
            "P5",
            "Candidate instrument evidence",
            "phase_reports/proportionality_USPA-0257_evidence_package.md; data/proportionality/USPA-0257_evidence_register.csv",
        ),
    ]
    for key, title, paths in refs:
        add("reference", f"[{key}] {title}. {paths}")
    add(
        "paragraph",
        "The release includes a claim ledger with table hashes and row selectors, a figure ledger linking each selected figure to its source population and specification, and a reproduction entry point with pinned dependencies. Archival source manifests preserve original provenance paths; the release entry point and reader links use the included files. The documented release check repeats the frozen analyses in a fresh environment outside the original checkout.",
    )
    figures = [
        (
            "Figure 1 Reported HIPS and FTIR predicted EC relationships",
            "figures/diagnostic/01_site_relationships.png",
            "The frozen diagnostic physical-filter pairs are shown with descriptive within-site OLS relationships. HIPS and EC have different units; these panels do not assert a 1:1 physical relationship. Source P1.",
            "data/diagnostic/analysis_points.parquet",
            "diagnostic; 545 physical filters",
            "P1",
        ),
        (
            "Figure 2 Paired proportionality comparison",
            "figures/proportionality/01_paired_proportionality.png",
            "Quarter-specific proportional MAE minus intercept-model MAE in reported HIPS units. Positive bars favor an intercept. Models share identical held-out filters; site-specific vertical scales should be read separately. Source P3.",
            "data/proportionality/proportionality_blocks.parquet",
            "diagnostic; leave_quarter_out; baseline",
            "P3",
        ),
        (
            "Figure 3 Later period prediction",
            "figures/proportionality/03_later_period_models.png",
            "Training-median, proportional and intercept-model MAE using strictly earlier training filters. Unavailable folds remain shown. These evaluations overlap the primary record and are not independent replications. Directional errors are reported in Table 3. Source P3.",
            "data/proportionality/proportionality_blocks.parquet",
            "diagnostic; later_period; baseline",
            "P3",
        ),
        (
            "Figure 4 Common ID 11 training comparisons",
            "figures/proportionality/06_id11_common_test_errors.png",
            "Intercept-model errors on the same 127 later ID-11 test filters under two training choices. The aggregate within-ID-11 proportional-versus-intercept comparison is separately reported in the Results. Restricted training does not improve every quarter; identifier and date remain confounded. Source P4.",
            "data/proportionality/id11_training_blocks.parquet",
            "Addis diagnostic; later_period; common_support",
            "P4",
        ),
    ]
    figrows = []
    for title, path, caption, data, pop, spec in figures:
        add("figure", title, path=path, caption=caption)
        figrows.append(
            {
                "figure": title,
                "path": path,
                "figure_sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                "source_table": data,
                "source_table_sha256": hashlib.sha256((ROOT / data).read_bytes()).hexdigest(),
                "population_selector": pop,
                "specification_reference": spec,
            }
        )
    pd.DataFrame(claims).to_csv(ROOT / "claim_ledger.csv", index=False)
    pd.DataFrame(figrows).to_csv(ROOT / "figure_ledger.csv", index=False)
    pd.DataFrame([dict(reference=k, title=t, paths=p) for k, t, p in refs]).to_csv(
        ROOT / "source_reference_ledger.csv", index=False
    )
    (ROOT / "manuscript_content.json").write_text(
        json.dumps(sections, indent=2, ensure_ascii=False) + "\n"
    )
    parts = []
    for s in sections:
        kind = s["kind"]
        if kind == "title":
            parts.append("# " + s["text"])
        elif kind == "heading":
            parts.append("## " + s["text"])
        elif kind == "subheading":
            parts.append("### " + s["text"])
        elif kind == "table":
            parts += ["**" + s["text"] + "**", mdtable(s["rows"], s["columns"])]
        elif kind == "figure":
            parts += ["## " + s["text"], f"![{s['text']}]({s['path']})", s["caption"]]
        elif kind == "reference":
            txt = s["text"]
            for item in re.findall(r"(?:data|specifications|phase_reports)/[^;\s]+", txt):
                txt = txt.replace(item, f"[{Path(item).name}]({item})")
            parts.append(txt)
        else:
            parts.append(s["text"])
    (ROOT / "manuscript.md").write_text("\n\n".join(parts) + "\n")
    (
        ROOT / "coverage_map.md"
    ).write_text("""# Coverage of the frozen baseline and consolidated report

The baseline represents the completed **filter-only** results and their audit
trail. It does not mean every part of the earlier measurement-comparison roadmap
has been validated.

## Three different meanings

| Term | Meaning in this release |
| --- | --- |
| Frozen study baseline | Saved physical-filter identities, reported values, flags, population memberships and specified analysis rules. These are kept fixed across the reported comparisons. |
| Prediction baseline | HIPS training median, estimated within each training fold. The proportional and intercept-bearing fits are separate comparisons. |
| FTIR spectral baselining | Upstream spectral preprocessing. This release does not reconstruct or verify it; the exact FTIR product definitions and processing mappings still need authoritative evidence. |

## Agreed work and where it appears

| Part of the work | Representation in this release | Status and limit |
| --- | --- | --- |
| Physical-filter identities and selection | [Frozen points](data/diagnostic/analysis_points.parquet), source-row links, manuscript Methods/Table 1 | Frozen and reproduced for the filter-only analysis. These identities do not establish active collection intervals. |
| Diagnostic relationships and denominator sensitivity | [Diagnostic report](phase_reports/diagnostic_results_report.md), Figure 1, saved MDL memberships and figures | Included; diagnostic and ratio populations remain distinct. |
| Withheld-quarter and later-period stability | [Stability report](phase_reports/stability_filter_relationship_stability_report.md), Results and block tables | Included; the two schemes revisit overlapping records and are not independent replications. |
| Proportional versus intercept-bearing prediction | [Proportionality report](phase_reports/proportionality_proportionality_temporal_transfer_report.md), Table 2 and Figures 2–3 | Completed under the frozen specification; site differences and failures are retained. |
| ID-11 model form and training composition | [ID-11 summary](data/proportionality/id11_training_summary.parquet), Results and Figure 4 | Both findings included separately; identifier definitions remain unresolved. |
| Weighting, bias and influence | Table 3, full block ledgers and the stability influence outputs | Included; no further error-driven exclusion or weighting selection. |
| Graphs and detailed notes | [Portable notebook](notebooks/filter_only_results.ipynb), main figures and all three phase figure folders | Included. Earlier slide decks remain historical outputs; they have not been refreshed by this consolidation. |
| Upstream EC roles, FTIR processing definitions and training membership | [Existing packet](phase_reports/proportionality_upstream_questions_v2_draft.md), manuscript scope discussion | Reviewed but unsent; recipient/channel and authoritative responses are still needed. No new metadata correction is made. |
| Active collection, clocks, session-46 processing and export scaling | [USPA-0257 evidence package](phase_reports/proportionality_USPA-0257_evidence_package.md) | Candidate review remains unresolved. The limited status screen is not a quality approval or proof of observed coverage. |
| Aethalometer/filter interval and absorption comparison | Candidate evidence and preserved numerical modules | Not a completed scientific comparison. A qualifying interval still requires the documented evidence gates; optical conversion must be established for an absorption comparison. |
| Independent EC validation and instrument calibration | Explicit scope limits in the manuscript | Not established by this filter-only analysis or by a future single accepted interval. |
| Reproduction outside the original checkout | [Entry point](reproduce.py), [pinned environment](requirements.lock), manifests and release checks | Reproduces frozen downstream analyses; it does not recreate FTIR predictions from raw spectra. |

The consolidated narrative is the [scientific draft](manuscript.md). The
[claim ledger](claim_ledger.csv) and [figure ledger](figure_ledger.csv) identify
the frozen tables, selections and hashes supporting the reported results.
""")
    print(
        "Manuscript generated with",
        len(claims),
        "traceable numeric cells and",
        len(figrows),
        "figure records",
    )


if __name__ == "__main__":
    main()
