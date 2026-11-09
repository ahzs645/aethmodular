"""Reconstruct the six closeout figures' populations from saved small tables.

Inputs are read-only. This does not fit models, regenerate aggregates, reproduce
the scientific acceptance test, or execute the gallery. Optional source scores
must have a corrected receipt bound to this decision bundle and recorded QC.
Each run publishes figure-ledger.json and figure-ledger.md in a new directory;
an existing output directory is never reused or overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path[:0] = [str(AREA / "scripts"), str(ROOT)]
from aethmodular_cli import env  # noqa: E402
import candidate_metrics as cm  # noqa: E402
from config import BASE_FILTER_ID_PATTERN, BASE_FILTER_ID_REPL, resolve_seasons, season_for_month  # noqa: E402
from outliers import EXCLUDED_SAMPLES, MANUAL_OUTLIERS, apply_exclusion_flags  # noqa: E402

PROTOCOL = "closeout_saved_figure_population_audit_v1"
BASE = Path("research/ftir_hips_chem/output/tables/closeout_20261107")
PRIMARY_MODELS = ("raw", "spline", "vibes", "vibes_deflate", "vibes_anchorhp", "epo")
TARGET_MODELS = ("vibes k18", "vibes_anchorhp k18")
SEASON_OUTCOMES = {
    "HIPS BC": "hips_bc", "VIBES k18 (before)": "vibes k18",
    "VIBES anchored k18 (after)": "vibes_anchorhp k18", "SPARTAN published": "published",
}
SOURCE_FILES = (
    "gallery/app/src/lib/closeoutReview.ts",
    "gallery/app/src/pages/experimental/ContinuumStudy.tsx",
    "gallery/app/src/pages/experimental/shared.tsx",
    "gallery/app/src/lib/markedRows.tsx", "gallery/app/src/lib/exclusionMode.ts",
    "gallery/app/src/lib/filterId.ts", "gallery/app/src/lib/stats.ts",
    "gallery/app/src/lib/sourceEvaluation.ts", "gallery/app/src/components/Crossplot.tsx",
    "gallery/app/src/App.tsx", "gallery/data/export_continuum_baseline.py",
    "research/ftir_hips_chem/scripts/config.py", "research/ftir_hips_chem/scripts/outliers.py",
    "research/ftir_hips_chem/scripts/candidate_metrics.py",
    "research/ftir_hips_chem/workflows/run_transfer_diagnosis_20261102.py",
    "research/ftir_hips_chem/workflows/run_intercept_fixes_20261103.py",
    "research/ftir_hips_chem/workflows/run_hips_response_20261103.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def display_path(path):
    """Use the central formatter; omit external machine/account directories."""
    path = Path(path)
    displayed = env.display_path(path)
    return f"external/{path.name}" if displayed.startswith(("/", "~/")) else displayed


class Inputs:
    """Hash only the small files used here, then check they stayed unchanged."""

    def __init__(self):
        self.hashes = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        if path not in self.hashes:
            self.hashes[path] = cm.sha256_file(path)
        actual = self.hashes[path]
        require(expected is None or actual == expected, f"input hash differs: {display_path(path)}")
        return actual

    def json(self, path):
        self.add(path)
        return json.loads(Path(path).read_text())

    def csv(self, path):
        self.add(path)
        return pd.read_csv(path, dtype={"filter_id": str, "site": str})

    def manifest_member(self, bundle, manifest, name):
        entry = manifest.get("files", {}).get(name)
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        require(isinstance(expected, str) and bool(expected), f"bundle manifest lacks {name}")
        return self.add(Path(bundle) / name, expected)

    def unchanged(self):
        for path, digest in self.hashes.items():
            require(cm.sha256_file(path) == digest, f"input changed during audit: {display_path(path)}")

    def report(self):
        return [{"path": display_path(path), "sha256": digest}
                for path, digest in sorted(self.hashes.items())]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def physical(value):
    require(value is not None and str(value).strip(), "missing physical-filter ID")
    return re.sub(BASE_FILTER_ID_PATTERN, BASE_FILTER_ID_REPL, str(value).strip())


def table(data, name):
    columns = data.get(name)
    require(isinstance(columns, dict) and bool(columns), f"missing columnar table: {name}")
    require(all(isinstance(v, list) for v in columns.values()), f"{name}: columns must be lists")
    require(len({len(v) for v in columns.values()}) == 1, f"{name}: unequal column lengths")
    return [dict(zip(columns, values)) for values in zip(*columns.values())]


def identity(rows, key="filter_id"):
    ids = [physical(r[key]) for r in rows]
    counts = Counter(ids)
    return dict(rows=len(ids), physical_filters=len(counts),
                duplicate_physical_ids={k: n for k, n in counts.items() if n > 1},
                sorted_unique_ids_sha256=cm.ids_sha256(sorted(counts)))


def indexed(rows, label, key="filter_id"):
    require(not identity(rows, key)["duplicate_physical_ids"], f"{label}: duplicate physical-filter IDs")
    return {physical(row[key]): row for row in rows}


def groups(rows):
    def count(key, transform=str):
        values = {}
        for row in rows:
            values.setdefault(transform(row[key]), set()).add(physical(row["filter_id"]))
        return {key: len(ids) for key, ids in sorted(values.items())}
    return dict(seasons=count("season"), lots=count("lot", lambda v: str(int(float(v)))),
                date_range=[min(r["date"] for r in rows), max(r["date"] for r in rows)] if rows else None)


def booleans(values, label):
    # Reuse the strict shared parser; string "False" must not become truthy.
    try:
        return cm.recorded_source_qc(pd.DataFrame({"raw_qc_ok": list(values)}))[0]
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error


def compact_population(population):
    return {key: value for key, value in population.items() if key != "common_filter_ids"}


def verified_source_summary(directory, bundle, inputs=None):
    """Bind saved corrected scores to this bundle without refitting or rescoring.

    The shared reader verifies all declared output hashes. These extra checks
    bind that internally valid receipt to the supplied metadata, grid, manifest,
    recorded QC, common IDs and reported availability. Full projection/fit
    lineage and held-out-label perturbation remain a separate acceptance step.
    """
    inputs = Inputs() if inputs is None else inputs
    directory, bundle = Path(directory), Path(bundle)
    inputs.add(directory / "baseline_scoreboard_receipt.json")
    receipt = cm.verified_score_receipt(directory)
    require(receipt.get("status") == "corrected", "supplied source scores lack a verified corrected receipt")
    require(receipt.get("schema_version") == 3, "source-score audit requires schema 3 with recorded QC")
    provenance = receipt.get("provenance", {})
    require(provenance.get("epo_protocol") == "epo_train_only_site_cv_v2", "source-score EPO protocol differs")
    manifest = inputs.json(bundle / "MANIFEST.json")
    for key, name in (("improve_metadata_sha256", "improve_meta.csv"),
                      ("wavenumber_grid_sha256", "wn_full.npy")):
        require(provenance.get(key) == inputs.manifest_member(bundle, manifest, name),
                f"source-score {key} does not match supplied decision bundle")
    require(provenance.get("decision_manifest_sha256") == inputs.add(bundle / "MANIFEST.json"),
            "source-score decision manifest does not match supplied decision bundle")
    for name, entry in receipt["outputs"].items():
        inputs.add(directory / name, entry["sha256"])

    im = inputs.csv(bundle / "improve_meta.csv")
    ids = cm.validate_physical_ids(im.filter_id, label="decision bundle")
    qc_mask, qc_policy = cm.recorded_source_qc(im)
    population = receipt.get("population", {})
    require(receipt.get("source_qc_policy") == qc_policy == population.get("source_qc_policy"),
            "source-score recorded QC policy differs from supplied decision bundle")
    require(population.get("filter_ids_sha256") == cm.ids_sha256(ids), "source-score physical-filter identity differs")
    require(population.get("k") == 18 and population.get("unit") == "physical_filter",
            "closeout source scores require k=18 and physical-filter units")
    require(set(population.get("compared_models", [])) == set(PRIMARY_MODELS), "source-score primary candidates differ")
    y = im.ec_ug.to_numpy(float)
    reference = np.isfinite(y) & (y < 150)
    eligible = reference & qc_mask
    expected_counts = dict(n_total=len(ids), n_reference_domain=int(reference.sum()), n_eligible=int(eligible.sum()),
                           n_excluded_raw_qc=int((reference & ~qc_mask).sum()),
                           n_missing_reference=int((~np.isfinite(y)).sum()),
                           n_outside_domain=int((np.isfinite(y) & (y >= 150)).sum()))
    for name, expected in expected_counts.items():
        require(population.get(name) == expected, f"source-score population differs: {name}")

    ledger = inputs.csv(directory / "baseline_scoreboard_filters.csv")
    require(np.array_equal(cm.validate_physical_ids(ledger.filter_id), ids), "source-score ledger IDs/order differ")
    require(np.array_equal(ledger.site.astype(str), im.site.astype(str)), "source-score ledger site IDs differ")
    for name, expected in (("raw_qc_ok", qc_mask), ("reference_domain", reference), ("eligible", eligible)):
        require(np.array_equal(booleans(ledger[name], name), expected), f"source-score ledger {name} differs")
    require(np.all(booleans(ledger.raw_qc_flag_present, "raw_qc_flag_present") == qc_policy["flag_present"]),
            "source-score ledger QC-presence flag differs")
    available = {model: eligible & np.isfinite(ledger[f"{model}_predicted_ug"].to_numpy(float)) for model in PRIMARY_MODELS}
    common = eligible & np.logical_and.reduce(list(available.values()))
    require(np.array_equal(booleans(ledger.common, "common"), common), "source-score common mask differs")
    common_ids = ids[common].tolist()
    require(population.get("common_filter_ids") == common_ids
            and population.get("common_filter_ids_sha256") == cm.ids_sha256(common_ids)
            and population.get("n_common") == len(common_ids), "source-score common physical-filter population differs")
    require(population.get("n_sites") == len(set(im.site[common])), "source-score common site count differs")
    folds = ledger.fold_id.to_numpy(float)
    require(np.isfinite(folds).all() and np.all(folds == folds.astype(int)) and set(folds) == set(range(5)),
            "source-score ledger lacks five integer folds")
    require(ledger.assign(fold=folds).groupby("site").fold.nunique().max() == 1, "source-score site crosses folds")
    common_coverage = len(common_ids) / int(eligible.sum()) if eligible.any() else None
    require(population.get("common_coverage") == common_coverage, "source-score common coverage differs")

    scores = cm.export_scoreboard(inputs.csv(directory / "baseline_scoreboard.csv"), receipt)
    require(len(scores) == len(PRIMARY_MODELS) and set(scores.model) == set(PRIMARY_MODELS) and scores.k.eq(18).all(),
            "source scoreboard must have six unique primary candidates at k=18")
    require(scores.loc[scores.model.eq("epo"), "evaluation_status"].eq("corrected").all(), "EPO score is not marked corrected")
    for row in scores.to_dict("records"):
        model = row["model"]
        require(np.array_equal(booleans(ledger[f"{model}_available"], model), available[model]),
                f"{model}: recorded availability differs")
        counts = {**{key: expected_counts[key] for key in ("n_eligible", "n_reference_domain", "n_excluded_raw_qc")},
                  "n_common": len(common_ids), "n_available": int(available[model].sum()),
                  "n_failed": int((eligible & ~available[model]).sum())}
        for name, expected in counts.items():
            require(row.get(name) == expected, f"{model}: scoreboard {name} differs")
        for name, expected in (("coverage", available[model].sum() / eligible.sum() if eligible.any() else None),
                               ("common_coverage", common_coverage)):
            require((expected is None and pd.isna(row.get(name))) or
                    (expected is not None and math.isclose(row.get(name, math.nan), expected, rel_tol=1e-12, abs_tol=1e-12)),
                    f"{model}: scoreboard {name} differs")
    control = receipt.get("matched_resolution_control")
    require(isinstance(control, dict), "schema-3 source receipt lacks its separate matched-resolution control")
    require(control.get("primary_common_filter_ids_sha256") == population["common_filter_ids_sha256"],
            "control's primary population hash differs")
    same_common = control.get("population", {}).get("common_filter_ids_sha256") == population["common_filter_ids_sha256"]
    require(control.get("same_common_filters_as_primary") is same_common, "control's common-population comparison differs")
    require({r.get("model") for r in control.get("summary", [])} == {"anchorhp_bin2_base", "epo_bin2"},
            "matched-resolution control candidates differ")
    inputs.unchanged()
    return dict(status="verified_corrected", receipt=dict(path=display_path(directory / "baseline_scoreboard_receipt.json"),
                sha256=inputs.add(directory / "baseline_scoreboard_receipt.json"), protocol=receipt["protocol"], schema_version=3),
                population=compact_population(population), recorded_qc=qc_policy,
                metric_definitions=cm.METRIC_DEFINITIONS,
                scores=json.loads(scores.to_json(orient="records", double_precision=15)),
                matched_resolution_control={**control, "population": compact_population(control["population"])},
                validation_scope="Output hashes, bundle identity, recorded QC, common population and availability; saved error values are read, not recomputed.",
                limitations=receipt.get("limitations", []))


def percent_interval(effect, log_se):
    require(finite(effect) and effect > -1 and finite(log_se) and log_se >= 0, "invalid saved seasonal effect or log SE")
    beta, margin = math.log1p(effect), 1.645 * log_se
    values = dict(v=100 * effect, lo=100 * math.expm1(beta - margin), hi=100 * math.expm1(beta + margin))
    require(all(finite(v) for v in values.values()), "nonfinite reconstructed seasonal interval")
    return values


def bin_counts(rows, *keys):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return [{**dict(zip(keys, key)), "exported_bins": len(group),
             "displayed_bins": sum(r["n"] >= 5 for r in group),
             "exported_bin_members": sum(r["n"] for r in group),
             "displayed_bin_members": sum(r["n"] for r in group if r["n"] >= 5),
             "omitted_bins": [{k: r[k] for k in ("lo", "hi", "n")} for r in group if r["n"] < 5]}
            for key, group in grouped.items()]


def build_ledger(frozen_export, decision_bundle, filters_export, source_scores=None):
    inputs = Inputs()
    data, focus = inputs.json(frozen_export), inputs.json(filters_export)
    bundle = Path(decision_bundle)
    manifest = inputs.json(bundle / "MANIFEST.json")
    frames = {}
    for group in ("improve", "addis", "network"):
        inputs.manifest_member(bundle, manifest, f"{group}_meta.csv")
        frames[group] = inputs.csv(bundle / f"{group}_meta.csv")
    addis = table(data, "addis_filters")
    exported = indexed(addis, "frozen Addis")
    focus_by = indexed(focus["rows"], "focus filters", "id")
    network = indexed(frames["network"].to_dict("records"), "network metadata")
    addis_meta = indexed(frames["addis"].to_dict("records"), "Addis metadata")
    require(set(exported) == set(addis_meta), "frozen Addis IDs differ from decision metadata")
    require(set(exported) <= set(network), "network metadata lacks frozen Addis IDs")
    counted = [r for r in addis if not focus_by.get(physical(r["filter_id"]), {}).get("excluded", False)]
    target = {model: [r for r in counted if finite(r["aeth_ir"]) and finite(r[model])] for model in TARGET_MODELS}
    calendar_id = data["settings"]["calendar"]
    require(calendar_id == "belg_feb", "this bounded closeout audit requires the saved belg_feb calendar")
    calendar = {"id": calendar_id, "months": {name.split()[0]: spec["months"] for name, spec in resolve_seasons(calendar_id).items()}}
    for row in addis:
        require(row["season"] == season_for_month(int(row["date"][5:7]), calendar_id).split()[0],
                f"saved season differs from {calendar_id}: {row['filter_id']}")

    # Reconstruct historical model eligibility independently of today's focus exclusion map.
    season_base = [r for r in addis if not r["excluded"] and all(finite(r[k]) for k in ("hips_bc", "exp_tor_hips", "aeth_ir"))]
    require(all(r["aeth_ir"] > 0 for r in season_base), "saved seasonal log-regression has nonpositive aeth values")
    season_models = []
    for saved in table(data, "season_aeth"):
        require(saved["y"] in SEASON_OUTCOMES, "unrecognized saved seasonal outcome")
        rows = [r for r in season_base if finite(r[SEASON_OUTCOMES[saved["y"]]]) and r[SEASON_OUTCOMES[saved["y"]]] > 0]
        require(len(rows) == saved["n"], f"seasonal cohort reconstruction differs: {saved['y']}")
        season_models.append(dict(model=saved["y"], saved_n=saved["n"], population=identity(rows), **groups(rows),
            bega_vs_belg=percent_interval(saved["bega_vs_belg"], saved["bega_se"]),
            kiremt_vs_belg=percent_interval(saved["kiremt_vs_belg"], saved["kiremt_se"])))
    missing_season = [r["filter_id"] for r in addis if finite(r["aeth_ir"]) and r not in season_base]
    response = [r for r in addis if not network[physical(r["filter_id"])]["excluded"]
                and network[physical(r["filter_id"])]["fabs_Mm1"] * network[physical(r["filter_id"])]["volume_m3"] > 0]

    canonical_frame = pd.DataFrame({"id": [r["filter_id"] for r in addis], "date": pd.to_datetime([r["date"] for r in addis])})
    canonical = apply_exclusion_flags(canonical_frame, "Addis_Ababa").set_index("id")
    discrepancies = []
    for key, row in exported.items():
        flags = dict(frozen_addis=bool(row["excluded"]), decision_addis=bool(addis_meta[key]["excluded"]),
                     decision_network=bool(network[key]["excluded"]),
                     focus=bool(focus_by[key]["excluded"]) if key in focus_by else None,
                     canonical=bool(canonical.loc[row["filter_id"], "is_excluded"]))
        if len({v for v in flags.values() if v is not None}) > 1:
            discrepancies.append(dict(filter_id=key, date=row["date"], excluded=flags))

    continuum_rows = [r for r in table(data, "continuum") if str(r["window"]) == "1880"]
    require(continuum_rows, "default 1880 continuum window is absent")
    continuum = bin_counts(continuum_rows, "prep", "group")
    curves = bin_counts(table(data, "tau_curves"), "curve")
    sites = table(data, "response_sites")
    heavy = sorted(sites, key=lambda r: r["tau_median"], reverse=True)[:12]
    source = (verified_source_summary(source_scores, bundle, inputs) if source_scores is not None else
              dict(status="pending", reason="No --source-scores supplied; corrected common-filter source evaluation is pending.",
                   population=None, recorded_qc=None, scores=[],
                   historical_fit_rows=data["settings"]["fit"]["fit_rows"],
                   historical_metric="improve_error is median absolute error; EPO source scores are withheld",
                   limitations=["Historical fit count is not a verified common-score population.",
                                "Source metrics are not inferred from target comparisons or old scalar errors."]))
    ledger = dict(
        schema_version=1, protocol=PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(),
        status="reconstructed_from_saved_tables",
        frozen_export=dict(path=display_path(frozen_export), sha256=inputs.add(frozen_export), generated=data["generated"], schema_version=data["schema_version"]),
        defaults=dict(network="focus", excluded="hide", hand_exclusions=[], legend="all", calendar=calendar,
                      continuum_window_cm1=[1850, 1910], target=dict(x="aeth_ir", before=TARGET_MODELS[0], after=TARGET_MODELS[1], view="pair")),
        identity_hash_definition="SHA-256 of the JSON array of sorted unique physical IDs, using candidate_metrics.ids_sha256; source-score ordered hashes retain their own receipt convention.",
        figures=[
            dict(id="xc-coverage", population=identity(counted), total_export=identity(addis), **groups(counted),
                 hips_available=identity([r for r in counted if finite(r["hips_bc"])])["physical_filters"],
                 aeth_available=identity([r for r in counted if finite(r["aeth_ir"])])["physical_filters"],
                 known_to_focus_map=sum(physical(r["filter_id"]) in focus_by for r in addis),
                 unknown_to_focus_map=sum(physical(r["filter_id"]) not in focus_by for r in addis),
                 excluded_rows=len(addis)-len(counted), metric="Unique physical filters by saved season, not scans.",
                 inclusion="Default focus exclusion map; unknown IDs count. No hand edits, date/site/season filtering, or legend removal reconstructed."),
            dict(id="xc-source", reference="IMPROVE TOR EC; micrograms per filter; fixed 18 components; site-grouped cross-validation", **source),
            dict(id="xc-continuum", panels=continuum, displayed_bin_medians=sum(p["displayed_bins"] for p in continuum),
                 metric="Median of per-filter mean absorbance at 1850–1910 cm^-1; AU multiplied by 1000 for display.",
                 reference="HIPS-derived loading at MAC 10; displayed bins require n>=5; populations repeat across preprocessing panels.",
                 count_limit="Bin n is group.size(), not an explicitly recorded finite-absorbance denominator.",
                 synthetic=dict(study="2–3 November light ambient-filter injections", reviewed_workflow_default_parents=16,
                    default_parent_lots={"251": 10, "248": 6}, background_variants=3, shapes=["flat", "nu^1", "nu^2"],
                    amplitude="0.05 AU at 2000 cm^-1", fit_jobs_from_workflow_defaults=192,
                    parent_ids_available=False, fit_statuses_available=False,
                    saved_medians=table(data, "injection"),
                    limits="Workflow defaults and displayed parent count are documented, not certified by a per-parent ledger. The 87 background blanks and later independent-blank control are different populations."),
                 noise="Scaled MAD across light ambient IMPROVE filters below 1 microgram HIPS; not replicate-blank uncertainty."),
            dict(id="xc-season-aeth", outcomes=season_models, omitted_from_aeth_available=missing_season,
                 reference="Saved log(outcome) on log(MA350 IR BCc) plus season intercepts; Belg is reference.",
                 inclusion="Historical keep: finite HIPS and inferred comparator, not excluded; available aeth and positive outcome. Independent of current live exclusions.",
                 interval="100*expm1(log1p(saved effect) +/- 1.645*saved log SE); approximate 90%, reconstructed from rounded summaries without refitting.",
                 lot_check=table(data, "lot_vs_aeth")),
            dict(id="xc-network", curves=curves, fitted_response=table(data, "response_fit"),
                 reference="IMPROVE TOR-derived response transferred to Addis; no measured Addis TOR is created.",
                 x="sqrt(HIPS mass * reference mass)/(10 * deposit area): geometric-mean optical depth, distinct from raw HIPS site depth.",
                 y="Within-site HIPS/reference relative to geometric-depth bin (0.2,0.3].",
                 selection="Positive HIPS; reference >0.3 microgram; depth in (0.05,3]; >=8 eligible records/site; display n>=5/bin. TOR curves additionally OP>0 and finite TOR<150. SPARTAN curves omit Ethiopia.",
                 curve_interval="approximate 90%", parameter_interval=f"80% (10th–90th percentiles), {data['settings']['response']['bootstrap']} site-bootstrap resamples",
                 addis_eligible=identity(response), addis_aeth_available=sum(finite(r["aeth_ir"]) for r in response),
                 modelwise_response_n_recorded=False, saved_response_rows=len(table(data, "response_addis")),
                 site_table=dict(eligible_sites=len(sites), sum_n=sum(r["n"] for r in sites), displayed_sites=len(heavy),
                     displayed_sum_n=sum(r["n"] for r in heavy), selected=[{k:r[k] for k in ("code", "n", "tau_median")} for r in heavy]),
                 limits="Curves, parametric fits, modelwise response summaries and the site table use different, overlapping selections."),
            dict(id="xc-addis", counted_context=identity(counted),
                 candidates=[dict(model=model, population=identity(rows), **groups(rows)) for model, rows in target.items()],
                 same_physical_pairs=identity(target[TARGET_MODELS[0]])["sorted_unique_ids_sha256"] == identity(target[TARGET_MODELS[1]])["sorted_unique_ids_sha256"],
                 missing_aeth=sum(not finite(r["aeth_ir"]) for r in counted),
                 reference="MA350 IR BCc, filter-window overlap-weighted means with >=12 h coverage; optical, not thermal reference.",
                 metric="Predicted FTIR EC and optical BC in micrograms/m^3; pooled live Deming, lambda=1 by default.",
                 model_version="Historical full-source k=18 target models: production VIBES (87 blanks, rank 70) and VIBES after PTFE anchoring; not implied Newton products.",
                 saved_fits=[r for r in table(data, "addis_fits") if r["x"] == "aeth_ir" and r["model"] in TARGET_MODELS],
                 limits="Saved fits are read, not recomputed. The browser fits rounded export rows and responds to inclusion/legend changes. Absolute Addis EC accuracy remains unresolved."),
        ],
        identity_checks={name: identity(table(data, name)) for name in ("addis_filters", "improve_filters", "tau_points")},
        canonical_exclusions=dict(source="research/ftir_hips_chem/scripts/outliers.py", registry=EXCLUDED_SAMPLES["Addis_Ababa"],
            manual_thresholds=MANUAL_OUTLIERS["Addis_Ababa"]["remove_criteria"], discrepancies=discrepancies,
            handling="Record differences; do not change historical tables or introduce exclusions. ID-aware canonical matching avoids excluding another filter on the same date."),
        evidence_boundaries=[
            "This is table/metadata reconstruction, not execution of the gallery, browser tests or CI; source-file hashes do not establish test success.",
            "No scientific model, spectral correction, response curve, regression or aggregate was refitted.",
            "Metadata hashes are checked against the decision manifest; spectral arrays are not opened or accepted by this audit.",
            "Corrected source attachment verifies byte identity, QC and population consistency; verify_closeout_source_20261107.py supplies the separate fit/projection/perturbation acceptance.",
            "Aggregate tables omit their member IDs; their n values alone do not certify independence or failed-fit coverage.",
            "Coverage and target figures can react to live exclusions; aggregate source, continuum, seasonal and response summaries stay fixed.",
            "Section links reopen active exports; preserve these input hashes and the release receipt with exported figures.",
        ],
        reviewed_implementation=[{"path": display_path(ROOT / name), "sha256": cm.sha256_file(ROOT / name)} for name in SOURCE_FILES],
        audit_code={"path": display_path(Path(__file__)), "sha256": cm.sha256_file(Path(__file__))},
    )
    inputs.unchanged()
    ledger["inputs"] = inputs.report()
    return ledger


def render_markdown(ledger):
    coverage, source, continuum, season, response, target = ledger["figures"]
    cohort = season["outcomes"][0]
    a, b = target["candidates"]
    source_population = (f"{source['population']['n_common']:,} common physical filters / {source['population']['n_sites']} sites; "
                         f"{source['population']['n_eligible']:,} eligible after recorded QC."
                         if source["status"] == "verified_corrected" else
                         f"**Pending**: no corrected-score receipt supplied. Historical fit pool {source['historical_fit_rows']:,} is not a verified common-score count.")
    rows = [
        ("1. Coverage", f"{coverage['population']['physical_filters']} counted physical filters / {coverage['total_export']['rows']} exported rows; {coverage['aeth_available']} aeth values.", "Unique physical filters, not scans; saved belg_feb calendar."),
        ("2. Source performance", source_population, "IMPROVE TOR EC, µg/filter, fixed k=18; read error metrics with common coverage and recorded QC."),
        ("3. Baseline behavior", f"{continuum['displayed_bin_medians']} displayed bin medians across preprocessing panels.", "1850–1910 cm⁻¹ mean, then bin median; AU ×1000. HIPS loading at MAC 10; n≥5 bins."),
        ("4. Seasonal aeth contrasts", "; ".join(f"{r['model']}: {r['saved_n']}" for r in season['outcomes']), "Relative to Belg; historical complete cases also require inferred-comparator availability. Approximate 90% intervals transformed from saved log SEs."),
        ("5. Optical response", f"{response['addis_eligible']['physical_filters']} Addis response-eligible; {response['addis_aeth_available']} aeth matches. {response['site_table']['displayed_sites']}/{response['site_table']['eligible_sites']} sites shown.", "Transferred IMPROVE response; geometric-mean depth on the curves, raw HIPS depth in the site table. Curve bands 90%; parameter ranges 80%."),
        ("6. Candidate target comparison", f"{a['population']['physical_filters']} / {b['population']['physical_filters']} physical pairs; same IDs: {target['same_physical_pairs']}.", "Production/anchored VIBES against MA350 IR BCc; historical k=18 target models, Deming λ=1. Absolute Addis EC accuracy unresolved."),
    ]
    out = ["# Closeout figure population ledger", "", f"Protocol: `{ledger['protocol']}`. Generated `{ledger['created_utc']}`.", "",
           f"Frozen export generated `{ledger['frozen_export']['generated']}`; SHA-256 `{ledger['frozen_export']['sha256']}`.", "",
           "This audit reconstructs saved populations and transforms saved interval summaries. It does not fit models, execute the gallery or certify UI/CI tests.", "",
           "| Figure | Population / coverage | Metric, reference and limits |", "|---|---|---|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    out += ["", "## Cohort and inclusion details", "",
            f"Coverage seasons: `{coverage['seasons']}`; lots: `{coverage['lots']}`; dates: `{coverage['date_range']}`. "
            f"{coverage['unknown_to_focus_map']} export IDs are unknown to the focus map and count under its default rule.", "",
            f"Seasonal first-outcome cohort: `{cohort['seasons']}`, lots `{cohort['lots']}`, dates `{cohort['date_range']}`. "
            f"Aeth-available rows omitted by its historical complete-case gate: {', '.join(season['omitted_from_aeth_available']) or 'none'}.", "",
            f"Target first-panel pairs: `{a['seasons']}`, lots `{a['lots']}`, dates `{a['date_range']}`; {target['missing_aeth']} counted rows lack aeth values.", "",
            "Canonical exclusions are read from `scripts/outliers.py`; no flags or historical summaries are rewritten. "
            "The JSON records the current registry and differences between frozen Addis, decision Addis/network, focus and canonical flags."]
    for row in ledger["canonical_exclusions"]["discrepancies"]:
        out += ["", f"- `{row['filter_id']}` ({row['date']}): `{row['excluded']}`."]
    if source["status"] == "verified_corrected":
        qc = source["recorded_qc"]
        out += ["", "## Verified source-score attachment", "",
                f"Recorded QC: `{qc['protocol']}`, flag present **{qc['flag_present']}**, {qc['n_pass']:,} pass / {qc['n_excluded']:,} excluded among {qc['n_total']:,} metadata rows. "
                f"{source['population']['n_excluded_raw_qc']} QC exclusions lie inside the TOR reference domain. Error values below are read from the hash-bound saved scoreboard, not recomputed.", "",
                "| Candidate | MedAE | MAE | RMSE | Mean signed bias | P95 absolute error | Available / eligible |", "|---|---:|---:|---:|---:|---:|---:|"]
        def fmt(value):
            return "Unavailable" if value is None else f"{value:.4f}"
        for r in source["scores"]:
            out += [f"| {r['model']} | " + " | ".join(fmt(r[k]) for k in ("improve_median_absolute_error", "improve_mae", "improve_rmse", "improve_mean_bias", "improve_p95_absolute_error")) + f" | {r['n_available']} / {r['n_eligible']} |"]
        control = source["matched_resolution_control"]
        out += ["", f"The matched-resolution control is separate from the six primary candidates: {control['population']['n_common']} common filters; same common IDs as primary: **{control['same_common_filters_as_primary']}**. Consult its receipt for its two saved score rows."]
    else:
        out += ["", "## Source evaluation pending", "", source["reason"],
                "Legacy `improve_error` means median absolute error. MAE, RMSE, signed bias, tail error and common coverage cannot be reconstructed from that scalar; historical EPO source scores remain withheld."]
    out += ["", "## Fixed aggregate populations", "", "| Continuum preprocessing / group | Exported members | Displayed members | Displayed bins |", "|---|---:|---:|---:|"]
    out += [f"| {r['prep']} / {r['group']} | {r['exported_bin_members']} | {r['displayed_bin_members']} | {r['displayed_bins']} |" for r in continuum["panels"]]
    out += ["", "These are repeated views of the same filters; do not sum members across preprocessing panels. Bin n is membership, not a separately recorded finite-absorbance denominator.", "",
            "| Optical curve | Exported bin members | Displayed members | Displayed bins |", "|---|---:|---:|---:|"]
    out += [f"| {r['curve']} | {r['exported_bin_members']} | {r['displayed_bin_members']} | {r['displayed_bins']} |" for r in response["curves"]]
    out += ["", "The synthetic display reports the 2–3 November ambient-parent study. Its reviewed defaults are 16 parents, three backgrounds and three additions plus unspiked spectra: 192 fitting jobs. Parent IDs and fit diagnostics are absent from the export. The 87 background-training blanks and later independent-blank control are distinct populations.", "",
            "## Evidence boundaries", ""]
    out += [f"- {text}" for text in ledger["evidence_boundaries"]]
    out += ["", "Input hashes, physical-ID hashes, reconstructed intervals, source-score details and reviewed source-file hashes are in `figure-ledger.json`.", ""]
    return "\n".join(out)


def audit(frozen_export, decision_bundle, filters_export, output, source_scores=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"audit output exists: {display_path(output)}; choose a new --output directory")
    ledger = build_ledger(frozen_export, decision_bundle, filters_export, source_scores)
    json_text = json.dumps(ledger, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    markdown = render_markdown(ledger)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive reservation: a concurrent or prior run cannot be replaced.
    for name, contents in (("figure-ledger.json", json_text), ("figure-ledger.md", markdown)):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(contents)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--frozen-export", type=Path, default=BASE / "frozen_public/continuum_baseline.json")
    parser.add_argument("--decision-bundle", type=Path, default=BASE / "decision_bundle")
    parser.add_argument("--filters-export", type=Path, default=BASE / "frozen_public/filters.json")
    parser.add_argument("--source-scores", type=Path, help="optional schema-3 corrected source-score directory; omission records pending")
    parser.add_argument("--output", type=Path, default=BASE / "figure_audit", help="new directory for figure-ledger.json and figure-ledger.md; never overwritten")
    args = parser.parse_args()
    try:
        ledger = audit(resolve_path(args.frozen_export), resolve_path(args.decision_bundle), resolve_path(args.filters_export),
                       resolve_path(args.output), resolve_path(args.source_scores) if args.source_scores else None)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(dict(status=ledger["status"], source_scores=ledger["figures"][1]["status"],
                          output=display_path(resolve_path(args.output))), ensure_ascii=False))


if __name__ == "__main__":
    main()
