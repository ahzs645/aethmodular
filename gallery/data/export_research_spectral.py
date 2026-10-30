"""Export the spectral and calibration diagnostics for the private Research tab.

    uv run --locked --no-sync python gallery/data/export_research_spectral.py

Writes one JSON per page to gallery/app/public/data/research/spectral/ (a
private folder: the Research section is left out of public builds). The pages
bring in notebooks whose results the gallery had only in part: the 1600/1617
cm⁻¹ band (ftir_12, ftir_38), MAC 6 vs 10 (ftir_16, ftir_19), the provisional
Addis series (ftir_29, ftir_30), applicability domain and analog quality
(ftir_36, ftir_39, ftir_50, char_12), the network spectral map (ftir_52), the
June 2026 Ethiopia calibrations (ftir_ec_calibration_2026_06_25 07/09/10/12/15,
spartan_ec_2026_06_16 08) and the committee backup figures (ftir_53).

Nothing is refitted. Each table is read from, in order of preference:

1. the CSV the notebook wrote, when its gitignored ``output/`` folder exists
   (after ``python scripts/build_notebooks.py <n>`` on a machine with the Drive);
2. otherwise the table the executed notebook displayed, which is committed with
   the notebook (``notebook_outputs.ExecutedNotebook``);
3. tracked CSVs (``ftir_ec_calibration_2026_06_25/tables``,
   ``spartan_ec_2026_06_16/tables``).

Where the Drive's "Davis Data" FTIR folder resolves (pls_transfer.FTIRTransferPaths,
which honours AETHMODULAR_DRIVE_ROOT), spartan_ec 08's cleaning is recomputed on all
906 calibration filters and gated on reproducing 08's tracked removals.

Every table records which of these it came from. Per-filter tables that only a
re-run can produce are listed under ``missing`` with the command that makes
them, and the page shows that list. Whatever the gallery already exports
(calibration_runs.json: per-filter predictions, MAC 6/10/17 fits, site spectra)
is read by the pages directly and not duplicated here.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meeting_bundle import columns  # noqa: E402
from notebook_outputs import ExecutedNotebook  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "research/ftir_ec_phase3"
CHAR = ROOT / "research/ftir_etad_char"
JUNE25 = ROOT / "research/ftir_ec_calibration_2026_06_25"
JUNE16 = ROOT / "research/spartan_ec_2026_06_16"
OUT = ROOT / "gallery/app/public/data/research/spectral"

# The gallery calls the USPA site JPL (useGalleryData rewrites calibration exports the same way).
RELABEL = {"Pasadena": "JPL"}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


# What a phase-3 re-run needs first (checked against the Davis Data mirror, 27 Oct 2026).
PREREQUISITES = {
    "12": "ftir_11's output/tables/ftir11 (ftir_11 needs the phase-2 ftir_09/ftir_10 tables in research/ftir_hips_chem/output/tables; "
          "ftir_09 also reads Downloads Backup/HIPS_all_SPARTAN_batches.csv, which is not in the Davis Data folder)",
    "19": "ftir_11 and ftir_13 outputs and the phase-2 ftir_10 predictions (same chain as ftir_12)",
    "38": "output/tables/variation_closure (ftir_35) and calibration_explorer/targets (built from the SPARTAN FTIR and HIPS pulls)",
    "39": "ftir_36's output/tables/ftir36, output/corrected/*_corrected_df6.npz (spline baseline over the pool) and calibration_explorer/targets",
    "52": "output/corrected/*_corrected_df6.npz and calibration_explorer/targets",
}


def prerequisite(notebook: str) -> str | None:
    return PREREQUISITES.get(re.match(r"(?:ftir|char)_(\d+)", notebook).group(1))


def rebuild_command(folder: Path, notebook: str) -> str:
    """How a phase-3 (or char) notebook is re-executed so its output/tables CSVs exist."""
    number = re.match(r"(?:ftir|char)_(\d+)", notebook).group(1)
    if folder == PHASE3:
        return f"cd {rel(PHASE3)} && uv run python scripts/build_notebooks.py {number}"
    return f"cd {rel(folder)} && uv run jupyter nbconvert --to notebook --execute --inplace {notebook}.ipynb"


def relabel(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: _relabel(v) if isinstance(v, str) else v)
    return out


def _relabel(text: str) -> str:
    for old, new in RELABEL.items():
        text = text.replace(old, new)
    return text


class Page:
    """One page's tables, findings text, sources and missing per-filter inputs."""

    def __init__(self, key: str, title: str, out: Path = OUT, exporter: Path | None = None) -> None:
        self.key, self.title, self.out = key, title, out
        self.exporter = exporter or Path(__file__)
        self.tables: dict[str, pd.DataFrame] = {}
        self.text: dict[str, str] = {}
        self.sources: list[dict] = []
        self.missing: list[dict] = []
        self._notebooks: dict[Path, ExecutedNotebook] = {}

    def notebook(self, folder: Path, name: str) -> ExecutedNotebook:
        path = folder / f"{name}.ipynb"
        if path not in self._notebooks:
            self._notebooks[path] = ExecutedNotebook(path)
        return self._notebooks[path]

    def add(self, name: str, frame: pd.DataFrame, **source) -> pd.DataFrame:
        frame = relabel(frame.reset_index(drop=True))
        self.tables[name] = frame
        self.sources.append({"table": name, "rows": len(frame), **source})
        return frame

    def read(self, folder: Path, notebook: str, marker: str, *, nth: int = 0,
             csv: str | None = None, index: str | None = None) -> tuple[pd.DataFrame, dict]:
        """A table the notebook both wrote (csv, relative to the notebook's folder) and displayed,
        with where it came from. `index` names the displayed frame's index column (a row label);
        an unnamed numeric index is dropped."""
        if csv and (folder / csv).exists():
            frame = pd.read_csv(folder / csv)
            source = {"notebook": notebook, "read": "csv", "path": rel(folder / csv)}
        else:
            frame = self.notebook(folder, notebook).table(marker, nth)
            source = {"notebook": notebook, "read": "executed notebook output", "path": rel(folder / f"{notebook}.ipynb"), "cell": marker}
        unnamed = [c for c in ("index", "Unnamed: 0") if c in frame.columns]
        if index and unnamed:
            frame = frame.rename(columns={unnamed[0]: index})
        elif unnamed and pd.api.types.is_numeric_dtype(frame[unnamed[0]]):
            frame = frame.drop(columns=unnamed[0])
        return frame, source

    def shown(self, name: str, folder: Path, notebook: str, marker: str, **how) -> pd.DataFrame:
        frame, source = self.read(folder, notebook, marker, **how)
        return self.add(name, frame, **source)

    def stream(self, folder: Path, notebook: str, marker: str) -> str:
        return self.notebook(folder, notebook).stream(marker)

    def optional(self, name: str, folder: Path, notebook: str, csv: str, what: str,
                 transform=None) -> pd.DataFrame | None:
        """A per-filter table only a re-run writes: added when present, else listed as missing."""
        path = folder / csv
        if path.exists():
            frame = pd.read_csv(path)
            return self.add(name, transform(frame) if transform else frame, notebook=notebook, read="csv", path=rel(path))
        self.missing.append({"table": name, "what": what, "needs": rel(path), "run": rebuild_command(folder, notebook), "after": prerequisite(notebook)})
        return None

    def write(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        bundle = {
            "schema_version": 1,
            "id": self.key,
            "title": self.title,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generated_from": rel(self.exporter),
            "file": f"{self.out.relative_to(ROOT / 'gallery/app/public/data').as_posix()}/{self.key}.json",
            "text": self.text,
            "sources": self.sources,
            "missing": self.missing,
            "tables": {name: columns(frame) for name, frame in self.tables.items()},
        }
        path = self.out / f"{self.key}.json"
        path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
        print(f"wrote {rel(path)} ({path.stat().st_size / 1024:.0f} KB): "
              + ", ".join(f"{k} {len(v)}" for k, v in self.tables.items())
              + (f"; missing {len(self.missing)}" if self.missing else ""))
        return path


def number(pattern: str, text: str, cast=float):
    """One captured value from printed notebook output; a changed print format fails loudly."""
    hit = re.search(pattern, text)
    if not hit:
        raise ValueError(f"pattern {pattern!r} not found in notebook output")
    return cast(hit.group(1))


def long(frame: pd.DataFrame, id_cols: list[str], var: str, value: str) -> pd.DataFrame:
    return frame.melt(id_vars=id_cols, var_name=var, value_name=value)


def ranked_sites(frame: pd.DataFrame, column: str, measure: str) -> pd.DataFrame:
    """'NOGA1 (0.9902), CHAS1 (0.9885), …' cells into one row per site. A notebook display cuts
    long strings with '...'; only the complete 'SITE (value)' entries are kept."""
    rows = []
    for target, cell in zip(frame["target"], frame[column]):
        for rank, (site, value) in enumerate(re.findall(r"([A-Z]{4}\d)\s*\(([\d.]+)\)", str(cell)), start=1):
            rows.append({"target": target, "measure": measure, "rank": rank, "site": site, "value": float(value)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 1600/1617 band (ftir_12, ftir_38, ftir_50)
def band_1617() -> Page:
    p = Page("band_1617", "The 1600/1617 cm⁻¹ band")
    nb12, nb38, nb50 = "ftir_12_band_1600_identity", "ftir_38_1617_and_dust_attribution", "ftir_50_spectral_comparison_methods"
    p.shown("groups", PHASE3, nb12, "group_audit.to_csv", csv="output/tables/ftir12/group_audit.csv")
    centers = p.shown("centers", PHASE3, nb12, "center_stats.to_csv", csv="output/tables/ftir12/peak_center_1600_stats.csv")
    cov = p.shown("covariation", PHASE3, nb12, "covariation.to_csv", csv="output/tables/ftir12/band1600_covariation.csv")
    if set(cov["group"]) == {"Addis"}:
        p.missing.append({"table": "covariation", "what": "companion-band covariation for the three IMPROVE groups (the notebook displayed Addis only)",
                          "needs": rel(PHASE3 / "output/tables/ftir12/band1600_covariation.csv"), "run": rebuild_command(PHASE3, nb12), "after": prerequisite(nb12)})
    p.shown("ratios", PHASE3, nb12, "ratios.to_csv", csv="output/tables/ftir12/band1600_ratio_context.csv")
    p.shown("tracking", PHASE3, nb12, "tracking.to_csv", csv="output/tables/ftir12/addis_band_vs_absorption.csv")
    sites = p.shown("site_bands", PHASE3, nb38, "neutral_band_and_pmf()", nth=0, csv="output/tables/ftir38/neutral_band_cross_site_summary.csv")
    pmf = p.shown("pmf_tests", PHASE3, nb38, "neutral_band_and_pmf()", nth=1, csv="output/tables/ftir38/addis_pmf_band_tests.csv")
    p.optional("dust_tests", PHASE3, nb38, "output/tables/ftir38/canonical_soil_residual_tests.csv",
               "canonical-soil and potassium partial r against each locked configuration's residual (ftir_38 drew it but displayed no table)")
    bands, source = p.read(PHASE3, nb50, "bands.to_csv", csv="output/tables/ftir50/band_correlations.csv")
    p.add("band_r", long(bands, ["target"], "band", "r"), **source)

    addis = centers.set_index("group").loc["Addis"]
    improve_min = centers[centers["group"] != "Addis"]["center_median"].min()
    charcoal = pmf.set_index("predictor").loc["charcoal_frac"]
    p.text["finding"] = (
        f"Addis 1600-band peaks sit at {addis.center_p25:.0f}–{addis.center_p75:.0f} cm⁻¹ (IQR, n = {int(addis.n_used)}), "
        f"below every IMPROVE group (medians ≥ {improve_min:.0f}). Under a neutral baseline the feature stays at "
        f"Addis ({sites.set_index('site').loc['Addis', 'median_peak_cm1']:.0f}) and Bishoftu "
        f"({sites.set_index('site').loc['Bishoftu', 'median_peak_cm1']:.0f}), while Delhi and Beijing peak at the 1680 edge. "
        f"Its per-filter strength has no adjusted association with the PMF charcoal fraction "
        f"(partial r = {charcoal.partial_r:.2f}, q = {charcoal.fdr_q:.2f}, n = {int(charcoal.n)}).")
    return p


# ---------------------------------------------------------------- MAC 6 vs 10 (ftir_16, ftir_19)
def _deciles(frame: pd.DataFrame) -> pd.DataFrame:
    """ftir_16's decile table: '(lo, hi]' interval labels into bounds and a readable label."""
    bounds = frame["OC_EC"].astype(str).str.extract(r"\(([-\d.e]+),\s*([-\d.e]+)\]").astype(float)
    out = frame.drop(columns="OC_EC").assign(decile=np.arange(1, len(frame) + 1), OC_EC_lo=bounds[0], OC_EC_hi=bounds[1])
    out["OC/EC decile"] = [f"{i} · {lo:.3g}–{hi:.3g}" for i, lo, hi in zip(out.decile, out.OC_EC_lo, out.OC_EC_hi)]
    return out


def mac_choice() -> Page:
    p = Page("mac_choice", "MAC 6 vs 10")
    nb16, nb19 = "ftir_16_mac_decision_prep", "ftir_19_mac_effect_on_calibrations"
    p.shown("setups", PHASE3, nb19, "metrics.to_csv(TABLE_DIR / 'mac_effect_metrics.csv'", index="setup")
    p.optional("setup_filters", PHASE3, nb19, "output/tables/ftir19/addis_predictions_all_setups.csv",
               "ftir_19's per-filter predictions for its six setups (the gallery draws the same pivot from calibration_runs.json)")
    p.shown("adama_scenarios", PHASE3, nb16, "implied_mac.to_csv", csv="output/tables/ftir16/addis_implied_mac_under_adama_composition.csv")
    p.shown("power", PHASE3, nb16, "power.to_csv", csv="output/tables/ftir16/quartz_campaign_power_by_season.csv")
    bridge_csv = PHASE3 / "output/tables/ftir16/improve_implied_mac_bridge.csv"
    if bridge_csv.exists():
        bridge = pd.read_csv(bridge_csv, usecols=["implied_MAC", "OC_EC"])
        table = (bridge.groupby(pd.qcut(bridge["OC_EC"], 10), observed=True)["implied_MAC"]
                 .agg(n="size", MAC_p25=lambda v: v.quantile(.25), MAC_median="median", MAC_p75=lambda v: v.quantile(.75))
                 .reset_index().assign(OC_EC=lambda d: d["OC_EC"].astype(str)))
        p.add("bridge", _deciles(table), notebook=nb16, read="csv", path=rel(bridge_csv))
        overall = (len(bridge), bridge.implied_MAC.median(), bridge.implied_MAC.quantile(.25), bridge.implied_MAC.quantile(.75))
        like = bridge[bridge.OC_EC <= 2.27]["implied_MAC"]
        addis_like = (len(like), like.median(), like.quantile(.25), like.quantile(.75))
    else:
        p.add("bridge", _deciles(p.notebook(PHASE3, nb16).table("bridge_improve.to_csv")), notebook=nb16,
              read="executed notebook output", path=rel(PHASE3 / f"{nb16}.ipynb"), cell="bridge_improve.to_csv")
        text = p.stream(PHASE3, nb16, "bridge_improve.to_csv")
        overall = (number(r"IMPROVE implied MAC: n=(\d+)", text, int), number(r"IMPROVE implied MAC: n=\d+, median=([\d.]+)", text),
                   number(r"IMPROVE implied MAC:.*?p25-p75 ([\d.]+)-", text), number(r"IMPROVE implied MAC:.*?p25-p75 [\d.]+-([\d.]+)", text))
        addis_like = (number(r"subset: n=(\d+)", text, int), number(r"subset: n=\d+, implied MAC median=([\d.]+)", text),
                      number(r"subset:.*?p25-p75 ([\d.]+)-", text), number(r"subset:.*?p25-p75 [\d.]+-([\d.]+)", text))
    p.add("bridge_summary", pd.DataFrame([
        {"population": "All matched IMPROVE filters", "n": overall[0], "median": overall[1], "p25": overall[2], "p75": overall[3]},
        {"population": "Addis-like OC/EC ≤ 2.27", "n": addis_like[0], "median": addis_like[1], "p25": addis_like[2], "p75": addis_like[3]},
    ]), notebook=nb16, read=p.sources[-1]["read"], path=p.sources[-1]["path"])
    chem = p.stream(PHASE3, nb16, "joined.to_csv")
    p.add("chemspec", pd.DataFrame([{
        "n base filters": number(r"base-joined n=(\d+)", chem, int),
        "r (ChemSpec EC vs Fabs/10)": number(r"Fabs/10: r=([\d.]+)", chem),
        "median ChemSpec EC / Fabs": number(r"ChemSpec/Fabs median=([\d.]+)", chem),
        "implied MAC": number(r"MAC implied by ChemSpec convention: median ([\d.]+)", chem),
    }]), notebook=nb16, read="executed notebook output", path=rel(PHASE3 / f"{nb16}.ipynb"), cell="joined.to_csv")
    p.text["bridge"] = (
        f"On {overall[0]:,} matched IMPROVE filters the implied MAC (Fabs ÷ TOR EC) has median {overall[1]:.2f} "
        f"(IQR {overall[2]:.2f}–{overall[3]:.2f}); at Addis-like OC/EC ≤ 2.27 (n = {addis_like[0]:,}) it is "
        f"{addis_like[1]:.2f} (IQR {addis_like[2]:.2f}–{addis_like[3]:.2f}): MAC 10 sits at the centre, MAC 6 in the lower tail.")
    return p


# ---------------------------------------------------------------- provisional Addis series (ftir_29, ftir_53 §4)
def provisional_series() -> Page:
    p = Page("provisional_series", "Provisional Addis EC")
    nb29, nb53 = "ftir_29_provisional_addis_ec_series", "ftir_53_committee_backup_figures"
    p.shown("seasonal", PHASE3, nb29, "seasonal.to_csv", csv="output/tables/ftir29/seasonal_summary.csv", index="season")
    p.shown("offset_ci", PHASE3, nb29, "ci.to_csv", csv="output/tables/ftir29/c_bootstrap_ci.csv")
    p.shown("summary", PHASE3, nb29, "summary.to_csv", csv="output/tables/ftir29/provisional_summary.csv")
    p.shown("kiremt_dry_ftir51", PHASE3, nb53, "kiremt_dry_all_variants.csv", csv="output/tables/ftir53/kiremt_dry_all_variants.csv")
    return p


# ---------------------------------------------------------------- applicability domain (ftir_36, ftir_39, ftir_50, char_12)
def applicability() -> Page:
    p = Page("applicability", "Applicability domain")
    nb36, nb39, nb50, nbc = ("ftir_36_domain_invariant_pls", "ftir_39_pls_applicability_domain",
                             "ftir_50_spectral_comparison_methods", "char_12_weakley_transferability")
    p.shown("protocol", PHASE3, nb36, "selection.to_csv(OUT / 'frozen_protocol.csv'", nth=0, csv="output/tables/ftir36/frozen_protocol.csv")
    p.shown("lambda", PHASE3, nb36, "selection.to_csv(OUT / 'frozen_protocol.csv'", nth=1, index="multiplier")
    p.shown("source_validation", PHASE3, nb36, "source_validation.to_csv", csv="output/tables/ftir36/outer_source_validation.csv")
    p.shown("target_summary", PHASE3, nb36, "target_summary.to_csv", csv="output/tables/ftir36/locked_target_summary.csv")
    p.shown("dipls_vs_pls", PHASE3, nb36, "improvements.to_csv", nth=1, csv="output/tables/ftir36/dipls_vs_pls.csv")
    domain = p.shown("domain_summary", PHASE3, nb39, "applicability_domain_analysis()", csv="output/tables/ftir39/locked_target_domain_summary.csv")
    p.optional("domain_filters", PHASE3, nb39, "output/tables/ftir39/locked_target_domain_per_filter.csv",
               "per-filter T² and Q source percentiles with |PLS − HIPS/10| for the 14 Addis and 26 Delhi locked holdouts")
    p.shown("t2q", PHASE3, nb50, "domain.to_csv", csv="output/tables/ftir50/domain_diagnostics.csv")
    p.shown("hubness", PHASE3, nb50, "hub.to_csv", csv="output/tables/ftir50/hubness.csv")
    p.shown("neighbours", PHASE3, nb50, "neighbours.to_csv", csv="output/tables/ftir50/nearest_analogs.csv")
    agreement, source = p.read(PHASE3, nb50, "display(agreement)", index="metric")
    p.add("metric_agreement", long(agreement, ["metric"], "against", "rho"), **source)
    p.shown("escape", CHAR, nbc, "display(esc_tbl", csv="output/tables/char12/escape_by_group.csv", index="group")
    p.shown("lot_effect", CHAR, nbc, "display(lot_tbl", csv="output/tables/char12/lot_effect.csv")
    p.shown("raw_scale", CHAR, nbc, "display(scatter_proxy)", csv="output/tables/char12/raw_scale_comparison.csv", index="population")

    winner = domain[(domain.config == "addis_winner") & (domain.target == "addis_holdout")].iloc[0]
    p.text["finding"] = (
        f"The Addis OC/EC + AIRSpec winner places {winner.outside_source_95_pct:.0f} % of the {int(winner.n)} locked Addis holdouts "
        f"outside the source 95 % envelope, yet misses them by RMSE {winner.RMSE_inside:.2f} µg/m³: in-domain is a warning system, "
        "not a guarantee. The derivative-space candidates put 38–64 % of their targets outside.")
    return p


# ---------------------------------------------------------------- the network as a spectral map (ftir_52)
def network_map() -> Page:
    p = Page("network_map", "Network spectral map")
    nb = "ftir_52_network_spectral_map"
    out = "output/tables/ftir52/"
    classes = p.shown("classes", PHASE3, nb, "classes.to_csv", csv=out + "ward_classes.csv")
    mags = p.stream(PHASE3, nb, "classes.to_csv")
    peak = {int(k): float(v) for k, v in re.findall(r"(\d+): ([\d.]+)", re.search(r"median peak absorbance by class: \{(.*?)\}", mags).group(1))}
    p.tables["classes"] = classes.assign(peak_abs_median=classes.cluster.astype(int).map(peak))
    target_cols = {"addis_n": "Addis (ETAD)", "etbi_n": "Bishoftu (ETBI)", "indh_n": "Delhi (INDH)", "chts_n": "Beijing (CHTS)", "uspa_n": "JPL (USPA)"}
    share = classes.melt(id_vars=["cluster"], value_vars=list(target_cols), var_name="target", value_name="n_filters")
    share["target"] = share["target"].map(target_cols)
    share["share"] = share["n_filters"] / share.groupby("target")["n_filters"].transform("sum")
    share["Ward class"] = "class " + share["cluster"].astype(int).astype(str)
    p.add("class_share", share, notebook=nb, read=p.sources[-1]["read"], path=p.sources[-1]["path"])
    sil = p.stream(PHASE3, nb, "linkage(S_all, method='ward')")
    scores = {int(k): float(v) for k, v in re.findall(r"(\d+): ([\d.]+)", re.search(r"silhouette by k: \{(.*?)\}", sil).group(1))}
    p.add("silhouette", pd.DataFrame({"k": list(scores), "silhouette": list(scores.values()), "series": "silhouette (4,000-filter subsample)"}),
          notebook=nb, read="executed notebook output", path=rel(PHASE3 / f"{nb}.ipynb"), cell="linkage(S_all, method='ward')")
    p.shown("smoke_by_loading", PHASE3, nb, "loading_matched.to_csv", csv=out + "smoke_enrichment_by_loading.csv")
    near, source = p.read(PHASE3, nb, "site_near.to_csv", nth=0, csv=out + "nearest_sites.csv")
    mix, mix_source = p.read(PHASE3, nb, "site_near.to_csv", nth=1)
    p.add("nearest_sites", pd.concat([ranked_sites(near, "nearest_IMPROVE_sites", "most like (median-spectrum r)"),
                                      ranked_sites(near, "most_unlike", "least like (median-spectrum r)"),
                                      ranked_sites(mix, "nearest_by_class_mix", "most like (Ward-class mix, JS distance)")]), **source)
    p.shown("threshold", PHASE3, nb, "thr.to_csv", csv=out + "threshold_coverage.csv")
    thr = p.stream(PHASE3, nb, "thr.to_csv")
    p.add("threshold_precision", pd.DataFrame([
        {"criterion": name, "base rate": float(lo) / 100, "strictest cutoff": float(hi) / 100}
        for name, lo, hi in re.findall(r"^\s+(same Ward class|EC within 25%|same site)\s+base rate ([\d.]+)% -> at the strictest cutoff ([\d.]+)%", thr, re.M)]),
        notebook=nb, read="executed notebook output", path=rel(PHASE3 / f"{nb}.ipynb"), cell="thr.to_csv")
    p.shown("top_pairs", PHASE3, nb, "imp_df.to_csv", nth=0)
    p.shown("least_pairs", PHASE3, nb, "imp_df.to_csv", nth=1)
    p.shown("isolated", PHASE3, nb, "imp_df.to_csv", nth=2)
    p.shown("typical", PHASE3, nb, "imp_df.to_csv", nth=3)
    moved = p.stream(PHASE3, nb, "moved.to_csv")
    p.add("raw_vs_baselined", pd.DataFrame([{
        "sites in both spaces": number(r"(\d+) sites in both spaces", moved, int),
        "Spearman ρ, raw vs baselined site matrices": number(r"rho = ([\d.]+)", moved),
        "median pair r, raw": number(r"raw pair similarity: median ([\d.]+)", moved),
        "median pair r, baselined": number(r"baselined: ([\d.]+)", moved),
        "sites whose nearest site changes": number(r"after baselining: (\d+)/", moved, int),
    }]), notebook=nb, read="executed notebook output", path=rel(PHASE3 / f"{nb}.ipynb"), cell="moved.to_csv")
    p.shown("season_r", PHASE3, nb, "season_analogs.to_csv", nth=0, index="season")
    p.shown("season_analogs", PHASE3, nb, "season_analogs.to_csv", nth=1, csv=out + "season_analogs.csv")
    p.shown("season_loading", PHASE3, nb, "season_analogs_loading_matched.csv", nth=0, index="season")
    p.shown("season_analogs_matched", PHASE3, nb, "season_analogs_loading_matched.csv", nth=1, csv=out + "season_analogs_loading_matched.csv")
    overlaps = []
    for search, marker in (("all filters", "season_analogs.to_csv"), ("EC 3–12 µg band", "season_analogs_loading_matched.csv")):
        for a, b, pct in re.findall(r"^\s+(\S+ \(\S+\))\s+vs (\S+ \(\S+\))\s+([\d.]+)%", p.stream(PHASE3, nb, marker), re.M):
            overlaps.append({"search": search, "pair": f"{a} vs {b}", "overlap": float(pct) / 100})
    p.add("season_overlap", pd.DataFrame(overlaps), notebook=nb, read="executed notebook output", path=rel(PHASE3 / f"{nb}.ipynb"),
          cell="season_analogs.to_csv / season_analogs_loading_matched.csv")
    p.optional("site_correlation", PHASE3, nb, out + "site_median_correlation.csv",
               "the full site × site correlation of median corrected spectra (157 IMPROVE sites + 5 SPARTAN), for a dendrogram")

    smoke = classes.loc[classes.pct_smoke_lineage.idxmax()]
    enrichment = number(r"Most smoke-lineage-enriched class: \d+ \(([\d.]+)x the library rate", mags)
    p.text["finding"] = (
        f"Silhouette picks k = {max(scores, key=scores.get)}. Two of the four Ward classes are low-deposit spectra "
        f"(median peak absorbance {min(peak.values()):.4f}–{sorted(peak.values())[1]:.4f}), so the top split is deposit and noise, not chemistry. "
        f"Class {int(smoke.cluster)} holds {enrichment:.1f}× the library share of the deployed model's smoke lineage "
        f"(top sites {smoke.top_sites}); Addis has {int(smoke.addis_n)} of 239 filters in it, Delhi {int(smoke.indh_n)} of 152.")
    return p


# ---------------------------------------------------------------- June 2026 Ethiopia calibrations (07, 09, 10, 12, 15; spartan_ec 08)
def ethiopia_calibrations() -> Page:
    p = Page("ethiopia_calibrations", "Ethiopia calibrations, June 2026")
    t = JUNE25 / "tables"
    rec = pd.read_csv(t / "etad_ec_recommended.csv")
    variants = pd.read_csv(t / "etad_ec_by_variant_conc.csv")
    comps = pd.read_csv(t / "ethiopia_new_ec_by_component.csv")
    lots = pd.read_csv(t / "ethiopia_new_ec_both_lots.csv")
    # every table holds one row per ETAD spectrum in the same order (19 filters were scanned twice)
    for frame in (variants, comps, lots):
        if not frame["MediaId"].equals(rec["MediaId"]):
            raise ValueError("the June 2026 per-spectrum tables are not in the same spectrum order")
    families = (
        ("07 · which samples (k = 20)", variants, {c: c for c in variants.columns if c.startswith("CAL-")}),
        ("09 · component count (906, lot 251)", comps, {c: c.replace("EC_ugm3__EC_all_nofilt_", "all 906, ").replace("_kmin", " (first major min)")
                                                        for c in comps.columns if c.startswith("EC_ugm3__")}),
        ("10 · both filter lots", lots, {**{c: c.replace("EC_bothlots_", "935 both lots, ") + (" (first major min)" if c.endswith("k30") else "")
                                            for c in lots.columns if c.startswith("EC_bothlots_k")},
                                         "EC_lot251only_recommended": "906 lot 251 only, k30"}),
        ("15 · recommended", rec, {"EC_ugm3_old": "OLD rds_EC 906, k20", "EC_ugm3_new": "NEW lot-248+251, flag-filtered, k10"}),
    )
    # wide (one row per spectrum, one column per calibration) keeps the file small; the page melts it
    wide = rec[["MediaId", "ExternalFilterId", "EC_hips"]].rename(columns={"EC_hips": "HIPS Fabs/10"})
    index = []
    for family, frame, cols in families:
        for source_col, label in cols.items():
            key = f"c{len(index)}"
            wide[key] = frame[source_col].to_numpy(float)
            index.append({"column": key, "family": family, "calibration": label})
    printed = p.notebook(JUNE25, "15_downstream_recommended_calibration").stream("summ = pd.DataFrame(")
    for label, col in (("OLD rds_EC 906", "EC_ugm3_old"), ("NEW lot-248+251", "EC_ugm3_new")):
        if not np.isclose(number(rf"{re.escape(label)}\s+([\d.]+)", printed), rec[col].median(), atol=0.005):
            raise ValueError(f"etad_ec_recommended.csv does not reproduce notebook 15's printed median for {label}")
    p.add("calibrations", pd.DataFrame(index), notebook="ftir_ec_calibration_2026_06_25 07/09/10/15", read="tracked csv", path=rel(t))
    p.add("per_filter", wide, notebook="ftir_ec_calibration_2026_06_25 07/09/10/15", read="tracked csv",
          path=", ".join(rel(t / f) for f in ("etad_ec_by_variant_conc.csv", "ethiopia_new_ec_by_component.csv",
                                               "ethiopia_new_ec_both_lots.csv", "etad_ec_recommended.csv")))
    trainset = pd.read_csv(t / "etad_ec_both_lots_median_by_trainset.csv")
    p.add("trainset_medians", long(trainset, ["ncomp"], "training set", "median EC"), notebook="10_ethiopia_ec_both_lots",
          read="tracked csv", path=rel(t / "etad_ec_both_lots_median_by_trainset.csv"))
    p.add("component_summary", pd.read_csv(t / "etad_ec_component_summary.csv").rename(columns={"Unnamed: 0": "calibration"}),
          notebook="09_ethiopia_component_ec", read="tracked csv", path=rel(t / "etad_ec_component_summary.csv"))
    p.add("variant_hips_fits", pd.read_csv(t / "etad_newec_vs_hips_fits.csv"), notebook="08_adama_hips_crossplots",
          read="tracked csv", path=rel(t / "etad_newec_vs_hips_fits.csv"))

    nb12 = ExecutedNotebook(JUNE25 / "12_dedicated_ethiopia_calibration.ipynb")
    steps = []
    for cell, step in (("b0 = fit_hips(Xr, yr, Xeth, 20)", "current"), ("for k in [10, 15, 20]:", "1a · dedicated lot-248+251 data"),
                       ("for N in [1000, 2000, 4000]:", "1b · PCA Addis neighbourhood"), ("variants = {", "3 · Weakley 2nd derivative")):
        for line in nb12.stream(cell).splitlines():
            hit = re.match(r"^(.*?)\s+k=(\d+)[^:]*:\s+slope=([-\d.]+)\s+intercept=([-+\d.]+)\s+r2=([-\d.]+)(?:\s+neg-EC=([\d.]+)%)?", line.strip())
            if hit:
                steps.append({"step": step, "calibration": f"{hit.group(1).strip()} k={hit.group(2)}", "slope": float(hit.group(3)),
                              "intercept": float(hit.group(4)), "R2": float(hit.group(5)),
                              "negative EC %": float(hit.group(6)) if hit.group(6) else None})
    p.add("dedicated_progression", pd.DataFrame(steps), notebook="12_dedicated_ethiopia_calibration", read="executed notebook output (printed)",
          path=rel(JUNE25 / "12_dedicated_ethiopia_calibration.ipynb"))

    nb15 = ExecutedNotebook(JUNE25 / "15_downstream_recommended_calibration.ipynb")
    adama = [line.split() for line in nb15.stream("ad[\"TOR_EC\"] = ").splitlines() if re.match(r"^\s*\d{4} \d{4}-\d\d-\d\d", line)]
    p.add("adama", pd.DataFrame([{"barcode": a[0], "date": a[1], "TOR EC": float(a[2]), "OLD rds_EC 906": float(a[3]),
                                  "NEW lot-248+251": float(a[4])} for a in adama]),
          notebook="15_downstream_recommended_calibration", read="executed notebook output (printed)",
          path=rel(JUNE25 / "15_downstream_recommended_calibration.ipynb"))

    # spartan_ec 08: which calibration filters the iterative 3σ cleaning removes
    nb08 = ExecutedNotebook(JUNE16 / "08_outlier_removal_explainer.ipynb")
    rounds = re.findall(r"\((\d+), (\d+), ([\d.]+), (\d+)\)", nb08.stream("removal rounds"))
    p.add("cleaning_rounds", pd.DataFrame([{"round": int(a), "n before": int(b), "3σ threshold (µg)": float(c), "dropped": int(d)} for a, b, c, d in rounds]),
          notebook="08_outlier_removal_explainer", read="executed notebook output (printed)", path=rel(JUNE16 / "08_outlier_removal_explainer.ipynb"))
    fits = re.findall(r"(BEFORE|AFTER)\s*: n=(\d+) R²=([\d.]+) RMSE=([\d.]+) Err=(\d+)%", nb08.stream("pred_clean=fitpred(keep)"))
    p.add("cleaning_fit", pd.DataFrame([{"stage": s, "n": int(n), "R2": float(r2), "RMSE (µg)": float(rmse), "mean |error| %": float(err)}
                                        for s, n, r2, rmse, err in fits]),
          notebook="08_outlier_removal_explainer", read="executed notebook output (printed)", path=rel(JUNE16 / "08_outlier_removal_explainer.ipynb"))
    removed = pd.read_csv(JUNE16 / "tables/ec_removed_filters.csv")
    p.add("removed", removed.assign(FilterId=removed.FilterId.astype(str)), notebook="08_outlier_removal_explainer",
          read="tracked csv", path=rel(JUNE16 / "tables/ec_removed_filters.csv"))
    cleaned = cleaning_all(removed, p.tables["cleaning_rounds"])
    if cleaned is not None:
        p.add("calibration_all", cleaned, notebook="08_outlier_removal_explainer", read="recomputed from the Drive (FTIR/apps, FTIR/notebook/extracted)",
              path="pls_transfer.load_current_pls_model(ftir_dir, 'EC'), checked against tables/ec_removed_filters.csv")
    else:
        p.missing.append({"table": "calibration_all", "what": "measured vs predicted for all 906 calibration filters (08's before/after scatter); only the 61 removed are tracked",
                          "needs": "<Davis Data>/FTIR/apps/apps_data.npz and FTIR/notebook/extracted/EC_*.csv",
                          "run": "AETHMODULAR_DRIVE_ROOT=<your My Drive> uv run --locked --no-sync python gallery/data/export_research_spectral.py"})

    smoke_tables(p)
    adama_tables(p)

    old, new = (rec.EC_ugm3_old.median(), rec.EC_ugm3_new.median())
    p.text["finding"] = (
        f"Moving from the 906-filter lot-251 set to the full lot-248+251 mirror lowers the median ETAD EC from {old:.2f} to {new:.2f} µg/m³ "
        f"and takes the HIPS crossplot from 2.27x − 5.25 toward ~1.2x − 2.3 (notebooks 12 and 15). Which samples train the model swings the median "
        f"about 2× (07); the component count alone swings it up to 5× (09); adding 29 lot-248 filters moves it 3 % (10).")
    return p


CLEANING_K = 20  # spartan_ec 08: "component count fixed at K=20"


def cleaning_all(removed: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame | None:
    """spartan_ec 08's cleaning on all 906 calibration filters, where the Drive's FTIR folder resolves.

    The 906-filter EC set is rebuilt by pls_transfer.load_current_pls_model (the same set as 08's
    rds_EC export); 08's rule is then run as it is written (fit at K = 20, drop |residual| > 3 × the
    kept residuals' spread, refit, at most nine rounds). Nothing is published unless the rounds equal
    the ones 08 printed and the removed filters equal its tracked list in measured, predicted and site.
    08 labels filters by MatchedFilterId and pls_transfer by another FilterId, so rows are keyed by
    AnalysisId and the check is on values.
    """
    sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
    from pls_transfer import FTIRTransferPaths, load_current_pls_model
    from sklearn.cross_decomposition import PLSRegression

    ftir_dir = FTIRTransferPaths.defaults().ftir_dir
    if not (ftir_dir / "apps/apps_data.npz").exists():
        return None
    model = load_current_pls_model(ftir_dir, "EC")
    X, y = model.X, model.y

    def fitted(mask: np.ndarray) -> np.ndarray:
        return PLSRegression(n_components=CLEANING_K, scale=False).fit(X[mask], y[mask]).predict(X[mask]).ravel()

    keep, round_of, seen = np.ones(len(y), bool), np.zeros(len(y), int), []
    for number in range(1, 10):
        residual = y[keep] - fitted(keep)
        threshold = 3 * residual.std()
        drop = np.abs(residual) > threshold
        seen.append((number, int(keep.sum()), round(float(threshold), 1), int(drop.sum())))
        if not drop.any():
            break
        index = np.where(keep)[0][drop]
        round_of[index], keep[index] = number, False
    if seen != [tuple(r) for r in rounds[["round", "n before", "3σ threshold (µg)", "dropped"]].itertuples(index=False)]:
        raise ValueError("the 906-filter set does not reproduce spartan_ec 08's printed cleaning rounds")
    full = fitted(np.ones(len(y), bool))
    out = pd.DataFrame({"AnalysisId": model.analysis_ids, "Site": model.sites, "measured": y, "predicted": full,
                        "round": round_of, "status": np.where(keep, "kept", "removed by 3σ cleaning")})
    gone = out[~keep].assign(m=lambda d: d.measured.round(1), q=lambda d: d.predicted.round(1))
    check = removed.merge(gone, left_on=["measured", "predicted", "Site", "round"], right_on=["m", "q", "Site", "round"])
    if len(check) != len(removed) or len(gone) != len(removed):
        raise ValueError("the recomputed removals do not match spartan_ec 08's tracked ec_removed_filters.csv")
    return out


def _polyfit(x: pd.Series, y: pd.Series) -> tuple[int, float, float, float]:
    """The July07 notebooks' reg_stats: np.polyfit of y on x over finite pairs, R² = 1 − SS_res / SS_tot."""
    ok = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    xs, ys = x.to_numpy(float)[ok], y.to_numpy(float)[ok]
    slope, intercept = np.polyfit(xs, ys, 1)
    r2 = 1 - np.sum((ys - (slope * xs + intercept)) ** 2) / np.sum((ys - ys.mean()) ** 2)
    return int(ok.sum()), float(slope), float(intercept), float(r2)


def smoke_tables(p: Page) -> None:
    """SPARTAN's lot-251 biomass ("smoke") tool calibration on the Addis spectra, against HIPS (July07, spartan_ec 02).

    Naming: July07's "regular CAL-0" is the 906-filter IMPROVE set refitted at k = 20 (the gallery's
    "Biomass-smoke (906)" preset trains on the same set); its "smoke" / "biomass" is a different model, the
    coefficients SPARTAN's calibration tool exported for lot 251 (tool_EC_coeffs_lot251_biomass.csv). Both
    tables are per-filter and are joined to the gallery's Addis rows in the browser (by date, and by
    ExternalFilterId without its scan suffix), so the HIPS side follows the reader's MAC and exclusions.
    """
    july = ROOT / "research/July07"
    addis = pd.read_csv(july / "tables/addis_regular_before_cal0_vs_smoke.csv")
    etad = pd.read_csv(july / "tables/etad_regular_vs_smoke_ec.csv")
    verdict_addis = pd.read_csv(july / "tables/regular_before_cal0_vs_smoke_verdict.csv")
    verdict_etad = pd.read_csv(july / "tables/regular_vs_smoke_verdict.csv")
    # each saved verdict must come back from its per-filter table (before-CAL-0 regresses HIPS on EC; the other EC on HIPS)
    both = addis.dropna(subset=["ec_regular", "ec_smoke", "hips_fabs"])
    for label, col in (("Regular reported EC (pre-CAL-0)", "ec_regular"), ("Smoke-only (biomass)", "ec_smoke"), ("CAL-0 refit (reference)", "ec_cal0")):
        n, slope, intercept, r2 = _polyfit(both[col], both["hips_fabs"])
        saved = verdict_addis.set_index("calibration").loc[label]
        if (n, round(slope, 2), round(intercept, 2), round(r2, 3)) != (saved.n, saved.slope, saved.intercept, saved.R2):
            raise ValueError(f"addis_regular_before_cal0_vs_smoke.csv does not reproduce July07's verdict for {label}")
    for label, col in (("Regular (all-data CAL-0)", "EC_regular"), ("Smoke-only (biomass)", "EC_smoke"), ("Deployed general (reported)", "EC_reported")):
        n, slope, intercept, r2 = _polyfit(etad["EC_hips"], etad[col])
        saved = verdict_etad.set_index("calibration").loc[label]
        if (n, round(slope, 2), round(intercept, 2), round(r2, 3)) != (saved.n, saved.slope, saved.intercept, saved.R2):
            raise ValueError(f"etad_regular_vs_smoke_ec.csv does not reproduce July07's verdict for {label}")
    p.add("smoke_addis", addis, notebook="July07/regular_before_cal0_vs_smoke_hips", read="tracked csv",
          path=rel(july / "tables/addis_regular_before_cal0_vs_smoke.csv"))
    p.add("smoke_etad", etad, notebook="July07/regular_vs_smoke_ec_vs_hips", read="tracked csv",
          path=rel(july / "tables/etad_regular_vs_smoke_ec.csv"))
    p.add("smoke_verdict_addis", verdict_addis, notebook="July07/regular_before_cal0_vs_smoke_hips", read="tracked csv",
          path=rel(july / "tables/regular_before_cal0_vs_smoke_verdict.csv"))
    p.add("smoke_verdict_etad", verdict_etad, notebook="July07/regular_vs_smoke_ec_vs_hips", read="tracked csv",
          path=rel(july / "tables/regular_vs_smoke_verdict.csv"))

    # spartan_ec 02 ran with PLACEHOLDER = True: its EC_biomass is the general EC × 1.18 on the higher-OC half
    # (is_biomass), not a calibration. Only the real columns go out: which Addis filters are lot 251, with their
    # general EC and HIPS; the page puts July07's lot-251 tool EC where the placeholder stood.
    lot = pd.read_csv(JUNE16 / "tables/biomass_vs_general_ec.csv")
    placeholder = np.where(lot.is_biomass, lot.EC_general * 1.18, lot.EC_general)
    if not np.allclose(lot.EC_biomass, placeholder):
        raise ValueError("biomass_vs_general_ec.csv no longer holds 02's placeholder: export its EC_biomass")
    p.add("lot251_set", lot[["FilterId", "LotId", "SampleDate", "EC_general", "OC_ftir", "HIPS_Fabs"]].assign(LotId=lot.LotId.astype(int)),
          notebook="02_biomass_calibration_comparison", read="tracked csv", note="placeholder biomass columns left out",
          path=rel(JUNE16 / "tables/biomass_vs_general_ec.csv"))


def adama_tables(p: Page) -> None:
    """The five Adama PTFE filters against the co-located quartz TOR, under every calibration tried on them.

    The quartz and PTFE filters pair by sample date (June 02's table). The June 2026 notebooks (02, 08) work
    in µg/m³; the spartan_ec tables (03's TOR fractions, and the tool / local / faithful / optimized /
    stabilized biomass EC) are µg on each filter, and the page converts with the PTFE volume June 08 saved.
    """
    t = JUNE25 / "tables"
    tor = pd.read_csv(t / "adama_tor_vs_ftir.csv")
    variants = pd.read_csv(t / "adama_new_ec_by_variant.csv")
    han = pd.read_csv(JUNE16 / "tables/adama_char_soot_classification.csv")
    spartan = {name: pd.read_csv(JUNE16 / f"tables/adama_ec_{name}_comparison.csv") for name in ("calibration", "faithful", "optimized", "stabilized")}
    # the pairings and units the page relies on: µg ÷ PTFE volume is June's µg/m³, and every table's TOR is the date-paired quartz
    volume = variants.set_index("FilterId")["volume_m3"]
    quartz = tor.set_index("ptfe_FilterId")["quartz_FilterId"]
    for name, frame in spartan.items():
        f = frame.set_index("FilterId")
        if not np.allclose(f.EC_general / volume.reindex(f.index), variants.set_index("FilterId").FTIR_EC_general_orig.reindex(f.index), atol=1e-3):
            raise ValueError(f"adama_ec_{name}_comparison.csv's EC_general is not June 08's general EC × the PTFE volume")
        if not np.allclose(f.ECTR, han.set_index("sample").EC_TOR.reindex(quartz.reindex(f.index)).to_numpy(), atol=1e-3):
            raise ValueError(f"adama_ec_{name}_comparison.csv's ECTR is not the date-paired quartz filter's TOR EC")
    p.add("adama_tor", tor.drop(columns=[c for c in tor.columns if tor[c].isna().all()]), notebook="02_adama_tor_vs_ftir",
          read="tracked csv", path=rel(t / "adama_tor_vs_ftir.csv"))
    p.add("adama_char_soot", pd.read_csv(t / "adama_char_soot_opr_opt.csv"), notebook="03_adama_char_soot",
          read="tracked csv", path=rel(t / "adama_char_soot_opr_opt.csv"))
    p.add("adama_variants", variants.assign(barcode=variants.barcode.astype(str)), notebook="08_adama_hips_crossplots",
          read="tracked csv", path=rel(t / "adama_new_ec_by_variant.csv"))
    p.add("adama_han", han, notebook="03_adama_han_char_soot", read="tracked csv",
          path=rel(JUNE16 / "tables/adama_char_soot_classification.csv"))
    writers = {"calibration": "_build_adama_comparison.py", "faithful": "_build_faithful_replica_adama.py",
               "optimized": "05_calibration_optimization", "stabilized": "07_calibration_optimization_story"}
    for name, frame in spartan.items():
        p.add(f"adama_{name}", frame, notebook=writers[name], read="tracked csv", path=rel(JUNE16 / f"tables/adama_ec_{name}_comparison.csv"))


# ---------------------------------------------------------------- committee backup figures (ftir_53)
def committee() -> Page:
    p = Page("committee", "Does the diagnosis generalise")
    nb = "ftir_53_committee_backup_figures"
    out = "output/tables/ftir53/"
    p.shown("five_site", PHASE3, nb, "five_site_table.to_csv", csv=out + "five_site_york_line_quad.csv")
    p.shown("bishoftu", PHASE3, nb, "bish.to_csv", csv=out + "bishoftu_replication_fits.csv")
    p.shown("protocol_swing", PHASE3, nb, "sw.to_csv", csv=out + "protocol_intercept_swing.csv")
    p.shown("kiremt_dry", PHASE3, nb, "kiremt_dry_all_variants.csv", csv=out + "kiremt_dry_all_variants.csv")
    p.shown("mass_term", PHASE3, nb, "mass_term_by_site.csv", csv=out + "mass_term_by_site.csv")
    return p


PAGES = (band_1617, mac_choice, provisional_series, applicability, network_map, ethiopia_calibrations, committee)


def main() -> None:
    for build in PAGES:
        build().write()


if __name__ == "__main__":
    main()
