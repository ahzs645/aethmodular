"""What counts as EC: target definitions and Adama. ftir_42, ftir_48, ftir_41, ftir_44 (ftir_40 the design companion).

The same spectra, cohort, protocol and folds with only the prediction target changed (EC_TOR, EC_TOT, OC,
TC, the pyrolysis split Δ = OP_TOT − OP_TOR and the routes that compose EC_TOT from it), and the five Adama
quartz/PTFE pairs on which thermal OC/EC under both optical conventions, FTIR carbon and HIPS absorption sit
on one table. Held-out scores are against IMPROVE TOR/TOT (µg/filter): MAC-free. The Addis lines are against
HIPS Fabs/10, which the page re-expresses at the reader's MAC (slope × MAC/10; RMSE through the fixed
cohort's HIPS moments).

The notebooks never write the held-out per-filter predictions behind their crossplots (ftir_42 [12],
ftir_48 [9], [14]); ftir_42's per-filter Addis predictions and ftir_41's thermal fractions are CSV only.
"""
from __future__ import annotations

import re

import pandas as pd

from ._common import PHASE3, number, page, rel

KEY = "targets"
NB40 = "ftir_40_quartz_tor_campaign_design"
NB41 = "ftir_41_adama_three_method_reconciliation"
NB42 = "ftir_42_target_definition_experiment"
NB44 = "ftir_44_adama_through_locked_calibrations"
NB48 = "ftir_48_pyrolysis_split_target"
COHORTS = ("Lowest-OC/EC (800) + AIRSpec", "Full pool (no selection) + AIRSpec")


def never_written(p, table: str, notebook: str, what: str, needs: str, cell: str) -> None:
    """A per-filter table the notebook computes but never saves: a re-run alone does not make it."""
    number_ = re.match(r"ftir_(\d+)", notebook).group(1)
    p.missing.append({"table": table, "what": f"{what} (never written: add a to_csv to cell {cell} first)",
                      "needs": rel(PHASE3) + "/" + needs, "run": f"cd {rel(PHASE3)} && uv run python scripts/build_notebooks.py {number_}", "after": None})


def full_label(label: str) -> str:
    """A display-truncated set label ('Full pool (no selection) + AIRSpec — held-out ...') made whole again."""
    if label.endswith("..."):
        for cohort in COHORTS:
            if label.startswith(cohort):
                return f"{cohort} — held-out rows"
    return label


def adama_wide(p) -> None:
    """ftir_44 [7]: the CSV is one row per date; the notebook displayed it transposed."""
    csv = PHASE3 / "output/tables/ftir44/adama_three_method_with_phase3_models.csv"
    if csv.exists():
        p.add("adama_models", pd.read_csv(csv), notebook=NB44, read="csv", path=rel(csv))
        return
    marker = "table.to_csv(OUT / 'adama_three_method_with_phase3_models.csv')"
    shown = p.notebook(PHASE3, NB44).table(marker)
    wide = shown.set_index(shown.columns[0]).T.rename_axis("date").reset_index()
    for col in wide.columns:
        if col not in ("date", "FilterId_ptfe", "FilterId_quartz", "flag"):
            wide[col] = pd.to_numeric(wide[col])
    wide["flag"] = wide["flag"].map(lambda v: None if v in (None, "None") else v)
    p.add("adama_models", wide, notebook=NB44, read="executed notebook output (transposed back)", path=rel(PHASE3 / f"{NB44}.ipynb"), cell=marker)


def build():
    p = page(KEY, "What counts as EC: target definitions and Adama")
    o41, o42, o44, o48 = (f"output/tables/ftir{n}/" for n in (41, 42, 44, 48))

    # ftir_42: four targets, two cohorts; TC direct vs summed; Addis by EC convention
    targets = p.shown("target_definition_summary", PHASE3, NB42, "summary.to_csv(OUT / 'target_definition_summary.csv'", csv=o42 + "target_definition_summary.csv")
    tc = p.shown("tc_direct_vs_summed", PHASE3, NB42, "sum_table.to_csv", csv=o42 + "tc_direct_vs_summed.csv")
    conv = p.shown("addis_by_ec_convention", PHASE3, NB42, "addis_table.to_csv(OUT / 'addis_by_ec_convention.csv'", csv=o42 + "addis_by_ec_convention.csv")
    pool = p.stream(PHASE3, NB42, "tor = pd.read_csv(")
    p.add("pool_ratio", pd.DataFrame([{"set": "usable mirror (site,date)", "median": number(r"EC_TOT/EC_TOR: median ([\d.]+)", pool),
                                       "q25": number(r"IQR ([\d.]+)-", pool), "q75": number(r"IQR [\d.]+-([\d.]+)", pool)}]),
          notebook=NB42, read="executed notebook output (printed)", path=rel(PHASE3 / f"{NB42}.ipynb"), cell="tor = pd.read_csv(")
    p.optional("addis_predictions_all_targets", PHASE3, NB42, o42 + "addis_predictions_all_targets.csv",
               "ftir_42's per-filter Addis predictions by target (its TOT/TOR prediction-ratio histogram [14]; joined to calibration_runs ids it would follow the MAC)")
    never_written(p, "heldout_by_target", NB42, "ftir_42's held-out IMPROVE predictions per filter for TC direct vs summed and the EC/TC share [12]",
                  o42 + "heldout_predictions.csv", "[11]")

    # ftir_48: the split target Δ, routes to EC_TOT, the error decomposition, Adama Δ, Addis by route
    delta = p.shown("delta_ratio_distribution", PHASE3, NB48, "delta_dist.to_csv", nth=0, csv=o48 + "delta_ratio_distribution.csv")
    p.tables["delta_ratio_distribution"] = delta.assign(set=delta["set"].map(full_label))
    p.shown("delta_amounts", PHASE3, NB48, "delta_dist.to_csv", nth=1)
    split = p.shown("split_target_summary", PHASE3, NB48, "summary.to_csv(OUT / 'split_target_summary.csv'", csv=o48 + "split_target_summary.csv")
    p.shown("rescale_factor", PHASE3, NB48, "routes_table.to_csv", nth=0, csv=o48 + "rescale_factor.csv")
    routes = p.shown("ec_tot_routes", PHASE3, NB48, "routes_table.to_csv", nth=1, csv=o48 + "ec_tot_routes.csv")
    per_site_csv = PHASE3 / o48 / "ec_tot_routes_per_site.csv"
    if per_site_csv.exists():
        p.add("ec_tot_routes_per_site", pd.read_csv(per_site_csv), notebook=NB48, read="csv", path=rel(per_site_csv))
    else:
        wide, source = p.read(PHASE3, NB48, "ps_wide = ps.pivot(")
        route_cols = [c for c in wide.columns if c.startswith("(")]
        long_ = wide.melt(id_vars=["site", "n", "EC_TOT_mean_ug"], value_vars=route_cols, var_name="route", value_name="RMSE_ug")
        p.add("ec_tot_routes_per_site", long_.assign(cohort=COHORTS[0]), **source)
    p.shown("composed_error_decomposition", PHASE3, NB48, "decomp.to_csv", csv=o48 + "composed_error_decomposition.csv")
    p.shown("adama_delta_check", PHASE3, NB48, "adama.to_csv(OUT / 'adama_delta_check.csv'", csv=o48 + "adama_delta_check.csv")
    addis = p.shown("addis_ec_tot_routes", PHASE3, NB48, "addis_table.to_csv(OUT / 'addis_ec_tot_routes.csv'", csv=o48 + "addis_ec_tot_routes.csv")
    never_written(p, "heldout_split_targets", NB48, "ftir_48's held-out IMPROVE predictions per filter for the six split targets [9] and the four EC_TOT routes [14]",
                  o48 + "heldout_predictions.csv", "[8] and [11]")
    never_written(p, "addis_routes_per_filter", NB48, "ftir_48's per-filter Addis EC_TOT by route [22] (the slopes are on this page; the points are not saved)",
                  o48 + "addis_predictions_by_route.csv", "[21]")

    # ftir_42's Addis lines are ftir_48's first two rows per cohort (ftir_48 asserts it); check the copies agree
    for cohort in COHORTS:
        for target, route in (("EC_TOR", "EC_TOR model (ftir_42)"), ("EC_TOT", "(a) direct EC_TOT model (ftir_42)")):
            a = conv[(conv.target == target) & conv.cohort.str.startswith(cohort)].iloc[0]
            b = addis[(addis.cohort == cohort) & (addis.route == route)].iloc[0]
            if not all(abs(a[c] - b[c]) < 0.0015 for c in ("slope", "intercept", "R2", "RMSE")):
                raise ValueError(f"ftir_42 and ftir_48 disagree on the Addis {target} line for {cohort}")

    # ftir_41: the five Adama pairs, three methods
    p.shown("adama_pairs", PHASE3, NB41, "ledger.to_csv", csv=o41 + "pairing_ledger.csv")
    ref = p.shown("adama_reference", PHASE3, NB41, "ref_table.to_csv", csv=o41 + "reference_definition_table.csv")
    p.optional("adama_thermal_fractions", PHASE3, NB41, o41 + "pyrolysis_summary.csv",
               "ftir_41's thermal-fraction shares (OC1–OC4, EC1–EC3, OP) per Adama date [8], from the Drive's DAVIS/Adama TOR Carbon_concs_Batch54.csv")
    budget = p.shown("adama_carbon_budget", PHASE3, NB41, "budget.to_csv", csv=o41 + "carbon_budget_table.csv")
    implied = p.shown("adama_implied_mac", PHASE3, NB41, "mac_table.to_csv", csv=o41 + "implied_mac_table.csv")

    # ftir_44: the Adama filters through the locked phase-3 calibrations
    adama_wide(p)
    ratios = p.shown("adama_model_ratios", PHASE3, NB44, "ratio_table.to_csv", csv=o44 + "model_over_reference_ratios.csv")
    p.shown("adama_oc_tc", PHASE3, NB44, "carbon.to_csv", csv=o44 + "adama_oc_tc_ratios.csv")

    # ftir_40: the quartz campaign design (no model fitted; the companion that says how many pairs)
    p.shown("campaign_seasons", PHASE3, NB40, "campaign_design_analysis()", nth=0)
    p.shown("campaign_allocation", PHASE3, NB40, "campaign_design_analysis()", nth=1)

    t = targets.set_index(["cohort", "target"])
    lo = COHORTS[0]
    s = split.set_index(["cohort", "target"])
    r = routes.set_index(["cohort", "route"])
    rt = ratios.set_index(["model", "reference"])
    unflagged = implied[implied.flag.isna()]
    p.text["finding"] = (
        f"Only the target changes. On the lowest-OC/EC cohort FTIR predicts total carbon best (held-out-site R² {t.loc[(lo, 'TC'), 'heldout_R2']:.3f}) "
        f"and EC under the transmittance convention worst (EC_TOT {t.loc[(lo, 'EC_TOT'), 'heldout_R2']:.3f} vs EC_TOR {t.loc[(lo, 'EC_TOR'), 'heldout_R2']:.3f}); "
        f"without composition selection EC_TOT collapses (R² {t.loc[(COHORTS[1], 'EC_TOT'), 'heldout_R2']:.3f}). The OC/EC split itself is unlearnable "
        f"out of site (EC/TC share R² {tc.set_index('cohort').loc[lo, 'EC_share_R2']:.3f}) (ftir_42). The pyrolysis correction Δ is predictable on the cohort "
        f"(R² {s.loc[(lo, 'DELTA'), 'heldout_R2']:.3f}), and composing EC_TOR − Δ does no worse than a direct EC_TOT model "
        f"(R² {r.loc[(lo, '(b) composed EC_TOR_hat − DELTA_hat'), 'R2']:.3f} vs {r.loc[(lo, '(a) direct EC_TOT model'), 'R2']:.3f}): "
        f"EC_TOT is hard because of the carbon, not the correction (ftir_48). At Adama, EC_TOT is "
        f"{-ref.EC_TT_vs_TR_pct.median():.0f} % below EC_TOR, the FTIR carbon sum is {budget.tc_ratio.median():.2f}× thermal TC, the implied MAC is "
        f"{unflagged.implied_MAC_vs_ECTOR.median():.1f} m²/g against EC_TOR (unflagged median) (ftir_41), and the locked 800 + AIRSpec calibration reads "
        f"{rt.loc[('Locked 800 + AIRSpec', 'TOR'), 'median_ratio_unflagged']:.2f}× EC_TOR where the deployed one reads "
        f"{rt.loc[('Deployed FTIR EC (lot 241a)', 'TOR'), 'median_ratio_unflagged']:.2f}× (ftir_44). Five pairs, one month, two flagged.")
    return p
