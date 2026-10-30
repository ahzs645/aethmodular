"""June 2026 calibration variants and chasing the origin.

research/ftir_ec_calibration_2026_06_25 05, 06, 09_component_selection_fixed, 10_calibration_variants_effect
(CAL-0…CAL-7 on the 906 IMPROVE filters), 11 and 13 (the ETAD-vs-HIPS line against the (1, 0) target),
14 (choosing the calibration data), and research/spartan_ec_2026_06_16 05, 07, 08 (calibration stability).

The June notebooks were committed without their figures; their tables are tracked under tables/ and
their printed results are in the executed notebooks. Everything here is one of those two. The
ETAD calibrations themselves (per-spectrum EC by variant, components, lots) are on Research →
Ethiopia calibrations (export_research_spectral.ethiopia_calibrations) and are not repeated.

HIPS fits (11, 13, 14) are OLS of predicted ETAD EC on HIPS Fabs/10, exported at MAC 10; the page
re-expresses the slopes at the reader's MAC (intercepts and R² do not move).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ._common import JUNE, SPARTAN_EC, number, page, rel

KEY = "june"

NB05, NB06 = "05_calibration_variants", "06_calibration_variants_components"
NB09, NB10 = "09_component_selection_fixed", "10_calibration_variants_effect"
NB11, NB13, NB14 = "11_hips_intercept_calibrations", "13_chase_origin_and_biomass", "14_choosing_calibration_data"
SP05, SP07, SP08 = "05_calibration_optimization", "07_calibration_optimization_story", "08_outlier_removal_explainer"

# the variants as notebooks 06 and 10 define them (their markdown naming tables)
VARIANTS = {
    "CAL-0": ("all 906, no removal", "candidate"),
    "CAL-1": ("iterative 3σ trim on the absolute residual (3 rounds at k = 21)", "candidate"),
    "CAL-2": ("only the filters CAL-1 removed (inverse)", "diagnostic"),
    "CAL-3": ("only filters below the 1:1 line, predicted < measured (inverse)", "diagnostic"),
    "CAL-4": ("only measured EC ≥ 70 µg", "diagnostic"),
    "CAL-5": ("EC inside a placeholder Ethiopia band", "candidate"),
    "CAL-6": ("iterative 3σ trim on the relative residual", "candidate"),
    "CAL-7": ("drop the 3 filters with EC > 250 µg", "candidate"),
}


def cal(name: str) -> str:
    return re.match(r"(CAL-\d)", name).group(1)


def printed(p, folder, nb: str, marker: str) -> tuple[str, dict]:
    """A cell's printed output and where it came from."""
    return p.stream(folder, nb, marker), {"notebook": nb, "read": "executed notebook output (printed)", "path": rel(folder / f"{nb}.ipynb"), "cell": marker}


def build():
    p = page(KEY, "June 2026 calibration variants and chasing the origin")
    t = JUNE / "tables"

    # ------------------------------------------------------------ CAL-0's RMSECV curve and how k is picked
    p.add("rmsecv_curve", pd.read_csv(t / "ec_rmsecv_curve.csv").rename(columns={"n_components": "k"}),
          notebook=NB05, read="tracked csv", path=rel(t / "ec_rmsecv_curve.csv"))
    s05, src05 = printed(p, JUNE, NB05, "k_first = first_major_min(rv, rel_tol=0.02)")
    s06, _ = printed(p, JUNE, NB06, "def k_within(ks, rv, frac=0.05)")
    s09, _ = printed(p, JUNE, NB09, "pct_raw = 100 * rc_raw / y.mean()")
    comp06 = pd.read_csv(t / "component_selection_by_variant.csv").assign(cal=lambda d: d.name.map(cal)).set_index("cal")
    five, ten = "05/06 · 5-fold shuffled CV", "09 · 10-fold interleaved CV"
    picks = [
        {"cv": five, "rule": "first major minimum, 2 % tolerance (05)", "spectra": "raw",
         "k": number(r"first major RMSECV min -> (\d+)", s05, int), "RMSECV": number(r"first major RMSECV min -> \d+ components \(RMSECV ([\d.]+)\)", s05)},
        {"cv": five, "rule": "global minimum (05)", "spectra": "raw",
         "k": number(r"global RMSECV min\s+-> (\d+)", s05, int), "RMSECV": number(r"global RMSECV min\s+-> \d+ components \(RMSECV ([\d.]+)\)", s05)},
        {"cv": five, "rule": "Wold's R (06)", "spectra": "raw", "k": int(comp06.loc["CAL-0", "k_wold"]), "RMSECV": None},
        {"cv": five, "rule": "first k within 5 % of the minimum (06)", "spectra": "raw",
         "k": number(r"raw spectra:\s+reaches within 5% of its best RMSECV at k=(\d+)", s06, int), "RMSECV": number(r"raw spectra:.*?\(best ([\d.]+)", s06)},
        {"cv": five, "rule": "first k within 5 % of the minimum (06)", "spectra": "2nd derivative",
         "k": number(r"2nd-derivative spectra: reaches within 5% of its best RMSECV at k=(\d+)", s06, int), "RMSECV": number(r"2nd-derivative spectra:.*?\(best ([\d.]+)", s06)},
        {"cv": ten, "rule": "first k within 5 % of the minimum (09 k*)", "spectra": "raw",
         "k": number(r"raw k\*=(\d+)", s09, int), "RMSECV": number(r"raw k\*=\d+ \(min ([\d.]+)", s09)},
        {"cv": ten, "rule": "first k within 5 % of the minimum (09 k*)", "spectra": "2nd derivative",
         "k": number(r"2nd-deriv k\*=(\d+)", s09, int), "RMSECV": number(r"2nd-deriv k\*=\d+ \(min ([\d.]+)", s09)},
    ]
    p.add("rule_picks", pd.DataFrame(picks), notebook=f"{NB05}, {NB06}, {NB09}", read="executed notebook output (printed)",
          path=", ".join(rel(JUNE / f"{nb}.ipynb") for nb in (NB05, NB06, NB09)))
    if not np.isclose(p.tables["rule_picks"].RMSECV.iloc[0], p.tables["rmsecv_curve"].RMSECV.min(), atol=0.005):
        raise ValueError("05's printed RMSECV minimum does not match tables/ec_rmsecv_curve.csv")

    # ------------------------------------------------------------ the variants, one row each
    res06 = pd.read_csv(t / "calibration_variants_results.csv").assign(cal=lambda d: d.name.map(cal)).set_index("cal")
    cor09 = pd.read_csv(t / "component_selection_corrected.csv").assign(cal=lambda d: d.name.map(cal)).set_index("cal")
    eff10 = pd.read_csv(t / "calibration_variants_effect.csv").assign(cal=lambda d: d.calibration.map(cal)).set_index("cal")
    names = {**res06.name.to_dict(), **eff10.calibration.to_dict()}
    rows = []
    for c, (rule, role) in VARIANTS.items():
        g = lambda frame, col: frame.loc[c, col] if c in frame.index else None  # noqa: E731
        rows.append({
            "variant": c, "name": names[c].replace(f"{c} ", ""), "rule": rule, "role": role,
            "n": g(eff10, "n_train") if c in eff10.index else g(cor09, "n"), "mean EC (µg)": g(cor09, "mean_EC"),
            "k fit (06)": g(res06, "k_fit"), "in-sample R² (06)": g(res06, "R2"), "in-sample RMSE (06, µg)": g(res06, "RMSE"),
            "RMSECV min (06, µg)": g(res06, "RMSECV_min"), "k first min (06)": g(comp06, "k_firstmin"), "k Wold (06)": g(comp06, "k_wold"),
            "k* (09/10)": g(eff10, "k*") if c in eff10.index else g(cor09, "k*_corrected"),
            "RMSECV min (09, µg)": g(cor09, "RMSECV_min_ug"),
            "%RMSECV (09/10)": g(eff10, "pctRMSECV") if c in eff10.index else g(cor09, "pctRMSECV_min"),
            "median predicted EC on the 906 (10, µg)": g(eff10, "median_pred_EC"), "ratio to CAL-0 (10)": g(eff10, "ratio_to_CAL0"),
            "% negative on the 906 (10)": g(eff10, "pct_negative"),
        })
    variants = pd.DataFrame(rows)
    # 09 and 10 report the same k* and %RMSECV for the variants they share
    both = cor09.index.intersection(eff10.index)
    if not (cor09.loc[both, "k*_corrected"].to_numpy() == eff10.loc[both, "k*"].to_numpy()).all():
        raise ValueError("notebooks 09 and 10 disagree on k* for the shared variants")
    p.add("variants", variants, notebook=f"{NB06}, {NB09}, {NB10}", read="tracked csv",
          path=", ".join(rel(t / f) for f in ("calibration_variants_results.csv", "component_selection_by_variant.csv",
                                               "component_selection_corrected.csv", "calibration_variants_effect.csv")))
    p.add("plan", pd.read_csv(t / "calibration_variants_plan.csv"), notebook=NB05, read="tracked csv", path=rel(t / "calibration_variants_plan.csv"))
    p.add("retention", pd.read_csv(t / "ec_threshold_retention.csv"), notebook=NB05, read="tracked csv", path=rel(t / "ec_threshold_retention.csv"))
    p.add("built", pd.read_csv(t / "ec_variants_built.csv"), notebook=NB05, read="tracked csv", path=rel(t / "ec_variants_built.csv"))

    # ------------------------------------------------------------ chasing the origin (HIPS Fabs/10, MAC 10)
    sweep = pd.read_csv(t / "hips_intercept_sweep.csv")
    sweep["k"] = sweep.setting.str.extract(r"k=(\d+)", expand=False).astype(int)
    sweep["training set"] = sweep.calibration.str.replace("<=", "≤", regex=False)
    p.add("hips_sweep", sweep.drop(columns="calibration"), notebook=NB11, read="tracked csv", path=rel(t / "hips_intercept_sweep.csv"))
    s11, src11 = printed(p, JUNE, NB11, "def offset_correct(M)")
    offsets = []
    for label, sr, ir, rr, sb, ib, rb in re.findall(r"^(.+?)\s+raw: slope=([-\d.]+) intercept=([-+\d.]+) r2=([-\d.]+)\s+\| baseline-corrected: slope=([-\d.]+) intercept=([-+\d.]+) r2=([-\d.]+)", s11, re.M):
        for spectra, s, i, r in (("raw", sr, ir, rr), ("offset-corrected (3900–4000 cm⁻¹)", sb, ib, rb)):
            offsets.append({"calibration": label.strip().replace("<=", "≤"), "spectra": spectra, "slope": float(s), "intercept": float(i), "r2": float(r)})
    if len(offsets) != 4:
        raise ValueError("notebook 11's baseline-offset printout changed")
    p.add("offset_check", pd.DataFrame(offsets), **src11)

    front = pd.read_csv(t / "hips_origin_frontier.csv").rename(columns={"dist_to_(1,0)": "distance at MAC 10"})
    p.add("frontier", front, notebook=NB13, read="tracked csv", path=rel(t / "hips_origin_frontier.csv"))
    # the other lot-248+251 (k = 10) fits the notebooks printed: biomass-like subsets (13), provenance filter (14)
    s13, src13 = printed(p, JUNE, NB13, "dist = np.sqrt(((Fi - eth_prof)**2).sum(1))")
    s14a, src14a = printed(p, JUNE, NB14, "training filters ever physically-flagged")
    subsets = []
    for line in s13.splitlines():
        hit = re.match(r"^(full lot-248\+251|biomass-like N=(\d+)).*?slope=([-\d.]+) int=([-+\d.]+) r2=([-\d.]+)", line.strip())
        if hit:
            subsets.append({"family": "13 · organic-profile (biomass-like) subset", "selection": "full set" if hit.group(2) is None else f"biomass-like {hit.group(2)}",
                            "n": None if hit.group(2) is None else int(hit.group(2)),
                            "slope": float(hit.group(3)), "intercept": float(hit.group(4)), "r2": float(hit.group(5))})
    for label, n, s, i, r in re.findall(r"^\s+(full set|drop physical-flagged)\s+n=\s*(\d+): slope=([-\d.]+) int=([-+\d.]+) r2=([-\d.]+)", s14a, re.M):
        subsets.append({"family": "14 · provenance filter", "selection": label, "n": int(n), "slope": float(s), "intercept": float(i), "r2": float(r)})
    if len(subsets) != 6:
        raise ValueError("notebook 13/14's printed subset fits changed")
    p.add("subsets", pd.DataFrame(subsets), notebook=f"{NB13}, {NB14}", read="executed notebook output (printed)",
          path=f"{rel(JUNE / f'{NB13}.ipynb')}, {rel(JUNE / f'{NB14}.ipynb')}")
    p.add("selection", pd.read_csv(t / "calibration_data_selection_summary.csv"), notebook=NB14, read="tracked csv",
          path=rel(t / "calibration_data_selection_summary.csv"))

    s14b, _ = printed(p, JUNE, NB14, "ETAD beyond training 99th-pct leverage")
    s14c, _ = printed(p, JUNE, NB14, "def rmse_loso(k=10)")
    flagged = number(r"physically-flagged: (\d+) of", s14a, int)
    checks = [
        {"check": "A · provenance", "measure": "training filters ever flagged for a physical problem", "value": flagged,
         "of": number(r"physically-flagged: \d+ of (\d+)", s14a, int), "unit": "filters"},
        {"check": "B · coverage", "measure": "ETAD spectra beyond the training 99th-percentile leverage", "value": number(r"leverage: (\d+)%", s14b), "of": None, "unit": "%"},
        {"check": "B · coverage", "measure": "median leverage, ETAD", "value": number(r"ETAD median leverage ([\d.]+)", s14b), "of": None, "unit": "whitened PCA distance"},
        {"check": "B · coverage", "measure": "median leverage, training", "value": number(r"vs training ([\d.]+)", s14b), "of": None, "unit": "whitened PCA distance"},
        {"check": "C · transfer", "measure": "random 5-fold RMSE (TOR EC)", "value": number(r"random 5-fold RMSE:\s+([\d.]+)", s14c), "of": None, "unit": "µg"},
        {"check": "C · transfer", "measure": "leave-one-site-out RMSE (TOR EC)", "value": number(r"leave-one-site-out RMSE:\s+([\d.]+)", s14c), "of": None, "unit": "µg"},
    ]
    p.add("checks", pd.DataFrame(checks), notebook=NB14, read="executed notebook output (printed)", path=rel(JUNE / f"{NB14}.ipynb"),
          cell="physically-flagged; 99th-pct leverage; rmse_loso")

    # ------------------------------------------------------------ calibration stability (spartan_ec 05, 07, 08; June 06)
    sp05r, _ = printed(p, SPARTAN_EC, SP05, "SIGMA = 3.0")
    sp07r, _ = printed(p, SPARTAN_EC, SP07, "3σ trimming rounds (n_before, threshold µg, dropped)")
    sp08r, _ = printed(p, SPARTAN_EC, SP08, "removal rounds")
    runs = []
    k05 = number(r"round 1: n=\d+\s+K=(\d+)", sp05r, int)
    for rnd, n, thr, d in re.findall(r"round (\d+): n=(\d+)\s+K=\d+.*?threshold=([\d.]+)µg\s+-> drop (\d+)", sp05r):
        runs.append({"run": f"spartan_ec 05 · K = {k05}, 5 rounds max", "round": int(rnd), "n before": int(n), "3σ threshold (µg)": float(thr), "dropped": int(d)})
    k07 = 18  # 07's trimming fits at K = 18 (fit_coef(X[keep], y[keep], 18))
    for rnd, (n, thr, d) in enumerate(re.findall(r"\((\d+), np\.float64\(([\d.]+)\), (\d+)\)", sp07r), start=1):
        runs.append({"run": f"spartan_ec 07 · K = {k07}, 6 rounds max", "round": rnd, "n before": int(n), "3σ threshold (µg)": float(thr), "dropped": int(d)})
    for rnd, n, thr, d in re.findall(r"\((\d+), (\d+), ([\d.]+), (\d+)\)", sp08r):
        runs.append({"run": "spartan_ec 08 · K = 20, until none", "round": int(rnd), "n before": int(n), "3σ threshold (µg)": float(thr), "dropped": int(d)})
    runs = pd.DataFrame(runs)
    if runs.groupby("run").size().to_dict() != {f"spartan_ec 05 · K = {k05}, 5 rounds max": 5, f"spartan_ec 07 · K = {k07}, 6 rounds max": 6, "spartan_ec 08 · K = 20, until none": 9}:
        raise ValueError("the printed cleaning rounds of spartan_ec 05/07/08 changed")
    s06c, _ = printed(p, JUNE, NB06, "cleaned_mask = keep")
    june_kept = number(r"CAL-1 cleaned: (\d+) kept", s06c, int)
    june_k = number(r"CAL-0 baseline: k=(\d+)", s06c, int)
    p.add("cleaning_runs", runs, notebook=f"spartan_ec {SP05}, {SP07}, {SP08}", read="executed notebook output (printed)",
          path=", ".join(rel(SPARTAN_EC / f"{nb}.ipynb") for nb in (SP05, SP07, SP08)))
    p.add("june06_clean", pd.DataFrame([{"run": f"June 06 · CAL-1, k = {june_k}, 3 rounds max", "k": june_k, "kept": june_kept, "removed": 906 - june_kept}]),
          notebook=NB06, read="executed notebook output (printed)", path=rel(JUNE / f"{NB06}.ipynb"), cell="cleaned_mask = keep")

    fits = []
    sp05f, _ = printed(p, SPARTAN_EC, SP05, "curve0 = rmsep_curve(Xa, y, KS)")
    sp05o, _ = printed(p, SPARTAN_EC, SP05, "curveF = rmsep_curve(Xc, yc, KS)")
    k, cv, r2, rmse = re.search(r"best K=(\d+)\s+CV-RMSEP=([\d.]+)\s+in-sample R2=([\d.]+) RMSE=([\d.]+)", sp05f).groups()
    fits.append({"run": "spartan_ec 05", "stage": "before cleaning", "n": 906, "K": int(k), "CV-RMSEP (µg)": float(cv), "R²": float(r2), "RMSE (µg)": float(rmse)})
    n, k, cv, r2, rmse = re.search(r"OPTIMIZED: n=(\d+)\s+K=(\d+)\s+CV-RMSEP=([\d.]+)\s+R²=([\d.]+)\s+RMSE=([\d.]+)", sp05o).groups()
    fits.append({"run": "spartan_ec 05", "stage": "cleaned, K re-picked", "n": int(n), "K": int(k), "CV-RMSEP (µg)": float(cv), "R²": float(r2), "RMSE (µg)": float(rmse)})
    for tag, n, r2, rmse in re.findall(r"^\s+(full|cleaned)\s+n=(\d+)\s+R²=([\d.]+)\s+RMSE=([\d.]+)", sp07r, re.M):
        fits.append({"run": "spartan_ec 07", "stage": "before cleaning" if tag == "full" else "cleaned", "n": int(n), "K": k07, "CV-RMSEP (µg)": None, "R²": float(r2), "RMSE (µg)": float(rmse)})
    sp08f, _ = printed(p, SPARTAN_EC, SP08, "pred_clean=fitpred(keep)")
    for stage, n, r2, rmse in re.findall(r"(BEFORE|AFTER)\s*: n=(\d+) R²=([\d.]+) RMSE=([\d.]+)", sp08f):
        fits.append({"run": "spartan_ec 08", "stage": "before cleaning" if stage == "BEFORE" else "cleaned", "n": int(n), "K": 20, "CV-RMSEP (µg)": None, "R²": float(r2), "RMSE (µg)": float(rmse)})
    if len(fits) != 6:
        raise ValueError("the printed before/after fits of spartan_ec 05/07/08 changed")
    p.add("cleaning_fits", pd.DataFrame(fits), notebook=f"spartan_ec {SP05}, {SP07}, {SP08}", read="executed notebook output (printed)",
          path=", ".join(rel(SPARTAN_EC / f"{nb}.ipynb") for nb in (SP05, SP07, SP08)))

    s7k, _ = printed(p, SPARTAN_EC, SP07, "argmins=KS[np.nanargmin(CVMAT,axis=1)]")
    s7r, _ = printed(p, SPARTAN_EC, SP07, "Kgrid=np.arange(8,41,2)")
    s7v, _ = printed(p, SPARTAN_EC, SP07, "flat=KS[mC<=mn*1.05]")
    runs_n = number(r"argmin-K across (\d+) CV runs", s7k, int)
    facts = [
        {"question": f"where one 10-fold CV run's RMSEP minimum lands ({runs_n} fold splits)", "low": number(r"min (\d+), max", s7k, int), "high": number(r"max (\d+), most common", s7k, int),
         "value": number(r"most common (\d+)", s7k, int), "unit": "K (most common)"},
        {"question": f"the averaged curve's flat valley (within 5 % of its minimum, {runs_n} runs)", "low": number(r"K = (\d+)–\d+", s7v, int), "high": number(r"K = \d+–(\d+)", s7v, int),
         "value": number(r"robust optimum (\d+)", s7k, int), "unit": "K (averaged minimum)"},
        {"question": "median Adama biomass/general EC ratio over K = 8…40", "low": number(r"ranges ([-\d.]+) –", s7r), "high": number(r"ranges [-\d.]+ – ([-\d.]+)", s7r),
         "value": None, "unit": "ratio"},
    ]
    p.add("k_stability", pd.DataFrame(facts), notebook=f"spartan_ec {SP07}", read="executed notebook output (printed)", path=rel(SPARTAN_EC / f"{SP07}.ipynb"),
          cell="argmins; Kgrid; flat valley")
    s7a, src7a = printed(p, SPARTAN_EC, SP07, "approaches[\"cleaned, ensemble 15–25\"]=ratio_ens")
    appr = []
    for line in s7a.splitlines():
        hit = re.match(r"^\s*(.+?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)$", line)
        if hit and not line.strip().startswith("approach"):
            appr.append({"approach": hit.group(1).strip(), "median ratio": float(hit.group(2)), "spread (max − min over the 5 filters)": float(hit.group(3)), "sd": float(hit.group(4))})
    if len(appr) != 3:
        raise ValueError("spartan_ec 07's approaches table changed")
    p.add("approaches", pd.DataFrame(appr), **src7a)

    st = SPARTAN_EC / "tables"
    opt = pd.read_csv(st / "adama_ec_optimized_comparison.csv")
    stab = pd.read_csv(st / "adama_ec_stabilized_comparison.csv")
    if not np.allclose(opt.set_index("FilterId").EC_general, stab.set_index("FilterId").EC_general):
        raise ValueError("spartan_ec 05 and 07 disagree on the general-calibration Adama EC")
    adama = pd.concat([
        opt.assign(calibration="general (lot 241a)", EC=opt.EC_general),
        opt.assign(calibration="biomass, before cleaning (05)", EC=opt.EC_biomass_baseline),
        opt.assign(calibration="biomass, cleaned, K re-picked (05)", EC=opt.EC_biomass_optimized),
        stab.assign(calibration="biomass, cleaned, K 15–25 ensemble (07)", EC=stab.EC_biomass),
    ])[["FilterId", "calibration", "EC", "ECTR", "char_soot"]].rename(columns={"ECTR": "TOR EC", "char_soot": "char/soot"})
    p.add("adama", adama, notebook=f"spartan_ec {SP05}, {SP07}", read="tracked csv",
          path=f"{rel(st / 'adama_ec_optimized_comparison.csv')}, {rel(st / 'adama_ec_stabilized_comparison.csv')}")

    # ------------------------------------------------------------ what only a re-run can add (the notebooks saved PNGs only)
    june = rel(JUNE)
    training = "the gitignored 906-filter training set spartan_ec_2026_06_16/data/rds_EC_X.csv and rds_EC_Ymeasured.csv (local only)"
    drive = "the Drive: ETAD spectra (DAVIS/ETAD FTIR), Spartan/SPARTAN_HIPS_Batch1-51.v2.csv and the lot-248+251 FTIR/local_db mirror"

    def rerun(folder, nb):
        return f"cd {rel(folder)} && uv run python _build_{nb}.py && uv run jupyter nbconvert --to notebook --execute --inplace {nb}.ipynb"

    p.missing += [
        {"table": "variant_cv_curves", "what": "the RMSECV curve of every variant (06 [15], 5-fold; 09 [7], 10-fold %RMSECV) and CAL-0 raw against 2nd-derivative (06 [19], 09 [9]); only the picked k and the minimum are tracked",
         "needs": f"{june}/tables/component_selection_curves.csv", "run": rerun(JUNE, NB09),
         "after": f"a line in _build_{NB09}.py that writes the curves it now only draws (figures/fig09_component_selection_pct.png, fig09_raw_vs_d2.png), and {training}"},
        {"table": "variant_predictions", "what": "per-filter predictions of the 906 filters under CAL-0…CAL-7 (06 [9] and 10 [7] 1:1 grids, 06 [12]/[13] inverse sets, 10 [9] predicted-EC distributions)",
         "needs": f"{june}/tables/calibration_variants_predictions.csv", "run": rerun(JUNE, NB10),
         "after": f"a line in _build_{NB10}.py that saves the in-sample and applied predictions it plots, and {training}"},
        {"table": "origin_predictions", "what": "per-spectrum ETAD EC for train EC ≤ 50 k = 10 and EC ≤ 30 k = 5 (11 [7]) and for drop-CF and SNV (13 [7]), for the before/after crossplots",
         "needs": f"{june}/tables/etad_origin_predictions.csv", "run": rerun(JUNE, NB13),
         "after": f"lines in _build_{NB11}.py and _build_{NB13}.py that save those predictions, and {drive}"},
        {"table": "leverage", "what": "per-spectrum leverage (whitened PCA distance) of the 319 ETAD and 13,012 training spectra, for 14 [5]'s histogram",
         "needs": f"{june}/tables/etad_coverage_leverage.csv", "run": rerun(JUNE, NB14),
         "after": f"a line in _build_{NB14}.py that writes the leverage arrays, and {drive}"},
        {"table": "rmsep_by_k", "what": "RMSEP against K: spartan_ec 05 [7]/[15] before and after cleaning, 07 [3] eight single CV runs and 07 [11] their mean ± sd",
         "needs": f"{rel(SPARTAN_EC)}/tables/rmsep_by_k.csv", "run": rerun(SPARTAN_EC, SP07),
         "after": f"lines in _build_{SP05}.py and _build_{SP07}.py that write the curves they draw (figures/fig07_1_kvariance.png, fig07_3_repeatedcv.png), and spartan_ec_2026_06_16/data/ (gitignored)"},
        {"table": "adama_by_k", "what": "the Adama biomass/general ratio per filter against K (07 [6]) and per approach (07 [15])",
         "needs": f"{rel(SPARTAN_EC)}/tables/adama_ratio_by_k.csv", "run": rerun(SPARTAN_EC, SP07),
         "after": f"a line in _build_{SP07}.py that writes the ratios it plots (figures/fig07_2_adama_vs_k.png, fig07_4_approaches.png), and spartan_ec_2026_06_16/data/ (gitignored)"},
        {"table": "cleaned_refit", "what": "the cleaned refit's predictions for the kept filters (spartan_ec 05 [15], 08 [7]) and 05 [10]'s residual diagnostics",
         "needs": f"{rel(SPARTAN_EC)}/tables/ec_cleaned_refit.csv", "run": f"cd {rel(SPARTAN_EC)} && uv run python _build_08_removal_explainer.py && uv run jupyter nbconvert --to notebook --execute --inplace {SP08}.ipynb",
         "after": "a line in _build_08_removal_explainer.py that saves pred_clean; the full-fit predictions of all 906 are on Research → Ethiopia calibrations where the Drive resolves"},
    ]

    # ------------------------------------------------------------ findings, read off the tables above
    v = variants.set_index("variant")
    eff = v.dropna(subset=["median predicted EC on the 906 (10, µg)"])
    cand = eff[eff.role == "candidate"]["median predicted EC on the 906 (10, µg)"]
    f = front.set_index("correction")
    removed = {**{r: int(g["dropped"].sum()) for r, g in runs.groupby("run")}, "June 06": 906 - june_kept}
    p.text["finding"] = (
        f"On the 906 IMPROVE filters the component count is set by the rule as much as by the data: CAL-0's curve gives k = "
        f"{picks[0]['k']} under 05's 2 % rule, {picks[3]['k']} under 06's 5 % rule and {picks[5]['k']} under 09's 10-fold CV, and "
        f"{picks[6]['k']} on 2nd-derivative spectra at about the same error. CAL-1's 3σ trim has the best in-sample R² "
        f"({v.loc['CAL-1', 'in-sample R² (06)']:.2f}) but a worse %RMSECV than keeping everything ({v.loc['CAL-1', '%RMSECV (09/10)']:.0f} % vs "
        f"{v.loc['CAL-0', '%RMSECV (09/10)']:.0f} %). Applied to the same 906 filters the candidate calibrations' median EC spans "
        f"{cand.min():.1f}–{cand.max():.1f} µg. Against HIPS no change reaches slope 1 and intercept 0 together (at MAC 10): dropping the "
        f"Teflon CF band keeps slope {f.loc['drop CF (Teflon)', 'slope']:.2f} at intercept {f.loc['drop CF (Teflon)', 'intercept']:.2f}, SNV reaches "
        f"{f.loc['SNV', 'intercept']:.2f} at slope {f.loc['SNV', 'slope']:.2f}. The same 3σ rule removes {min(removed.values())}–{max(removed.values())} "
        f"filters of 906 depending on K and when it stops (June 06's three rounds at k = {june_k} keep {june_kept}; spartan_ec 08's nine at K = 20 keep {906 - removed['spartan_ec 08 · K = 20, until none']}).")
    return p
