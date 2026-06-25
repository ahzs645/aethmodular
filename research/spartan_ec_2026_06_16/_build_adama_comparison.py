"""Adama (5 PTFE filters) EC: general vs biomass calibration, vs quartz TOR.

Calibrations applied to the 5 Adama PTFE spectra:
  * general  = lot 241a (==245) production coefficients (data/adama/cal_lot241a_245_EC.csv)
  * biomass_tool  = the tool's exported lot-251 biomass coeffs (data/tool_EC_coeffs_lot251_biomass.csv)
  * biomass_local = local rebuild on lot-251 IMPROVE biomass spectra WITH sigma-scaled outlier
                    removal (the robustness check for Mona's removal step)

Paired with the co-located quartz TOR (ECTR, char/soot, OC/EC) by sample date.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import ftir_pls_calibration as f

HERE = Path(__file__).resolve().parent
WN = re.compile(r"^[+-]?\d+(\.\d+)?$")

# --- Adama PTFE spectra (rows map in order to the FTIR filter list, validated) ---
sp = pd.read_csv(HERE / "data/adama/adama_ptfe_spectra_batch54.csv")
wn = [c for c in sp.columns if WN.match(str(c).strip())]
wv = np.array([float(c) for c in wn]); o = np.argsort(wv); wv_s = wv[o]; cols_s = [wn[i] for i in o]

def apply_tool(calpath, tol=0.3):
    cal = pd.read_csv(calpath); ic = float(cal.loc[cal.Wavenumber == 0, "b"].iloc[0]); co = cal.loc[cal.Wavenumber != 0]
    cw = co.Wavenumber.to_numpy(float); cb = co.b.to_numpy(float)
    idx = np.clip(np.searchsorted(wv_s, cw), 1, len(wv_s) - 1)
    pick = np.where(np.abs(cw - wv_s[idx - 1]) <= np.abs(cw - wv_s[idx]), idx - 1, idx)
    ok = np.abs(cw - wv_s[pick]) <= tol
    return sp[[cols_s[i] for i in pick[ok]]].to_numpy(float) @ cb[ok] + ic

def apply_result(res, tol=0.3):  # CalibrationResult -> predictions on the 5 Adama spectra
    cw = res.wavenumbers; cb = res.coefficients
    idx = np.clip(np.searchsorted(wv_s, cw), 1, len(wv_s) - 1)
    pick = np.where(np.abs(cw - wv_s[idx - 1]) <= np.abs(cw - wv_s[idx]), idx - 1, idx)
    ok = np.abs(cw - wv_s[pick]) <= tol
    return sp[[cols_s[i] for i in pick[ok]]].to_numpy(float) @ cb[ok] + res.intercept

# --- build the local lot-251 biomass calibration WITH sigma-scaled removal ---
spec, swn = f.load_spectra_csv(HERE / "data/spectra_lot251_biomass_2021-2025.csv")
spec["date"] = pd.to_datetime(spec.SampleDate).dt.tz_localize(None).dt.normalize()
ref = pd.read_csv(HERE / "../../research/ftir_hips_chem/output/improve_high_fabs_comparison/improve_valid_cleaned.csv", low_memory=False)
ref["date"] = pd.to_datetime(ref.Date).dt.normalize()
refd = ref.sort_values("POC").drop_duplicates(["SiteCode", "date"])[["SiteCode", "date", "EC_loading_ug"]]
train = spec.merge(refd, left_on=["Site", "date"], right_on=["SiteCode", "date"], how="left").dropna(subset=["EC_loading_ug"])

res_sig = f.build_calibration(train, swn, "EC_loading_ug", species="EC",
                             ncomp_range=range(6, 41, 2), residual_sigma=3.0, n_sigma_rounds=3, cv=5)
print("=== local lot-251 biomass EC rebuild WITH sigma=3 removal ===")
print(f"  n kept: {res_sig.stats['n']} | dropped neg: {len(res_sig.dropped['measured_negative'])} | "
      f"dropped resid: {len(res_sig.dropped['residual'])} | n_comp: {res_sig.n_components} | "
      f"R2: {res_sig.stats['r2']:.3f} | RMSE: {res_sig.stats['rmse']:.2f}")
for h in res_sig.history:
    print("   trim:", h)

# --- assemble the 5-filter comparison ---
ft = pd.read_csv(HERE / "data/adama/adama_ptfe_ftir_batch54.csv")
pt = ft[ft.Parameter == "EC_ftir"][["FilterId", "SampleDate", "MassLoading_ug"]].reset_index(drop=True)
pt["date"] = pd.to_datetime(pt.SampleDate).dt.normalize()
pt["EC_general"] = apply_tool(HERE / "data/adama/cal_lot241a_245_EC.csv")
pt["EC_biomass_tool"] = apply_tool(HERE / "data/tool_EC_coeffs_lot251_biomass.csv")
pt["EC_biomass_local_sig"] = apply_result(res_sig)

t = pd.read_csv(HERE / "data/adama/adama_quartz_tor_batch54.csv")
tw = t.pivot_table(index=["FilterId", "SampleDate"], columns="Parameter", values="MassLoading_ug").reset_index()
tw["date"] = pd.to_datetime(tw.SampleDate).dt.normalize()
tw["char_soot"] = (tw.EC1 - tw.OPTR) / (tw.EC2 + tw.EC3); tw["OC_EC"] = tw.OCTR / tw.ECTR
m = pt.merge(tw[["date", "ECTR", "char_soot", "OC_EC"]], on="date", how="left")
m["ratio_tool"] = m.EC_biomass_tool / m.EC_general
m["ratio_local"] = m.EC_biomass_local_sig / m.EC_general

cols = ["FilterId", "date", "EC_general", "EC_biomass_tool", "EC_biomass_local_sig",
        "ratio_tool", "ratio_local", "ECTR", "char_soot", "OC_EC"]
print("\n=== Adama EC: general vs biomass(tool) vs biomass(local+sigma), vs TOR ===")
print(m[cols].round(2).to_string(index=False))
print(f"\nmedian ratios  tool={m.ratio_tool.median():.2f}  local+sigma={m.ratio_local.median():.2f}")
m[cols].to_csv(HERE / "tables/adama_ec_calibration_comparison.csv", index=False)
print("wrote tables/adama_ec_calibration_comparison.csv")
