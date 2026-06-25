"""Faithful local replica of the tool's biomass EC calibration, built on the tool's
EXACT training data (extracted from pls-EC-2026-06-24.RDS), then applied to Adama.

We proved sklearn PLS(scale=False) == the tool's R kernelpls to ~1e-10 on identical
X/Y, so a sklearn fit on the RDS X/Y is an exact stand-in for the tool — no server.

Outputs three EC predictions for the 5 Adama PTFE filters:
  * general            = lot 241a (==245) production coeffs
  * biomass_faithful   = sklearn PLS on tool's exact X/Y, K by min CV-RMSEP, NO removal
                         (this reproduces the tool's exported coefficients)
  * biomass_faithful_sig = same but WITH Mona's sigma-scaled outlier removal, on the
                         CORRECT Y (the earlier sigma run used an approximate Y)
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import ftir_pls_calibration as f

HERE = Path(__file__).resolve().parent
WN = re.compile(r"^[+-]?\d+(\.\d+)?$")

# --- tool's EXACT training data from the RDS ---
X = pd.read_csv(HERE / "data/rds_EC_X.csv").drop(columns=["id"])
y = pd.read_csv(HERE / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy(float)
wvs = pd.read_csv(HERE / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy(float)  # column order
assert X.shape[1] == len(wvs), (X.shape, len(wvs))
wn = [f"{w:.6f}" for w in wvs]
df = X.copy(); df.columns = wn; df["EC"] = y
print(f"tool exact training set: X={X.shape}  Y n={len(y)} median={np.median(y):.2f}")

# --- faithful (no removal) and faithful + sigma removal ---
res = f.build_calibration(df, wn, "EC", species="EC", ncomp_range=range(6, 41, 2),
                          residual_thresholds=(), drop_measured_negatives=False, cv=5)
res_sig = f.build_calibration(df, wn, "EC", species="EC", ncomp_range=range(6, 41, 2),
                              residual_sigma=3.0, n_sigma_rounds=3,
                              drop_measured_negatives=False, cv=5)
print(f"faithful (no removal): n={res.stats['n']} K={res.n_components} R2={res.stats['r2']:.3f} RMSE={res.stats['rmse']:.2f}")
print(f"faithful + sigma=3   : n={res_sig.stats['n']} K={res_sig.n_components} R2={res_sig.stats['r2']:.3f} RMSE={res_sig.stats['rmse']:.2f} (dropped {len(res_sig.dropped['residual'])})")

# sanity: faithful coeffs should match the tool's exported coeffs
tool = pd.read_csv(HERE / "data/tool_EC_coeffs_lot251_biomass.csv")
tb = tool.loc[tool.Wavenumber != 0].set_index(tool.loc[tool.Wavenumber != 0].Wavenumber.round(4)).b
tool_b = np.array([tb.get(round(float(w), 4), np.nan) for w in res.wavenumbers])
mask = ~np.isnan(tool_b)
print(f"faithful vs tool exported coeffs: r={np.corrcoef(res.coefficients[mask], tool_b[mask])[0,1]:.6f} "
      f"(K_faithful={res.n_components}; tool export was K=18)")

# --- apply all to the 5 Adama PTFE spectra ---
sp = pd.read_csv(HERE / "data/adama/adama_ptfe_spectra_batch54.csv")
awn = [c for c in sp.columns if WN.match(str(c).strip())]
av = np.array([float(c) for c in awn]); o = np.argsort(av); av_s = av[o]; ac = [awn[i] for i in o]
def apply_res(r, tol=0.3):
    cw = r.wavenumbers
    idx = np.clip(np.searchsorted(av_s, cw), 1, len(av_s) - 1)
    pick = np.where(np.abs(cw - av_s[idx - 1]) <= np.abs(cw - av_s[idx]), idx - 1, idx)
    ok = np.abs(cw - av_s[pick]) <= tol
    return sp[[ac[i] for i in pick[ok]]].to_numpy(float) @ r.coefficients[ok] + r.intercept
def apply_tool(calpath, tol=0.3):
    cal = pd.read_csv(calpath); ic = float(cal.loc[cal.Wavenumber == 0, "b"].iloc[0]); co = cal.loc[cal.Wavenumber != 0]
    cw = co.Wavenumber.to_numpy(float); cb = co.b.to_numpy(float)
    idx = np.clip(np.searchsorted(av_s, cw), 1, len(av_s) - 1)
    pick = np.where(np.abs(cw - av_s[idx - 1]) <= np.abs(cw - av_s[idx]), idx - 1, idx)
    ok = np.abs(cw - av_s[pick]) <= tol
    return sp[[ac[i] for i in pick[ok]]].to_numpy(float) @ cb[ok] + ic

ft = pd.read_csv(HERE / "data/adama/adama_ptfe_ftir_batch54.csv")
pt = ft[ft.Parameter == "EC_ftir"][["FilterId", "SampleDate"]].reset_index(drop=True)
pt["date"] = pd.to_datetime(pt.SampleDate).dt.normalize()
pt["EC_general"] = apply_tool(HERE / "data/adama/cal_lot241a_245_EC.csv")
pt["EC_biomass_faithful"] = apply_res(res)
pt["EC_biomass_faithful_sig"] = apply_res(res_sig)

t = pd.read_csv(HERE / "data/adama/adama_quartz_tor_batch54.csv")
tw = t.pivot_table(index=["FilterId", "SampleDate"], columns="Parameter", values="MassLoading_ug").reset_index()
tw["date"] = pd.to_datetime(tw.SampleDate).dt.normalize()
tw["char_soot"] = (tw.EC1 - tw.OPTR) / (tw.EC2 + tw.EC3)
m = pt.merge(tw[["date", "ECTR", "char_soot"]], on="date", how="left")
m["ratio_faithful"] = m.EC_biomass_faithful / m.EC_general
m["ratio_faithful_sig"] = m.EC_biomass_faithful_sig / m.EC_general

cols = ["FilterId", "date", "EC_general", "EC_biomass_faithful", "EC_biomass_faithful_sig",
        "ratio_faithful", "ratio_faithful_sig", "ECTR", "char_soot"]
print("\n=== Adama EC: general vs faithful biomass (exact X/Y) vs +sigma, vs TOR ===")
print(m[cols].round(2).to_string(index=False))
print(f"\nmedian ratio  faithful={m.ratio_faithful.median():.2f}  faithful+sigma={m.ratio_faithful_sig.median():.2f}")
m[cols].to_csv(HERE / "tables/adama_ec_faithful_comparison.csv", index=False)
print("wrote tables/adama_ec_faithful_comparison.csv")
