"""28 Oct 2026 meeting: why Dry and Belg fail, and why the IMPROVE look-alikes err differently from Addis.

Works through the analysis items of the AAAR master to-do list (email to Ann, 22 Oct):

  1 ocec      season calibrations from IMPROVE filters chosen by TOR OC/EC instead of spectral shape
  2 deficit   filter by filter: does the Addis error track the organic signal per µg EC?
  3 dust      dust (SPARTAN Al, Si) by season and against the error
  4 mac       the MAC each season would need for slope 1, and the look-alikes' own MAC
  5 highwn    the 3600-4000 cm⁻¹ residual against sulfate and humidity
  6 loading   the heaviest IMPROVE filters as an out-of-range test set; error against loading
  7 dark      TOR EC vs HIPS on IMPROVE filters where optical pyrolysis reads zero
  8 vip       VIP of the raw (SPARTAN-like) and baseline-corrected calibrations
  9 urban     a calibration from the six urban IMPROVE sites only
  10 speciation  Addis chemistry vs IMPROVE (all, and each season's look-alikes)
  11 blh      concentrations normalised by ERA5 boundary-layer height

Run from research/ftir_hips_chem (about 5 min):
    ~/anaconda3/bin/python workflows/run_meeting_20261028.py

Inputs are the poster's (see ann_followup_20261021.ipynb): the frozen spline (AIRSpec) run and its raw-spectra
bundle, the rank-70 meeting tables (seasonal look-alike sets and Figure 9's seasonal predictions), and the 189 Addis
filters with a published SPARTAN FTIR EC. Addis fits are Deming with the poster's λ (σx from HIPS uncertainty, σy
0.531); IMPROVE fits use the look-alike error model of run_meeting_followup_20260917.analog_panels. Writes
output/tables/meeting_20261028/; the gallery page reads it through gallery/data/export_meeting_20261028.py.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.cross_decomposition import PLSRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import analog_bias as ab  # noqa: E402
import raw_vs_spline_20261020 as rv  # noqa: E402
import run_meeting_followup_20260917 as rmf  # noqa: E402
from pls_transfer import vip_scores  # noqa: E402

AREA = HERE.parent
OUT = AREA / "output/tables/meeting_20261028"
M = AREA / "output/tables/vibes_fullrange/gallery_rank70_20261002/meeting"
FED = AREA / "output/tables/improve_validated_archive/raw"
WEATHER = AREA.parents[1] / "gallery/app/public/data/weather"
SEASONS = ab.SEASONS
N = ab.N_ANALOGS
URBAN = {"ATLA1": "Atlanta", "BIRM1": "Birmingham", "FRES1": "Fresno", "PHOE1": "Phoenix",
         "PITT1": "Pittsburgh", "PUSO1": "Seattle (Puget Sound)"}
BANDS = {"O-H / N-H 3000-3600": (3600, 3000), "C-H 2800-3000": (3000, 2800), "carbonyl 1650-1800": (1800, 1650),
         "1600 band 1500-1650": (1650, 1500), "high-wn 3600-4000": (4000, 3600)}
ORGANIC = ["O-H / N-H 3000-3600", "C-H 2800-3000", "carbonyl 1650-1800"]
CHEM = {"ChemSpec_Sulfate_Ion_PM2.5": "sulfate", "ChemSpec_Nitrate_Ion_PM2.5": "nitrate",
        "ChemSpec_Ammonium_Ion_PM2.5": "ammonium", "ChemSpec_OC_PM2.5": "OC (FTIR)", "ChemSpec_OM_PM2.5": "OM",
        "ChemSpec_Aluminum_PM2.5": "Al", "ChemSpec_Silicon_PM2.5": "Si", "ChemSpec_Titanium_PM2.5": "Ti",
        "ChemSpec_Iron_PM2.5": "Fe", "ChemSpec_Calcium_PM2.5": "Ca", "ChemSpec_Potassium_Ion_PM2.5": "K+",
        "ChemSpec_Filter_PM2.5_mass": "PM2.5"}
FED_PARAMS = {"SO4f": "sulfate", "NO3f": "nitrate", "OCf": "OC", "ECf": "EC", "SOILf": "soil", "MF": "PM2.5",
              "ALf": "Al", "SIf": "Si"}


def log(msg):
    print(msg, flush=True)


def save(frame, name):
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{name}.csv", index=False)
    log(f"  wrote {name}.csv ({len(frame)} rows)")


# ------------------------------------------------------------------ inputs (as ann_followup_20261021.ipynb)
def load():
    ctx = ab.load_context(fullrange=False)
    cases = pd.read_csv(ab.RUN / "cases.csv")
    lib = ctx.lib
    d = dict(ctx=ctx, lib=lib, wn_full=np.load(rv.BUNDLE_FULL / "data/wn.npy"), wn=np.load(ab.RUN / "wn.npy"))
    d["X_raw"] = rv.raw_rows(cases, lib.row, rv.BUNDLE_FULL)
    XA_raw = rv.raw_rows(cases, ctx.addis.row, rv.BUNDLE_FULL)
    d["X"], XA = ctx.X["AIRSpec"], ctx.XA["AIRSpec"]
    d["qc"] = rv.qc_keep(rv.raw_rows(cases, lib.row), ctx)

    dep = pd.read_csv(rv.DEPLOYED)
    dep = dep[dep.fit_id.eq("deployed|SPARTAN|current SPARTAN calibration")].dropna(subset=["ftir_ec_ugm3", "Fabs"])
    coh = ctx.addis.MediaId.isin(dep.MediaId).to_numpy()
    A = ctx.addis.loc[coh].copy().reset_index(drop=True)
    A["vol"] = A.Volume_m3.astype(float)
    A["hips"] = A.Fabs / ab.MAC_VALUE
    A["spartan"] = A[["MediaId"]].merge(dep[["MediaId", "ftir_ec_ugm3"]], on="MediaId", how="left").ftir_ec_ugm3.to_numpy()
    A["ec_ug"] = A.hips * A.vol
    A["cartridge"] = A.filter_id.str.rsplit("-", n=1).str[0]
    d["coh"], d["A"], d["XA_raw"], d["XA"] = coh, A, XA_raw[coh], XA[coh]

    an = pd.read_csv(M / "analog_predictions.csv")
    fit = "allimprove|AIRSpec|all-IMPROVE lot 251"
    pos = pd.Series(np.arange(len(lib)), index=lib.filter_id.astype(int).to_numpy())

    def rows_of(group, role=None):
        q = an[an.fit_id.eq(f"{fit}|{group}")]
        if role:
            q = q[q.role.eq(role)]
        return pos.loc[q.filter_id.astype(int)].to_numpy()

    d["L97"] = rows_of("All Addis", "test")
    d["LSEA"] = {s: rows_of(s) for s in SEASONS}
    assert len(d["L97"]) == 97

    # Figure 9's season-specific predictions (seasonal|AIRSpec|locked) for the 189
    allp = pd.read_csv(M / "addis_predictions.csv")
    seas = np.full(len(A), np.nan)
    for s in SEASONS:
        q = allp[allp.fit_id.eq(f"seasonal|AIRSpec|locked|{s}") & allp.season.eq(s)].set_index("MediaId").ftir_ec_ugm3
        m = A.season.eq(s).to_numpy()
        seas[m] = q.reindex(A.MediaId[m]).to_numpy()
    assert np.isfinite(seas).all()
    A["pred_season"] = seas
    log(f"Addis {len(A)} filters; IMPROVE library {len(lib)}; look-alikes 97; seasonal sets "
        + ", ".join(f"{s.split()[0]} {len(v)}" for s, v in d["LSEA"].items()))
    return d


def addis_fit(d, x, y):
    ctx = d["ctx"]
    st = ab.xy_stats(x, y, lam=ctx.lam, sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
    return {"slope": st["deming_slope"], "intercept": st["deming_intercept"], "R2": st["R2"], "n": st["n"]}


def season_rows(d, pred, **base):
    """Deming FTIR vs HIPS for each season and all of Addis, the poster's readout."""
    A, rows = d["A"], []
    for s in SEASONS + ["All Addis"]:
        m = np.ones(len(A), bool) if s == "All Addis" else A.season.eq(s).to_numpy()
        rows.append({**base, "season": s, **addis_fit(d, A.hips.to_numpy()[m], pred[m])})
    return rows


def band_areas(X, wn):
    return pd.DataFrame({name: X[:, (wn <= hi) & (wn > lo)].sum(1) for name, (hi, lo) in BANDS.items()})


def partial_spearman(x, y, z):
    """Spearman ρ of x and y with the (rank-linear) effect of z removed from both."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def spartan_chem(A):
    from data_matching import load_filter_data

    f = load_filter_data()
    e = f.loc[f.Site.eq("ETAD") & f.Parameter.isin(CHEM)].copy()
    e["ugm3"] = np.where(e.Concentration_Units.eq("ng/m3"), e.Concentration / 1000, e.Concentration)
    wide = e.pivot_table(index="FilterId", columns="Parameter", values="ugm3", aggfunc="mean").rename(columns=CHEM)
    return A[["cartridge"]].merge(wide, left_on="cartridge", right_index=True, how="left").drop(columns="cartridge")


# ------------------------------------------------------------------ 1. OC/EC-chosen calibration sets
def stage_ocec(d):
    ctx, lib, A, X = d["ctx"], d["lib"], d["A"], d["X"]
    chem = d["chem"]
    A_ocec = chem["OC (FTIR)"].to_numpy() / A.hips.to_numpy()
    ok = ((lib.tor_ec_ugm3 > 0) & (lib.tor_oc_ugm3 > 0) & lib.tor_oc_ec.notna() & (lib.y >= 1)).to_numpy()
    ocec = lib.tor_oc_ec.to_numpy()
    gm = ab.group_masks(ctx)
    target = {s: float(np.nanmedian(A_ocec[A.season.eq(s).to_numpy()])) for s in SEASONS}
    target["All Addis"] = float(np.nanmedian(A_ocec))

    lowest = np.flatnonzero(ok)[np.argsort(ocec[ok], kind="stable")][:N]
    cohorts = []
    for s in SEASONS:
        spectral, _ = ab.correlation_analogs(ctx, "AIRSpec", gm[s], ab.LOCKED_MASK)
        near = np.flatnonzero(ok)[np.argsort(np.abs(np.log(ocec[ok] / target[s])), kind="stable")][:N]
        top, _ = ab.correlation_analogs(ctx, "AIRSpec", gm[s], ab.LOCKED_MASK, n=4 * N)
        top = top[ok[top]]
        both = top[np.argsort(ocec[top], kind="stable")][:N]
        cohorts += [(s, "spectral look-alikes (Figure 9)", spectral), (s, "lowest TOR OC/EC", lowest),
                    (s, "TOR OC/EC nearest this season", near), (s, "spectral top 2,000, lowest OC/EC", both)]

    fits, summary, members, vips = [], [], [], {}
    vol = A.vol.to_numpy()
    for s, label, pos in cohorts:
        L = lib.iloc[pos]
        f = ab.fit_cohort(X[pos], L.y, L.Site)
        test = ~f.train
        tor = ab.xy_stats(L.y.to_numpy()[test], f.predict(X[pos])[test])
        pred = f.predict(d["XA"]) / vol
        m = A.season.eq(s).to_numpy()
        fits.append({"selection": label, "season": s, "k": f.k, "TOR_test_R2": tor.get("R2"),
                     **addis_fit(d, A.hips.to_numpy()[m], pred[m])})
        summary.append({"selection": label, "season": s, "n": len(pos), "TOR OC/EC median": float(np.median(ocec[pos])),
                        "TOR EC µg median": float(np.median(L.y)), "Addis OC/EC target": target[s],
                        "sites": int(L.Site.nunique())})
        members.append(pd.DataFrame({"selection": label, "season": s, "filter_id": L.filter_id.to_numpy()}))
        vips[f"{label}|{s}"] = vip_scores(f.model)
        if label.startswith("spectral look-alikes"):
            # the refit must reproduce Figure 9's season-specific predictions
            assert np.allclose(pred[m], A.pred_season.to_numpy()[m], atol=1e-6), s
    fits = pd.DataFrame(fits)
    save(fits, "ocec_fits")
    save(pd.DataFrame(summary), "ocec_cohorts")
    save(pd.concat(members), "ocec_members")
    dist = pd.DataFrame({"population": "IMPROVE (TOR, EC ≥ 1 µg)", "oc_ec": ocec[ok]})
    dist = pd.concat([dist, pd.DataFrame({"population": [f"Addis {s.split()[0]}" for s in A.season],
                                          "oc_ec": A_ocec})])
    save(dist.dropna(), "ocec_distribution")
    d["vip_ocec"] = vips


# ------------------------------------------------------------------ 2. organic signal vs the error
def stage_deficit(d):
    A, XA, wn = d["A"], d["XA"], d["wn"]
    vol, hips = A.vol.to_numpy(), A.hips.to_numpy()
    bands = band_areas(XA, wn).div(vol, axis=0)          # absorbance per m³
    d["bands_A"] = bands
    preds = {"season-specific (Figure 9)": A.pred_season.to_numpy(), "SPARTAN published": A.spartan.to_numpy(),
             "one baseline-corrected (Figure 7)": d["pred_one"]}
    rows, points = [], []
    org = bands[ORGANIC].rank(pct=True).mean(1).to_numpy()   # one organic index: mean percentile of the three bands
    for cal, y in preds.items():
        for s in SEASONS + ["All Addis"]:
            m = np.ones(len(A), bool) if s == "All Addis" else A.season.eq(s).to_numpy()
            st = addis_fit(d, hips[m], y[m])
            resid = y[m] - (st["intercept"] + st["slope"] * hips[m])
            for b in ORGANIC + ["organic index"]:
                v = org[m] if b == "organic index" else bands[b].to_numpy()[m]
                rows.append({"calibration": cal, "season": s, "band": b, "n": int(m.sum()),
                             "partial_rho": partial_spearman(y[m], v, hips[m]),
                             "rho_ratio": spearmanr(y[m] / hips[m], v / hips[m]).correlation})
            if cal != "SPARTAN published" and s != "All Addis":
                # for the scatter: error against the organic signal, both with HIPS EC's effect removed
                ov = bands["C-H 2800-3000"].to_numpy()[m]
                ores = ov - np.polyval(np.polyfit(hips[m], ov, 1), hips[m])
                points.append(pd.DataFrame({"calibration": cal, "season": s, "MediaId": A.MediaId[m].to_numpy(),
                                            "error": resid, "ch_resid": ores * 1000,
                                            "ch_per_ug_ec": ov / hips[m] * 1000}))
    save(pd.DataFrame(rows), "deficit_correlations")
    save(pd.concat(points), "deficit_points")


# ------------------------------------------------------------------ 3. dust
def stage_dust(d):
    A, chem, hips = d["A"], d["chem"], d["A"].hips.to_numpy()
    out, rows = [], []
    for s in SEASONS:
        m = A.season.eq(s).to_numpy()
        c = chem[m]
        out.append({"season": s, "n": int(m.sum()), "Al": c.Al.median(), "Si": c.Si.median(), "Ti": c.Ti.median(),
                    "Fe": c.Fe.median(), "HIPS EC": np.median(hips[m]), "Si / EC": np.nanmedian(c.Si / hips[m]),
                    "Al / EC": np.nanmedian(c.Al / hips[m]), "PM2.5": c["PM2.5"].median()})
        for cal, y in (("season-specific (Figure 9)", A.pred_season.to_numpy()), ("SPARTAN published", A.spartan.to_numpy()),
                       ("one baseline-corrected (Figure 7)", d["pred_one"])):
            for el in ("Al", "Si"):
                v = c[el].to_numpy()
                ok = np.isfinite(v)
                rows.append({"calibration": cal, "season": s, "element": el, "n": int(ok.sum()),
                             "partial_rho": partial_spearman(y[m][ok], v[ok], hips[m][ok])})
    save(pd.DataFrame(out), "dust_by_season")
    save(pd.DataFrame(rows), "dust_correlations")


# ------------------------------------------------------------------ 4. MAC
def stage_mac(d):
    A, lib, ctx = d["A"], d["lib"], d["ctx"]
    rows = []
    preds = {"SPARTAN published": A.spartan.to_numpy(), "one baseline-corrected (Figure 7)": d["pred_one"],
             "season-specific (Figure 9)": A.pred_season.to_numpy()}
    for s in SEASONS:
        m = A.season.eq(s).to_numpy()
        L = lib.iloc[d["LSEA"][s]]
        ok = (L.tor_ec_ugm3 > 0.05) & (L.fabs_Mm1 > 0)
        mac_la = float(np.median(L.fabs_Mm1[ok] / L.tor_ec_ugm3[ok]))
        for cal, y in preds.items():
            base = addis_fit(d, A.hips.to_numpy()[m], y[m])
            x = A.Fabs.to_numpy()[m] / mac_la
            st = ab.xy_stats(x, y[m], lam=(ab.SIGMA_Y_PROXY / (ctx.sigma_x * ab.MAC_VALUE / mac_la)) ** 2,
                             sigma_x=ctx.sigma_x * ab.MAC_VALUE / mac_la, sigma_y=ab.SIGMA_Y_PROXY)
            rows.append({"season": s, "calibration": cal, "slope at MAC 10": base["slope"],
                         "intercept at MAC 10": base["intercept"], "MAC for slope 1": ab.MAC_VALUE / base["slope"],
                         "look-alike MAC (Fabs / TOR EC)": mac_la, "slope at look-alike MAC": st["deming_slope"],
                         "intercept at look-alike MAC": st["deming_intercept"]})
    save(pd.DataFrame(rows), "mac_by_season")


# ------------------------------------------------------------------ 5. the 3600-4000 cm⁻¹ residual
def stage_highwn(d):
    A, chem, bands, lib = d["A"], d["chem"], d["bands_A"], d["lib"]
    w = weather_for(A)
    hw = bands["high-wn 3600-4000"].to_numpy()
    rows = []
    for s in SEASONS + ["All Addis"]:
        m = np.ones(len(A), bool) if s == "All Addis" else A.season.eq(s).to_numpy()
        for name, v in (("sulfate", chem.sulfate.to_numpy()), ("ammonium", chem.ammonium.to_numpy()),
                        ("PM2.5", chem["PM2.5"].to_numpy()), ("HIPS EC", A.hips.to_numpy()),
                        ("Si", chem.Si.to_numpy()), ("relative humidity (ERA5)", w["Relative humidity (ERA5)"].to_numpy())):
            ok = m & np.isfinite(v)
            rows.append({"season": s, "against": name, "n": int(ok.sum()), "rho": spearmanr(hw[ok], v[ok]).correlation})
    save(pd.DataFrame(rows), "highwn_correlations")
    # the band itself, Addis by season vs each season's look-alikes, per m³ and per µg EC
    lb = band_areas(d["X"], d["wn"])["high-wn 3600-4000"].to_numpy()
    out = []
    for s in SEASONS:
        m = A.season.eq(s).to_numpy()
        L = d["LSEA"][s]
        out.append({"season": s, "Addis per m³": np.median(hw[m]), "look-alikes per m³": np.median(lb[L] / lib.tor_volume_m3.to_numpy()[L]),
                    "Addis per µg EC": np.median(hw[m] * A.vol.to_numpy()[m] / A.ec_ug.to_numpy()[m]),
                    "look-alikes per µg EC": np.median(lb[L] / lib.y.to_numpy()[L])})
    out = pd.DataFrame(out)
    out["ratio per µg EC"] = out["Addis per µg EC"] / out["look-alikes per µg EC"]
    save(out, "highwn_by_season")
    pts = pd.DataFrame({"season": A.season, "MediaId": A.MediaId, "highwn": hw * 1000, "sulfate": chem.sulfate,
                        "Si": chem.Si, "rh": w["Relative humidity (ERA5)"].to_numpy()})
    save(pts, "highwn_points")


# ------------------------------------------------------------------ 6. loading
def stage_loading(d):
    lib, X, A = d["lib"], d["X"], d["A"]
    lot = lib.lot.eq(251).to_numpy()
    heavy = lot & (lib.y >= 20).to_numpy()
    light = np.flatnonzero(lot & ~heavy)
    f_light = ab.fit_cohort(X[light], lib.y.to_numpy()[light], lib.Site.to_numpy()[light])
    test = lib.split.eq("test").to_numpy() & lot
    vol, y = lib.tor_volume_m3.to_numpy(), lib.y.to_numpy()
    hips_ug = lib.fabs_Mm1.to_numpy() / ab.MAC_VALUE * vol
    sets = {
        "IMPROVE held-out sites (all-IMPROVE calibration)": (np.flatnonzero(test), d["f_one"].predict(X[test])),
        "IMPROVE ≥ 20 µg EC (calibration trained below 20 µg)": (np.flatnonzero(heavy), f_light.predict(X[heavy])),
    }
    rows, fits = [], []
    edges = np.array([0, 2, 5, 10, 20, 40, 80, 400])
    for name, (pos, pred) in sets.items():
        for ref, x in (("TOR", y[pos]), ("HIPS / 10", hips_ug[pos])):
            ok = np.isfinite(x) & (x > 0)
            st = ab.xy_stats(x[ok], pred[ok])
            fits.append({"population": name, "reference": ref, "n": int(ok.sum()), "slope": st["ols_slope"],
                         "intercept µg": st["ols_intercept"], "R2": st["R2"], "median FTIR / reference": np.median(pred[ok] / x[ok])})
            b = np.digitize(x[ok], edges) - 1
            for i in np.unique(b):
                q = b == i
                if q.sum() >= 5:
                    r = pred[ok][q] / x[ok][q]
                    rows.append({"population": f"{name.split(' (')[0]}", "reference": ref, "bin_lo": edges[i], "bin_hi": edges[i + 1],
                                 "loading": float(np.median(x[ok][q])), "ratio": float(np.median(r)),
                                 "lo": float(np.percentile(r, 25)), "hi": float(np.percentile(r, 75)), "n": int(q.sum())})
    # Addis on the same mass scale (HIPS EC on the filter), three calibrations
    for cal, p in (("one baseline-corrected (Figure 7)", d["pred_one"]), ("SPARTAN published", A.spartan.to_numpy())):
        x, pred = A.ec_ug.to_numpy(), p * A.vol.to_numpy()
        b = np.digitize(x, edges) - 1
        for i in np.unique(b):
            q = b == i
            if q.sum() >= 5:
                r = pred[q] / x[q]
                rows.append({"population": f"Addis, {cal.split(' (')[0]}", "reference": "HIPS / 10", "bin_lo": edges[i],
                             "bin_hi": edges[i + 1], "loading": float(np.median(x[q])), "ratio": float(np.median(r)),
                             "lo": float(np.percentile(r, 25)), "hi": float(np.percentile(r, 75)), "n": int(q.sum())})
    save(pd.DataFrame(rows), "loading_bins")
    fits = pd.DataFrame(fits)
    fits["k"] = [d["f_one"].k] * 2 + [f_light.k] * 2
    save(fits, "loading_fits")


# ------------------------------------------------------------------ 7. TOR on dark filters
def stage_dark(d):
    lib = d["lib"]
    ok = (lib.tor_ec_ugm3 > 0.02) & (lib.fabs_Mm1 > 0) & lib.tor_optr_ugm3.notna()
    L = lib.loc[ok].copy()
    L["op_zero"] = L.tor_optr_ugm3 <= 0
    L["hips_ug"] = L.fabs_Mm1 / ab.MAC_VALUE * L.tor_volume_m3
    L["ratio"] = L.tor_ec_ugm3 / (L.fabs_Mm1 / ab.MAC_VALUE)
    edges = np.array([0, 1, 2, 5, 10, 20, 40, 400])
    L["bin"] = np.digitize(L.hips_ug, edges) - 1
    rows = []
    for (b, z), g in L.groupby(["bin", "op_zero"]):
        if len(g) >= 10:
            rows.append({"group": "OP = 0" if z else "OP > 0", "bin_lo": edges[b], "bin_hi": edges[b + 1],
                         "loading": g.hips_ug.median(), "ratio": g.ratio.median(), "lo": g.ratio.quantile(.25),
                         "hi": g.ratio.quantile(.75), "n": len(g)})
    save(pd.DataFrame(rows), "dark_bins")
    share = [{"set": "IMPROVE library", "n": len(L), "OP = 0 share": L.op_zero.mean(), "median TOR EC / (HIPS/10)": L.ratio.median()}]
    rowpos = pd.Series(np.arange(len(lib)))
    for s, pos in [("97 look-alikes", d["L97"])] + [(f"{s.split()[0]} look-alikes", d["LSEA"][s]) for s in SEASONS]:
        g = L.loc[L.index.isin(rowpos.iloc[pos].to_numpy())]
        share.append({"set": s, "n": len(g), "OP = 0 share": g.op_zero.mean(), "median TOR EC / (HIPS/10)": g.ratio.median()})
    save(pd.DataFrame(share), "dark_share")


# ------------------------------------------------------------------ 8. VIP
def stage_vip(d):
    wn, wn_full = d["wn"], d["wn_full"]
    out = [pd.DataFrame({"model": "raw, SPARTAN-like (k 18)", "wn": wn_full, "vip": vip_scores(d["m_raw"])}),
           pd.DataFrame({"model": f"baseline-corrected, all IMPROVE (k {d['f_one'].k})", "wn": wn, "vip": vip_scores(d["f_one"].model)})]
    for s in SEASONS:
        out.append(pd.DataFrame({"model": f"baseline-corrected, {s.split()[0]} look-alikes",
                                 "wn": wn, "vip": d["vip_ocec"][f"spectral look-alikes (Figure 9)|{s}"]}))
        out.append(pd.DataFrame({"model": f"baseline-corrected, {s.split()[0]} lowest OC/EC",
                                 "wn": wn, "vip": d["vip_ocec"][f"lowest TOR OC/EC|{s}"]}))
    v = pd.concat(out)
    save(v, "vip_spectra")
    regions = {"high-wn 3600-4000": (4000, 3600), "O-H / N-H 3000-3600": (3600, 3000), "C-H 2800-3000": (3000, 2800),
               "2000-2800 (featureless)": (2800, 2000), "carbonyl 1650-1800": (1800, 1650), "1425-1650": (1650, 1425),
               "PTFE 1050-1250": (1250, 1050), "other below 1425": (1425, 0)}
    rows = []
    for model, g in v.groupby("model", sort=False):
        tot = (g.vip ** 2).sum()
        for r, (hi, lo) in regions.items():
            sel = (g.wn <= hi) & (g.wn > lo)
            if r == "other below 1425":
                sel &= ~((g.wn <= 1250) & (g.wn > 1050))
            rows.append({"model": model, "region": r, "share": float((g.vip[sel] ** 2).sum() / tot)})
    save(pd.DataFrame(rows), "vip_regions")


# ------------------------------------------------------------------ 9. urban IMPROVE
def stage_urban(d):
    lib, X, A = d["lib"], d["X"], d["A"]
    pos = np.flatnonzero(lib.Site.isin(URBAN).to_numpy())
    L = lib.iloc[pos]
    # six sites are too few for site-grouped folds: the AQRC app's protocol (interleaved 10-fold CV, k within
    # tolerance), with leave-one-site-out predictions at that k as the IMPROVE check
    f = ab.fit_cohort(X[pos], L.y, L.Site, mode="app")
    y, sites, loso = L.y.to_numpy(), L.Site.to_numpy(), np.full(len(pos), np.nan)
    for s in URBAN:
        m = sites == s
        loso[m] = PLSRegression(n_components=f.k, scale=False).fit(X[pos][~m], y[~m]).predict(X[pos][m]).ravel()
    tor = ab.xy_stats(y, loso)
    pred = f.predict(d["XA"]) / A.vol.to_numpy()
    rows = season_rows(d, pred, calibration=f"urban IMPROVE only ({len(pos)} filters, k {f.k})",
                       TOR_test_R2=tor.get("R2"))
    rows += season_rows(d, d["pred_one"], calibration=f"all IMPROVE (Figure 7, k {d['f_one'].k})")
    rows += season_rows(d, A.pred_season.to_numpy(), calibration="season-specific look-alikes (Figure 9)")
    save(pd.DataFrame(rows), "urban_fits")
    site = L.groupby("Site").agg(n=("y", "size"), ec_ug=("y", "median"), oc_ec=("tor_oc_ec", "median")).reset_index()
    site["name"] = site.Site.map(URBAN)
    save(site, "urban_sites")


# ------------------------------------------------------------------ 10. speciation
def load_fed(lib):
    want = lib[["Site", "date"]].drop_duplicates()
    years = sorted(want.date.dt.year.unique())
    frames = []
    for y in years:
        z = FED / f"IMPAER_{y}.txt.zip"
        if not z.exists():
            continue
        with zipfile.ZipFile(z) as zz:
            with zz.open(zz.namelist()[0]) as fh:
                t = pd.read_csv(fh, sep="|", usecols=["SiteCode", "FactDate", "ParamCode", "FactValue", "POC"])
        t = t[t.ParamCode.isin(FED_PARAMS) & t.POC.eq(1)]
        frames.append(t)
    t = pd.concat(frames)
    t["date"] = pd.to_datetime(t.FactDate)
    t.loc[t.FactValue <= -900, "FactValue"] = np.nan
    wide = t.pivot_table(index=["SiteCode", "date"], columns="ParamCode", values="FactValue", aggfunc="mean").rename(columns=FED_PARAMS)
    return lib[["Site", "date"]].merge(wide, left_on=["Site", "date"], right_index=True, how="left")


def stage_speciation(d):
    lib, A, chem = d["lib"], d["A"], d["chem"]
    fed = load_fed(lib)
    log(f"  FED speciation matched for {fed.sulfate.notna().sum()} of {len(lib)} library filters")
    species = ["EC", "OC", "sulfate", "nitrate", "soil"]
    addis = pd.DataFrame({"EC": A.hips, "OC": chem["OC (FTIR)"], "sulfate": chem.sulfate, "nitrate": chem.nitrate,
                          "soil": 2.20 * chem.Al + 2.49 * chem.Si + 1.63 * chem.Ca + 2.42 * chem.Fe + 1.94 * chem.Ti,
                          "PM2.5": chem["PM2.5"]})
    sets = [("Addis " + s.split()[0], addis[A.season.eq(s).to_numpy()]) for s in SEASONS]
    sets += [("IMPROVE, all library", fed)] + [(f"IMPROVE, {s.split()[0]} look-alikes", fed.iloc[d["LSEA"][s]]) for s in SEASONS]
    sets += [("IMPROVE, 97 look-alikes", fed.iloc[d["L97"]]), ("IMPROVE, urban sites", fed[lib.Site.isin(URBAN).to_numpy()])]
    rows = []
    for name, g in sets:
        g = g[g["PM2.5"] > 0]
        for sp in species:
            rows.append({"population": name, "species": sp, "median µg/m³": g[sp].median(),
                         "median share of PM2.5": (g[sp] / g["PM2.5"]).median(), "n": int(g[sp].notna().sum())})
        rows.append({"population": name, "species": "OC / EC", "median µg/m³": (g.OC / g.EC).median(),
                     "median share of PM2.5": np.nan, "n": int(g.OC.notna().sum())})
    save(pd.DataFrame(rows), "speciation")


# ------------------------------------------------------------------ 11. boundary-layer height
def weather_for(A):
    meta = json.loads((WEATHER / "filters.json").read_text())
    rows = json.loads((WEATHER / "filters/ETAD.json").read_text())["rows"]
    w = pd.DataFrame.from_dict(rows, orient="index", columns=meta["fields"])
    return A[["cartridge"]].merge(w, left_on="cartridge", right_index=True, how="left").drop(columns="cartridge")


def stage_blh(d):
    A, chem = d["A"], d["chem"]
    w = weather_for(A)
    blh = w["Boundary-layer height (ERA5)"].to_numpy()
    log(f"  BLH for {np.isfinite(blh).sum()} of {len(A)} Addis filters")
    rows = []
    for s in SEASONS:
        m = A.season.eq(s).to_numpy() & np.isfinite(blh)
        for name, v in (("HIPS EC", A.hips.to_numpy()), ("PM2.5", chem["PM2.5"].to_numpy()),
                        ("sulfate", chem.sulfate.to_numpy()), ("OC (FTIR)", chem["OC (FTIR)"].to_numpy())):
            rows.append({"season": s, "quantity": name, "n": int(m.sum()), "BLH m": float(np.median(blh[m])),
                         "median µg/m³": float(np.nanmedian(v[m])), "median mg/m² (× BLH)": float(np.nanmedian(v[m] * blh[m] / 1000))})
    out = pd.DataFrame(rows)
    for q, g in out.groupby("quantity"):
        out.loc[g.index, "relative to Dry"] = g["median µg/m³"] / g["median µg/m³"].iloc[0]
        out.loc[g.index, "relative to Dry (× BLH)"] = g["median mg/m² (× BLH)"] / g["median mg/m² (× BLH)"].iloc[0]
    save(out, "blh_by_season")
    err = A.pred_season.to_numpy() - A.hips.to_numpy()
    r = []
    for s in SEASONS + ["All Addis"]:
        m = (np.ones(len(A), bool) if s == "All Addis" else A.season.eq(s).to_numpy()) & np.isfinite(blh)
        r.append({"season": s, "n": int(m.sum()), "rho BLH vs HIPS EC": spearmanr(blh[m], A.hips[m]).correlation,
                  "partial rho BLH vs FTIR (season-specific)": partial_spearman(A.pred_season.to_numpy()[m], blh[m], A.hips.to_numpy()[m]),
                  "median error": float(np.median(err[m]))})
    save(pd.DataFrame(r), "blh_error")
    save(pd.DataFrame({"season": A.season, "blh": blh, "hips": A.hips, "pm25": chem["PM2.5"]}), "blh_points")


# ------------------------------------------------------------------ shared calibrations
def calibrations(d):
    lib, X, A = d["lib"], d["X"], d["A"]
    lot = lib.lot.eq(251).to_numpy()
    train = lib.split.eq("train").to_numpy()
    idx = np.flatnonzero(lot)
    d["f_one"] = f = ab.fit_cohort(X[idx], lib.y.to_numpy()[idx], lib.Site.to_numpy()[idx], train=train[idx])
    d["pred_one"] = f.predict(d["XA"]) / A.vol.to_numpy()
    tr = np.flatnonzero(lot & train & d["qc"])
    d["m_raw"] = PLSRegression(n_components=18, scale=False).fit(d["X_raw"][tr], lib.y.to_numpy()[tr])
    st = addis_fit(d, A.hips.to_numpy(), d["pred_one"])
    assert abs(st["intercept"] + 1.21) < 0.01 and abs(st["slope"] - 0.59) < 0.01, st   # poster Figure 7
    log(f"all-IMPROVE baseline-corrected calibration: k {f.k}, Addis {st['slope']:.2f}x {st['intercept']:+.2f}")


STAGES = {"ocec": stage_ocec, "deficit": stage_deficit, "dust": stage_dust, "mac": stage_mac, "highwn": stage_highwn,
          "loading": stage_loading, "dark": stage_dark, "vip": stage_vip, "urban": stage_urban,
          "speciation": stage_speciation, "blh": stage_blh}


def main(argv):
    only = set(argv) or set(STAGES)
    d = load()
    d["chem"] = spartan_chem(d["A"])
    calibrations(d)
    if "vip" in only:
        only.add("ocec")              # the VIP stage reuses the OC/EC stage's seasonal models
    if "highwn" in only:
        only.add("deficit")           # band areas
    for name, fn in STAGES.items():
        if name in only:
            log(f"[{name}]")
            fn(d)
    (OUT / "provenance.json").write_text(json.dumps({
        "script": "research/ftir_hips_chem/workflows/run_meeting_20261028.py", "stages": sorted(only),
        "addis_filters": len(d["A"]), "library_filters": len(d["lib"]), "urban_sites": URBAN,
        "generated": pd.Timestamp.now(tz="UTC").isoformat()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
