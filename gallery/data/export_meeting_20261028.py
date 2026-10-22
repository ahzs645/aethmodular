"""Export the 28 Oct 2026 meeting page (AAAR master to-do list, analysis items 1-11) for the gallery.

Run from research/ftir_hips_chem after the workflow:

    cd research/ftir_hips_chem
    ~/anaconda3/bin/python workflows/run_meeting_20261028.py
    ../../.venv/bin/python ../../gallery/data/export_meeting_20261028.py

Reads output/tables/meeting_20261028/ and writes gallery/app/public/data/meetings/meeting-20261028.json
(see meeting_bundle.py). The figures are declared in gallery/app/src/meetings/meeting-20261028.meeting.ts.
Nothing is refitted here; every number in the text is read from the tables the figures draw.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from meeting_bundle import write_bundle  # noqa: E402

T = ROOT / "research/ftir_hips_chem/output/tables/meeting_20261028"
SEASONS = ["Dry (Oct-Feb)", "Belg (Mar-May)", "Kiremt (Jun-Sep)"]
SHORT = {s: s.split()[0] for s in SEASONS}
C_SPEC, C_OCEC, C_BOTH, C_URBAN, C_ALL = "#8F8C84", "#B23327", "#7A4FA3", "#2C6E9E", "#22252A"
COLORS = {"Dry (Oct-Feb)": "#E67E22", "Belg (Mar-May)": "#27AE60", "Kiremt (Jun-Sep)": "#3498DB",
          "Dry": "#E67E22", "Belg": "#27AE60", "Kiremt": "#3498DB", "All Addis": "#C49442",
          "Addis Dry": "#E67E22", "Addis Belg": "#27AE60", "Addis Kiremt": "#3498DB",
          "IMPROVE (TOR, EC ≥ 1 µg)": "#8a8f98",
          "spectral look-alikes (Figure 9)": C_SPEC, "500 lowest TOR OC/EC": C_OCEC,
          "spectral top 2,000, lowest OC/EC": C_BOTH, "urban IMPROVE only": C_URBAN, "all IMPROVE (Figure 7)": C_ALL,
          "IMPROVE held-out sites (vs TOR)": "#8a8f98", "IMPROVE ≥ 20 µg EC, out of range (vs TOR)": "#5B5852",
          "Addis, baseline-corrected (vs HIPS)": "#C49442", "Addis, SPARTAN published (vs HIPS)": "#B23327",
          "OP > 0": "#8a8f98", "OP = 0": "#B23327",
          "raw, SPARTAN-like (k 18)": "#B23327", "baseline-corrected, all IMPROVE (k 8)": "#22252A",
          "Dry look-alikes": "#E67E22", "Belg look-alikes": "#27AE60", "Kiremt look-alikes": "#3498DB",
          "500 lowest OC/EC": C_OCEC}

r = lambda name: pd.read_csv(T / f"{name}.csv")
f2 = lambda v: f"{v:.2f}"
line = lambda s, i: f"{s:.2f}x {i:+.2f}"

# ------------------------------------------------------------------ 1 + 9. which IMPROVE filters calibrate
oc = r("ocec_fits")
oc = oc[oc.selection.isin(["spectral look-alikes (Figure 9)", "lowest TOR OC/EC", "spectral top 2,000, lowest OC/EC"])]
oc["calibration"] = oc.selection.replace({"lowest TOR OC/EC": "500 lowest TOR OC/EC"})
ur = r("urban_fits")
ur["calibration"] = ur.calibration.str.replace(r"urban IMPROVE only.*", "urban IMPROVE only", regex=True).str.replace(
    r"all IMPROVE \(Figure 7.*", "all IMPROVE (Figure 7)", regex=True)
ur = ur[ur.calibration.isin(["urban IMPROVE only", "all IMPROVE (Figure 7)"]) & ur.season.isin(SEASONS)]
cals = pd.concat([oc[["calibration", "season", "k", "TOR_test_R2", "slope", "intercept", "R2", "n"]],
                  ur[["calibration", "season", "TOR_test_R2", "slope", "intercept", "R2", "n"]]], ignore_index=True)
order = ["all IMPROVE (Figure 7)", "spectral look-alikes (Figure 9)", "500 lowest TOR OC/EC",
         "spectral top 2,000, lowest OC/EC", "urban IMPROVE only"]
cals["o"] = cals.calibration.map(order.index)
cals = cals.sort_values(["o", "season"], key=lambda s: s.map(SEASONS.index) if s.name == "season" else s).drop(columns="o")
cals["season"] = cals.season.map(SHORT)
g = cals.set_index(["calibration", "season"])
cohorts = r("ocec_cohorts")
dist = r("ocec_distribution")
imp = dist[dist.population.str.startswith("IMPROVE")].oc_ec
targets = cohorts.drop_duplicates("season").set_index("season")["Addis OC/EC target"]
low = cohorts[cohorts.selection.eq("lowest TOR OC/EC")].iloc[0]
urban_sites = r("urban_sites")
urban_r2 = r("urban_fits").dropna(subset=["TOR_test_R2"]).TOR_test_R2.iloc[0]

def cmp(cal, s):
    return line(g.loc[(cal, s), "slope"], g.loc[(cal, s), "intercept"]) + f" (R² {g.loc[(cal, s), 'R2']:.2f})"

text_ocec = f"""**Choosing the calibration filters by chemistry helps Belg most, Dry a little, and costs Kiremt some of its
Figure 9 gain.** Addis OC/EC (SPARTAN OC over HIPS EC) has a median of {targets[SEASONS[0]]:.2f} in Dry,
{targets[SEASONS[1]]:.2f} in Belg and {targets[SEASONS[2]]:.2f} in Kiremt. Only {(imp < 1).mean():.1%} of the IMPROVE filters
with at least 1 µg of EC have an OC/EC below 1 (median {imp.median():.1f}). So the filters "nearest Addis" are simply
the lowest-OC/EC filters in IMPROVE (median {low['TOR OC/EC median']:.2f}, from {int(low.sites)} sites), and one set
serves every season. The table below the chart has every fit.

""" + f"""* **Belg** moves from {cmp('spectral look-alikes (Figure 9)', 'Belg')} to {cmp('500 lowest TOR OC/EC', 'Belg')}: the
  positive intercept is gone and R² nearly triples.
* **Dry** improves only in slope ({g.loc[('spectral look-alikes (Figure 9)', 'Dry'), 'slope']:.2f} to
  {g.loc[('500 lowest TOR OC/EC', 'Dry'), 'slope']:.2f}). It stays the worst season under every selection.
* **Kiremt** does worse than with its own spectral look-alikes ({g.loc[('spectral look-alikes (Figure 9)', 'Kiremt'), 'slope']:.2f}
  to {g.loc[('500 lowest TOR OC/EC', 'Kiremt'), 'slope']:.2f}).
* The **six urban IMPROVE sites** ({', '.join(urban_sites.name)}; median OC/EC {urban_sites.oc_ec.median():.1f},
  {int(urban_sites.n.sum())} filters) give almost the same answer as the low-OC/EC set, and predict their own sites well
  (leave-one-site-out TOR R² {urban_r2:.2f}).
* Combining both (the 2,000 most spectrally similar, then the 500 with the lowest OC/EC) lands in between.

Both chemistry-based sets also predict held-out IMPROVE better than the spectral look-alikes (TOR R²
{g.loc[('500 lowest TOR OC/EC', 'Dry'), 'TOR_test_R2']:.2f} vs {g.loc[('spectral look-alikes (Figure 9)', 'Dry'), 'TOR_test_R2']:.2f}
to {g.loc[('spectral look-alikes (Figure 9)', 'Belg'), 'TOR_test_R2']:.2f}). No selection reaches slope 1 in Dry."""

# ------------------------------------------------------------------ 2. deficit
dc = r("deficit_correlations")
dci = dc[dc.band.eq("organic index")].copy()
dci["season"] = dci.season.map(lambda s: SHORT.get(s, s))
dp = r("deficit_points")
dp["season"] = dp.season.map(SHORT)
pr = dci.set_index(["calibration", "season"]).partial_rho
text_deficit = f"""**Yes, in Dry and Belg; not in Kiremt.** For each Addis filter, the organic signal is the C–H, O–H and
carbonyl band areas (baseline-corrected, per m³, averaged as percentiles). The test is the partial Spearman
correlation between FTIR EC and that signal, with HIPS EC held fixed, so a positive value means filters with more
organic material per unit EC are read higher.

* Season-specific calibrations (Figure 9): Dry {pr[('season-specific (Figure 9)', 'Dry')]:.2f},
  Belg {pr[('season-specific (Figure 9)', 'Belg')]:.2f}, Kiremt {pr[('season-specific (Figure 9)', 'Kiremt')]:.2f}.
* SPARTAN's published EC shows the same pattern (Dry {pr[('SPARTAN published', 'Dry')]:.2f},
  Belg {pr[('SPARTAN published', 'Belg')]:.2f}, Kiremt {pr[('SPARTAN published', 'Kiremt')]:.2f}).

So within Dry and Belg the filters with the least organic signal for their EC are the ones read lowest, as the
organic-deficit explanation predicts. In Kiremt, which already carries the most organic signal per unit EC
(21 Oct notebook), the error no longer depends on it. This is a correlation within 51–73 filters per season, not
a proof of mechanism."""

# ------------------------------------------------------------------ 3. dust
dust = r("dust_by_season")
dust["season"] = dust.season.map(SHORT)
dcor = r("dust_correlations")
dcor["season"] = dcor.season.map(SHORT)
dsi = dcor[dcor.element.eq("Si")]
ds = dust.set_index("season")
text_dust = f"""**Dust does not explain Dry and Belg.** Dust relative to EC is about the same in every season
(Si/EC {ds['Si / EC'].min():.2f}–{ds['Si / EC'].max():.2f}, Al/EC {ds['Al / EC'].min():.2f}–{ds['Al / EC'].max():.2f}). Kiremt
actually has the most dust in absolute terms (Si {ds.loc['Kiremt', 'Si']:.2f} µg/m³ vs {ds.loc['Dry', 'Si']:.2f} in Dry).
With HIPS EC held fixed, more dust goes with slightly *lower* FTIR EC in every season and calibration (partial ρ
{dsi.partial_rho.min():.2f} to {dsi.partial_rho.max():.2f}). The strongest link is SPARTAN's published EC in Kiremt
({dsi.set_index(['calibration', 'season']).partial_rho[('SPARTAN published', 'Kiremt')]:.2f}). Note that the
baseline-corrected spectra stop at 1425 cm⁻¹, so the Si-O band near 1035 cm⁻¹ is not in those calibrations at all.
Only the raw (SPARTAN) calibration sees it."""

# ------------------------------------------------------------------ 4. MAC
mac = r("mac_by_season")
mac["season"] = mac.season.map(SHORT)
ms = mac[mac.calibration.eq("season-specific (Figure 9)")].set_index("season")
text_mac = f"""**MAC does not explain it either.** HIPS EC is Fabs / 10 in every season. The IMPROVE look-alikes' own
ratio of Fabs to TOR EC is close to 10 (Dry {ms.loc['Dry', 'look-alike MAC (Fabs / TOR EC)']:.1f}, Belg
{ms.loc['Belg', 'look-alike MAC (Fabs / TOR EC)']:.1f}, Kiremt {ms.loc['Kiremt', 'look-alike MAC (Fabs / TOR EC)']:.1f} m²/g).
Using it instead of 10 moves the season-specific slopes only a little (Dry {ms.loc['Dry', 'slope at MAC 10']:.2f} to
{ms.loc['Dry', 'slope at look-alike MAC']:.2f}). Reaching slope 1 would need a MAC of
{ms.loc['Dry', 'MAC for slope 1']:.0f} in Dry and {ms.loc['Belg', 'MAC for slope 1']:.0f} in Belg, two to three times
any plausible value. Kiremt would need {ms.loc['Kiremt', 'MAC for slope 1']:.1f}. As expected, the intercepts do not
change with MAC."""

# ------------------------------------------------------------------ 5. high-wn residual
hc = r("highwn_correlations")
hc["season"] = hc.season.map(lambda s: SHORT.get(s, s))
hp = r("highwn_points")
hp["season"] = hp.season.map(SHORT)
h = hc.set_index(["against", "season"]).rho
text_highwn = f"""**The 3600–4000 cm⁻¹ residual tracks dust, not sulfate.** After baseline correction, Addis keeps some
absorbance above 3600 cm⁻¹. Within every season it follows SPARTAN Si closely (Spearman ρ Dry {h[('Si', 'Dry')]:.2f},
Belg {h[('Si', 'Belg')]:.2f}, Kiremt {h[('Si', 'Kiremt')]:.2f}). It follows sulfate only weakly
({h[('sulfate', 'Dry')]:.2f}, {h[('sulfate', 'Belg')]:.2f}, {h[('sulfate', 'Kiremt')]:.2f}), and goes down, not up, with
humidity. So it looks like scattering by mineral dust rather than the sulfate-related scattering Anwar (2025)
describes. Kiremt, where this residual is largest relative to its look-alikes (21 Oct notebook), is also the
dustiest season."""

# ------------------------------------------------------------------ 6. loading
lb = r("loading_bins")
lb["series"] = lb.population.map({
    "IMPROVE held-out sites": "IMPROVE held-out sites (vs TOR)", "IMPROVE ≥ 20 µg EC": "IMPROVE ≥ 20 µg EC, out of range (vs TOR)",
    "Addis, one baseline-corrected": "Addis, baseline-corrected (vs HIPS)", "Addis, SPARTAN published": "Addis, SPARTAN published (vs HIPS)"})
lb = lb[(lb.reference.eq("TOR") & lb.population.str.startswith("IMPROVE")) | lb.population.str.startswith("Addis")]
lf = r("loading_fits")
li = lb.set_index(["series", "bin_lo"])
heavy = lf[lf.population.str.startswith("IMPROVE ≥ 20") & lf.reference.eq("TOR")].iloc[0]
a20 = li.loc[("Addis, baseline-corrected (vs HIPS)", 20)]
i20 = li.loc[("IMPROVE ≥ 20 µg EC, out of range (vs TOR)", 20)]
i5 = li.loc[("IMPROVE held-out sites (vs TOR)", 5)]
text_loading = f"""**Loading explains part of it, not most.** PLS underpredicts heavy filters even within IMPROVE. On
held-out IMPROVE sites the median FTIR/TOR ratio falls from {i5.ratio:.2f} at 5–10 µg EC to
{li.loc[('IMPROVE held-out sites (vs TOR)', 20)].ratio:.2f} at 20–40 µg. A calibration trained only on filters below
20 µg, predicting the {int(heavy.n)} heavier ones, reads {i20.ratio:.2f} of TOR at 20–40 µg and
{li.loc[('IMPROVE ≥ 20 µg EC, out of range (vs TOR)', 40)].ratio:.2f} at 40–80 µg.

At the same 20–40 µg the baseline-corrected calibration reads Addis at {a20.ratio:.2f} of HIPS. So out-of-range
loading costs IMPROVE about a quarter of its EC, while Addis loses about two thirds; loading is part of the story,
and composition is the larger part. SPARTAN's raw calibration reads Addis at
{li.loc[('Addis, SPARTAN published (vs HIPS)', 20)].ratio:.2f} at 20–40 µg and
{li.loc[('Addis, SPARTAN published (vs HIPS)', 40)].ratio:.2f} at 40–80 µg. It overshoots as loading rises, which is
the steep slope in Figure 4."""

# ------------------------------------------------------------------ 7. TOR on dark filters
dk = r("dark_bins")
dsh = r("dark_share")
dki = dk.set_index(["group", "bin_lo"]).ratio
text_dark = f"""**No sign that TOR EC runs high on dark IMPROVE filters.** Where optical pyrolysis reads zero
({dsh.iloc[0]['OP = 0 share']:.1%} of the library), TOR EC relative to HIPS/10 is *lower*, not higher, than on other
filters at the same loading (2–5 µg: {dki[('OP = 0', 2)]:.2f} vs {dki[('OP > 0', 2)]:.2f}; 5–10 µg: {dki[('OP = 0', 5)]:.2f}
vs {dki[('OP > 0', 5)]:.2f}). The look-alikes are rarely OP-zero filters (none of the 97; {dsh.set_index('set').loc['Dry look-alikes', 'OP = 0 share']:.1%}
of Dry's, {dsh.set_index('set').loc['Kiremt look-alikes', 'OP = 0 share']:.1%} of Kiremt's). One thing to keep in mind: above 20 µg, TOR runs above
HIPS/10 in IMPROVE ({dki[('OP > 0', 20)]:.2f} at 20–40 µg, {dki[('OP > 0', 40)]:.2f} above 40). If Addis TOR behaved the
same way, Addis TOR EC would be higher than HIPS/10, and the FTIR shortfall against TOR would be larger, not smaller."""

# ------------------------------------------------------------------ 8. VIP
vip = r("vip_spectra")
vip = vip[~vip.model.str.contains("lowest OC/EC") | vip.model.str.contains("Dry")]
vip["model"] = vip.model.replace({"baseline-corrected, Dry lowest OC/EC": "500 lowest OC/EC",
                                  "baseline-corrected, Dry look-alikes": "Dry look-alikes",
                                  "baseline-corrected, Belg look-alikes": "Belg look-alikes",
                                  "baseline-corrected, Kiremt look-alikes": "Kiremt look-alikes"})
vr = r("vip_regions")
vr = vr[~vr.model.str.contains("lowest OC/EC") | vr.model.str.contains("Dry")]
vr["model"] = vr.model.replace({"baseline-corrected, Dry lowest OC/EC": "500 lowest OC/EC",
                                "baseline-corrected, Dry look-alikes": "Dry look-alikes",
                                "baseline-corrected, Belg look-alikes": "Belg look-alikes",
                                "baseline-corrected, Kiremt look-alikes": "Kiremt look-alikes"})
v = vr.set_index(["model", "region"]).share
RAW, ONE = "raw, SPARTAN-like (k 18)", "baseline-corrected, all IMPROVE (k 8)"
below = v[(RAW, "PTFE 1050-1250")] + v[(RAW, "other below 1425")]
text_vip = f"""**The raw (SPARTAN-like) calibration reads EC from different parts of the spectrum than the
baseline-corrected ones.** VIP² shares by region:

* Raw: {below:.0%} below 1425 cm⁻¹ (PTFE {v[(RAW, 'PTFE 1050-1250')]:.0%}), {v[(RAW, 'high-wn 3600-4000')]:.0%} at
  3600–4000, and only {v[(RAW, 'C-H 2800-3000')]:.0%} on C–H and {v[(RAW, 'O-H / N-H 3000-3600')]:.0%} on O–H.
* Baseline-corrected, all IMPROVE: {v[(ONE, 'O-H / N-H 3000-3600')]:.0%} O–H, {v[(ONE, 'C-H 2800-3000')]:.0%} C–H,
  {v[(ONE, 'carbonyl 1650-1800')]:.0%} carbonyl.

Section 5 found that the 3600–4000 cm⁻¹ region carries Addis dust scattering. So the raw calibration leans on the
regions where Addis differs from IMPROVE in background and dust, not in EC. That is a plausible reason it overshoots
on heavy Addis filters (Section 6). The baseline-corrected calibrations lean on the organic bands, which Addis has
little of per unit EC. Kiremt's look-alike calibration puts the most weight on C–H
({v[('Kiremt look-alikes', 'C-H 2800-3000')]:.0%}); the low-OC/EC set puts the most on carbonyl
({v[('500 lowest OC/EC', 'carbonyl 1650-1800')]:.0%})."""

# ------------------------------------------------------------------ 10. speciation
sp = r("speciation")
share = sp[sp.species.ne("OC / EC")].copy()
ocr = sp[sp.species.eq("OC / EC")][["population", "median µg/m³"]].rename(columns={"median µg/m³": "OC / EC"})
pops = ["Addis Dry", "Addis Belg", "Addis Kiremt", "IMPROVE, Dry look-alikes", "IMPROVE, Belg look-alikes",
        "IMPROVE, Kiremt look-alikes", "IMPROVE, 97 look-alikes", "IMPROVE, urban sites", "IMPROVE, all library"]
share = share[share.population.isin(pops)]
share["o"] = share.population.map(pops.index)
share = share.sort_values("o").drop(columns="o")
spec_tab = share.pivot(index="population", columns="species", values="median share of PM2.5").reindex(pops).reset_index()
spec_tab = spec_tab.merge(ocr, on="population")
st_ = spec_tab.set_index("population")
text_spec = f"""**Addis is EC-rich and organic-poor next to every IMPROVE group, including its own look-alikes.** Median
shares of PM2.5:

* EC is {st_.loc['Addis Kiremt', 'EC']:.0%}–{st_.loc['Addis Dry', 'EC']:.0%} at Addis against
  {st_.loc['IMPROVE, Dry look-alikes', 'EC']:.0%}–{st_.loc['IMPROVE, Kiremt look-alikes', 'EC']:.0%} in the look-alikes.
* OC is {st_.loc['Addis Dry', 'OC']:.0%}–{st_.loc['Addis Kiremt', 'OC']:.0%} against
  {st_.loc['IMPROVE, Dry look-alikes', 'OC']:.0%}–{st_.loc['IMPROVE, Kiremt look-alikes', 'OC']:.0%}.
* OC/EC is {st_.loc['Addis Dry', 'OC / EC']:.2f}–{st_.loc['Addis Kiremt', 'OC / EC']:.2f} against
  {st_.loc['IMPROVE, Kiremt look-alikes', 'OC / EC']:.1f}–{st_.loc['IMPROVE, Dry look-alikes', 'OC / EC']:.1f}.
* Soil is a bigger share at Addis ({st_.loc['Addis Kiremt', 'soil']:.0%}–{st_.loc['Addis Dry', 'soil']:.0%}).
* Sulfate is a smaller share, except that Dry's look-alikes are sulfate-rich ({st_.loc['IMPROVE, Dry look-alikes', 'sulfate']:.0%}).
* The urban IMPROVE sites are the closest IMPROVE group in OC/EC ({st_.loc['IMPROVE, urban sites', 'OC / EC']:.1f}) and in
  EC share ({st_.loc['IMPROVE, urban sites', 'EC']:.0%}).

Caveat: Addis OC is SPARTAN's FTIR OC and Addis EC is HIPS/10; IMPROVE OC and EC are TOR. Soil uses the IMPROVE
formula on SPARTAN elements."""

# ------------------------------------------------------------------ 11. boundary layer
bl = r("blh_by_season")
bl["season"] = bl.season.map(SHORT)
be = r("blh_error")
be["season"] = be.season.map(lambda s: SHORT.get(s, s))
b = bl.set_index(["quantity", "season"])
text_blh = f"""**Kiremt's higher concentrations are mostly a shallower boundary layer.** The ERA5 boundary-layer height,
averaged over each filter's sampling window, has a median of {b.loc[('HIPS EC', 'Dry'), 'BLH m']:.0f} m in Dry,
{b.loc[('HIPS EC', 'Belg'), 'BLH m']:.0f} m in Belg and {b.loc[('HIPS EC', 'Kiremt'), 'BLH m']:.0f} m in Kiremt.
Concentrations relative to Dry, before and after multiplying by the boundary-layer height:

* HIPS EC: {b.loc[('HIPS EC', 'Kiremt'), 'relative to Dry']:.2f} becomes {b.loc[('HIPS EC', 'Kiremt'), 'relative to Dry (× BLH)']:.2f}.
* PM2.5: {b.loc[('PM2.5', 'Kiremt'), 'relative to Dry']:.2f} becomes {b.loc[('PM2.5', 'Kiremt'), 'relative to Dry (× BLH)']:.2f}.
* OC: {b.loc[('OC (FTIR)', 'Kiremt'), 'relative to Dry']:.2f} becomes {b.loc[('OC (FTIR)', 'Kiremt'), 'relative to Dry (× BLH)']:.2f}.

So per column of air there is no more EC in Kiremt than in Dry. OC stays somewhat higher, which fits more smoke
in Kiremt. The boundary-layer height says little about the calibration error: with HIPS EC held fixed, its partial
correlation with the season-specific FTIR EC is {be.set_index('season').loc['Dry', 'partial rho BLH vs FTIR (season-specific)']:.2f}
in Dry and near zero in Belg and Kiremt. (Concentration × boundary-layer height is a rough column measure: one
surface site, and the height averaged over each filter's window.)"""

# ------------------------------------------------------------------ summary
intro = f"""Work on the analysis items of the AAAR master to-do list (email to Ann, 22 Oct). Unless stated otherwise:
the 189 Addis filters on the poster; spline baseline-corrected spectra (1425–4000 cm⁻¹); Deming fits of FTIR EC
against HIPS EC (Fabs / MAC 10) with the poster's error ratio. The season-specific calibrations are Figure 9's,
refitted here and checked to reproduce it exactly. Code: `research/ftir_hips_chem/workflows/run_meeting_20261028.py`.

**What we found**

1. **Choose by chemistry, not spectral shape.** The 500 IMPROVE filters with the lowest TOR OC/EC fix most of Belg
   ({cmp('spectral look-alikes (Figure 9)', 'Belg')} becomes {cmp('500 lowest TOR OC/EC', 'Belg')}) and help Dry a
   little. The six urban IMPROVE sites give nearly the same result. Kiremt does worse than with its spectral
   look-alikes, and no selection gets Dry near slope 1.
2. **Within Dry and Belg, the filters with the least organic signal are read lowest** (partial ρ
   {pr[('season-specific (Figure 9)', 'Dry')]:.2f} and {pr[('season-specific (Figure 9)', 'Belg')]:.2f}). Kiremt shows no
   such link ({pr[('season-specific (Figure 9)', 'Kiremt')]:.2f}). This supports the organic-deficit explanation.
3. **It is not dust, MAC, or TOR on dark filters.** Dust/EC is the same in every season. Slope 1 would need a MAC of
   {ms.loc['Dry', 'MAC for slope 1']:.0f}–{ms.loc['Belg', 'MAC for slope 1']:.0f} in Dry and Belg. OP-zero IMPROVE
   filters have lower, not higher, TOR/HIPS.
4. **Loading explains part of it.** Out-of-range IMPROVE filters read about {i20.ratio:.2f} of TOR, while Addis reads
   {a20.ratio:.2f} of HIPS at the same loading.
5. **The raw (SPARTAN) calibration leans on background and dust regions** ({below:.0%} of VIP² below 1425 cm⁻¹,
   {v[(RAW, 'high-wn 3600-4000')]:.0%} at 3600–4000). The 3600–4000 cm⁻¹ residual at Addis tracks dust (Si), not sulfate.
6. **Kiremt's higher EC is mostly a shallower boundary layer.** Per column of air, Kiremt has no more EC than Dry
   ({b.loc[('HIPS EC', 'Kiremt'), 'relative to Dry (× BLH)']:.2f} of Dry)."""

next_steps = f"""* **Build calibration sets from chemistry and spectra together.** For example, low OC/EC and urban sites
  as the base, with the heaviest filters added, then read the result on Dry with repeated site splits (not one split).
* **Test whether the organic deficit is the cause, not just a correlation.** Add organic absorbance to the IMPROVE
  filters, or take it away from Addis, and see whether the Addis prediction moves as expected.
* **Pull the raw calibration off the background regions.** Refit it on raw spectra without 3600–4000 cm⁻¹ and
  without the region below 1425, and see whether the Addis slope drops toward 1 without baseline correction.
* **Loading:** a calibration that includes the heavy IMPROVE filters (≥ 20 µg), or weights them up.
* **Before relying on the OC/EC result:** Addis OC is SPARTAN's FTIR OC. Check it against the Addis city TOR
  filters when they arrive."""

tables = {
    "cals": cals,
    "ocec_dist": dist[dist.oc_ec > 0].assign(oc_ec=lambda d: d.oc_ec.clip(upper=200)),
    "deficit": dci[["calibration", "season", "partial_rho"]],
    "deficit_points": dp[dp.calibration.eq("season-specific (Figure 9)")],
    "dust": dust,
    "dust_rho": dsi[["calibration", "season", "partial_rho"]],
    "mac": mac,
    "highwn_rho": hc,
    "highwn_points": hp,
    "loading_bins": lb[["series", "loading", "ratio", "lo", "hi", "n"]],
    "loading_fits": lf,
    "dark_bins": dk,
    "dark_share": dsh,
    "vip": vip,
    "vip_regions": vr,
    "speciation": share,
    "speciation_table": spec_tab,
    "blh": bl,
    "blh_error": be,
}
text = {"intro": intro, "ocec": text_ocec, "deficit": text_deficit, "dust": text_dust, "mac": text_mac,
        "highwn": text_highwn, "loading": text_loading, "dark": text_dark, "vip": text_vip, "speciation": text_spec,
        "blh": text_blh, "next": next_steps}
# the gallery's Markdown (lib/markdown.tsx) knows "- " bullets only
text = {k: re.sub(r"(?m)^(\s*)(?:\*|\d+\.) (?=\S)", r"\1- ", v) for k, v in text.items()}
write_bundle("meeting-20261028", generated_from="research/ftir_hips_chem/workflows/run_meeting_20261028.py",
             tables=tables, text=text, colors=COLORS, digits={"vip": 3, "ocec_dist": 3})
