"""Export the 9 Oct 2026 Ann meeting follow-up for the gallery's spec-driven Meeting page.

Run from research/ftir_hips_chem with the repo environment:

    cd research/ftir_hips_chem && ../../.venv/bin/python ../../gallery/data/export_meeting_20261009.py

Reads the same saved rank-70 outputs as research/ftir_hips_chem/ann_followup_20261009.ipynb
and writes gallery/app/public/data/meetings/ann-20261009.json (see meeting_bundle.py).
The figures themselves are declared in gallery/app/src/meetings/ann-20261009.ts.
Nothing is refitted except the Addis band areas, exactly as in the notebook. The
answer text is built from the same tables the figures draw, so prose and charts agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))

from config import MAC_VALUE  # noqa: E402
from data_matching import load_filter_data  # noqa: E402
from meeting_bundle import write_bundle  # noqa: E402
from vibes_current import MIGRATION, load_context  # noqa: E402

T = MIGRATION / "meeting"
SEASONS = ["Dry (Oct-Feb)", "Belg (Mar-May)", "Kiremt (Jun-Sep)"]
GROUPS = ["All Addis"] + SEASONS
METHODS = {"AIRSpec": "Spline", "VIBES-full-r70": "VIBES", "VIBES-full-cut-r70": "VIBES cut"}
COLORS = {"Dry (Oct-Feb)": "#E67E22", "Belg (Mar-May)": "#27AE60", "Kiremt (Jun-Sep)": "#3498DB",
          "All Addis": "#e0a800", "Spline": "#2C6E9E", "VIBES": "#d95f0e", "VIBES cut": "#7A4FA3",
          "IMPROVE pool": "#8a8f98", "IMPROVE": "#8a8f98", "Addis": "#e0a800",
          "Addis Dry": "#E67E22", "Addis Belg": "#27AE60", "Addis Kiremt": "#3498DB"}

rep = pd.read_csv(T / "repeated_splits.csv")
stats = pd.read_csv(T / "panel_stats.csv")
refs = pd.read_csv(T / "improve_references.csv")
addis_eval = pd.read_csv(T / "addis_evaluation.csv")

# ---------------------------------------------------------------- 1. self-prediction
splits = rep.assign(baseline=rep.method.map(METHODS))[
    ["baseline", "method", "group", "repeat", "k", "TOR_test_R2", "ftir_vs_tor_slope", "ftir_vs_tor_intercept",
     "addis_slope", "addis_intercept", "addis_R2"]]

ft = stats.loc[stats.family.eq("allimprove") & stats.panel.eq("ftir_vs_fabs") & stats["mask"].eq("locked")].copy()
ft["model"] = ft.fit_id.str.split("|").str[2]
ft["baseline"] = ft.method.map(METHODS)
pick = pd.concat([
    ft.loc[ft.population.eq("pool_test")].assign(population="every held-out IMPROVE filter"),
    ft.loc[ft.population.eq("analog_test") & ft.group.eq("All Addis")].assign(population="held-out all-Addis analogs"),
    ft.loc[ft.population.eq("addis") & ft.group.eq("All Addis") & ft.eval_group.eq("All Addis")].assign(population="Addis (test set)"),
])
allimprove = pick[["model", "baseline", "k", "population", "deming_slope", "deming_intercept", "n"]].rename(
    columns={"deming_slope": "slope", "deming_intercept": "intercept"})

med = rep.groupby("method").median(numeric_only=True)
sp, vb = med.loc["AIRSpec"], med.loc["VIBES-full-r70"]
answer_self = f"""**Yes, IMPROVE predicts itself, under both baselines.** Across seasons the held-out IMPROVE
filters have a near-zero intercept (FTIR vs TOR {rep.ftir_vs_tor_intercept.min():.2f} to {rep.ftir_vs_tor_intercept.max():.2f} µg/m³
over all splits, median {rep.ftir_vs_tor_intercept.median():.2f}) and slopes of 0.5–0.9, the expected PLS shrinkage at low
loadings. VIBES predicts IMPROVE *better* than spline (median cross-validation TOR R² {vb.TOR_test_R2:.2f} vs
{sp.TOR_test_R2:.2f}), yet its Addis intercept is far worse (median {vb.addis_intercept:.2f} vs {sp.addis_intercept:.2f} µg/m³).

The calibration sets are internally consistent; the problem is in how the Addis spectra transfer. IMPROVE
validation cannot see it, which matches the 8 Oct finding that the Addis intercept grows with the number of PLS
factors while IMPROVE test R² stays flat."""

# ---------------------------------------------------------------- 2. window
seasonal = (rep.groupby(["group", "method"])[["addis_intercept", "addis_slope", "k"]].median()
            .unstack("method").reindex(GROUPS))
seasonal.columns = [f'{METHODS[m]} {c.replace("addis_", "")}' for c, m in seasonal.columns]
s = seasonal
closed = ((s["VIBES cut intercept"] - s["VIBES intercept"]) / (s["Spline intercept"] - s["VIBES intercept"])).clip(lower=0) * 100
al_long = ft.loc[ft.population.eq("addis") & (ft.group == ft.eval_group), ["model", "group", "baseline", "deming_intercept", "deming_slope", "k"]]
al = al_long.pivot_table(index=["model", "group"], columns="baseline", values=["deming_intercept", "k"], aggfunc="first")
a, kk = al["deming_intercept"], al["k"]
L, B = ("all-IMPROVE lot 251", "All Addis"), ("all-IMPROVE both lots (frozen Colab)", "All Addis")
answer_window = f"""**Mostly not the range.** For the seasonal calibrations, cutting VIBES to 4000–1425 cm⁻¹ closes only
{closed.min():.0f}–{closed.max():.0f} % of the gap to spline (Kiremt {s.loc['Kiremt (Jun-Sep)', 'VIBES intercept']:.2f} → {s.loc['Kiremt (Jun-Sep)', 'VIBES cut intercept']:.2f} µg/m³,
spline {s.loc['Kiremt (Jun-Sep)', 'Spline intercept']:.2f}; All Addis {s.loc['All Addis', 'VIBES intercept']:.2f} → {s.loc['All Addis', 'VIBES cut intercept']:.2f}).

For the all-IMPROVE models the cut points in opposite directions, and both follow the number of PLS factors:
lot 251 full k {kk.loc[L, 'VIBES']:.0f} gives {a.loc[L, 'VIBES']:.2f}, cut k {kk.loc[L, 'VIBES cut']:.0f} gives {a.loc[L, 'VIBES cut']:.2f};
both lots full k {kk.loc[B, 'VIBES']:.0f} gives {a.loc[B, 'VIBES']:.2f}, cut k {kk.loc[B, 'VIBES cut']:.0f} gives {a.loc[B, 'VIBES cut']:.2f}.
The meeting impression that "sensitivity looks like spline" came from the shape of one crossplot, not the
intercept. **The VIBES correction above 1425 cm⁻¹, and how many factors the model uses, carry the bias; the
fingerprint region adds a little.**"""

# ---------------------------------------------------------------- 3. mass per filter
imp = pd.DataFrame({"population": "IMPROVE", "conc": refs.fabs_Mm1 / MAC_VALUE,
                    "mass": refs.fabs_Mm1 / MAC_VALUE * refs.tor_volume_m3})
ad = pd.DataFrame({"population": "Addis", "conc": addis_eval.Fabs / MAC_VALUE,
                   "mass": addis_eval.Fabs / MAC_VALUE * addis_eval.Volume_m3})
loading = pd.concat([imp, ad]).dropna()
rows = []
for col, name in [("conc", "HIPS Fabs / MAC 10, µg/m³"), ("mass", "HIPS Fabs / MAC 10 × volume, µg per filter")]:
    i, d = imp[col].dropna(), ad[col].dropna()
    rows.append({"measure": name, "IMPROVE median": i.median(), "IMPROVE p95": i.quantile(.95), "IMPROVE p99": i.quantile(.99),
                 "Addis median": d.median(), "Addis median, IMPROVE percentile": 100 * (i < d.median()).mean(),
                 "% of Addis above IMPROVE p99": 100 * (d > i.quantile(.99)).mean()})
mass_tbl = pd.DataFrame(rows)
c, m = mass_tbl.iloc[0], mass_tbl.iloc[1]
answer_mass = f"""**Still an extrapolation in mass, but a far smaller one.** The Addis median is
{c['Addis median'] / c['IMPROVE median']:.0f}× the IMPROVE median in µg/m³ but {m['Addis median'] / m['IMPROVE median']:.0f}× in µg per filter, and
{c['Addis median'] / c['IMPROVE p99']:.1f}× vs {m['Addis median'] / m['IMPROVE p99']:.1f}× the IMPROVE 99th percentile. {m['% of Addis above IMPROVE p99']:.0f} % of Addis filters
are still above IMPROVE's 99th percentile in mass (all of them in concentration), so the calibration sees few
filters as heavily loaded as Addis."""

# ---------------------------------------------------------------- 4. the hump
ctx, _ = load_context()
meth = "VIBES-full-r70"
wn = np.asarray(ctx.wn_by[meth]); XA = np.asarray(ctx.XA[meth]); X = np.asarray(ctx.X[meth])
ch = (wn >= 2800) & (wn <= 3000)
norm = lambda Z: Z / np.abs(Z[:, ch].sum(1, keepdims=True))  # noqa: E731
win = (wn >= 550) & (wn <= 1900)
spec_rows = []
for name, Z in [("IMPROVE pool", X)] + [(f"Addis {g.split()[0]}", XA[ctx.addis.season.eq(g).to_numpy()]) for g in SEASONS]:
    q = np.quantile(norm(Z), [0.25, 0.5, 0.75], axis=0)
    spec_rows.append(pd.DataFrame({"series": name, "wn": wn[win], "median": q[1][win], "q25": q[0][win], "q75": q[2][win], "n": len(Z)}))
hump = pd.concat(spec_rows)

CHEM = {"ChemSpec_Sulfate_Ion_PM2.5": "sulfate", "ChemSpec_Ammonium_Ion_PM2.5": "ammonium",
        "ChemSpec_Silicon_PM2.5": "Si", "ChemSpec_Aluminum_PM2.5": "Al", "ChemSpec_Calcium_PM2.5": "Ca",
        "ChemSpec_Potassium_Ion_PM2.5": "K⁺", "ChemSpec_Nitrate_Ion_PM2.5": "nitrate"}
f = load_filter_data()
e = f.loc[f.Site.eq("ETAD") & f.Parameter.isin(CHEM)]
wide = e.pivot_table(index="FilterId", columns="Parameter", values="Concentration", aggfunc="mean").rename(columns=CHEM)
cart = ctx.addis.ExternalFilterId.astype(str).str.rsplit("-", n=1).str[0].reset_index(drop=True)
chem = pd.DataFrame({"cartridge": cart}).merge(wide, left_on="cartridge", right_index=True, how="left")
vol = ctx.addis.Volume_m3.to_numpy(float)
BANDS = {"SO₄ ν4 600–640": (600, 640), "Si-O 1000–1060": (1000, 1060), "SO₄ ν3 1080–1140": (1080, 1140),
         "PTFE 1150–1250": (1150, 1250), "NH₄⁺ 1400–1430": (1400, 1430), "CH 2800–3000": (2800, 3000)}
area = pd.DataFrame({b: XA[:, (wn >= lo) & (wn <= hi)].sum(1) / vol for b, (lo, hi) in BANDS.items()})
band_chem = pd.DataFrame([{"band": b, "species": cn,
                           "rho": spearmanr(*area[[b]].join(chem[[cn]]).dropna().T.values).statistic,
                           "n": int(area[[b]].join(chem[[cn]]).dropna().shape[0])}
                          for b in BANDS for cn in CHEM.values()])
rho = band_chem.pivot(index="band", columns="species", values="rho")
near = lambda Z, lo, hi: float(np.median(norm(Z), 0)[(wn >= lo) & (wn <= hi)].max())  # noqa: E731
sio = {g: near(XA[ctx.addis.season.eq(g).to_numpy()], 1000, 1060) for g in SEASONS}
sio_imp = near(X, 1000, 1060)
answer_hump = f"""**The broad hump near 1000 cm⁻¹ is silicate dust (Si-O), not sulfate.**
- Its maximum sits near **1035 cm⁻¹**, and relative to CH it is {sio['Belg (Mar-May)'] / sio_imp:.1f}× the IMPROVE level in Belg,
  {sio['Kiremt (Jun-Sep)'] / sio_imp:.1f}× in Kiremt and {sio['Dry (Oct-Feb)'] / sio_imp:.1f}× in Dry. Within Addis the 1000–1060 band tracks
  **Si (ρ {rho.loc['Si-O 1000–1060', 'Si']:.2f}) and Al ({rho.loc['Si-O 1000–1060', 'Al']:.2f})**.
- The **sulfate** band (~1100 cm⁻¹) tracks sulfate ({rho.loc['SO₄ ν3 1080–1140', 'sulfate']:.2f}) and ammonium ({rho.loc['SO₄ ν3 1080–1140', 'ammonium']:.2f}),
  but relative to CH it is no stronger in Addis than in IMPROVE ({near(XA, 1080, 1120):.4f} vs {near(X, 1080, 1120):.4f}).
- Separately, sharp **PTFE** features at ~1153 and ~1210 cm⁻¹ are {near(XA, 1140, 1170) / near(X, 1140, 1170):.1f}× IMPROVE: filter-substrate
  signal the VIBES background leaves on Addis filters.
- These features explain what the Addis spectra look like; they have not yet been shown to explain the EC bias
  (dropping the PTFE window from VIBES calibrations on 8 Oct did **not** reduce the Addis intercept)."""

intro = """Four questions from walking through the Meeting tab with Ann on 9 Oct. Everything reads saved outputs of
the completed rank-70 runs; nothing is refitted except the Addis band areas in the last section.
Calibration set = IMPROVE filters a model is built on; cross-validation = IMPROVE sites held out of it;
test set = the Addis filters. Deming at MAC 10. "Spline" = AIRSpec; "VIBES cut" = the same VIBES correction
predicting from 4000–1425 cm⁻¹ only."""

still_open = """- **Chemical speciation side by side** (Ann): needs an IMPROVE sulfate pull; the local cache has EC, OC, Fe, soil and mass but no sulfate.
- **Boundary-layer normalisation of Addis concentrations**: ERA5 boundary-layer heights are pulled for SPARTAN; IMPROVE is not yet pulled.
- **Chile sites** (Talca, Santiago; the IMPROVE Chile sites): after AAAR.
- **Show all held-out filters**: done (Follow up meeting › Three crossplots, *IMPROVE filters: All held-out IMPROVE*)."""

write_bundle(
    "ann-20261009",
    generated_from="research/ftir_hips_chem/ann_followup_20261009.ipynb via gallery/data/export_meeting_20261009.py",
    colors=COLORS,
    text={"intro": intro, "self": answer_self, "window": answer_window, "mass": answer_mass,
          "hump": answer_hump, "open": still_open},
    tables={"splits": splits, "allimprove": allimprove, "allimprove_addis": al_long.rename(
                columns={"deming_intercept": "intercept", "deming_slope": "slope"}),
            "loading": loading, "loading_summary": mass_tbl, "hump": hump, "band_chem": band_chem},
    digits={"hump": 6, "loading": 4},
)
