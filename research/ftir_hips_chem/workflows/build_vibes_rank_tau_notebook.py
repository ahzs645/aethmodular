"""Build vibes_rank_tau_ec_20261116.ipynb from the outputs of run_vibes_rank_tau_ec_20261116.py (no fitting here)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(True)}


cells = [
    md("""
# VIBES rank and tau: how far fixed-model EC moves (16 Nov 2026)

Every network scan saved under the 87-blank background was refitted to convergence at ranks 10, 20, 40 and 70 and
tau 0.05, 0.1 and 0.2 by `workflows/run_vibes_rank_tau_ec_20261116.py`. EC is read with the frozen k 4 IMPROVE model,
which was trained on spectra corrected at rank 70 and tau 0.1. Nothing is refitted here.

**What this can and cannot show.** A change in EC here is the cost of correcting a spectrum one way and calibrating
another way. It does not say which setting gives the best EC. That needs the calibration refitted at each setting.
EC is in µg per filter; the bundle has no sampled volumes.

**What it shows** (4,376 scans, all certified at all 12 settings; 3,626 are PM2.5).

- **Rank 40 gives the same EC as rank 70.** Median change 0.01 %, largest 0.03 µg.
- **Rank 20 is close, rank 10 is not.** Rank 20 moves EC by a median 0.5 %. Rank 10 moves it by 3 %, and 17 % of scans move by more than 10 %.
- **tau moves every scan a little.** Median 1.2 % at tau 0.05 and 2.0 % at tau 0.2; 3 % and 6 % of scans move by more than 10 %.
- **For scale, the solver switch** moves the median scan by 0.003 %, and the 97 early stops by a median 5 %.
- **Neither check that needs no calibration picks another setting.** Repeat scans agree about as well at every setting.
  Field blanks are tightest at rank 70 and spread out as the rank falls.
- **The fitted noise level follows tau.** Its median roughly doubles each time tau doubles, so a noise level is only
  comparable between fits made at one tau.
- **With the calibration rebuilt at each tau (section 7), no tau is better at every k.** tau 0.1 stays.
- **One QC-failed training spectrum moves SPARTAN EC from the k 4 model by a median 17 % (section 8),** while IMPROVE
  scores stay the same.
- **Training without the heaviest filters steadies the model but does not pin EC down (section 9):** the cut-off moves
  EC by 3 to 30 %, and about 7 % of spread is left when IMPROVE sites are resampled.
"""),
    code("""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
AREA = Path(os.environ.get("AETH_AREA", "."))
R = AREA / "output/tables/vibes_rank_tau_ec_20261116"
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#3a3a38"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a95", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": .6, "lines.linewidth": 1.2})
pd.set_option("display.width", 220, "display.max_columns", 30)

scans = pd.read_csv(R / "scans.csv", dtype={"analysis_id": str})
fits = {}
for f in sorted(R.glob("fit_rank*_tau*.npz")):
    z = np.load(f)
    fits[(int(z["rank"]), float(z["tau"]))] = {k: z[k] for k in z.files}
REF = (70, 0.1)
ref = fits[REF]
pm = scans.filter_type.eq("PM2.5").to_numpy()
blank = scans.filter_type.eq("FB").to_numpy()
addis = pm & scans.code.eq("ETAD").to_numpy()
early = scans.premature_stop.to_numpy(bool)
print(len(scans), "scans:", int(pm.sum()), "PM2.5,", int(addis.sum()), "of them Addis,", int(blank.sum()), "field blanks")
pd.DataFrame([{"rank": k[0], "tau": k[1], "certified": int(f["ok"].sum()), "of": len(f["ok"])} for k, f in sorted(fits.items(), reverse=True)])
"""),
    md("""
## 1. PM2.5 scans against the network setting (rank 70, tau 0.1)

Percent changes use scans whose EC at the network setting is at least 1 µg, so a near-zero EC does not set the scale.
The slope and intercept are an ordinary fit of EC at the setting on EC at the network setting.
"""),
    code("""
def compare(key, mask):
    f = fits[key]
    m = mask & f["ok"] & ref["ok"]
    base, d = ref["ec"][m], f["ec"][m] - ref["ec"][m]
    loaded = base >= 1
    slope, intercept = np.polyfit(base, f["ec"][m], 1)
    return {"rank": key[0], "tau": key[1], "scans": int(m.sum()),
            "median change (µg)": np.median(d), "median |change| (µg)": np.median(np.abs(d)),
            "95th pct |change| (µg)": np.quantile(np.abs(d), .95), "largest |change| (µg)": np.abs(d).max(),
            "median |change| (%)": 100 * np.median(np.abs(d[loaded]) / base[loaded]),
            "moved > 10 % (share)": 100 * np.mean(np.abs(d[loaded]) / base[loaded] > .1),
            "slope": slope, "intercept (µg)": intercept, "r": np.corrcoef(base, f["ec"][m])[0, 1],
            "baseline change (% of corrected RMS)": 100 * np.median(f["baseline_change_rms"][m] / ref["corrected_rms"][m])}

keys = sorted(fits, key=lambda k: (-k[0], abs(k[1] - 0.1), k[1]))
print("Median EC at the network setting:", round(float(np.median(ref["ec"][pm & ref["ok"]])), 2), "µg per filter")
network = pd.DataFrame([compare(k, pm) for k in keys if k != REF]).round(3)
network
"""),
    code("""
grid = network.pivot(index="rank", columns="tau", values="median |change| (%)").reindex([70, 40, 20, 10])
grid.loc[70, 0.1] = 0.0
grid = grid[sorted(grid.columns)]
fig, ax = plt.subplots(figsize=(3.6, 3.0))
im = ax.imshow(grid.to_numpy(), cmap="Blues", vmin=0)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid.to_numpy()[i, j]
        ax.text(j, i, "network" if (grid.index[i], grid.columns[j]) == REF else f"{v:.1f} %", ha="center", va="center",
                color="white" if v > grid.to_numpy().max() * .55 else INK)
ax.set_xticks(range(grid.shape[1]), [f"{t:g}" for t in grid.columns]); ax.set_yticks(range(grid.shape[0]), grid.index)
ax.set(xlabel="tau", ylabel="Rank", title="Median |EC change|, PM2.5 scans"); ax.grid(False)
fig.tight_layout(); plt.show()
"""),
    code("""
panels = [((70, 0.05), "tau 0.05, rank 70"), ((70, 0.2), "tau 0.2, rank 70"), ((40, 0.1), "rank 40, tau 0.1"), ((10, 0.1), "rank 10, tau 0.1")]
fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1))
for ax, (key, title) in zip(axes, panels):
    m = pm & fits[key]["ok"] & ref["ok"]
    x, y = ref["ec"][m], fits[key]["ec"][m]
    lo, hi = np.quantile(np.r_[x, y], [.001, .999])
    ax.plot([lo, hi], [lo, hi], color="#9a9a95", lw=.8)
    ax.scatter(x, y, s=4, color=BLUE, alpha=.35, linewidth=0)
    a_ = addis[m]
    ax.scatter(x[a_], y[a_], s=6, color=ORANGE, alpha=.8, linewidth=0, label="Addis")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), title=title, xlabel="EC at rank 70, tau 0.1 (µg)")
    ax.set_box_aspect(1)
axes[0].set_ylabel("EC at this setting (µg)"); axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout(); plt.show()
"""),
    md("## 2. Addis PM2.5 scans"),
    code("""
pd.DataFrame([compare(k, addis) for k in keys if k != REF]).round(3)
"""),
    md("""
## 3. Field blanks

A field blank carries no deposit, so its EC should be near zero at any setting. The spread is the more useful number:
the fixed model's intercept belongs to rank 70 and tau 0.1.
"""),
    code("""
def blanks(key):
    f = fits[key]
    v = f["ec"][blank & f["ok"]]
    return {"rank": key[0], "tau": key[1], "blank scans": len(v), "median EC (µg)": np.median(v),
            "interquartile range (µg)": np.subtract(*np.quantile(v, [.75, .25])), "5th pct": np.quantile(v, .05), "95th pct": np.quantile(v, .95)}
pd.DataFrame([blanks(k) for k in keys]).round(3)
"""),
    md("""
## 4. Repeat scans of one filter

Filters with two or more PM2.5 scans, first two scans. If one setting were clearly better, repeat scans should agree
more closely there. The table leaves out the 12 filters the lab review marks as having one weak scan
(`gallery/app/src/lib/nativeScanReview.ts`).
"""),
    code("""
REVIEWED = {"CHTS-0777-8", "CHTS-0814-5", "CHTS-0830-5", "ETAD-0037-5", "ETAD-0139-3", "INDH-0209-1", "INDH-0265-1",
            "KRSE-0136-8", "PRFJ-0073-1", "ZAJB-0121-1", "ZAJB-0283-3", "ZAJB-0286-6"}
index = scans[pm & ~scans.filter_id.isin(REVIEWED)].groupby("filter_id").indices
pairs = np.array([scans.index[pm & ~scans.filter_id.isin(REVIEWED)][v[:2]] for v in index.values() if len(v) >= 2])
def repeats(key):
    f = fits[key]
    ok = f["ok"][pairs].all(1)
    a, b = f["ec"][pairs[ok, 0]], f["ec"][pairs[ok, 1]]
    return {"rank": key[0], "tau": key[1], "filters": int(ok.sum()), "median |scan 1 - scan 2| (µg)": np.median(np.abs(a - b)),
            "90th pct (µg)": np.quantile(np.abs(a - b), .9), "r": np.corrcoef(a, b)[0, 1]}
pd.DataFrame([repeats(k) for k in keys]).round(3)
"""),
    md("""
## 5. Where the corrected spectrum moves

Mean corrected absorbance in three windows, median over PM2.5 scans. The 2000-2500 cm⁻¹ window has no aerosol bands,
so its level shows how far above the baseline the spectrum is left to sit.
"""),
    code("""
names = [n for n in ref if n.startswith("mean ")]
def levels(key):
    f = fits[key]
    m = pm & f["ok"]
    return {"rank": key[0], "tau": key[1], **{n[5:] + " (×1e-3)": 1e3 * np.median(f[n][m]) for n in names},
            "noise level, median (×1e-4)": 1e4 * np.median(f["sigma"][m])}
pd.DataFrame([levels(k) for k in keys]).round(3)
"""),
    md("## 6. Early stops of the October run, for scale\nThe same comparison for the solver switch itself: October baseline against converged, both at rank 70 and tau 0.1."),
    code("""
d = pd.read_csv(AREA / "output/tables/network_baselines_v2_20261103/vibes_fit_diagnostics.csv")
d = d[d.vibes_model.eq("b87") & d.filter_type.eq("PM2.5")]
ch = (d.ec_newton_baseline_frozen_model_ug - d.ec_saved_baseline_frozen_model_ug)
loaded = d.ec_saved_baseline_frozen_model_ug >= 1
pd.DataFrame([{"group": name, "scans": int(m.sum()), "median |change| (µg)": ch[m].abs().median(),
               "median |change| (%)": 100 * (ch[m & loaded].abs() / d.ec_saved_baseline_frozen_model_ug[m & loaded]).median(),
               "largest |change| (µg)": ch[m].abs().max()}
              for name, m in [("All PM2.5", d.premature_stop | True), ("Not an early stop", ~d.premature_stop), ("Early stop", d.premature_stop),
                              ("Addis early stop", d.premature_stop & d.code.eq("ETAD"))]]).round(4)
"""),
    md("""
## 7. Which tau is best once the calibration matches it

Sections 1 to 6 read EC with a model trained at tau 0.1. Here the IMPROVE calibration is rebuilt at each tau
(`workflows/run_vibes_tau_calibration_20261116.py`, rank 70, 12,393 IMPROVE and 233 Addis spectra, all certified), so each
tau is scored with its own model. At tau 0.1 the rebuild returns the 3 Nov calibration: k 4, test RMSE 3.850.

**What it shows.**

- **No tau wins at every k.** Scores are compared at fixed k, with sites resampled to see which gaps are larger than chance.
- **tau 0.05 is never better than tau 0.1.** It is worse in 3 of 12 comparisons and no different in the other 9.
- **tau 0.2 is better at low k and worse at high k.** At k 4 and k 8 its median error is lower (by 0.05 and 0.12 µg on the
  held-out split). At k 11 and k 18 its RMSE is higher (by 0.15 and 0.09 µg). At the frozen k 4 its RMSE is no different.
- **The choice of k matters far more than tau.** Going from k 4 to k 11 lowers the held-out RMSE from about 3.87 to
  about 3.2 µg at every tau. The largest gap between tau values at one k is 0.15 µg.
- **The selection rule is not stable.** It picks k 4 at tau 0.1 and 0.2 and k 11 at tau 0.05, and it changes again
  (4, 8, 3) when one QC-failed training spectrum is left out. A tau should not be judged at the k the rule picks.
- **Addis against HIPS does not separate them either.** The slope follows k, from about 0.43 at k 4 to about 1.8 at k 18,
  and the three tau values stay within about 0.1 of each other.
"""),
    code("""
C = AREA / "output/tables/vibes_tau_calibration_20261116"
TAU_COLOR = {0.05: BLUE, 0.1: INK, 0.2: ORANGE}
scores = pd.read_csv(C / "calibration_scores.csv")
rule = scores[scores.selected].pivot(index="training", columns="tau", values="k").rename(index={"all": "whole train split (frozen procedure)", "qc": "without the QC-failed spectrum"})
print("k picked by the selection rule")
rule
"""),
    md("Held-out IMPROVE test split (2,327 filters), each tau with its own model, at fixed k. Train split without the QC-failed spectrum."),
    code("""
q = scores[scores.training.eq("qc") & scores.k.isin([4, 8, 11, 12, 18])]
q.pivot(index="k", columns="tau", values=["RMSE", "MAE", "median |error|", "bias"]).round(3)
"""),
    code("""
curves = pd.read_csv(C / "calibration_cv_curves.csv")
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.1))
for tau, color in TAU_COLOR.items():
    c = curves[curves.tau.eq(tau) & curves.training.eq("qc")]
    axes[0].plot(c.n_components, c.rmse_mean, color=color, label=f"tau {tau:g}")
    s_ = scores[scores.tau.eq(tau) & scores.training.eq("qc")]
    axes[1].plot(s_.k, s_.RMSE, color=color)
t = pd.read_csv(C / "transfer_scores.csv")
for tau, color in TAU_COLOR.items():
    axes[2].plot(t[t.tau.eq(tau)].k, t[t.tau.eq(tau)]["Addis slope vs HIPS"], color=color)
axes[2].axhline(1, color="#9a9a95", lw=.8)
axes[0].set(xlabel="PLS components (k)", ylabel="RMSE (µg)", title="Site-grouped CV on the train split", xlim=(1, 25)); axes[0].legend(frameon=False)
axes[1].set(xlabel="PLS components (k)", ylabel="RMSE (µg)", title="Held-out test split", xlim=(1, 25))
axes[2].set(xlabel="PLS components (k)", ylabel="Slope of FTIR EC on HIPS", title="Addis against HIPS", xlim=(1, 25))
fig.tight_layout(); plt.show()
"""),
    md("""
Is any difference from tau 0.1 larger than chance? Sites are resampled with replacement 2,000 times and the same
filters are compared under both tau values. A negative number means the tau has the smaller error.
"""),
    code("""
boot = pd.read_csv(C / "bootstrap.csv").rename(columns={"difference": "difference from tau 0.1 (µg)", "lo": "5 %", "hi": "95 %", "excludes_zero": "interval excludes zero"})
boot.round(3)
"""),
    md("Out-of-site scoring over all eligible IMPROVE filters (site-grouped 5-fold) and the Addis prediction against HIPS, as in the 2 Nov transfer diagnosis."),
    code("""
t[t.k.isin([4, 5, 8, 12, 18])].pivot(index="k", columns="tau", values=["IMPROVE out-of-site median |error| (µg)", "Addis slope vs HIPS", "Addis intercept vs HIPS (µg)", "Addis R2"]).round(3)
"""),
    md("""
**A side finding.** The train split holds one spectrum the decision bundle marks as failing raw QC (IMPROVE filter
2008808, TONT1; its corrected absorbance spans 3.7). The 3 Nov transfer table was
made with it in training; today's harness leaves it out. With it in, this notebook's code returns the 3 Nov predictions
to 2e-8. Leaving it out moves the Addis slope at k 5 from 0.41 to 0.61, so that table's low-k rows should be reread.
The frozen k 4 model was also trained with it; section 8 measures what that does.
"""),
    md("""
## 8. What the QC-failed spectrum does to network EC at k 4

The k 4 model is fitted twice on the train split of the converged tau 0.1 spectra, with and without IMPROVE filter
2008808, and both are applied to the converged network run (`qc_spectrum_effect` in
`workflows/run_vibes_tau_calibration_20261116.py`). The model with the spectrum is the 3 Nov rebuilt k 4 model (to 6e-7 µg).

**What it shows.**

- **IMPROVE cannot tell the two models apart.** Held-out RMSE is 3.850 with the spectrum and 3.873 without it.
- **SPARTAN EC is not the same.** PM2.5 EC moves by a median 17 %, and 64 % of scans move by more than 10 %. Site medians
  move from -64 % (PRFJ) to +51 % (ILHA). Addis moves 2 % at the median.
- **Field blanks drop from 1.27 to 0.30 µg.** That is closer to zero, but the blank level wanders with k for both models,
  so it does not prove the cleaned model right.
- **Nothing here picks a winner.** Without the spectrum the repeat-scan gap is larger (0.074 against 0.051 µg) and the
  Addis fit against HIPS is slightly looser (R² 0.67 against 0.71) with the same slope (0.41).
- **The spectrum is 364 times farther from the mean than a typical train spectrum.** The next most extreme train spectra
  are 19 to 33 times farther; they are very loaded filters that passed QC, and dropping one of them alone moves network
  EC by up to 6 %. Dropping 100 random spectra moves it by 0.25 % at the median.
- **The effect shrinks as k grows:** 17 % at k 4, 8 % at k 8, 3 % at k 12 and 2.5 % at k 25.

So EC from a k 4 model depends on a handful of extreme training spectra, and IMPROVE scores do not show it.
"""),
    code("""
Q = lambda name: pd.read_csv(C / f"qc_spectrum_{name}.csv")
Q("groups").round(3)
"""),
    code("""
Q("checks").round(3)
"""),
    code("""
sites = Q("sites")
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.barh(sites.site, sites.change_pct, color=np.where(sites.change_pct < 0, BLUE, ORANGE), height=.7)
ax.axvline(0, color="#9a9a95", lw=.8); ax.grid(axis="y", visible=False)
ax.set(xlabel="Change in the site's median PM2.5 EC when the spectrum is left out (%)", title="k 4 model with and without IMPROVE filter 2008808")
fig.tight_layout(); plt.show()
"""),
    code("""
by_k = Q("by_k")
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1))
axes[0].plot(by_k.k, by_k.network_median_abs_pct, color=BLUE, label="SPARTAN PM2.5 scans")
axes[0].plot(by_k.k, by_k.improve_test_median_abs_pct, color=ORANGE, label="IMPROVE held-out filters")
axes[0].set(xlabel="PLS components (k)", ylabel="Median |EC change| (%)", title="How far EC moves when the spectrum is left out", ylim=(0, None)); axes[0].legend(frameon=False)
axes[1].plot(by_k.k, by_k.heldout_rmse_with, color=INK, label="with the spectrum")
axes[1].plot(by_k.k, by_k.heldout_rmse_without, color=ORANGE, label="without it")
axes[1].set(xlabel="PLS components (k)", ylabel="Held-out RMSE (µg)", title="IMPROVE scores barely differ"); axes[1].legend(frameon=False)
fig.tight_layout(); plt.show()
"""),
    md("The train spectra farthest from the mean spectrum. For the QC-failed one the last column is the change from leaving it out; for the others, from dropping that one spectrum from the cleaned split."),
    code("""
Q("leverage").round(2)
"""),
    md("""
## 9. Is a k 4 model trained without the heaviest filters stable on SPARTAN?

One model per training set (`training_domain_stability` in the same workflow): every QC-passed train spectrum, only those
with thermal EC below 150, 100, 50 and 25 µg, and the 99 % of spectra nearest the mean. No network PM2.5 scan reads above
50 µg and 1.5 % read above 25 µg, so the heavy training filters lie outside SPARTAN's range. Each model is tested by
dropping one of its six most extreme spectra, and by refitting on IMPROVE sites drawn with replacement 30 times.

**What it shows.**

- **One spectrum no longer matters.** The largest change from dropping one spectrum falls from 6 % (all) to 3 % (below 150 µg),
  2 % (below 50) and 0.4 % (below 25).
- **The training sites still matter.** With IMPROVE sites resampled, a scan's EC spreads by 14 % with every spectrum in and
  by 8 % below 25 µg. A site's median EC spreads by 11 % and 6 %. The worst site goes from 61 % (PRFJ) to 13 %.
- **Each cut-off gives a different EC.** The models differ from each other by a median 3 to 30 % on network scans. Addis's
  median goes from 8.1 µg (all) to 5.9 µg (below 25 µg).
- **Accuracy on light filters improves.** Held-out RMSE on IMPROVE filters below 25 µg falls from 3.04 to 2.66 µg at k 4.
- **The Addis slope against HIPS follows the cut-off as much as it follows k:** 0.41 with every spectrum, 0.23 below 25 µg
  at k 4; 1.45 and 0.72 at k 12.
- **A higher k does not remove the spread:** below 25 µg it is 7.6, 7.0 and 6.6 % at k 4, 8 and 12.

So trimming makes the model steadier, but it does not pin EC down: the cut-off is itself a choice that moves EC, and about
7 % of site-to-site resampling spread is left.
"""),
    code("""
dom = pd.read_csv(C / "domain_policies.csv")
cols = {"training": "Training set", "train_spectra": "Train spectra", "heaviest_ug": "Heaviest (µg)", "one_spectrum_max_pct": "One spectrum, max (%)",
        "scan_spread_pct": "Scan spread (%)", "site_spread_median_pct": "Site spread, median (%)", "site_spread_max_pct": "Site spread, max (%)",
        "site_spread_max_site": "Worst site", "heldout_rmse_below25": "Held-out RMSE < 25 µg", "addis_slope": "Addis slope", "blank_median": "Blank (µg)",
        "repeat_gap": "Repeat gap (µg)", "network_median_ec": "Network median EC (µg)", "change_from_all_pct": "Change from all (%)"}
dom[dom.k.eq(4)][list(cols)].rename(columns=cols).round(2)
"""),
    code("""
order = list(dom.training.drop_duplicates())
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
for k, color in zip((4, 8, 12), (BLUE, INK, ORANGE)):
    d_ = dom[dom.k.eq(k)].set_index("training").loc[order]
    axes[0].plot(range(len(order)), d_.scan_spread_pct, marker="o", ms=4, color=color, label=f"k {k}")
    axes[1].plot(range(len(order)), d_.heldout_rmse_below25, marker="o", ms=4, color=color)
    axes[2].plot(range(len(order)), d_.addis_slope, marker="o", ms=4, color=color)
short = [o.replace("EC below ", "< ").replace("All QC-passed", "All").replace("Nearest 99 % of spectra", "Nearest 99 %") for o in order]
for ax, title, label in zip(axes, ("EC spread of a scan, IMPROVE sites resampled", "Held-out RMSE, IMPROVE filters below 25 µg", "Addis slope against HIPS"), ("%", "µg", "slope")):
    ax.set_xticks(range(len(order)), short, rotation=30, ha="right"); ax.set(title=title, ylabel=label)
axes[2].axhline(1, color="#9a9a95", lw=.8); axes[0].set_ylim(0, None); axes[0].legend(frameon=False)
fig.tight_layout(); plt.show()
"""),
    md("How far two training sets' network PM2.5 EC differ at k 4 (median |difference|, percent of the row's EC)."),
    code("""
pairs_ = pd.read_csv(C / "domain_pairs.csv")
pairs_[pairs_.k.eq(4)].pivot(index="training", columns="against", values="median_abs_pct").loc[order, order].round(1)
"""),
    code("""
ds = pd.read_csv(C / "domain_sites.csv")
ds.pivot(index="site", columns="training", values="median_ec")[order].round(2)
"""),
    md("""
## 10. Does the Addis slope still follow k once the training set is held fixed?

Earlier work read the Addis slope of FTIR EC on HIPS BC as following the number of PLS components. Section 9 showed the
slope also moves with the training set, so the earlier reading is repeated here with the training set held fixed
(`addis_by_training_set`): one model per k 1 to 25 for each of the six sets, applied to the 233 Addis spectra. The band
is the 5th to 95th percentile over 30 refits on IMPROVE sites drawn with replacement.

**What it shows.**

- **The slope follows k under every training set.** It rises about fivefold from k 4 to k 18 with every spectrum in
  (0.41 to 2.11) and three and a half fold below 25 µg (0.23 to 0.81). The intercept falls with it.
- **The training set moves it too, at every k.** Below 25 µg the slope is 0.38 to 0.56 of the slope with every spectrum
  in, and it never reaches 1. Below 50 µg it reaches 1 at k 11; the other four sets reach it at k 8 or 9.
- **Cut-offs of 150 and 100 µg are not distinguishable from "all".** From k 2 up they stay inside the band that
  resampling IMPROVE sites draws around it (0.29 to 0.61 at k 4, 1.16 to 1.72 at k 12).
- **The training sites alone move the slope by 20 to 50 %** (the width of that band).
- **R² does not choose.** It stays between 0.60 and 0.78 from k 3 up under every set. k and the training set change the
  scale of Addis EC (median 8.1 µg at k 4 and 18.0 µg at k 12 with every spectrum in; 5.9 and 13.0 µg below 25 µg), not
  how closely it follows HIPS.

So "the k that matches HIPS" is not a property of k alone. HIPS reads low on dark filters, so a slope of 1 is a
comparison, not a target.
"""),
    code("""
dk = pd.read_csv(C / "domain_addis_by_k.csv")
show = dk[dk.k.isin([4, 8, 12, 18])].assign(slope=lambda f: f.addis_slope.round(2).astype(str) + " (" + f.slope_lo.round(2).astype(str) + " to " + f.slope_hi.round(2).astype(str) + ")")
show.pivot(index="training", columns="k", values="slope").loc[order].rename(columns=lambda k: f"Slope at k {k} (5th to 95th pct)")
"""),
    code("""
shades = dict(zip(order, ("#1f3b63", "#2f6aa8", "#4f9bc9", ORANGE, "#e45756", "#7f7f7f")))
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
for name in order:
    d_ = dk[dk.training.eq(name)]
    axes[0].fill_between(d_.k, d_.slope_lo, d_.slope_hi, color=shades[name], alpha=.12, lw=0)
    axes[0].plot(d_.k, d_.addis_slope, color=shades[name], lw=1.3, label=name)
    axes[1].plot(d_.k, d_.addis_median_ec, color=shades[name], lw=1.3)
    axes[2].plot(d_.k, d_.addis_r2, color=shades[name], lw=1.3)
axes[0].axhline(1, color="#9a9a95", lw=.8)
for ax, title, label in zip(axes, ("Addis slope of FTIR EC on HIPS BC", "Median FTIR EC of the Addis filters", "R² against HIPS BC"), ("slope", "µg", "R²")):
    ax.axvline(4, color="#9a9a95", lw=.8, ls=":"); ax.set(title=title, xlabel="PLS components (k)", ylabel=label)
axes[2].set_ylim(0, 1); axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); plt.show()
"""),
    code("""
ratio = dk.pivot(index="k", columns="training", values="addis_slope")
reach = {name: (int(ratio.index[(ratio[name] >= 1).to_numpy().argmax()]) if (ratio[name] >= 1).any() else None) for name in order}
pd.DataFrame({"First k with slope of at least 1": reach,
              "Slope relative to all, smallest over k": (ratio[order].div(ratio["All QC-passed"], axis=0)).min().round(2),
              "largest over k": (ratio[order].div(ratio["All QC-passed"], axis=0)).max().round(2)})
"""),
    md("""
## 11. What to do with it

1. **Keep the frozen model as it is for now.** A trimmed training set is steadier and more accurate on light IMPROVE
   filters, but it lowers Addis EC by about a quarter and halves the Addis slope, and nothing measured here says the
   lower reading is more correct. That needs a reference SPARTAN does not have yet (thermal EC on SPARTAN filters).
2. **Quote EC from this calibration with a spread.** The table below is the spread to quote for a site's median EC from
   the k 4 model with every QC-passed spectrum in: the standard deviation over refits on resampled IMPROVE sites. It is
   under 10 % at 12 of 28 sites, 10 to 20 % at 13, and above 20 % at PRFJ, AEAZ and ILHA. It does not include the choice
   of k or of training set, which move EC by more.
3. **The training set and k are one decision, for Ann and Satoshi.** Both move the Addis comparison with HIPS, so a
   result of the form "the slope follows k" holds only for a stated training set (`docs/open-items.md`).
"""),
    code("""
spread = ds[ds.training.eq("All QC-passed")].set_index("site")[["median_ec", "spread_pct"]]
spread = spread.assign(spread_ug=spread.median_ec * spread.spread_pct / 100).sort_values("spread_pct", ascending=False)
print((spread.spread_pct < 10).sum(), "sites under 10 %,", spread.spread_pct.between(10, 20).sum(), "between 10 and 20 %,", (spread.spread_pct > 20).sum(), "above 20 %")
spread.rename(columns={"median_ec": "Median EC (µg)", "spread_pct": "Spread (%)", "spread_ug": "Spread (µg)"}).round(2)
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
out = HERE / "vibes_rank_tau_ec_20261116.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out, file=sys.stderr)
