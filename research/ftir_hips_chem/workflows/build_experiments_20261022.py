"""Build experiments_20261022.ipynb: the follow-up tests from the 22 Oct protocol review (tests 3-6, the season
calendar check and the Addis TOR study), each a section with saved tables for the gallery's Experimental tab.

It runs on the decision workbook's Drive bundle and reuses that workbook's setup, data loading, PLS toolbox and
out-of-site predictions verbatim (copied from ``decision_workbook_20261022.ipynb`` at build time, so the two cannot
drift), then adds its own sections. Build, execute, export (run from research/ftir_hips_chem):

    ~/anaconda3/bin/python workflows/build_experiments_20261022.py
    ~/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 experiments_20261022.ipynb
    ../../.venv/bin/python ../../gallery/data/export_experiments.py

Tables go to ``<Drive>/Aethmodular VIBES/decision_workbook_20261022/experiments/<date-time>/``.
"""
from pathlib import Path

import nbformat as nbf

AREA = Path(__file__).resolve().parents[1]
SOURCE = AREA / "decision_workbook_20261022.ipynb"
OUT = AREA / "experiments_20261022.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def borrowed(prefixes):
    """The decision workbook's code cells that start with each prefix, in that order (outputs dropped)."""
    nb = nbf.read(SOURCE, as_version=4)
    out = []
    for p in prefixes:
        hits = [c for c in nb.cells if c.cell_type == "code" and c.source.lstrip().startswith(p)]
        if len(hits) != 1:
            raise ValueError(f"expected one decision-workbook cell starting {p!r}, found {len(hits)}")
        out.append(code(hits[0].source))
    return out


cells = [md(r"""# Follow-up experiments (22 Oct 2026)

The protocol review of 22 Oct listed tests that can run on data we already have. The flag study (tests 1 and 2) is
Section 6b of the decision workbook. This notebook runs the rest, each judged the same way as there: **against TOR
inside IMPROVE, holding out whole sites** (or years), with SPARTAN Addis read against HIPS only as a cross-check.

| Section | Test | Question |
|---|---|---|
| 3 | Loading-matched analogs | Does choosing calibration neighbours by deposited mass as well as spectral shape move Addis toward HIPS while IMPROVE holds? |
| 4 | Attribution grid | How much do preprocessing, spectral window, calibration pool and k each move IMPROVE error and the Addis intercept, and do they interact? |
| 5 | Error-predicting selector | Does predicting each model's error (and abstaining when all look risky) beat picking a winner? |
| 6 | Earlier years → later years | Does the calibration transfer to a later year as well as to a new site? |
| 7 | Season calendar | Do the Addis seasonal slopes survive moving February from Dry to Belg? |
| 8 | Adama TOR study | Where do the Adama pilot pairs sit against IMPROVE, the SPARTAN network and SPARTAN Addis, where would they need to sit to test the Addis problem, how precise can ~10 pairs be, and what counts as a pass? |

The first cells are copied from the decision workbook (data, toolbox, out-of-site predictions), so every number here
uses the same models and folds."""),
         *borrowed(["# 0. Where the data is"]),
         code(r'''RESULTS = HOME / "experiments" / time.strftime("%Y%m%d-%H%M")      # this notebook's own results folder'''),
         *borrowed(["# Settings", "import numpy as np", "class PLS:", "rng = np.random.default_rng(SEED)\nFOLDS"])]

cells.append(code(r'''# shared for every section: weighed PM2.5 per filter, the Adama pilot, percentiles in IMPROVE, a table store
mass = pd.read_csv(BUNDLE / "improve_mass.csv")
im["pm_ug"] = im[["filter_id"]].merge(mass, on="filter_id", how="left", validate="one_to_one").pm25_ugm3.to_numpy() * im.volume_m3.to_numpy()
am["pm_ug"] = am.pm25_ugm3 * am.volume_m3
ad = pd.read_csv(BUNDLE / "adama_meta.csv")
ad["flag"] = ad.flag.fillna("")                                  # "" (unflagged) reads back as missing
XD = np.load(BUNDLE / "adama_spline.npy").astype(float)
HIPS_OK = ADDIS & am.hips_bc.notna().to_numpy() & (am.hips_bc > 0).to_numpy()
vol_a = am.volume_m3.where(am.volume_m3 > 0).to_numpy()
year = im.date.dt.year.to_numpy()
SAVE = {}


def pct_of(ref, v):
    ref = np.sort(np.asarray(ref, float)[np.isfinite(ref)])
    v = np.asarray(v, float)
    return np.where(np.isfinite(v), 100 * np.searchsorted(ref, v) / len(ref), np.nan)


def deming_row(x, yy, label):
    ok = np.isfinite(x) & np.isfinite(yy)
    s, i = deming(x[ok], yy[ok], LAMBDA_ADDIS)
    return {"model": label, "n": int(ok.sum()), "slope": s, "intercept": i, "ratio (median)": float(np.median(yy[ok] / x[ok]))}


ALL = {p: PLS().fit(XI[p], y) for p in PREPS_ALL}
k_main, k_raw = KSEL[MAIN], KSEL["raw"]
print(f"IMPROVE weighed PM2.5 on {np.isfinite(im.pm_ug).mean():.1%}; Addis with HIPS {HIPS_OK.sum()}; Adama pairs {len(ad)}; k: {KSEL}")'''))

# ---------------------------------------------------------------------------------------------- 3
cells.append(md(r"""## 3. Loading-matched analogs

The decision workbook's local calibration (each filter fitted on its 500 most similar IMPROVE raw spectra) wins inside
IMPROVE but moves Addis further from HIPS, and Addis filters are several times heavier than their neighbours. Here the
neighbours are chosen three more ways and compared on the same filters:

* **shape**: the 500 most correlated raw spectra (the workbook's rule);
* **shape + loading**: of the 2,000 most correlated, the 500 closest in weighed PM2.5 per filter (log scale);
* **loading only**: the 500 closest in weighed PM2.5, whatever their shape (a control);
* **shape + HIPS loading**: of the 2,000 most correlated, the 500 closest in HIPS BC per filter (absorption loading). HIPS
  only chooses the neighbours, the prediction is still from the spectrum, but read the Addis comparison with that in mind.

IMPROVE: 600 filters (300 from the heaviest tenth, 300 others), neighbours from other site folds only, against TOR.
Addis: every evaluation filter, neighbours from all of IMPROVE, against HIPS (cross-check). If **shape + loading**
moves Addis toward HIPS while IMPROVE holds, loading extrapolation is the mechanism; if not, it is ruled out."""))

cells.append(code(r'''mask_raw, mask_sp = corr_mask("raw"), corr_mask("spline")
Z_lib, ZS_lib = unit_rows(XI["raw"], mask_raw), unit_rows(XI["spline"], mask_sp)
LPM = np.log(np.clip(im.pm_ug.to_numpy(), 1, None))
POOL = 2000
LHIPS = np.log(np.clip(im.hips_ug.to_numpy(), 0.05, None))
RULES = ["shape", "shape + loading", "loading only", "shape + HIPS loading"]


def neighbours(z, pm_t, rule, allowed, hips_t=np.nan):
    c = np.where(allowed, Z_lib @ z, -np.inf)
    if rule == "shape + HIPS loading":
        if not np.isfinite(hips_t):
            return np.argpartition(-c, N_SELECT)[:N_SELECT]
        cand = np.argpartition(-c, POOL)[:POOL]
        return cand[np.argsort(np.abs(LHIPS[cand] - np.log(max(hips_t, 0.05))), kind="stable")[:N_SELECT]]
    if rule == "shape" or (rule == "shape + loading" and not np.isfinite(pm_t)):
        return np.argpartition(-c, N_SELECT)[:N_SELECT]
    d = np.abs(LPM - np.log(max(pm_t, 1))) if np.isfinite(pm_t) else np.zeros(len(LPM))
    d = np.where(np.isfinite(d) & allowed, d, np.inf)
    if rule == "loading only":
        return np.argpartition(d, N_SELECT)[:N_SELECT]
    cand = np.argpartition(-c, POOL)[:POOL]
    return cand[np.argsort(d[cand], kind="stable")[:N_SELECT]]


def local_all(Xraw, Xsp, pm_t, hips_t, allowed_of):
    """per rule: local prediction (µg/filter), PM and HIPS loading ÷ neighbours' median, shape similarity"""
    Zt, Zs = unit_rows(Xraw, mask_raw), unit_rows(Xsp, mask_sp)
    out = {r: {"pred": np.empty(len(Xraw)), "pm_ratio": np.empty(len(Xraw)), "hips_ratio": np.empty(len(Xraw)), "sim": np.empty(len(Xraw))} for r in RULES}
    for i in range(len(Xraw)):
        allowed = allowed_of(i)
        for r in RULES:
            nn = neighbours(Zt[i], pm_t[i], r, allowed, hips_t[i])
            o = out[r]
            o["pred"][i] = PLS(k_raw).fit(XI["raw"][nn], y[nn]).predict(Xraw[i:i + 1])[0, k_raw - 1]
            o["pm_ratio"][i] = pm_t[i] / np.nanmedian(im.pm_ug.to_numpy()[nn])
            o["hips_ratio"][i] = hips_t[i] / np.nanmedian(im.hips_ug.to_numpy()[nn])
            o["sim"][i] = np.median(ZS_lib[nn] @ Zs[i])
    return out


t0 = time.time()
rng = np.random.default_rng(SEED + 3)
heavy_cut = np.nanpercentile(im.pm_ug, 90)
heavy = np.flatnonzero(im.pm_ug.to_numpy() >= heavy_cut)
other = np.flatnonzero(im.pm_ug.to_numpy() < heavy_cut)
S3 = np.concatenate([rng.choice(heavy, 300, replace=False), rng.choice(other, 300, replace=False)])
L_I = local_all(XI["raw"][S3], XI["spline"][S3], im.pm_ug.to_numpy()[S3], im.hips_ug.to_numpy()[S3], lambda i: FOLDS != FOLDS[S3[i]])
L_A = local_all(XA["raw"], XA["spline"], am.pm_ug.to_numpy(), am.hips_ug.to_numpy(), lambda i: np.ones(len(y), bool))
print(f"local models for {len(S3) + len(am)} filters × {len(RULES)} rules in {time.time() - t0:.0f} s")

grp = np.where(np.isin(S3, heavy), "heaviest tenth", "other filters")
rows = []
for name, pred in [("full calibration (raw)", OOF["raw"][S3, k_raw - 1])] + [(f"local: {r}", L_I[r]["pred"]) for r in RULES]:
    for g in ["heaviest tenth", "other filters", "all"]:
        m = np.ones(len(S3), bool) if g == "all" else grp == g
        sc = score(pred[m], y[S3][m])
        r_ = name.replace("local: ", "")
        rows.append({"model": name, "group": g, **sc,
                     "PM ÷ neighbours (median)": float(np.nanmedian(L_I[r_]["pm_ratio"][m])) if r_ in L_I else np.nan,
                     "shape similarity (median)": float(np.nanmedian(L_I[r_]["sim"][m])) if r_ in L_I else np.nan})
E3_IMPROVE = pd.DataFrame(rows)

xa = am.hips_bc.to_numpy()
full_raw_a = ALL["raw"].predict(XA["raw"])[:, k_raw - 1] / vol_a
rows = [deming_row(xa[HIPS_OK], am.ec_published_ugm3.to_numpy()[HIPS_OK], "SPARTAN published"),
        deming_row(xa[HIPS_OK], full_raw_a[HIPS_OK], "full calibration (raw)")]
for r in RULES:
    row = deming_row(xa[HIPS_OK], (L_A[r]["pred"] / vol_a)[HIPS_OK], f"local: {r}")
    row.update({"PM ÷ neighbours (median)": float(np.nanmedian(L_A[r]["pm_ratio"][HIPS_OK])), "HIPS ÷ neighbours (median)": float(np.nanmedian(L_A[r]["hips_ratio"][HIPS_OK])),
                "shape similarity (median)": float(np.nanmedian(L_A[r]["sim"][HIPS_OK]))})
    rows.append(row)
E3_ADDIS = pd.DataFrame(rows)
E3_ADDIS_FILTERS = pd.DataFrame({"filter_id": am.filter_id, "season": am.season, "hips_bc": xa, "published": am.ec_published_ugm3, "full raw": full_raw_a,
                                 **{f"local {r}": L_A[r]["pred"] / vol_a for r in RULES}, **{f"PM ratio {r}": L_A[r]["pm_ratio"] for r in RULES},
                                 **{f"HIPS ratio {r}": L_A[r]["hips_ratio"] for r in RULES}})[HIPS_OK]
E3_IMPROVE_FILTERS = pd.DataFrame({"filter_id": im.filter_id.to_numpy()[S3], "site": sites[S3], "group": grp, "tor_ug": y[S3], "full raw": OOF["raw"][S3, k_raw - 1],
                                   **{f"local {r}": L_I[r]["pred"] for r in RULES}, **{f"PM ratio {r}": L_I[r]["pm_ratio"] for r in RULES}})
SAVE.update({"e3_improve": E3_IMPROVE, "e3_addis": E3_ADDIS, "e3_addis_filters": E3_ADDIS_FILTERS, "e3_improve_filters": E3_IMPROVE_FILTERS})
display(E3_IMPROVE.pivot(index="model", columns="group", values="rmse")); display(E3_ADDIS)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
bins = np.linspace(-1.2, 1.5, 50)
for r in RULES:
    ax[0].hist(np.log10(L_A[r]["pm_ratio"][HIPS_OK]), bins, histtype="step", lw=1.5, density=True, label=r)
ax[0].axvline(0, color="k", lw=.8); ax[0].set(xlabel="log10 (Addis PM2.5 ÷ neighbours' median)", title="did loading matching work?"); ax[0].legend(frameon=False, fontsize=8)
e = E3_ADDIS.set_index("model")
ax[1].barh(e.index, e.intercept); ax[1].axvline(0, color="k", lw=.8); ax[1].set(xlabel="Deming intercept vs HIPS BC (µg/m³)", title="Addis")
plt.tight_layout(); plt.show()'''))

# ---------------------------------------------------------------------------------------------- 4
cells.append(md(r"""## 4. Attribution grid

One factorial on the same filters, crossing **preprocessing** (raw, spline, and VIBES as an extra), **spectral window**
(1425–4000 cm⁻¹ or the full range where the preprocessing has one; spline exists only above 1425), **calibration pool**
(all IMPROVE, or the 500 lowest TOR OC/EC filters) and **k** (fixed 8, as on the poster, or the one-SE choice).
Each cell is scored against TOR on every IMPROVE filter (out-of-site, the common test set) and against HIPS on the
Addis evaluation filters. Effects are 2-level factorial contrasts on the complete sub-grids; an interaction is how much
one factor's effect changes with another."""))

cells.append(code(r'''COMBOS = [("raw", "full"), ("raw", "1425"), ("spline", "1425"), ("vibes", "full"), ("vibes", "1425")]
POOLS, KRULES = ["all IMPROVE", "500 lowest OC/EC"], ["k = 8", "one-SE k"]
ocec = (im.tor_oc_ugm3 / im.tor_ec_ugm3.where(im.tor_ec_ugm3 > 0)).to_numpy()
heavy_all = im.pm_ug.to_numpy() >= heavy_cut


def pool_rows(idx):
    """within a set of training rows, the members of each pool"""
    low = idx[np.argsort(np.where(np.isfinite(ocec[idx]), ocec[idx], np.inf))[:N_SELECT]]
    return {"all IMPROVE": idx, "500 lowest OC/EC": low}


t0 = time.time()
rows, preds_a = [], {}
for prep, win in COMBOS:
    cols = np.ones(XI[prep].shape[1], bool) if win == "full" else WN[prep] >= 1425
    X, XAa = XI[prep][:, cols], XA[prep][:, cols]
    for pool in POOLS:
        oof = np.full((len(y), K_MAX), np.nan)
        err = np.full((NF, K_MAX), np.nan)
        for f in range(NF):
            tr = pool_rows(np.flatnonzero(FOLDS != f))[pool]
            m = PLS().fit(X[tr], y[tr])
            te = FOLDS == f
            oof[te, :m.K] = m.predict(X[te])
            err[f, :m.K] = np.sqrt(((oof[te, :m.K] - y[te, None]) ** 2).mean(0))
        k_se = one_se(err)
        full = PLS().fit(X[pool_rows(np.arange(len(y)))[pool]], y[pool_rows(np.arange(len(y)))[pool]])
        for krule in KRULES:
            k = 8 if krule == "k = 8" else k_se
            p = oof[:, k - 1]
            pa = full.predict(XAa)[:, k - 1] / vol_a
            preds_a[(prep, win, pool, krule)] = pa
            d = deming_row(xa[HIPS_OK], pa[HIPS_OK], "")
            rows.append({"preprocessing": prep, "window": win, "pool": pool, "k rule": krule, "k": k,
                         "IMPROVE RMSE": rmse(p - y), "IMPROVE bias": float(np.mean(p - y)), "IMPROVE R²": 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2),
                         "IMPROVE RMSE heaviest tenth": rmse((p - y)[heavy_all]), "Addis slope": d["slope"], "Addis intercept": d["intercept"], "Addis n": d["n"]})
print(f"attribution grid: {len(rows)} cells in {time.time() - t0:.0f} s")
GRID4 = pd.DataFrame(rows)
pub = deming_row(xa[HIPS_OK], am.ec_published_ugm3.to_numpy()[HIPS_OK], "published")


def effects(sub, factors, outcomes):
    """2^3 factorial contrasts: main effects and two-way interactions (high level minus low level)"""
    lv = {f: sorted(sub[f].unique()) for f in factors}
    if any(len(v) != 2 for v in lv.values()) or len(sub) != 8:
        return pd.DataFrame()
    code_ = {f: np.where(sub[f] == lv[f][1], 1, -1) for f in factors}
    out = []
    terms = [(f,) for f in factors] + [(a, b) for i, a in enumerate(factors) for b in factors[i + 1:]]
    for t in terms:
        sign = np.prod([code_[f] for f in t], axis=0)
        row = {"term": " × ".join(t), "contrast": " × ".join(f"{lv[f][1]} vs {lv[f][0]}" for f in t)}
        for o in outcomes:
            row[o] = float((sign * sub[o].to_numpy()).sum() / 4)
        out.append(row)
    return pd.DataFrame(out)


OUTS = ["IMPROVE RMSE", "IMPROVE bias", "Addis intercept", "Addis slope"]
a = GRID4[GRID4.window.eq("1425") & GRID4.preprocessing.isin(["raw", "spline"])]
E_PREP = effects(a, ["preprocessing", "pool", "k rule"], OUTS).assign(grid="raw vs spline (1425–4000)")
b = GRID4[GRID4.preprocessing.eq("raw")]
E_WIN = effects(b, ["window", "pool", "k rule"], OUTS).assign(grid="full vs 1425 window (raw)")
EFFECTS4 = pd.concat([E_PREP, E_WIN], ignore_index=True)
SAVE.update({"e4_grid": GRID4, "e4_effects": EFFECTS4, "e4_published": pd.DataFrame([pub])})
display(GRID4.round(3)); display(EFFECTS4.round(3))'''))

# ---------------------------------------------------------------------------------------------- 5
cells.append(md(r"""## 5. A selector that predicts each model's error

The earlier selector learned which model wins and once picked a −587 µg prediction. Here, for 1,500 IMPROVE filters,
three candidates are compared: the full calibration (spline, the workbook's main model), the full raw calibration and
the local raw calibration (shape neighbours from other site folds). A regressor learns each candidate's |error| from
prediction-time inputs; the selector takes the candidate with the smallest predicted error, and optionally **abstains**
when even that is in the riskiest tenth. Everything is nested by site fold: the error models never see the fold they
score. Compared with each candidate alone, their average, a "winner" classifier and the oracle."""))

cells.append(code(r'''from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier as HGC
t0 = time.time()
rng = np.random.default_rng(SEED + 5)
S5 = np.sort(rng.choice(len(y), 1500, replace=False))
Zt5 = unit_rows(XI["raw"][S5], mask_raw)
loc5, sim5, heavy5 = np.empty(len(S5)), np.empty(len(S5)), np.empty(len(S5))
for j, i in enumerate(S5):
    nn = neighbours(Zt5[j], np.nan, "shape", FOLDS != FOLDS[i])
    loc5[j] = PLS(k_raw).fit(XI["raw"][nn], y[nn]).predict(XI["raw"][i:i + 1])[0, k_raw - 1]
    sim5[j] = np.median(ZS_lib[nn] @ unit_rows(XI["spline"][i:i + 1], mask_sp)[0])
    heavy5[j] = im.pm_ug.to_numpy()[i] / np.nanmedian(im.pm_ug.to_numpy()[nn])
CAND = {"full spline": OOF[MAIN][S5, k_main - 1], "full raw": OOF["raw"][S5, k_raw - 1], "local raw": loc5}
names = list(CAND)
P = np.column_stack([CAND[n] for n in names])
E = np.abs(P - y[S5, None])
t2_5, q_5 = np.empty(len(S5)), np.empty(len(S5))
for f in range(NF):
    m = FOLDS[S5] == f
    t2_5[m], q_5[m] = FOLD_MODELS[MAIN][f].distances(XI[MAIN][S5[m]], k_main)
F5 = spectral_features(XI["raw"][S5])
F5 = F5.assign(t2=t2_5, q=q_5, pm_ug=im.pm_ug.to_numpy()[S5], sim=sim5, heavier=heavy5, **{f"pred {n}": CAND[n] for n in names},
               spread=P.max(1) - P.min(1))
fold5 = FOLDS[S5]
risk = np.full_like(E, np.nan)
win_choice = np.full(len(S5), -1)
tau = np.full(len(S5), np.nan)
for f in range(NF):
    tr, te = fold5 != f, fold5 == f
    for j in range(len(names)):
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED)
        reg.fit(F5[tr], np.log1p(E[tr, j]))
        risk[te, j] = np.expm1(reg.predict(F5[te]))
        risk_tr = np.expm1(reg.predict(F5[tr])) if j == 0 else np.minimum(risk_tr, np.expm1(reg.predict(F5[tr])))
    tau[te] = np.quantile(risk_tr, 0.9)                       # abstain on the riskiest tenth, set on the training folds
    clf = HGC(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED).fit(F5[tr], E[tr].argmin(1))
    win_choice[te] = clf.predict(F5[te])
pick = risk.argmin(1)
sel = P[np.arange(len(S5)), pick]
winner = P[np.arange(len(S5)), win_choice]
oracle = P[np.arange(len(S5)), E.argmin(1)]
keep = risk.min(1) <= tau
yy = y[S5]


def strat(name, pred, kept=None):
    kept = np.ones(len(pred), bool) if kept is None else kept
    e = (pred - yy)[kept]
    return {"strategy": name, "kept share": kept.mean(), "RMSE": rmse(e), "MAE": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
            "99th pct |error|": float(np.percentile(np.abs(e), 99)), "max |error|": float(np.abs(e).max()), "share |error| > 10 µg": float((np.abs(e) > 10).mean())}


E5 = pd.DataFrame([strat(n, CAND[n]) for n in names] + [strat("average of three", P.mean(1)), strat("winner classifier", winner),
                   strat("error-predicting selector", sel), strat("selector, abstain on riskiest tenth", sel, keep), strat("oracle (best per filter)", oracle)])
order = np.argsort(risk.min(1))
COV5 = pd.DataFrame([{"kept share": c, "RMSE selector": rmse((sel - yy)[order[:int(c * len(order))]]), "RMSE oracle order": rmse(np.sort(np.abs(oracle - yy))[:int(c * len(order))])}
                     for c in np.round(np.arange(0.3, 1.0001, 0.02), 2)])
dec = pd.qcut(risk.min(1), 10, labels=False, duplicates="drop")
CAL5 = pd.DataFrame({"predicted |error|": risk.min(1), "actual |error|": np.abs(sel - yy), "decile": dec}).groupby("decile").mean().reset_index()
PICKS5 = pd.DataFrame({"model": names, "chosen share (selector)": [(pick == j).mean() for j in range(3)], "chosen share (winner)": [(win_choice == j).mean() for j in range(3)],
                       "best share (oracle)": [(E.argmin(1) == j).mean() for j in range(3)]})
FILT5 = pd.DataFrame({"filter_id": im.filter_id.to_numpy()[S5], "site": sites[S5], "tor_ug": yy, **{f"pred {n}": CAND[n] for n in names},
                      **{f"risk {n}": risk[:, j] for j, n in enumerate(names)}, "chosen": [names[j] for j in pick], "kept": keep})
print(f"selector in {time.time() - t0:.0f} s; Spearman(predicted, actual |error| of the chosen model) = {spearmanr(risk.min(1), np.abs(sel - yy)).statistic:.2f}")
SAVE.update({"e5_strategies": E5, "e5_coverage": COV5, "e5_calibration": CAL5, "e5_picks": PICKS5, "e5_filters": FILT5})
display(E5.round(3)); display(PICKS5.round(3))'''))

# ---------------------------------------------------------------------------------------------- 6
cells.append(md(r"""## 6. Earlier years → later years

Every other test holds out sites. Here the calibration is trained on 2020–2022 and tested on 2023, the only way to
ask whether it carries forward in time, and compared on equal-sized tests with new sites, random filters, and new
sites in the later year (both at once). New-site and random splits are repeated five times."""))

cells.append(code(r'''def fit_score(tr, te, prep):
    m = PLS().fit(XI[prep][tr], y[tr])
    p = m.predict(XI[prep][te])
    e = p - y[te, None]
    return np.sqrt((e ** 2).mean(0)), e.mean(0), 1 - (e ** 2).sum(0) / ((y[te] - y[te].mean()) ** 2).sum()


later = year == year.max()
n_te = int(later.sum())
rows = []
t0 = time.time()
for prep in ["spline", "raw"]:
    splits = [("later year (train 2020–22, test 2023)", ~later, later, 0)]
    for r in range(5):
        rr = np.random.default_rng(SEED + 60 + r)
        u = rr.permutation(np.unique(sites))
        held, cnt = [], 0
        for s_ in u:
            if cnt >= n_te:
                break
            held.append(s_); cnt += int((sites == s_).sum())
        hs = np.isin(sites, held)
        splits.append(("new sites (same test size)", ~hs, hs, r))
        rf = np.zeros(len(y), bool); rf[rr.choice(len(y), n_te, replace=False)] = True
        splits.append(("random filters (same test size)", ~rf, rf, r))
        both_te = later & hs
        splits.append(("new sites in the later year", ~later & ~hs, both_te, r))
    for name, tr, te, r in splits:
        rm, bi, r2 = fit_score(tr, te, prep)
        for k in range(1, len(rm) + 1):
            rows.append({"preprocessing": prep, "split": name, "repeat": r, "k": k, "n test": int(te.sum()), "RMSE": rm[k - 1], "bias": bi[k - 1], "R²": r2[k - 1]})
E6 = pd.DataFrame(rows)
print(f"time splits in {time.time() - t0:.0f} s")
E6_SUMMARY = (E6[E6.k == E6.preprocessing.map(KSEL)].groupby(["preprocessing", "split"])
              .agg(n_test=("n test", "mean"), RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std"), bias=("bias", "mean"), R2=("R²", "mean")).reset_index())
E6_YEARS = im.assign(year=year).groupby("year").agg(filters=("site", "size"), sites=("site", "nunique"), median_tor_ug=("ec_ug", "median")).reset_index()
SAVE.update({"e6_curves": E6, "e6_summary": E6_SUMMARY, "e6_years": E6_YEARS})
display(E6_SUMMARY.round(3)); display(E6_YEARS)'''))

# ---------------------------------------------------------------------------------------------- 7
cells.append(md(r"""## 7. Season calendar: does February belong to Dry or Belg?

The seasonal Addis results used the historical calendar with February in Dry; the site's current calendar puts it in
Belg. The seasonal Deming slopes of FTIR EC against HIPS BC are recomputed under both, for SPARTAN's published EC and
this workbook's full spline and raw calibrations, with 95 % bootstrap intervals."""))

cells.append(code(r'''mon = am.date.dt.month.to_numpy()
CAL = {"Feb in Dry (historical)": {**{m: "Dry" for m in (10, 11, 12, 1, 2)}, **{m: "Belg" for m in (3, 4, 5)}, **{m: "Kiremt" for m in (6, 7, 8, 9)}},
       "Feb in Belg (site calendar)": {**{m: "Dry" for m in (10, 11, 12, 1)}, **{m: "Belg" for m in (2, 3, 4, 5)}, **{m: "Kiremt" for m in (6, 7, 8, 9)}}}
MODELS7 = {"SPARTAN published": am.ec_published_ugm3.to_numpy(), "full spline": ALL[MAIN].predict(XA[MAIN])[:, k_main - 1] / vol_a,
           "full raw": full_raw_a}
rows = []
rng = np.random.default_rng(SEED + 7)
for cal, mp in CAL.items():
    seas = np.array([mp.get(m, "") for m in mon])
    for model, v in MODELS7.items():
        for s_ in ["Dry", "Belg", "Kiremt"]:
            m = HIPS_OK & (seas == s_) & np.isfinite(v)
            xs, ys = xa[m], v[m]
            sl, ic = deming(xs, ys, LAMBDA_ADDIS)
            boot = []
            for _ in range(300):
                b = rng.integers(0, len(xs), len(xs))
                boot.append(deming(xs[b], ys[b], LAMBDA_ADDIS)[0])
            rows.append({"calendar": cal, "model": model, "season": s_, "n": int(m.sum()), "slope": sl, "slope lo": np.percentile(boot, 2.5),
                         "slope hi": np.percentile(boot, 97.5), "intercept": ic})
E7 = pd.DataFrame(rows)
E7_FEB = pd.DataFrame({"filter_id": am.filter_id, "date": am.date.dt.date, "hips_bc": xa, **{k: v for k, v in MODELS7.items()}})[HIPS_OK & (mon == 2)]
SAVE.update({"e7_seasons": E7, "e7_february": E7_FEB})
display(E7.pivot_table(index=["model", "season"], columns="calendar", values=["n", "slope"]).round(2))'''))

# ---------------------------------------------------------------------------------------------- 8
cells.append(md(r"""## 8. Adama TOR study: where the pilot sits, where it would need to sit, and what would count as a pass

Five Adama quartz/PTFE date-candidate pairs (CSU AMOD sampler, lower flow, July 2024) are in hand, and about ten parallel
FTIR + TOR filters from Addis city are expected on the same kind of sampler. Before any new outcome is read:

* **Against IMPROVE, the SPARTAN network and SPARTAN Addis.** Every population on the same axes: EC and PM per filter,
  EC concentration, OC/EC, spectral distance, sampled volume and season.
* **Where it should be.** A pilot can only test the Addis problem if it sits where SPARTAN Addis sits. For each axis the
  Adama pairs are placed in SPARTAN Addis's range, and the volume needed to reach Addis-like loading per filter is worked out.

* **Where they sit.** The IMPROVE calibration set is the TOR network the calibration learned from. Each IMPROVE site,
  SPARTAN Addis and the Adama pilot are placed on the same axes (EC and PM per filter, spectral distance), and the
  IMPROVE sites whose filters are Addis's and Adama's nearest spectral neighbours are counted.
* **How precise ~10 pairs can be.** Resampling IMPROVE's own out-of-site errors on Addis-like filters gives the 95 %
  interval of a mean bias as a function of the number of pairs.
* **The contract.** Frozen models, the reference, the metrics and pass rules, written here, with the Adama pairs
  scored against it as a pilot (not the test)."""))

cells.append(code(r'''# distances on the main model: IMPROVE out-of-site (yardstick), Addis and Adama from the all-IMPROVE fit
t2_I, Q_I = np.zeros(len(y)), np.zeros(len(y))
for f in range(NF):
    te = FOLDS == f
    t2_I[te], Q_I[te] = FOLD_MODELS[MAIN][f].distances(XI[MAIN][te], k_main)
dist = lambda t2, q: np.maximum(pct_of(t2_I, t2), pct_of(Q_I, q))
t2_A, Q_A = ALL[MAIN].distances(XA[MAIN], k_main)
t2_D, Q_D = ALL[MAIN].distances(XD, k_main)
pred_D = ALL[MAIN].predict(XD)[:, k_main - 1] / ad.volume_m3.to_numpy()
dist_I, dist_A, dist_D = dist(t2_I, Q_I), dist(t2_A, Q_A), dist(t2_D, Q_D)
# the SPARTAN network (Addis comes from its frozen evaluation set), distances on the same model
NET_ROWS = (~nm.excluded & (nm.code != "ETAD") & np.isfinite(XN[MAIN]).all(axis=1)).to_numpy()
dist_N = np.full(len(nm), np.nan)
t2_N, Q_N = ALL[MAIN].distances(np.nan_to_num(XN[MAIN][NET_ROWS]), k_main)
dist_N[NET_ROWS] = dist(t2_N, Q_N)

# per filter, on common axes
PTS = pd.concat([
    pd.DataFrame({"population": "IMPROVE (TOR)", "site": sites, "id": im.filter_id.astype(str), "EC µg/filter": y, "EC µg/m³": im.tor_ec_ugm3,
                  "EC source": "TOR", "PM2.5 µg/filter": im.pm_ug, "OC/EC": ocec, "distance percentile": dist_I, "volume m³": im.volume_m3}),
    pd.DataFrame({"population": "SPARTAN Addis", "site": "ETAD", "id": am.filter_id, "EC µg/filter": am.ec_published_ugm3 * am.volume_m3, "EC µg/m³": am.ec_published_ugm3,
                  "EC source": "FTIR (published)", "PM2.5 µg/filter": am.pm_ug, "OC/EC": am.oc_ftir_ugm3 / am.ec_published_ugm3, "distance percentile": dist_A,
                  "volume m³": am.volume_m3, "HIPS BC µg/m³": am.hips_bc, "month": am.date.dt.month})[ADDIS],
    pd.DataFrame({"population": "Adama pilot (TOR)", "site": "ADAMA", "id": ad.ptfe_id, "EC µg/filter": ad.ec_tr_ugm3 * ad.quartz_volume_m3, "EC µg/m³": ad.ec_tr_ugm3,
                  "EC source": "TOR (ECTR, quartz volume)", "PM2.5 µg/filter": np.nan, "OC/EC": ad.oc_tr_ugm3 / ad.ec_tr_ugm3, "distance percentile": dist_D,
                  "volume m³": ad.quartz_volume_m3, "FTIR µg/m³ (this model)": pred_D, "flag": ad.flag, "month": pd.to_datetime(ad.date).dt.month}),
    pd.DataFrame({"population": "SPARTAN network", "site": nm.code, "id": nm.filter_id, "EC µg/filter": nm.ec_ftir_ugm3 * nm.volume_m3.where(nm.volume_m3 > 0),
                  "EC µg/m³": nm.ec_ftir_ugm3, "EC source": "FTIR (published)", "PM2.5 µg/filter": nm.pm25_ugm3 * nm.volume_m3.where(nm.volume_m3 > 0),
                  "OC/EC": nm.oc_ftir_ugm3 / nm.ec_ftir_ugm3.where(nm.ec_ftir_ugm3 > 0), "distance percentile": dist_N, "volume m³": nm.volume_m3.where(nm.volume_m3 > 0),
                  "month": nm.month, "season": nm.season})[NET_ROWS],
], ignore_index=True)

# IMPROVE sites as the TOR network, with Addis and Adama summarised the same way
NET = (PTS.groupby(["population", "site"]).agg(filters=("id", "size"), ec_ug=("EC µg/filter", "median"), ec_ugm3=("EC µg/m³", "median"),
                                              pm_ug=("PM2.5 µg/filter", "median"), ocec=("OC/EC", "median"), distance=("distance percentile", "median"),
                                              volume=("volume m³", "median")).reset_index())
NET["urban"] = NET.site.isin(["ATLA1", "BIRM1", "FRES1", "PHOE1", "PITT1", "PUSO1"])

# which TOR sites are Addis's and Adama's nearest spectral neighbours (spline shape, the 500 nearest per filter)
def neighbour_sites(Xsp):
    Zt = unit_rows(Xsp, mask_sp)
    counts = pd.Series(0.0, index=np.unique(sites))
    for z in Zt:
        nn = np.argpartition(-(ZS_lib @ z), N_SELECT)[:N_SELECT]
        counts = counts.add(pd.Series(sites[nn]).value_counts(normalize=True), fill_value=0)
    return counts / len(Zt)


NB = pd.DataFrame({"Addis": neighbour_sites(XA[MAIN][ADDIS]), "Adama": neighbour_sites(XD), "IMPROVE share of filters": pd.Series(sites).value_counts(normalize=True)}).fillna(0)
NB = NB.rename_axis("site").reset_index()

# precision: the 95 % interval of a mean relative bias with n pairs, resampling IMPROVE out-of-site errors
e_main = OOF[MAIN][:, k_main - 1] - y
like = (dist_I >= 60) & (pct_of(y, y) >= 60) & (y > 0.5)
rel_all, rel_like = (e_main / y)[y > 0.5], (e_main / y)[like]
rows = []
rng = np.random.default_rng(SEED + 8)
for n in [3, 5, 8, 10, 12, 15, 20, 25, 30, 36, 45, 60]:
    for name, pool in (("Addis-like IMPROVE filters", rel_like), ("all IMPROVE filters", rel_all)):
        means = pool[rng.integers(0, len(pool), (4000, n))].mean(1)
        lo, hi = np.percentile(means - means.mean(), [2.5, 97.5])
        rows.append({"pairs": n, "population": name, "half width (relative bias)": (hi - lo) / 2, "pool size": len(pool)})
PRECISION = pd.DataFrame(rows)

# where the pilot should be: for each axis, SPARTAN Addis's range and where each Adama pair falls in it
AX = ["EC µg/filter", "EC µg/m³", "OC/EC", "distance percentile", "volume m³"]
A_ = PTS[PTS.population == "SPARTAN Addis"]
D_ = PTS[PTS.population == "Adama pilot (TOR)"]
I_ = PTS[PTS.population == "IMPROVE (TOR)"]
N_ = PTS[PTS.population == "SPARTAN network"]
rows = []
for ax in AX:
    a = A_[ax].dropna().to_numpy()
    q5, q25, q50, q75, q95 = np.percentile(a, [5, 25, 50, 75, 95])
    dv = D_[ax].to_numpy()
    rows.append({"axis": ax, "Addis p5": q5, "Addis p25": q25, "Addis median": q50, "Addis p75": q75, "Addis p95": q95,
                 "Adama median": float(np.nanmedian(dv)), "Adama inside Addis 25-75": float(np.mean((dv >= q25) & (dv <= q75))),
                 "Adama inside Addis 5-95": float(np.mean((dv >= q5) & (dv <= q95))),
                 "Adama percentile in Addis": float(np.nanmedian(pct_of(a, dv))), "Adama percentile in IMPROVE": float(np.nanmedian(pct_of(I_[ax].to_numpy(), dv))),
                 "Adama percentile in SPARTAN network": float(np.nanmedian(pct_of(N_[ax].to_numpy(), dv))),
                 "IMPROVE median": float(np.nanmedian(I_[ax])), "SPARTAN network median": float(np.nanmedian(N_[ax]))})
TARGET = pd.DataFrame(rows)

# loading = concentration × volume: the volume that would bring the pilot to SPARTAN Addis's loading per filter
ec_ad, ec_a_ug = float(np.median(ad.ec_tr_ugm3)), A_["EC µg/filter"].dropna().to_numpy()
VOLUME = pd.DataFrame([{"target": name, "EC µg/filter": float(np.percentile(ec_a_ug, q)),
                        "volume at Adama's median EC (m³)": float(np.percentile(ec_a_ug, q)) / ec_ad,
                        "volume at SPARTAN Addis median EC (m³)": float(np.percentile(ec_a_ug, q)) / float(np.nanmedian(A_["EC µg/m³"])),
                        "Adama volume now (m³)": float(np.median(ad.quartz_volume_m3)), "hours at the Adama flow rate": float(np.percentile(ec_a_ug, q)) / ec_ad / float(np.median(ad.quartz_volume_m3)) * 24}
                       for name, q in (("Addis 25th percentile", 25), ("Addis median", 50), ("Addis 75th percentile", 75))])

# season: where in the year the FTIR-HIPS disagreement sits at Addis, and when the pilot was sampled (site calendar, Feb in Belg)
SEASON_OF = {**{m: "Dry" for m in (10, 11, 12, 1)}, **{m: "Belg" for m in (2, 3, 4, 5)}, **{m: "Kiremt" for m in (6, 7, 8, 9)}}
am_season = am.date.dt.month.map(SEASON_OF)
rows = []
for s_ in ["Dry", "Belg", "Kiremt"]:
    m = HIPS_OK & (am_season == s_).to_numpy()
    r = am.ec_published_ugm3.to_numpy()[m] / xa[m]
    sl, ic = deming(xa[m & np.isfinite(am.ec_published_ugm3.to_numpy())], am.ec_published_ugm3.to_numpy()[m & np.isfinite(am.ec_published_ugm3.to_numpy())], LAMBDA_ADDIS)
    rows.append({"season": s_, "Addis filters": int(m.sum()), "published ÷ HIPS (median)": float(np.nanmedian(r)), "published ÷ HIPS p25": float(np.nanpercentile(r, 25)),
                 "published ÷ HIPS p75": float(np.nanpercentile(r, 75)), "Deming slope": sl, "Deming intercept": ic,
                 "Adama pairs": int((pd.to_datetime(ad.date).dt.month.map(SEASON_OF) == s_).sum())})
SEASONS8 = pd.DataFrame(rows)

# the contract, written before the Addis city pairs are read (edit, then keep the saved file as the record)
CONTRACT = {
    "status": "DRAFT: proposed values to agree with Ann and Satoshi before the Addis city pairs arrive; not yet adopted",
    "written": time.strftime("%Y-%m-%d"),
    "reference": "TOR EC (IMPROVE_A protocol, ECTR) on collocated quartz filters, same sampling interval; concentration in µg/m³ with each filter's own volume",
    "frozen models": [f"full IMPROVE calibration, {MAIN}, k = {k_main}", f"full IMPROVE calibration, raw, k = {k_raw}",
                      "frozen AIRSpec / VIBES full-pool (addis_validation_freeze, run 83dcf32e…)", "SPARTAN published EC (deployed model, comparison only)"],
    "pass rules": [
        {"rule": "mean bias", "test": "|mean (FTIR − TOR) / TOR|", "limit": 0.20, "unit": "fraction of TOR"},
        {"rule": "Deming slope vs TOR", "test": "slope", "limit": [0.8, 1.2], "unit": "", "note": "reported only with ≥ 8 pairs spanning a factor of 3 in TOR EC"},
        {"rule": "intercept", "test": "|intercept|", "limit": 0.5, "unit": "µg/m³"},
        {"rule": "per-pair agreement", "test": "|FTIR − TOR| ≤ max(0.5 µg/m³, 30 % of TOR)", "limit": 0.8, "unit": "share of pairs"},
    ],
    "low-signal handling": "pairs with TOR EC below 3× the field-blank MDL are reported but excluded from the pass rules; negative FTIR predictions are kept as negative",
    "eligibility": "identity-confirmed quartz/PTFE pairs, start times within 1 h, volume ratio 0.8–1.25; decided from metadata before TOR is opened",
    "representativeness check": "before scoring: EC and PM per filter, and spectral distance, against SPARTAN Addis (Section 8 plots)",
    "precision": "with 10 pairs the 95 % interval on mean bias is about ± the 'Addis-like' half width at n = 10 (table PRECISION)",
    "caveat": "about 10 filters from a different site, sampler and flow: a check of the calibration, not a recalibration; they test the Addis problem only where they overlap SPARTAN Addis (season, loading per filter, OC/EC)",
}

# the Adama pilot against the contract (a pilot, not the test: date-candidate pairs, unconfirmed IDs)
PILOT = []
for model, col in (("frozen AIRSpec", ad.airspec_frozen_ugm3.to_numpy()), ("frozen VIBES", ad.vibes_frozen_ugm3.to_numpy()), (f"full {MAIN} (this notebook)", pred_D)):
    for ref, r in (("ECTR", ad.ec_tr_ugm3.to_numpy()), ("ECTT", ad.ec_tt_ugm3.to_numpy())):
        for scope, m in (("all five", np.ones(len(ad), bool)), ("three unflagged", ad.flag.eq("").to_numpy())):
            rel = (col[m] - r[m]) / r[m]
            ok = np.abs(col[m] - r[m]) <= np.maximum(0.5, 0.3 * r[m])
            PILOT.append({"model": model, "reference": ref, "pairs": scope, "n": int(m.sum()), "mean relative bias": float(rel.mean()),
                          "passes |bias| ≤ 20 %": bool(abs(rel.mean()) <= 0.2), "share of pairs within tolerance": float(ok.mean())})
PILOT = pd.DataFrame(PILOT)
ADAMA_PAIRS = ad.assign(**{f"FTIR {MAIN} (this notebook)": pred_D, "distance percentile": dist_D})
SAVE.update({"e8_points": PTS, "e8_network": NET, "e8_neighbour_sites": NB, "e8_precision": PRECISION, "e8_pilot": PILOT, "e8_adama_pairs": ADAMA_PAIRS,
             "e8_target": TARGET, "e8_volume": VOLUME, "e8_seasons": SEASONS8})
(RESULTS / "e8_contract.json").write_text(json.dumps(CONTRACT, indent=2, ensure_ascii=False))
print(f"Adama distance percentiles {np.round(dist_D)}; Addis median {np.nanmedian(dist_A):.0f}")
print(f"precision at 10 pairs: ±{PRECISION.query('pairs == 10')['half width (relative bias)'].round(2).tolist()} (Addis-like, all)")
display(NB.sort_values("Addis", ascending=False).head(10)); display(PILOT); display(TARGET.round(2)); display(VOLUME.round(1)); display(SEASONS8.round(2))'''))

cells.append(code(r'''# save every table of this run
for name, df in SAVE.items():
    df.to_csv(RESULTS / f"{name}.csv", index=False)
(RESULTS / "settings.json").write_text(json.dumps({"k_selected": KSEL, "MAIN": MAIN, "N_SELECT": N_SELECT, "POOL": POOL, "SEED": SEED,
                                                   "LAMBDA_ADDIS": LAMBDA_ADDIS, "bundle": MANIFEST["created"], "heavy_cut_pm_ug": float(heavy_cut)}, indent=2))
print(f"saved {len(list(RESULTS.iterdir()))} files to {RESULTS}")'''))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                    "language_info": {"name": "python"}, "colab": {"provenance": []}})
nbf.write(nb, OUT)
print(f"wrote {OUT}")
