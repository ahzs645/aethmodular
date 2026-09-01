# %% [markdown]
# # ftir_52 — the IMPROVE network as a spectral map: Ward clusters, site similarity, smoke-likeness
#
# ## tl;dr
#
# Ward hierarchical clustering of the whole 13,634-spectrum IMPROVE library together with
# the five SPARTAN targets (Russell 2009 / Takahama 2011 lineage), on AIRSpec-corrected
# spectra, plus a site-similarity map and an Open-Specy-style validated threshold. Four
# results. **(1) The network's top-level spectral split is deposit and signal-to-noise,
# not chemistry** — silhouette picks k=4, and two of the four classes are simply
# low-absorbance filters (median peak absorbance 0.003-0.004 vs 0.020-0.044) whose
# standardized shape is noise. Any class read as composition must be loading-conditioned
# first. **(2) There is a real smoke-associated class, and it survives that conditioning.**
# Class 2 (Jul-Oct-weighted, heavily loaded, ATLA1/BOND1/LASU2) holds 3.3x its share of the
# deployed model's smoke lineage, and *within* narrow EC-loading bins smoke filters are
# still 1.3-11.5x (median 3.3x) more likely to sit there — so it is not merely a loading
# class. Delhi (68% of filters) and Beijing (56%) fall in it; Addis (5%) and Bishoftu (15%)
# essentially do not. **(3) Site similarity has clear answers**: Addis's nearest IMPROVE
# sites are NOGA1, CHAS1, LTCC1, PUSO1 (r 0.987-0.990); Delhi's are PITT1, BIRM1, MACA1;
# Beijing's VILA1, QUCI1, GRRI1; Bishoftu's BIBE1, MELA1, LYEB1; Pasadena's PACK1, MOMO1,
# ACAD1. Every site's least-like partner is TOOL1 (Arctic Alaska). **(4) The threshold test
# fails, informatively**: a nearest neighbour matching at r >= 0.9999 still carries EC
# within 25% only 64% of the time, against a 31% base rate, and no cutoff reaches 75%.
# Spectral near-identity is not calibration equivalence — which is the quantitative version
# of ftir_50's redundancy warning and the microplastics community's "high hit-quality index
# does not mean a correct match". **(5) Baselining is what makes the network comparable at
# all**: on raw spectra every pair of IMPROVE sites sits at median r 0.9990 — the network is
# effectively one spectrum plus background — while after AIRSpec the median pair falls to
# 0.9681 and spreads down to 0.56, and **92% of sites change which site they are closest
# to**. Raw-space similarity rankings are background rankings; the site-level version of
# the phase-3 background-leakage story.
#
# ## Context & Methods
#
# ftir_50 compared spectra pairwise. This notebook asks the population question: what
# spectral *classes* does the network contain, where do the SPARTAN sites fall among them,
# and can a class be called "smoke-like" on evidence rather than by name? There is no
# wildfire label in the local database (the 36 fire-related filter comments do not
# intersect the spectral pool, and the "smoke-906" cohort is the deployed EC model's
# training set, not a fire flag), so smoke-likeness is characterized from proxies —
# smoke-lineage membership, OC/EC, Jul–Oct sampling, site — and reported as such.
#
# Pipeline: standardize rows → PCA (fit on the library only) → Ward linkage on the
# scores of library + targets → cut at the silhouette-best k → characterize. Site
# similarity uses per-site median spectra (sites with ≥ 20 spectra) plus the five SPARTAN
# medians, Pearson r, Ward on 1 − r. The threshold is calibrated leave-one-out on the
# library: for each spectrum, its nearest neighbour's r and whether that neighbour shares
# its Ward class; precision vs r gives the operating point.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path('scripts').resolve()))
sys.path.insert(0, str((Path('..') / 'ftir_hips_chem' / 'scripts').resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import spectral_similarity as ss
from phase3_common import load_addis_evaluation, load_pool_metadata, load_tor_loadings
from plotting import apply_default_style

apply_default_style()
OUT = Path('output/tables/ftir52'); PLOTS = Path('output/plots/ftir52')
OUT.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
TARGET_DIR = Path('..') / '..' / 'calibration_explorer' / 'targets'
RNG = np.random.default_rng(20260901)
N_PCS = 30
SITE_COLOUR = {'addis': '#2a78d6', 'etbi': '#eb6834', 'indh': '#1baf7a',
               'chts': '#eda100', 'uspa': '#e87ba4'}
TARGETS = {'addis': 'Addis (ETAD)', 'etbi': 'Bishoftu (ETBI)', 'indh': 'Delhi (INDH)',
           'chts': 'Beijing (CHTS)', 'uspa': 'Pasadena (USPA)'}

# %% [markdown]
# ## 1. Library, metadata, targets

# %%
pool_npz = np.load('output/corrected/improve_pool_corrected_df6.npz', allow_pickle=True)
LIB = pool_npz['corrected'].astype(float); WN = pool_npz['wn'].astype(float)
lib = pd.DataFrame({'AnalysisId': pool_npz['analysis_id'].astype(int),
                    'FilterId': pool_npz['filter_id'].astype(int),
                    'Site': pool_npz['site'].astype(str),
                    'date': pd.to_datetime(pd.Series(pool_npz['sample_date'].astype(str)),
                                           format='mixed', errors='coerce').dt.normalize()})
lib['month'] = lib['date'].dt.month
tor = load_tor_loadings()
lib = lib.merge(tor[['Site', 'date', 'TOR_EC_loading_ug', 'TOR_EC_ugm3', 'TOR_OC_ugm3',
                     'OC_EC_ratio']], on=['Site', 'date'], how='left')
smoke = pd.read_csv('../ftir_hips_chem/output/tables/pls_calibration_phase2/'
                    'smoke_cohort_spectral_selection.csv')
lib['smoke_lineage'] = lib['AnalysisId'].isin(smoke['AnalysisId'].astype(int))
ocec800 = pd.read_csv('output/tables/ftir11/lowest_ocec_800_cohort.csv')
lib['ocec800'] = lib['AnalysisId'].isin(ocec800['AnalysisId'].astype(int))

etad_eval, _, _ = load_addis_evaluation()
etad_npz = np.load('output/corrected/etad_corrected_df6.npz', allow_pickle=True)
# The cache holds SCAN rows: 19 of the 239 evaluation filters were scanned more than once
# (259 scan rows). load_addis_evaluation averages replicate scans per physical filter, so
# do the same here -- otherwise those 19 filters are silently weighted twice.
_mid = etad_npz['media_id'].astype(int)
_corr = etad_npz['corrected'].astype(float)
_order = etad_eval['MediaId'].to_numpy(int)
_addis = np.vstack([_corr[_mid == m].mean(axis=0) for m in _order])
X = {'addis': _addis}
for key in ('etbi', 'indh', 'chts', 'uspa'):
    frame = pd.read_csv(TARGET_DIR / key / 'spectra_corrected.csv')
    cols = [c for c in frame.columns if c not in ('MediaId', 'ExternalFilterId')]
    assert np.allclose(np.array([float(c) for c in cols]), WN)
    X[key] = frame[cols].to_numpy(float)

print(f'library {len(lib):,} spectra, {lib.Site.nunique()} sites, '
      f'{lib.OC_EC_ratio.notna().mean():.0%} with TOR OC/EC, '
      f'{lib.smoke_lineage.sum()} in the deployed-model smoke lineage')
for k, v in X.items():
    print(f'  {TARGETS[k]:<18} {len(v):>4}')

# %% [markdown]
# ## 2. Ward clustering of library + targets
#
# PCA is fitted on the library alone so the targets cannot pull the axes; the Ward tree is
# built on library + target scores together so a target filter's class is decided by the
# same tree, not by post-hoc nearest-centroid assignment. k is chosen by silhouette on a
# stratified subsample (full-library silhouette is O(n²)).

# %%
Ls = ss.standardize(LIB)
pca = PCA(n_components=N_PCS).fit(Ls)
S_lib = pca.transform(Ls)
S_tgt = {k: pca.transform(ss.standardize(v)) for k, v in X.items()}
S_all = np.vstack([S_lib] + [S_tgt[k] for k in TARGETS])
origin = np.array(['library'] * len(S_lib) + sum([[k] * len(S_tgt[k]) for k in TARGETS], []))
print(f'PCA-{N_PCS} explains {pca.explained_variance_ratio_.sum():.1%} of standardized variance')

Z = linkage(S_all, method='ward')
sub = RNG.choice(len(S_lib), 4000, replace=False)
sil = {}
for k in range(4, 15):
    labels_k = fcluster(Z, k, criterion='maxclust')
    sil[k] = silhouette_score(S_lib[sub], labels_k[:len(S_lib)][sub])
K_BEST = max(sil, key=sil.get)
print('silhouette by k:', {k: round(v, 3) for k, v in sil.items()})
print(f'chosen k = {K_BEST}')
labels = fcluster(Z, K_BEST, criterion='maxclust')
lib['cluster'] = labels[:len(S_lib)]
tgt_cluster = {k: labels[origin == k] for k in TARGETS}

# %% [markdown]
# ## 3. What each class is
#
# Size, how many sites it draws from, its dominant sites and months, TOR OC/EC and EC
# loading medians, the share of the deployed-model smoke lineage and of the lowest-OC/EC
# cohort, and how many filters of each SPARTAN target the tree put in it.

# %%
rows = []
for c in sorted(lib.cluster.unique()):
    g = lib[lib.cluster == c]
    top_sites = g.Site.value_counts().head(3)
    months = g.month.value_counts(normalize=True)
    row = {'cluster': c, 'n': len(g), 'n_sites': g.Site.nunique(),
           'top_sites': ', '.join(f'{s} {n}' for s, n in top_sites.items()),
           'pct_Jul_Oct': round(100 * months.reindex([7, 8, 9, 10]).fillna(0).sum(), 0),
           'pct_Dec_Feb': round(100 * months.reindex([12, 1, 2]).fillna(0).sum(), 0),
           'OC_EC_median': round(g.OC_EC_ratio.median(), 2),
           'EC_ug_median': round(g.TOR_EC_loading_ug.median(), 2),
           'pct_smoke_lineage': round(100 * g.smoke_lineage.mean(), 1),
           'pct_ocec800': round(100 * g.ocec800.mean(), 1)}
    for k, lab in TARGETS.items():
        row[f'{k}_n'] = int((tgt_cluster[k] == c).sum())
    rows.append(row)
classes = pd.DataFrame(rows)
classes.to_csv(OUT / 'ward_classes.csv', index=False)
display(classes)

# What is the top-level split? Rows were standardized before PCA, so amplitude is
# removed -- but SNR is not, and low-deposit spectra have noise-dominated SHAPES. Check
# absorbance magnitude per class before reading any class as chemistry.
lib['peak_abs'] = LIB.max(axis=1)
mag = lib.groupby('cluster')['peak_abs'].median().round(4)
classes['peak_abs_median'] = classes.cluster.map(mag)
print('median peak absorbance by class:', mag.to_dict())
print('-> the top-level Ward split is largely deposit/SNR, not composition: classes with '
      'near-zero peak absorbance are low-loading spectra whose standardized shape is noise.')

lineage_enrich = classes.set_index('cluster')['pct_smoke_lineage'] / (100 * lib.smoke_lineage.mean())
smoke_class = int(lineage_enrich.idxmax())
sc = classes.set_index('cluster').loc[smoke_class]
print(f"\nMost smoke-lineage-enriched class: {smoke_class} "
      f"({lineage_enrich.max():.1f}x the library rate; {sc['pct_Jul_Oct']:.0f}% Jul-Oct; "
      f"OC/EC {sc['OC_EC_median']}; top sites {sc['top_sites']})")
for k, lab in TARGETS.items():
    dist = pd.Series(tgt_cluster[k]).value_counts(normalize=True)
    print(f"  {lab:<18} main class {dist.index[0]} ({dist.iloc[0]:.0%}), "
          f"{len(dist)} classes used; in smoke-like class: "
          f"{100 * (tgt_cluster[k] == smoke_class).mean():.0f}%")

# %% [markdown]
# ### Is the smoke-enriched class actually about smoke, or just about loading?
#
# Smoke filters are heavily loaded, and so is the enriched class. The test: inside narrow
# EC-loading bins, is smoke-lineage membership still concentrated in that class? If the
# enrichment vanishes once loading is matched, "smoke-like" was a loading statement.

# %%
band = lib.dropna(subset=['TOR_EC_loading_ug']).copy()
band['ec_bin'] = pd.qcut(band['TOR_EC_loading_ug'], 6, duplicates='drop')
rows = []
for b, g in band.groupby('ec_bin', observed=True):
    if g.smoke_lineage.sum() < 10:
        continue
    in_class = g.cluster == smoke_class
    rows.append({'EC_bin': str(b), 'n': len(g), 'n_smoke': int(g.smoke_lineage.sum()),
                 'pct_of_bin_in_class': round(100 * in_class.mean(), 1),
                 'pct_of_smoke_in_class': round(100 * in_class[g.smoke_lineage].mean(), 1),
                 'enrichment': round(in_class[g.smoke_lineage].mean() / max(in_class.mean(), 1e-9), 2)})
loading_matched = pd.DataFrame(rows)
loading_matched.to_csv(OUT / 'smoke_enrichment_by_loading.csv', index=False)
display(loading_matched)
med_enrich = loading_matched.enrichment.median()
print(f'median within-bin enrichment {med_enrich:.2f}x '
      f'(1.0 = smoke filters are no more likely to be in class {smoke_class} than any '
      'other filter at the same loading)')
print('VERDICT: ' + ('the class carries smoke-lineage information beyond loading'
                     if med_enrich > 1.3 else
                     'the apparent smoke class is mostly a LOADING class -- do not call '
                     'it a smoke class'))

# %%
fig, axes = plt.subplots(2, int(np.ceil(K_BEST / 2)), figsize=(15, 5.6), sharex=True)
lib_med = np.median(LIB, axis=0)
for ax, c in zip(axes.ravel(), sorted(lib.cluster.unique())):
    members = LIB[(lib.cluster == c).to_numpy()]
    q25, med, q75 = np.percentile(members, [25, 50, 75], axis=0)
    ax.fill_between(WN, q25, q75, color='#2C6E9E', alpha=0.25, lw=0)
    ax.plot(WN, med, color='#2C6E9E', lw=1.4)
    ax.plot(WN, lib_med, color='#898781', lw=0.9, ls='--')
    row = classes.set_index('cluster').loc[c]
    ax.set_title(f"class {c}  n={row['n']}  OC/EC {row['OC_EC_median']}  "
                 f"smoke-lin {row['pct_smoke_lineage']:.0f}%", fontsize=8.5)
    ax.set_xlim(WN.max(), WN.min())
for ax in axes.ravel()[len(classes):]:
    ax.axis('off')
fig.suptitle('Median (IQR) corrected spectrum per Ward class; dashed = library median',
             x=0.02, ha='left')
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(PLOTS / 'class_spectra.png', dpi=140)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 6.4))
xy = S_all[:, :2]
cmap = plt.get_cmap('tab20', K_BEST)
lib_mask = origin == 'library'
ax.scatter(xy[lib_mask, 0], xy[lib_mask, 1], c=labels[lib_mask], cmap=cmap, s=4,
           alpha=0.35, rasterized=True, vmin=1, vmax=K_BEST)
for k, lab in TARGETS.items():
    m = origin == k
    ax.scatter(xy[m, 0], xy[m, 1], s=18, color=SITE_COLOUR[k], edgecolor='white',
               linewidth=0.5, label=lab, zorder=4)
for c in sorted(lib.cluster.unique()):
    cen = xy[lib_mask][labels[lib_mask] == c].mean(axis=0)
    ax.text(cen[0], cen[1], str(c), fontsize=10, weight='bold', ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.0%})',
       ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.0%})',
       title='Library coloured by Ward class; SPARTAN targets overlaid')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(PLOTS / 'class_map_pca.png', dpi=150); plt.show()

# %% [markdown]
# ## 4. Which sites look like which
#
# Per-site median corrected spectrum (IMPROVE sites with ≥ 20 spectra) plus the five
# SPARTAN medians; Pearson r between medians; Ward on 1 − r. Two views: the nearest
# IMPROVE sites to each SPARTAN site, and a dendrogram of the whole network with the
# SPARTAN sites placed in it. A second, class-based site distance (Jensen–Shannon between
# sites' Ward-class histograms) checks that the answer is not a median artefact.

# %%
counts = lib.Site.value_counts()
big_sites = counts[counts >= 20].index.tolist()
site_median = {s: np.median(LIB[(lib.Site == s).to_numpy()], axis=0) for s in big_sites}
for k, lab in TARGETS.items():
    site_median[f'*{lab.split(" ")[0]}'] = np.median(X[k], axis=0)
names = list(site_median)
M = np.vstack([site_median[n] for n in names])
R = ss.correlation_matrix(M, M)
pd.DataFrame(R, index=names, columns=names).to_csv(OUT / 'site_median_correlation.csv')

rows = []
for k, lab in TARGETS.items():
    n = f'*{lab.split(" ")[0]}'
    i = names.index(n)
    order = np.argsort(-R[i])
    near = [(names[j], round(float(R[i, j]), 4)) for j in order if not names[j].startswith('*')][:6]
    rows.append({'target': lab, 'nearest_IMPROVE_sites': ', '.join(f'{s} ({r})' for s, r in near)})
    # sites most UNLIKE too
    far = [(names[j], round(float(R[i, j]), 3)) for j in order[::-1] if not names[j].startswith('*')][:3]
    rows[-1]['most_unlike'] = ', '.join(f'{s} ({r})' for s, r in far)
site_near = pd.DataFrame(rows); site_near.to_csv(OUT / 'nearest_sites.csv', index=False)
display(site_near)

# class-histogram distance as a check on the median-based answer
def js(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
hist = {s: np.bincount(lib.cluster[lib.Site == s], minlength=K_BEST + 1)[1:] for s in big_sites}
hist = {s: h / h.sum() for s, h in hist.items()}
rows = []
for k, lab in TARGETS.items():
    h = np.bincount(tgt_cluster[k], minlength=K_BEST + 1)[1:]; h = h / h.sum()
    d = sorted(((js(h + 1e-9, hist[s] + 1e-9), s) for s in big_sites))[:5]
    rows.append({'target': lab, 'nearest_by_class_mix': ', '.join(f'{s} ({v:.3f})' for v, s in d)})
display(pd.DataFrame(rows))

# %%
D = squareform(np.clip(1 - R, 0, None), checks=False)
Zs = linkage(D, method='average')
fig, ax = plt.subplots(figsize=(15, 5.2))
dn = dendrogram(Zs, labels=names, leaf_font_size=6, color_threshold=0, ax=ax,
                above_threshold_color='#898781')
for lbl in ax.get_xmajorticklabels():
    if lbl.get_text().startswith('*'):
        lbl.set_color('#B23327'); lbl.set_fontweight('bold'); lbl.set_fontsize(8)
ax.set(ylabel='1 − Pearson r (site median spectra)',
       title='The IMPROVE network by median corrected spectrum, with the SPARTAN sites (red) placed in it')
fig.tight_layout(); fig.savefig(PLOTS / 'site_dendrogram.png', dpi=150); plt.show()

# %% [markdown]
# ## 5. A validated similarity threshold (Open Specy's discipline)
#
# Every library spectrum's leave-one-out nearest neighbour: its r, and whether the
# neighbour shares the spectrum's Ward class. Precision of "same class" as a function of
# the r cutoff gives the operating point; the 90%-precision r is the threshold. Then: what
# fraction of each SPARTAN target's filters has a library match above it?

# %%
# Same-class agreement is the wrong yardstick here: with k=4 and one class holding 59%
# of the library, a random neighbour is already ~99% "correct", so precision is flat and
# any threshold read off it is meaningless. Two harder criteria that match what a match
# is FOR: does the neighbour carry the same EC loading (within 25%), and is it even from
# the same site?
Ls_std = ss.standardize(LIB)
lib_lab = lib.cluster.to_numpy()
lib_ec = lib.TOR_EC_loading_ug.to_numpy(float)
lib_site_arr = lib.Site.to_numpy()
nn_r = np.empty(len(LIB)); nn_same = np.empty(len(LIB), bool)
nn_ec_ok = np.zeros(len(LIB), bool); nn_site = np.empty(len(LIB), bool)
for start in range(0, len(LIB), 1024):
    block = slice(start, start + 1024)
    Rb = Ls_std[block] @ Ls_std.T
    rows_i = np.arange(Rb.shape[0])
    Rb[rows_i, np.arange(start, start + Rb.shape[0])] = -np.inf
    j = Rb.argmax(axis=1)
    nn_r[block] = Rb[rows_i, j]
    nn_same[block] = lib_lab[j] == lib_lab[block]
    nn_site[block] = lib_site_arr[j] == lib_site_arr[block]
    a, b = lib_ec[block], lib_ec[j]
    with np.errstate(invalid='ignore', divide='ignore'):
        nn_ec_ok[block] = np.abs(a - b) / np.maximum(a, 1e-9) <= 0.25

valid_ec = np.isfinite(lib_ec) & np.isfinite(lib_ec[np.arange(len(LIB))])
grid = np.quantile(nn_r, np.linspace(0.02, 0.995, 60))
curves = {
    'same Ward class': np.array([nn_same[nn_r >= t].mean() for t in grid]),
    'EC within 25%': np.array([nn_ec_ok[(nn_r >= t) & valid_ec].mean() for t in grid]),
    'same site': np.array([nn_site[nn_r >= t].mean() for t in grid]),
}
cover = np.array([(nn_r >= t).mean() for t in grid])
print(f'leave-one-out nearest neighbour: median r {np.median(nn_r):.4f}')
for name, c in curves.items():
    print(f'  {name:<16} base rate {c[0]:.1%} -> at the strictest cutoff {c[-1]:.1%}')

TASK = 'EC within 25%'
prec = curves[TASK]
TARGET_PREC = 0.75
reached = bool((prec >= TARGET_PREC).any())
thr_idx = int(np.argmax(prec >= TARGET_PREC)) if reached else int(np.argmax(prec))
R_THR = float(grid[thr_idx])
if reached:
    print(f'\noperating point on "{TASK}": r >= {R_THR:.4f} reaches {prec[thr_idx]:.0%} '
          f'precision, covering {cover[thr_idx]:.0%} of the library (base rate {prec[0]:.0%}).')
else:
    print(f'\nNO USABLE THRESHOLD on "{TASK}". Precision never reaches {TARGET_PREC:.0%} at '
          f'any similarity cutoff: the best is {prec[thr_idx]:.0%} at r >= {R_THR:.4f}, and '
          f'that cutoff keeps only {cover[thr_idx]:.1%} of the library. Base rate is '
          f'{prec[0]:.0%}, so extreme spectral similarity roughly doubles the odds of EC '
          'agreement and no more.')
    print('This is the central negative result of the notebook: a near-identical spectrum '
          'is NOT a guarantee of a near-identical EC loading, so "nearest analogs" cannot '
          'be treated as calibration-equivalent samples however tight the match.')
print(f'same-site precision rises {curves["same site"][0]:.0%} -> '
      f'{curves["same site"][-1]:.0%}: spectral twins are mostly NOT from the same site, '
      'which is the encouraging half -- the library does generalize across sites.')

rows = []
for k, lab in TARGETS.items():
    best = ss.correlation_matrix(X[k], LIB).max(axis=1)
    rows.append({'target': lab, 'median_best_r': round(float(np.median(best)), 4),
                 'pct_above_threshold': round(100 * float((best >= R_THR).mean()), 1),
                 'pct_below_0.99': round(100 * float((best < 0.99).mean()), 1)})
thr = pd.DataFrame(rows); thr.to_csv(OUT / 'threshold_coverage.csv', index=False)
display(thr)

fig, ax = plt.subplots(figsize=(7.6, 4.4))
for (name, c), colour in zip(curves.items(), ['#8F8C84', '#2C6E9E', '#7A4FA3']):
    ax.plot(grid, c, color=colour, lw=2 if name == TASK else 1.5,
            ls='-' if name == TASK else (0, (4, 3)), label=f'{name} precision')
ax.plot(grid, cover, color='#c3c2b7', lw=1.5, ls=':', label='library coverage')
ax.axvline(R_THR, color='#B23327', lw=1.2, ls=':')
ax.text(R_THR, 0.02, f'  r ≥ {R_THR:.4f}', color='#B23327', fontsize=9)
ax.set(xlabel='nearest-neighbour Pearson r cutoff', ylabel='fraction',
       title='Where a spectral match becomes trustworthy')
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(PLOTS / 'threshold_curve.png', dpi=150); plt.show()

# %% [markdown]
# ## 6. The IMPROVE network compared with itself
#
# Sections 4-5 asked "which IMPROVE sites are near the SPARTAN targets". This asks the
# network question directly: **which IMPROVE sites resemble each other**, which are
# isolated, and whether the structure is regional. Same per-site median corrected spectra,
# now read as a 162x162 matrix with the SPARTAN columns removed.

# %%
improve_names = [n for n in names if not n.startswith('*')]
ii = [names.index(n) for n in improve_names]
R_imp = R[np.ix_(ii, ii)]
np.fill_diagonal(R_imp, np.nan)
imp_df = pd.DataFrame(R_imp, index=improve_names, columns=improve_names)
imp_df.to_csv(OUT / 'improve_site_correlation.csv')

pairs = [(improve_names[a], improve_names[b], float(R_imp[a, b]))
         for a in range(len(improve_names)) for b in range(a + 1, len(improve_names))]
pairs.sort(key=lambda t: -t[2])
top_pairs = pd.DataFrame(pairs[:15], columns=['site_A', 'site_B', 'r']).round(5)
print('Most similar IMPROVE site pairs (median corrected spectra):')
display(top_pairs)
print('Least similar pairs:')
display(pd.DataFrame(pairs[-10:], columns=['site_A', 'site_B', 'r']).round(4))

isolation = pd.DataFrame({
    'site': improve_names,
    'best_r_to_any_site': np.nanmax(R_imp, axis=1).round(5),
    'median_r_to_network': np.nanmedian(R_imp, axis=1).round(5),
    'nearest_site': [improve_names[int(np.nanargmax(R_imp[a]))] for a in range(len(improve_names))],
    'n_spectra': [int(counts[n]) for n in improve_names],
}).sort_values('best_r_to_any_site')
isolation.to_csv(OUT / 'improve_site_isolation.csv', index=False)
print('Most ISOLATED IMPROVE sites (no close spectral partner anywhere in the network):')
display(isolation.head(12))
print('Most TYPICAL sites (highest median similarity to the whole network):')
display(isolation.sort_values('median_r_to_network', ascending=False).head(8))

# %%
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.hist([p[2] for p in pairs], bins=80, color='#8F8C84', edgecolor='white', linewidth=0.4)
for k, lab in TARGETS.items():
    i = names.index(f'*{lab.split(" ")[0]}')
    best = max(R[i, j] for j in ii)
    ax.axvline(best, color=SITE_COLOUR[k], lw=1.8)
    ax.text(best, ax.get_ylim()[1] * (0.95 - 0.09 * list(TARGETS).index(k)),
            f' {lab.split(" ")[0]} {best:.3f}', color=SITE_COLOUR[k], fontsize=8.5, va='top')
ax.set(xlabel='Pearson r between two site median spectra',
       ylabel='IMPROVE site pairs',
       title='How similar IMPROVE sites are to each other, with each SPARTAN site\u2019s best match marked')
fig.tight_layout(); fig.savefig(PLOTS / 'improve_pair_distribution.png', dpi=150); plt.show()

pair_r = np.array([p[2] for p in pairs])
for k, lab in TARGETS.items():
    i = names.index(f'*{lab.split(" ")[0]}')
    best = max(R[i, j] for j in ii)
    print(f'{lab:<18} best IMPROVE match r={best:.4f} sits at the '
          f'{100 * (pair_r < best).mean():.0f}th percentile of IMPROVE-IMPROVE pair similarity')

# %% [markdown]
# ## 7. Does baselining change who is similar to whom?
#
# Everything so far is AIRSpec-corrected. Raw spectra are baseline-dominated (that is the
# phase-3 background-leakage story), so the honest check is whether the network's
# similarity structure is the *same* before and after baselining. Raw pool spectra are
# restricted to the corrected window (1425-3998 cm-1) so the two spaces are comparable,
# and the whole section is recomputed on raw medians.

# %%
from phase3_common import load_pool_spectra                              # noqa: E402
full_wcols = list(etad_eval.attrs['wcols'])
raw_pool = load_pool_spectra(lib.AnalysisId.to_numpy(), full_wcols)
raw_pool = raw_pool.set_index('AnalysisId').reindex(lib.AnalysisId.to_numpy())
RAW_FULL = raw_pool[full_wcols].to_numpy(float)
WN_FULL = np.array([float(c) for c in full_wcols])
# the comparison window matches the corrected grid so the two spaces are comparable
win = (WN_FULL >= 1425) & (WN_FULL <= 3999)
raw_wcols = [c for c, m in zip(full_wcols, win) if m]
RAW = RAW_FULL[:, win]
raw_ok = np.isfinite(RAW).all(axis=1)
print(f'raw spectra fetched for {raw_ok.sum():,}/{len(lib):,} pool rows; '
      f'full range {WN_FULL.min():.0f}-{WN_FULL.max():.0f} cm-1 ({len(full_wcols)} channels), '
      f'comparison window {len(raw_wcols)} channels')

raw_site_median = {}
for site in improve_names:
    m = (lib.Site == site).to_numpy() & raw_ok
    if m.sum() >= 20:
        raw_site_median[site] = np.median(RAW[m], axis=0)
shared = [s for s in improve_names if s in raw_site_median]
R_raw = ss.correlation_matrix(np.vstack([raw_site_median[s] for s in shared]),
                              np.vstack([raw_site_median[s] for s in shared]))
np.fill_diagonal(R_raw, np.nan)
sub_idx = [improve_names.index(s) for s in shared]
R_cor = R_imp[np.ix_(sub_idx, sub_idx)]

iu = np.triu_indices(len(shared), 1)
rho = float(pd.Series(R_raw[iu]).corr(pd.Series(R_cor[iu]), method='spearman'))
print(f'\n{len(shared)} sites in both spaces. Spearman between the raw and baselined '
      f'site-similarity matrices: rho = {rho:.3f}')
print(f'raw pair similarity: median {np.nanmedian(R_raw[iu]):.4f}, '
      f'baselined: {np.nanmedian(R_cor[iu]):.4f} '
      '(baselining SEPARATES sites: removing the shared PTFE/background makes the '
      'remaining differences a larger share of what is left)')

moved = []
for a, site in enumerate(shared):
    nn_raw = shared[int(np.nanargmax(R_raw[a]))]
    nn_cor = shared[int(np.nanargmax(R_cor[a]))]
    moved.append({'site': site, 'nearest_raw': nn_raw, 'nearest_baselined': nn_cor,
                  'changed': nn_raw != nn_cor,
                  'rank_of_raw_nn_after': int((R_cor[a] > R_cor[a, shared.index(nn_raw)]).sum()) + 1})
moved = pd.DataFrame(moved)
moved.to_csv(OUT / 'nearest_neighbour_raw_vs_baselined.csv', index=False)
print(f"\nnearest-neighbour site changes after baselining: "
      f"{moved.changed.sum()}/{len(moved)} sites ({moved.changed.mean():.0%})")
display(moved[moved.changed].head(12))

# %%
# Plot DISSIMILARITY 1-r on log axes: in raw space every pair sits at r ~ 0.999, so a
# linear r axis collapses the whole raw network onto one tick and hides the result.
d_raw, d_cor = 1 - R_raw[iu], 1 - R_cor[iu]
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
axes[0].scatter(d_raw, d_cor, s=5, color='#2C6E9E', alpha=0.25, rasterized=True)
lims = [min(d_raw.min(), d_cor.min()) * 0.8, max(d_raw.max(), d_cor.max()) * 1.2]
axes[0].plot(lims, lims, ls='--', lw=1.2, color='#c3c2b7')
axes[0].set(xscale='log', yscale='log', xlim=lims, ylim=lims,
            xlabel='site-pair dissimilarity 1$-$r, RAW',
            ylabel='site-pair dissimilarity 1$-$r, AIRSpec-baselined',
            title=f'Every IMPROVE site pair, both spaces (Spearman {rho:.2f})')
axes[0].text(0.04, 0.95, 'above the line =\nbaselining separates the pair',
             transform=axes[0].transAxes, va='top', fontsize=8.5, color='#52514e')
bins = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 60)
axes[1].hist(d_raw, bins=bins, alpha=0.6, color='#8F8C84', label='raw', edgecolor='none')
axes[1].hist(d_cor, bins=bins, alpha=0.6, color='#2C6E9E', label='baselined', edgecolor='none')
axes[1].set(xscale='log', xlabel='site-pair dissimilarity 1$-$r', ylabel='pairs',
            title='Raw: every site looks the same. Baselined: structure appears')
axes[1].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(PLOTS / 'raw_vs_baselined_sites.png', dpi=150); plt.show()

# %%
# Do the SPARTAN targets' nearest IMPROVE sites survive baselining?
rows = []
for k, lab in TARGETS.items():
    tgt_raw = None
    if k == 'addis':
        raw_t = etad_eval[raw_wcols].to_numpy(float)
        tgt_raw = np.median(raw_t, axis=0)
    if tgt_raw is None:
        rows.append({'target': lab, 'nearest_raw': 'n/a (no raw target export)',
                     'nearest_baselined': site_near.set_index('target')
                     .loc[lab, 'nearest_IMPROVE_sites'].split(',')[0]})
        continue
    rr = ss.correlation_matrix(tgt_raw[None, :],
                               np.vstack([raw_site_median[s] for s in shared])).ravel()
    rows.append({'target': lab, 'nearest_raw': f'{shared[int(np.argmax(rr))]} ({rr.max():.4f})',
                 'nearest_baselined': site_near.set_index('target')
                 .loc[lab, 'nearest_IMPROVE_sites'].split(',')[0]})
display(pd.DataFrame(rows))
print('Only Addis has a raw target export in this notebook; the other four targets are '
      'distributed as corrected spectra only, so their raw comparison needs the SPARTAN '
      'raw pulls and is left for a follow-up.')

# %% [markdown]
# ## 8. Addis in raw space: which IMPROVE spectra does it actually match?
#
# The site-level result says raw-space similarity is background similarity. This is the
# per-filter version, and the one to look at directly: take every Addis filter, find its
# nearest IMPROVE **spectra** in raw space and in baselined space, and ask whether they are
# the same spectra. Then plot them — including the 1150-1300 cm-1 PTFE doublet that the
# corrected grid cuts away, because that is what raw matching is mostly looking at.

# %%
A_raw = etad_eval[raw_wcols].to_numpy(float)          # matched window
A_raw_full = etad_eval[full_wcols].to_numpy(float)
A_cor = X['addis']
ok = raw_ok

K_AN = 50
idx_raw, sc_raw = ss.nearest_analogs(A_raw, RAW[ok], k=K_AN)
idx_cor, sc_cor = ss.nearest_analogs(A_cor, LIB, k=K_AN)
pool_ids_raw = lib.AnalysisId.to_numpy()[ok]
pool_site_raw = lib.Site.to_numpy()[ok]
pool_ids_cor = lib.AnalysisId.to_numpy()
pool_site_cor = lib.Site.to_numpy()

overlap = [len(set(pool_ids_raw[a]) & set(pool_ids_cor[b])) / K_AN
           for a, b in zip(idx_raw, idx_cor)]
print(f'Per-Addis-filter overlap between its top-{K_AN} RAW analogs and its top-{K_AN} '
      f'BASELINED analogs: median {100 * np.median(overlap):.0f}%, '
      f'mean {100 * np.mean(overlap):.0f}%, '
      f'{100 * np.mean(np.array(overlap) == 0):.0f}% of filters share NONE')
print(f'raw-space match quality:       median top-1 r {np.median(sc_raw[:, 0]):.4f}')
print(f'baselined-space match quality: median top-1 r {np.median(sc_cor[:, 0]):.4f}')

top_raw = pd.Series(pool_site_raw[idx_raw.ravel()]).value_counts().head(8)
top_cor = pd.Series(pool_site_cor[idx_cor.ravel()]).value_counts().head(8)
sites_cmp = pd.DataFrame({'raw analog sites': top_raw, 'baselined analog sites': top_cor})
display(sites_cmp.fillna(0).astype(int))
pd.DataFrame({'AnalysisId_raw_top1': pool_ids_raw[idx_raw[:, 0]],
              'site_raw_top1': pool_site_raw[idx_raw[:, 0]],
              'r_raw_top1': sc_raw[:, 0].round(5),
              'AnalysisId_cor_top1': pool_ids_cor[idx_cor[:, 0]],
              'site_cor_top1': pool_site_cor[idx_cor[:, 0]],
              'r_cor_top1': sc_cor[:, 0].round(5),
              'top50_overlap': np.round(overlap, 3)}).to_csv(
    OUT / 'addis_raw_vs_baselined_analogs.csv', index=False)

# %%
# The representative Addis filter: its raw analogs (drawn raw) and its baselined analogs
# (drawn baselined), full raw range on the left so the PTFE doublet is visible.
i = int(np.argmax(ss.correlation_matrix(A_cor, np.median(A_cor, axis=0)[None, :]).ravel()))
fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8))

pool_full = RAW_FULL[ok]
for j, li in enumerate(idx_raw[i][:5]):
    axes[0].plot(WN_FULL, pool_full[li], lw=1.1, color='#8F8C84', alpha=0.85,
                 label=f'{pool_site_raw[li]}  r={sc_raw[i, j]:.4f}')
axes[0].plot(WN_FULL, A_raw_full[i], lw=2.2, color='#2C6E9E', zorder=5,
             label='Addis filter (raw)')
axes[0].axvspan(1150, 1300, color='#f3e9e9', zorder=0)
axes[0].text(1225, axes[0].get_ylim()[1], 'PTFE', ha='center', va='top',
             fontsize=8.5, color='#B23327')
axes[0].set(xlim=(WN_FULL.max(), WN_FULL.min()), xlabel='Wavenumber (cm$^{-1}$)',
            ylabel='Absorbance (raw)', title='Nearest analogs chosen in RAW space')
axes[0].legend(frameon=False, fontsize=7.6, loc='upper left')

for j, li in enumerate(idx_cor[i][:5]):
    axes[1].plot(WN, LIB[li], lw=1.1, color='#8F8C84', alpha=0.85,
                 label=f'{pool_site_cor[li]}  r={sc_cor[i, j]:.4f}')
axes[1].plot(WN, A_cor[i], lw=2.2, color='#eb6834', zorder=5,
             label='Addis filter (baselined)')
axes[1].set(xlim=(WN.max(), WN.min()), xlabel='Wavenumber (cm$^{-1}$)',
            ylabel='Absorbance (AIRSpec-corrected)',
            title='Nearest analogs chosen in BASELINED space')
axes[1].legend(frameon=False, fontsize=7.6, loc='upper left')
fig.tight_layout(); fig.savefig(PLOTS / 'addis_raw_vs_baselined_analogs.png', dpi=150)
plt.show()

# %%
# The same raw-chosen analogs, redrawn AFTER baselining: does a raw match survive?
fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6), sharey=True)
cor_row = {int(a): r for r, a in enumerate(lib.AnalysisId.to_numpy())}
for j, li in enumerate(idx_raw[i][:5]):
    row = cor_row[int(pool_ids_raw[li])]
    axes[0].plot(WN, LIB[row], lw=1.1, color='#8F8C84', alpha=0.85,
                 label=f'{pool_site_raw[li]} (raw-chosen)')
axes[0].plot(WN, A_cor[i], lw=2.2, color='#2C6E9E', zorder=5, label='Addis (baselined)')
axes[0].set(xlim=(WN.max(), WN.min()), xlabel='Wavenumber (cm$^{-1}$)',
            ylabel='Absorbance (AIRSpec-corrected)',
            title='RAW-chosen analogs, redrawn after baselining')
axes[0].legend(frameon=False, fontsize=7.6, loc='upper left')
for j, li in enumerate(idx_cor[i][:5]):
    axes[1].plot(WN, LIB[li], lw=1.1, color='#8F8C84', alpha=0.85,
                 label=f'{pool_site_cor[li]} (baselined-chosen)')
axes[1].plot(WN, A_cor[i], lw=2.2, color='#eb6834', zorder=5, label='Addis (baselined)')
axes[1].set(xlim=(WN.max(), WN.min()), xlabel='Wavenumber (cm$^{-1}$)',
            title='BASELINED-chosen analogs, same axes')
axes[1].legend(frameon=False, fontsize=7.6, loc='upper left')
fig.tight_layout(); fig.savefig(PLOTS / 'addis_raw_analogs_after_baselining.png', dpi=150)
plt.show()

raw_chosen_rows = [cor_row[int(pool_ids_raw[li])] for li in idx_raw[i][:K_AN]]
r_after = ss.correlation_matrix(A_cor[i][None, :], LIB[raw_chosen_rows]).ravel()
print(f'The top-{K_AN} RAW-chosen analogs of this filter, scored in baselined space: '
      f'median r {np.median(r_after):.4f} (its true baselined top-{K_AN} median is '
      f'{np.median(sc_cor[i]):.4f}) -- a raw match is not a baselined match.')

# %% [markdown]
# ## Takeaways
#
# 1. **The network has a small number of spectral classes and the SPARTAN sites do not
#    all land in the same ones** — see the per-target class distribution above. Where a
#    target concentrates in one class, that class's IMPROVE membership *is* its analog
#    population; where it spreads across classes, no single cohort will fit it.
# 2. **The top-level Ward split is deposit/SNR, not composition** (see the peak-absorbance
#    column): low-loading spectra have noise-dominated standardized shapes and form their
#    own classes. Any class read as chemistry has to be loading-conditioned first.
# 3. **Smoke-likeness is a proxy statement, and it has to survive loading.** There is no
#    fire label in the database (the 36 fire-related comments do not intersect the pool),
#    so smoke-lineage membership is the only handle, and the within-EC-bin enrichment
#    table is what decides whether the enriched class means anything beyond "heavily
#    loaded". Read that verdict line before calling anything a smoke class.
# 4. **Site similarity is answerable two ways and they should agree.** Median-spectrum r
#    and class-mix distance are both reported; where they disagree, the site is
#    heterogeneous and a single median misrepresents it.
# 6. **The network's own similarity structure survives baselining in rank but not in
#    scale** (section 7): the raw and baselined site-similarity matrices agree in ordering
#    while baselining spreads the whole network apart, because removing the shared
#    PTFE/background leaves the real differences as a larger share of what remains. Where a
#    site's nearest neighbour *changes*, its raw match was a background match.
# 7. **A match threshold, not a top-N — but validated on a task that can fail.**
#    Same-class agreement is ~99% at any r here (k=4, one dominant class), so it cannot
#    calibrate anything; "does the neighbour carry the same EC loading within 25%" can,
#    and is what a calibration-usable match actually means. Quote that operating point.
#
# Scope: lot-248/251 IMPROVE library, AIRSpec-corrected, 1425–3998 cm⁻¹. Classes are
# spectral, not chemical identifications. The ward tree is one cut of one linkage; the
# silhouette scan shows how flat that choice is.
