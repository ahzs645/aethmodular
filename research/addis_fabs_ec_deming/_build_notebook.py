"""Builds addis_fabs_ec_deming.ipynb. Run once, then nbconvert --execute."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Addis Ababa — fAbs vs FTIR-EC: errors-in-variables fit & the EC shift that zeros the intercept

**This week's question.** The Addis fAbs-vs-FTIR-EC scatter has a stubborn intercept
(~28–29 Mm⁻¹) that won't go to zero. The working hypothesis (a reversal of the earlier
"HIPS is wrong" idea): **FTIR may underestimate EC in charcoal-heavy environments** because
fully-charred carbon has no IR-active bonds for FTIR to detect. If FTIR-EC is systematically
too low, the true EC sits further *right* on the x-axis, which would shift the cloud and could
collapse the intercept toward zero.

This notebook quantifies whether that is even plausible, in three steps:

1. **Re-fit accounting for error in both X and Y** — OLS assumes all error is in Y (fAbs). The
   intercept question is sensitive to X-error, so we use an errors-in-variables fit
   (orthogonal / Deming regression).
2. **Assume a measurement uncertainty on EC (and fAbs)** — the errors-in-variables fit needs an
   assumed uncertainty. This is *ordinary* measurement noise (≈0.2 µg/m³), **not** the
   big "missing char" effect (that is step 3).
3. **Solve for the x-shift that zeros the intercept** — holding the slope, how far right must all
   EC move for the best-fit intercept to land naturally at zero? And is that shift a **constant
   offset** or a **multiplier**? A constant additive shift is physically implausible (every filter
   would carry the same char), so the character of the shift tells us whether the charcoal
   hypothesis is realistic.

**Scope: Addis only — not Delhi.** Warren cautioned against over-interpreting intercepts; Delhi's
data is scattered with no points near the origin, so shifting it would be exactly that
over-interpretation. Addis is the one site with a tight enough fit to justify the exercise.""")

md(r"""## Regression method (from the Boris paper)

Boris et al. (2019) — Anne's first functional-groups paper — uses **orthogonal least-squares
regression** wherever two measured quantities are compared (calibration vs. gravimetric weights,
OM mass recovery, the van Krevelen slope). That is the "error in both X and Y" method Anne was
reaching for.

- **Orthogonal / total least squares (TLS)** minimizes the *perpendicular* distance to the line.
  In the strict sense it assumes equal X and Y error variances (or axes scaled so they are).
- **Deming regression** is the general version that weights by an assumed **error-variance ratio**
  λ = Var(Y-error) / Var(X-error). λ = 1 reduces to orthogonal regression.

Both are the same errors-in-variables family. The lever that moves the intercept is the assumed
error ratio, so we **state our assumed σ_EC and σ_fAbs** and show a sensitivity sweep over λ.""")

code(r"""import sys, warnings
warnings.filterwarnings('ignore')

# This notebook lives in research/addis_fabs_ec_deming/.
# Reusable logic lives in the sibling ftir_hips_chem project.
sys.path.insert(0, '../ftir_hips_chem/scripts')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import odr

from config import MAC_VALUE
from data_matching import (
    load_aethalometer_data, load_filter_data,
    add_base_filter_id, match_all_parameters,
)
from plotting import PlotConfig          # importing applies the white-background default style
from plotting.utils import calculate_regression_stats, deming, deming_lambda  # deming consolidated into scripts/

SITE_NAME, SITE_CODE = 'Addis_Ababa', 'ETAD'
print('MAC_VALUE =', MAC_VALUE)""")

md(r"""## Load and build the Addis pair set

`match_all_parameters` divides `HIPS_Fabs` by `MAC_VALUE` to express it as a BC-equivalent
(µg/m³). The ~28–29 intercept the meeting referred to is on the **raw fAbs in Mm⁻¹** axis, so we
multiply back by `MAC_VALUE` to recover Mm⁻¹.

- **x = FTIR-EC** (µg/m³)
- **y = fAbs** (Mm⁻¹)""")

code(r"""aeth = load_aethalometer_data()
filters = add_base_filter_id(load_filter_data())

matched = match_all_parameters(SITE_NAME, SITE_CODE, aeth[SITE_NAME], filters)

df = matched.dropna(subset=['hips_fabs', 'ftir_ec']).copy()
df['fabs_mm'] = df['hips_fabs'] * MAC_VALUE      # back to Mm⁻¹
df['ec'] = df['ftir_ec']                          # µg/m³
df = df[['date', 'ec', 'fabs_mm']].reset_index(drop=True)

x = df['ec'].values
y = df['fabs_mm'].values
print(f"n = {len(df)} paired Addis filters")
df[['ec', 'fabs_mm']].describe().round(3)""")

md(r"""## Step 1 — OLS baseline (all error assumed in Y)

This is the fit currently used. It is the reference the errors-in-variables fit is compared against.""")

code(r"""ols = calculate_regression_stats(x, y)   # y = fAbs regressed on x = EC
print("OLS  (y = fAbs  ~  x = EC):")
print(f"  slope     = {ols['slope']:.3f} Mm⁻¹ per µg/m³   (a MAC-like value)")
print(f"  intercept = {ols['intercept']:.3f} Mm⁻¹")
print(f"  R²        = {ols['r_squared']:.3f}   (r = {ols['correlation']:.3f},  n = {ols['n']})")""")

md(r"""## Step 2 — errors-in-variables fit (orthogonal / Deming)

Closed-form Deming with error-variance ratio λ = Var(Y-error)/Var(X-error), cross-checked against
`scipy.odr` (which minimizes weighted orthogonal distances given per-axis 1σ).

**Assumed uncertainties** (stated, ordinary measurement noise — not the missing-char effect):
- σ_EC = 0.2 µg/m³  (Anne's suggested value)
- σ_fAbs = 1.0 Mm⁻¹  (HIPS absorption measurement noise)""")

code(r"""# deming() / deming_lambda() are imported from plotting.utils (see setup cell).
SIGMA_EC   = 0.2    # µg/m³  — assumed EC measurement uncertainty
SIGMA_FABS = 1.0    # Mm⁻¹   — assumed fAbs measurement uncertainty
lam = deming_lambda(SIGMA_EC, SIGMA_FABS)

dem_slope, dem_int = deming(x, y, lam)

# Cross-check with scipy.odr (errors-in-variables; sx, sy are assumed 1-sigma)
linmodel = odr.Model(lambda B, xx: B[0] * xx + B[1])
data = odr.RealData(x, y, sx=SIGMA_EC, sy=SIGMA_FABS)
out = odr.ODR(data, linmodel, beta0=[ols['slope'], ols['intercept']]).run()
odr_slope, odr_int = out.beta

print(f"Assumed σ_EC = {SIGMA_EC} µg/m³,  σ_fAbs = {SIGMA_FABS} Mm⁻¹   ->   λ = {lam:.1f}")
print(f"Deming (closed form) : slope = {dem_slope:.3f} Mm⁻¹/(µg/m³),  intercept = {dem_int:.3f} Mm⁻¹")
print(f"scipy.odr (check)    : slope = {odr_slope:.3f} Mm⁻¹/(µg/m³),  intercept = {odr_int:.3f} Mm⁻¹")
print(f"\nFor reference, OLS    : slope = {ols['slope']:.3f},  intercept = {ols['intercept']:.3f}")""")

md(r"""### Sensitivity — how the slope and intercept move with the error ratio

The assumed error ratio is the lever. The sweep brackets the two OLS limits:
λ → ∞ is OLS of Y-on-X (all error in fAbs); λ → 0 is the inverse OLS (all error in EC);
λ = 1 is strict orthogonal regression.""")

code(r"""ratios = np.logspace(-1, 2.5, 80)          # σ_fAbs/σ_EC from 0.1 to ~316
slopes = np.array([deming(x, y, r ** 2)[0] for r in ratios])
ints   = np.array([deming(x, y, r ** 2)[1] for r in ratios])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].semilogx(ratios, slopes, color='C2')
ax[0].set_xlabel('σ_fAbs / σ_EC'); ax[0].set_ylabel('slope (Mm⁻¹ per µg/m³)')
ax[0].set_title('Deming slope vs assumed error ratio')

ax[1].semilogx(ratios, ints, color='C2')
ax[1].axhline(0, ls='--', c='grey', lw=1)
ax[1].set_xlabel('σ_fAbs / σ_EC'); ax[1].set_ylabel('intercept (Mm⁻¹)')
ax[1].set_title('Deming intercept vs assumed error ratio')

for a in ax:
    a.axvline(SIGMA_FABS / SIGMA_EC, ls=':', c='red', lw=1.5, label='assumed ratio')
    a.legend()
plt.tight_layout(); plt.show()

inv_slope = 1.0 / np.polyfit(y, x, 1)[0]   # inverse OLS (x-on-y), expressed as y/x slope
print(f"OLS Y-on-X  (λ→∞)  slope ≈ {ols['slope']:.2f},  intercept ≈ {ols['intercept']:.1f}")
print(f"orthogonal  (λ=1)  slope =  {deming(x, y, 1)[0]:.2f},  intercept =  {deming(x, y, 1)[1]:.1f}")
print(f"inverse OLS (λ→0)  slope ≈ {inv_slope:.2f}")""")

md(r"""## Step 3 — how far must EC shift to zero the intercept?

This is the key diagnostic, and it is **not** the same as forcing the line through the origin.
We **hold the slope fixed** and ask what transformation of EC makes the fixed-slope line pass
through zero:

- **Additive:**  EC → EC + Δ.  Fixed-slope intercept = b − slope·Δ = 0  ⟹  **Δ = b / slope**.
- **Multiplicative:**  EC → k·EC.  Fixed-slope intercept = ȳ − slope·k·x̄ = 0  ⟹  **k = ȳ / (slope·x̄)**.

These two are computed by **holding the slope fixed**. Note the multiplier `k` here only lands the
*mean* point on the through-origin line — it is the additive shift expressed as a ratio at the mean
(`k = 1 + Δ/x̄`). As the derivations below show, a multiplier does **not** actually zero the offset when
you re-fit; only the additive shift does. We report both for the OLS and Deming slopes.""")


code(r"""def shifts_to_zero_intercept(slope, intercept, xbar, ybar):
    delta = intercept / slope               # additive shift in µg/m³
    k = ybar / (slope * xbar)               # multiplicative factor
    return delta, k

xbar, ybar = x.mean(), y.mean()
rows = []
for name, s, b in [('OLS', ols['slope'], ols['intercept']),
                   ('Deming', dem_slope, dem_int)]:
    d, k = shifts_to_zero_intercept(s, b, xbar, ybar)
    rows.append({
        'fit': name,
        'slope (Mm⁻¹/µgm⁻³)': round(s, 3),
        'intercept (Mm⁻¹)': round(b, 2),
        'additive Δ (µg/m³)': round(d, 2),
        'Δ / mean EC': round(d / xbar, 2),
        'multiplier k': round(k, 2),
    })
shift_table = pd.DataFrame(rows)
print(f"mean EC = {xbar:.2f} µg/m³,  mean fAbs = {ybar:.2f} Mm⁻¹")
shift_table""")

code(r"""# Visualize: OLS line, Deming line, and the multiplicatively-shifted EC cloud
d_dem, k_dem = shifts_to_zero_intercept(dem_slope, dem_int, xbar, ybar)

fig, ax = plt.subplots(figsize=(8.5, 7))
ax.scatter(x, y, s=45, alpha=0.6, edgecolor='k', linewidth=0.3,
           color='C0', label=f'Addis filters (n={len(x)})')

xx = np.linspace(0, x.max() * 1.05, 100)
ax.plot(xx, ols['slope'] * xx + ols['intercept'], 'C1-', lw=2,
        label=f"OLS:    y = {ols['slope']:.2f}x + {ols['intercept']:.1f}")
ax.plot(xx, dem_slope * xx + dem_int, 'C2-', lw=2,
        label=f"Deming: y = {dem_slope:.2f}x + {dem_int:.1f}")

ax.scatter(x * k_dem, y, s=45, alpha=0.30, color='C3', marker='s',
           label=f'EC × {k_dem:.2f} (Deming-slope line → 0)')
ax.plot(xx, dem_slope * xx, 'C3--', lw=1.5)

ax.axhline(0, color='grey', lw=0.6); ax.axvline(0, color='grey', lw=0.6)
ax.set_xlabel('FTIR-EC (µg/m³)'); ax.set_ylabel('fAbs (Mm⁻¹)')
ax.set_title('Addis: fAbs vs FTIR-EC — OLS vs Deming, and the EC shift that zeros the intercept')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout(); plt.show()""")

md(r"""## Derivations — each correction worked separately, with graphs

Step 3 held the slope fixed and asked for the shift. Here we instead **re-fit** after transforming EC,
and derive from the OLS estimators what happens to slope and intercept under each correction. Start from
deviations measured about the means:

$$S_{xx}=\sum_i (x_i-\bar{x})^2,\quad S_{xy}=\sum_i (x_i-\bar{x})(y_i-\bar{y}),\quad
  m=\frac{S_{xy}}{S_{xx}},\quad b=\bar{y}-m\,\bar{x}.$$

The intercept is the fitted fAbs at EC = 0. Everything below turns on **how each transform changes the
deviations $(x_i-\bar{x})$**: the slope depends only on those, the intercept on the slope and $\bar{x}$.""")

code(r"""xbar, ybar = x.mean(), y.mean()
Sxx = np.sum((x - xbar) ** 2)
Sxy = np.sum((x - xbar) * (y - ybar))
slope = Sxy / Sxx
offset = ybar - slope * xbar
Delta = offset / slope                      # additive shift to zero the intercept
k = ybar / (slope * xbar)                   # mean-point multiplier
add_slope, add_int = np.polyfit(x + Delta, y, 1)
mul_slope, mul_int = np.polyfit(x * k, y, 1)
print(f"Sxx = {Sxx:.2f}   Sxy = {Sxy:.2f}")
print(f"slope  m = Sxy/Sxx = {slope:.4f} Mm⁻¹/(µg/m³)   offset b = ȳ - m·x̄ = {offset:.4f} Mm⁻¹")""")

md(r"""### 1. Additive:  $x' = x + \Delta$

Shifting moves the mean equally, so deviations are **unchanged** ($x_i'-\bar{x}'=x_i-\bar{x}$):
$$m' = \frac{S_{xy}}{S_{xx}} = m,\qquad b' = \bar{y}-m(\bar{x}+\Delta) = b - m\,\Delta.$$
Zero it: $\Delta = b/m \approx 7.11\ \mu g/m^3$. **Slope preserved, offset removable.**""")

code(r"""print(f"Δ = b/m = {Delta:.4f}")
print(f"refit slope     = {add_slope:.4f}  | predicted m       = {slope:.4f}")
print(f"refit intercept = {add_int:.4f}  | predicted b - m·Δ = {offset - slope*Delta:.4f}")""")

md(r"""### 2. Multiplicative:  $x' = k\,x$

Scaling stretches deviations by $k$ ($x_i'-\bar{x}'=k(x_i-\bar{x})$):
$$m' = \frac{k\,S_{xy}}{k^2 S_{xx}} = \frac{m}{k},\qquad
  b' = \bar{y}-\frac{m}{k}(k\bar{x}) = \bar{y}-m\bar{x} = b.$$
The intercept is **independent of $k$** — no multiplier removes the offset; it only flattens the slope.""")

code(r"""print(f"using k = {k:.4f}")
print(f"refit slope     = {mul_slope:.4f}  | predicted m/k = {slope/k:.4f}")
print(f"refit intercept = {mul_int:.4f}  | predicted b   = {offset:.4f}  (unchanged for ANY k)")""")

md(r"""### 3. Combined / affine:  $x' = k\,x + \Delta$

The additive part drops out of the deviations, so $m'=m/k$ as before, but the mean carries $\Delta$:
$$b' = \bar{y}-\frac{m}{k}(k\bar{x}+\Delta) = b - \frac{m}{k}\,\Delta.$$
Zero it: $\Delta = k\,(b/m) = k\cdot 7.11$. **The additive term does all the offset-removal; the
multiplier is free and only sets the slope $m/k$.** Pure additive is the $k=1$ case.""")

code(r"""k_c = 1.5
D_c = k_c * (offset / slope)
mc, bc = np.polyfit(k_c * x + D_c, y, 1)
print(f"using k = {k_c}, Δ = k·(b/m) = {D_c:.4f}")
print(f"refit slope     = {mc:.4f}  | predicted m/k         = {slope/k_c:.4f}")
print(f"refit intercept = {bc:.4f}  | predicted b - (m/k)·Δ = {offset - (slope/k_c)*D_c:.4f}")""")

md(r"""### The three corrections as graphs

Same y-axis (fAbs, Mm⁻¹); only the EC transform differs. Each panel is **re-fit** with OLS, so the
intercept you read off is the honest one (not a held-slope construction).""")

code(r"""def _panel(ax, xv, yv, title, xlabel, color):
    s, b = np.polyfit(xv, yv, 1)
    r = np.corrcoef(xv, yv)[0, 1]
    ax.scatter(xv, yv, s=35, alpha=0.55, edgecolor='k', linewidth=0.3, color=color)
    xx = np.linspace(0, xv.max() * 1.05, 100)
    ax.plot(xx, s * xx + b, color='k', lw=2)
    ax.axhline(0, color='grey', lw=0.6); ax.axvline(0, color='grey', lw=0.6)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=11)
    ax.text(0.04, 0.96, f"y = {s:.2f}x + {b:.1f}\nR² = {r**2:.3f}   n = {len(xv)}",
            transform=ax.transAxes, va='top', ha='left', fontsize=9.5,
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=True)
_panel(axes[0], x,         y, '1. Raw — fAbs vs FTIR-EC', 'FTIR-EC (µg/m³)', 'C0')
_panel(axes[1], x + Delta, y, f'2. Additive: EC + {Delta:.2f} µg/m³',
       f'EC + {Delta:.2f} (µg/m³)', 'C2')
_panel(axes[2], x * k,     y, f'3. Multiplicative: EC × {k:.2f}',
       f'EC × {k:.2f} (µg/m³)', 'C3')
axes[0].set_ylabel('fAbs (Mm⁻¹)')
fig.suptitle('Addis fAbs vs FTIR-EC — raw, additive, and multiplicative EC corrections (each re-fit)',
             fontsize=13, y=1.02)
plt.tight_layout(); plt.show()""")

md(r"""### Why the multiplier can't move the offset — the geometry

**Additive = rigid slide:** every filter moves the *same* distance, so the cloud and its fit translate
sideways without tilting (slope kept, line drops to the origin). **Multiplicative = stretch anchored at
EC = 0:** each filter moves a distance *proportional to its EC*, so the line pivots about its y-intercept
— the value at EC = 0 is pinned, the intercept can't change, only the slope flattens.""")

code(r"""idx = np.argsort(x)[np.linspace(0, len(x) - 1, 6).astype(int)]
xs, ys = x[idx], y[idx]
xx = np.linspace(0, (x * k).max() * 1.02, 100)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
ax[0].scatter(xs, ys, s=70, color='C0', edgecolor='k', zorder=3, label='raw filter')
ax[0].scatter(xs + Delta, ys, s=70, color='C2', edgecolor='k', zorder=3, label=f'EC + {Delta:.1f}')
for xi, yi in zip(xs, ys):
    ax[0].annotate('', xy=(xi + Delta, yi), xytext=(xi, yi),
                   arrowprops=dict(arrowstyle='->', color='0.4', lw=1.3))
ax[0].plot(xx, slope * xx + offset, 'C0-', lw=2, label=f'raw fit (int {offset:.0f})')
ax[0].plot(xx, slope * xx, 'C2--', lw=2, label='shifted fit (int 0)')
ax[0].set_title('Additive: every point slides the SAME distance →\nline keeps its slope, drops to origin')
ax[0].set_xlabel('EC (µg/m³)'); ax[0].set_ylabel('fAbs (Mm⁻¹)')

ax[1].scatter(xs, ys, s=70, color='C0', edgecolor='k', zorder=3, label='raw filter')
ax[1].scatter(xs * k, ys, s=70, color='C3', edgecolor='k', zorder=3, label=f'EC × {k:.1f}')
for xi, yi in zip(xs, ys):
    ax[1].annotate('', xy=(xi * k, yi), xytext=(xi, yi),
                   arrowprops=dict(arrowstyle='->', color='0.4', lw=1.3))
ax[1].plot(xx, slope * xx + offset, 'C0-', lw=2, label=f'raw fit (int {offset:.0f})')
ax[1].plot(xx, mul_slope * xx + mul_int, 'C3--', lw=2, label=f'scaled fit (int {mul_int:.0f})')
ax[1].scatter([0], [offset], s=120, color='k', marker='*', zorder=4)
ax[1].annotate('pivot at EC = 0\n(intercept pinned)', xy=(0, offset), xytext=(2.5, offset - 11),
               fontsize=9, arrowprops=dict(arrowstyle='->', color='k'))
ax[1].set_title('Multiplicative: each point moves ∝ its EC →\nline pivots about EC=0, intercept unchanged')
ax[1].set_xlabel('EC (µg/m³)')
for a in ax:
    a.axhline(0, color='grey', lw=0.6); a.axvline(0, color='grey', lw=0.6)
    a.legend(loc='lower right', fontsize=8.5)
plt.tight_layout(); plt.show()""")

md(r"""## Summary

The cell below prints the headline numbers. Narrative read (filled from the executed values):

- The intercept is real under OLS and only **shrinks**, not vanishes, under the errors-in-variables
  fit — so it is not purely an artifact of ignoring X-error.
- **Only a constant additive shift can zero the offset** (Δ ≈ 7 µg/m³ on every filter); the derivations
  show a multiplier leaves the intercept untouched and merely flattens the slope (the MAC). So the
  "≈2.4×" is the additive shift expressed as a ratio at the mean, **not** a correction that fixes the
  data.
- That required additive shift is **larger than the mean EC**, so attributing the offset to uniform
  missing char is physically hard — it would mean even the cleanest filters miss ~7 µg/m³ of char. A
  genuine constant-*factor* FTIR underestimate would show up as a wrong slope (MAC), not this offset.

**Caveats.** The fit/intercept depend on the assumed error ratio (see the sweep) and orientation.
This is a plausibility exercise for Addis only; do not extend it to Delhi.""")

code(r"""print('=== Addis fAbs-vs-FTIR-EC summary ===')
print(f'n = {len(x)} filters | mean EC = {xbar:.2f} µg/m³ | mean fAbs = {ybar:.2f} Mm⁻¹\n')
print(f"OLS    : slope {ols['slope']:.2f}, intercept {ols['intercept']:.1f} Mm⁻¹, R² {ols['r_squared']:.3f}")
print(f"Deming : slope {dem_slope:.2f}, intercept {dem_int:.1f} Mm⁻¹  (σ_EC={SIGMA_EC}, σ_fAbs={SIGMA_FABS}, λ={lam:.0f})\n")
print('To zero the intercept (slope held fixed):')
print(shift_table.to_string(index=False))""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
with open('addis_fabs_ec_deming.ipynb', 'w') as f:
    nbf.write(nb, f)
print('wrote addis_fabs_ec_deming.ipynb with', len(cells), 'cells')
