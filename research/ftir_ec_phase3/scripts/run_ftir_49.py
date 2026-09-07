# %% [markdown]
# # ftir_49 — the MA350 raw dual-spot chain at Jacros: correction diagnostics and the missingness benchmark
#
# ## tl;dr
#
# **Run status (2026-09-01): the raw 1-minute Drive CSV could not be hydrated (three
# attempts, ~164 of 556 MB streamed; Google Drive stalls at ≈ 2 MB/min), so this run is the
# daily-pickle fallback.** Every cell that needs the minute record prints `SKIPPED`; the
# script auto-switches to the full minute analysis (A2, A3, B4, B5, and the calendar-day
# window in B6) the next time it is built after `cat FILE > /dev/null` completes. What
# *could* be done from `df_Jacros_9am_resampled.pkl` (1,047 daily records, MA350-0238):
#
# - **Loading compensation is not a small correction at Jacros.** On daily means the
#   reported BCc is BC1 × **1.59** (IR), **1.74** (Red), **1.91** (Blue) — i.e. **37 % / 43 % /
#   48 %** of the IR / Red / Blue BCc is the DualSpot term K·ATN1, at median ATN1 of 31 / 41 /
#   53. The instrument's K is 0.011 (IR, 5–95 % range −0.015 to 0.037), 0.010 (Red), 0.009
#   (Blue). A **10 % error in K moves IR BCc by 5.9 % and Red BCc by 7.4 %**; a 25 % BCc shift
#   (the ≈ 2 µg/m³ FTIR–HIPS–MA350 yardstick) needs a **43 % (IR) / 34 % (Red)** K error.
# - **The two spots do not agree after compensation.** With the instrument's own K,
#   spot-2-compensated BC2/(1 − K·ATN2) is **0.925 × BCc at IR, 0.944 at Red, 0.981 at
#   Blue** (daily-mean medians); the daily-mean closure K\* is **0.0124 at IR vs the reported
#   0.0107** (+16 %), 0.0106 vs 0.0099 at Red. On this evidence the dual-spot chain carries a
#   **6–8 % spot-disagreement term** at the two channels the calibration uses — a third of
#   the yardstick, in the direction of BCc being *high* relative to spot 2. Minute-level
#   confirmation (by ATN and spot age) is the skipped A1 cell.
# - **The ±1-day tolerant window of `match_aeth_filter_data` is a first-order effect.** On
#   the **193** ETAD filter start dates with both numbers, the tolerant mean differs from the
#   9am–9am filter-day mean by a median **+3.7 %** (IR; IQR −7.2 to +19.5 %, 2.5–97.5 %
#   range −24.8 to +56.3 %); **103 of 193 (53 %) filter days move by more than 10 %, 41
#   (21 %) by more than 25 %, and 32 exceed 2,000 ng/m³** — as large as the discrepancy
#   under study, on one filter day in six. Red BCc behaves identically (106 / 44 / 31).
# - Not done in this run (needs minutes): tape-advance steps, T/RH/flow sensitivity and the
#   clean-mask test of the ftir_28 AAE(625,880) = 0.944 anomaly, the which-20 %-is-missing
#   benchmark, the AAE order-of-operations test, and the calendar-day window.
#
# ## Context & Methods
#
# Every Addis MA350 number the calibration work uses is a daily mean of the instrument's
# **BCc** column — the DualSpot loading-compensated output. Between the photodiodes and
# that daily mean sit four processing steps the repo has never audited on the raw
# 1-minute record: (i) the loading compensation itself (the running `K` that reconciles
# the two spots), (ii) the tape advance (a new spot, `K` reset to zero, a start-up
# transient), (iii) the instrument's sensitivity to temperature/RH transients and flow
# state (Elomaa et al. 2025, *Aerosol Res.* 3:293, on MA350 T/RH artefacts; Poland et al.
# 2025, *AS&T* 60(4), on equating spot-compensated absorption on both spots), and (iv) the
# daily aggregation with whatever minutes happen to be missing. The reviews' recurring
# question — *which* 20 % of minutes is missing, not how many — is the fourth step.
#
# This notebook works on the raw minute file (`Jacros_MA350_1-min_2022-2024_Cleaned.csv`,
# firmware 1.1, five channels 375/470/528/625/880 nm — MA350 wavelengths from
# `config.WAVELENGTHS_NM`, not the AE33 set). It has two parts:
#
# **Part A — dual-spot diagnostics.** Per wavelength: the distribution of the instrument's
# `K`; the closure between `BC1`, `BC2` and `BCc`; the *instantaneous* K\* that would make
# the two spot-compensated concentrations agree (the Poland et al. criterion) as a
# function of ATN1 and spot age; the BCc step across tape advances against a
# random-time control; and BCc(λ)/BCc(IR) ratios and AAE(470,880), AAE(625,880) binned by
# |dT/dt|, |dRH/dt|, flow state and loading. A "clean" mask (nominal flows, T/RH
# transients below their medians, ATN1 < 30, > 30 min from a tape advance) tests whether
# the Green/IR and Red/IR anomaly of ftir_28 (AAE(625,880) = 0.944 ± 0.060, below the
# AAE_BC ≈ 1 anchor) is an artefact of any of those states.
#
# **Part B — the missingness benchmark.** On 9am–9am days with ≥ 95 % minute coverage,
# the true daily mean is known; 10/20/40 % of the minutes are then removed under five
# patterns (random, one contiguous outage, rush-hour-only, high-pollution-only,
# low-pollution-only) and the recovered daily mean is scored. AAE from daily-aggregated
# absorption is compared with the daily mean of instantaneous AAE. Finally, on the real
# ETAD filter dates, the calendar-day mean, the 9am–9am mean and the ±1-day tolerant
# mean that `match_aeth_filter_data` uses are compared.
#
# The yardstick throughout is the FTIR–HIPS–MA350 discrepancy at Addis: ≈ 20 Mm⁻¹ in
# absorption, ≈ 2 µg/m³ in EC-equivalent, i.e. roughly a quarter of the ≈ 8 µg/m³ IR BCc
# daily mean.
#
# ### Key assumptions
#
# - The DualSpot algebra is the Drinovec et al. (2015) form the vendored engine
#   (`src/external/calibration.py`, aethpy) implements: `BC1 = BCc·(1 − K·ATN1)`,
#   `BC2 = BCc·(1 − K·ATN2)`, so `BCc = BC1/(1 − K·ATN1)`. Solving both for one K gives the
#   instantaneous closure value `K* = (BC1 − BC2)/(BC1·ATN2 − BC2·ATN1)`. The identity
#   `BCc = BC1/(1 − K·ATN1)` is checked against the file, not assumed.
# - Absorption is `b_ATN(λ) = BCc(λ)·σ_ATN(λ)` with σ_ATN parsed from the vendored
#   firmware table (as ftir_28 did); AAE from `optics.aae`. This is the attenuation scale,
#   not a filter-corrected absorption — AAE is invariant to the scale.
# - The instrument stamps local time (`Date local`, `Time local`, 12-hour clock); the
#   timezone-offset column is 0, so the stamps are taken as local, and the diurnal peak is
#   checked below as a sanity test.
# - A 9am–9am day is `[09:00 on D, 09:00 on D+1)`, labelled D. The processed pickle's
#   convention is verified against the minute data rather than assumed.
# - ETAD filter windows: SPARTAN filters are ~8 staggered 3-h sub-windows over ~9 days
#   (Snider et al. 2015); the repo treats them as a 24-h day starting on
#   `SamplingStartDate`. That convention is kept here so the comparison is to what the
#   calibrations actually consumed — the 24-h assumption is a known caveat, not a claim.
# - The minute file has no flow-factor or leak-compensation corrections applied; "nominal
#   flow" means total flow within 5 % of the setpoint and Flow1/Flow2 within 10 % of its
#   record median. The file's Flow1:Flow2 split is ≈ 55:45 (checked below) — the 3:1 spec
#   sometimes quoted for MA-series instruments does not describe this record.
# - Heavy intermediate results are cached to `output/tables/ftir49/*.parquet` on the first
#   run and loaded if present.

# %%
import ast
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path('scripts').resolve()))
sys.path.insert(0, str((Path('..') / 'ftir_hips_chem' / 'scripts').resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from config import WAVELENGTHS_NM, PROCESSED_SITES_DIR
from data_paths import aethalometry_dir
from optics import aae
from phase3_common import load_addis_evaluation
from plotting import apply_default_style

apply_default_style()
T0 = time.time()
REPO_ROOT = Path('..').resolve().parent
OUT = Path('output/tables/ftir49')
PLOTS = Path('output/plots/ftir49')
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

CHANNELS = ['UV', 'Blue', 'Green', 'Red', 'IR']
RNG = np.random.default_rng(20260901)
CACHE = OUT / 'minute_cache.parquet'

RAW_NAME = 'Jacros_MA350_1-min_2022-2024_Cleaned.csv'
try:
    RAW = aethalometry_dir() / 'Raw' / RAW_NAME
except Exception:  # pragma: no cover - env without the Drive mount resolver
    RAW = Path('/Users/ahmadjalil/Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/'
               'University/Research/Grad/Data/Davis Data/Aethalometry Data/Raw') / RAW_NAME

# sigma_ATN per channel, parsed from the vendored firmware module (never retyped)
calibration_src = (REPO_ROOT / 'src' / 'external' / 'calibration.py').read_text()
SIGMA_ATN = next(
    ast.literal_eval(node.value)
    for node in ast.walk(ast.parse(calibration_src))
    if isinstance(node, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'atn_cross_section_dict_MAx'
            for t in node.targets)
)
assert set(SIGMA_ATN) == set(CHANNELS)
print('sigma_ATN (m2/g) from src/external/calibration.py:', SIGMA_ATN)
print('wavelengths (nm) from config.WAVELENGTHS_NM   :', {c: WAVELENGTHS_NM[c] for c in CHANNELS})


def raw_is_hydrated(path: Path) -> bool:
    """Google Drive placeholders report the full size but allocate blocks only as the
    content streams in; a partially hydrated CSV would be silently truncated by read_csv."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_blocks * 512 >= 0.995 * st.st_size


MINUTE_DATA = CACHE.exists() or raw_is_hydrated(RAW)
SKIP_MSG = ('SKIPPED — this cell needs the 1-min record. The raw Drive CSV was not fully '
            'hydrated when the notebook ran (and no minute cache exists), so this part falls '
            'back to the daily 9am pickle where that is possible and is otherwise not done.')
print('minute record available:', MINUTE_DATA,
      '' if MINUTE_DATA else f'(allocated {os.stat(RAW).st_blocks * 512 / 1e6:.0f} of {os.stat(RAW).st_size / 1e6:.0f} MB)')

from IPython.core.magic import register_cell_magic


@register_cell_magic
def skip_unless_minute(line, cell):
    """Run the cell only when the minute record is available; otherwise print SKIP_MSG."""
    if MINUTE_DATA:
        get_ipython().run_cell(cell).raise_error()
    else:
        print(SKIP_MSG)

# %% [markdown]
# ## Data — the minute cache
#
# First pass: the 1.1-million-row CSV is read in chunks with only the needed columns and
# written to parquet. Later runs load the parquet. Derived columns (9am-day, tape
# segment, spot age, T/RH deltas) are cheap and recomputed every time.

# %%
%%skip_unless_minute
META_COLS = ['Session ID', 'Date local (yyyy/MM/dd)', 'Time local (hh:mm:ss)', 'Timebase (s)',
             'Status', 'Readable status', 'Tape position', 'Flow setpoint (mL/min)',
             'Flow total (mL/min)', 'Flow1 (mL/min)', 'Flow2 (mL/min)', 'Sample temp (C)',
             'Sample RH (%)', 'Internal pressure (Pa)', 'Internal temp (C)']
CHAN_COLS = [f'{c} {v}' for c in CHANNELS for v in ('ATN1', 'ATN2', 'K', 'BC1', 'BC2', 'BCc')]


def build_cache() -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(RAW, usecols=META_COLS + CHAN_COLS, chunksize=200_000,
                             low_memory=False):
        stamp = chunk['Date local (yyyy/MM/dd)'].astype(str).str.strip() + ' ' + \
            chunk['Time local (hh:mm:ss)'].astype(str).str.strip()
        t = pd.to_datetime(stamp, format='%Y-%m-%d %I:%M:%S %p', errors='coerce')
        bad = t.isna()
        if bad.any():
            t[bad] = pd.to_datetime(stamp[bad], errors='coerce')
        chunk = chunk.drop(columns=['Date local (yyyy/MM/dd)', 'Time local (hh:mm:ss)'])
        chunk.insert(0, 't', t)
        chunk['Readable status'] = chunk['Readable status'].astype(str)
        for col in chunk.columns:
            if col not in ('t', 'Readable status') and chunk[col].dtype == object:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df = df[df['t'].notna()].sort_values('t', kind='stable').reset_index(drop=True)
    df.to_parquet(CACHE, index=False)
    return df


if CACHE.exists():
    raw = pd.read_parquet(CACHE)
    print(f'loaded minute cache {CACHE} ({len(raw):,} rows)')
else:
    print(f'building minute cache from {RAW.name} ...')
    raw = build_cache()
    print(f'cached {len(raw):,} rows -> {CACHE}  [{time.time() - T0:.0f} s]')

n_dup = int(raw['t'].duplicated().sum())
raw = raw.drop_duplicates('t', keep='first').reset_index(drop=True)
print(f'{len(raw):,} minutes, {raw["t"].min()} to {raw["t"].max()}, '
      f'{n_dup} duplicate stamps dropped; timebase values {sorted(raw["Timebase (s)"].dropna().unique())}')

# %%
%%skip_unless_minute
df = raw
df['day_9am'] = (df['t'] - pd.Timedelta(hours=9)).dt.normalize()
df['min_9am'] = ((df['t'] - pd.Timedelta(hours=9)) - df['day_9am']).dt.total_seconds() // 60
df['hour'] = df['t'].dt.hour
df['cal_day'] = df['t'].dt.normalize()

# tape segments: a new segment whenever Tape position or Session ID changes
tape_change = (df['Tape position'].ne(df['Tape position'].shift())
               | df['Session ID'].ne(df['Session ID'].shift()))
tape_change.iloc[0] = True
df['segment'] = tape_change.cumsum()
seg_start = df.groupby('segment')['t'].transform('min')
seg_end = df.groupby('segment')['t'].transform('max')
df['age_min'] = (df['t'] - seg_start).dt.total_seconds() / 60
df['to_next_adv_min'] = (seg_end - df['t']).dt.total_seconds() / 60
df['near_advance'] = (df['age_min'] < 30) | (df['to_next_adv_min'] < 30)

# per-minute T/RH deltas, only across genuine 1-minute steps, then 10-min rolling mean
dt_min = df['t'].diff().dt.total_seconds() / 60
step_ok = dt_min.eq(1)
for col, name in (('Sample temp (C)', 'dT'), ('Sample RH (%)', 'dRH')):
    delta = df[col].diff().where(step_ok)
    df[name] = delta
    df[f'{name}_roll10_abs'] = delta.rolling(10, min_periods=5).mean().abs()

df['flow_ratio'] = df['Flow1 (mL/min)'] / df['Flow2 (mL/min)']
ratio_nominal = float(df['flow_ratio'].median())
setpoint = df['Flow setpoint (mL/min)']
df['flow_dev'] = (df['Flow total (mL/min)'] - setpoint) / setpoint
df['flow_nominal'] = (df['flow_dev'].abs() < 0.05) & \
    ((df['flow_ratio'] / ratio_nominal - 1).abs() < 0.10)

for c in CHANNELS:
    df[f'{c} b'] = df[f'{c} BCc'] * SIGMA_ATN[c] * 1e-3   # Mm^-1 on the attenuation scale

n_adv = int(df['segment'].nunique())
print(f'tape segments: {n_adv}  (median segment length '
      f'{df.groupby("segment").size().median():.0f} min)')
print(f'Flow1/Flow2 median = {ratio_nominal:.3f} (IQR {df["flow_ratio"].quantile(.25):.3f}–'
      f'{df["flow_ratio"].quantile(.75):.3f}); setpoint values {sorted(setpoint.dropna().unique())}')
print(f'nominal-flow minutes: {100 * df["flow_nominal"].mean():.1f} %')
hourly = df.groupby('hour')['IR BCc'].mean()
print('diurnal sanity check (IR BCc, ng/m3): peak hour %02d (%.0f), trough hour %02d (%.0f)'
      % (hourly.idxmax(), hourly.max(), hourly.idxmin(), hourly.min()))
print(f'status codes present: {sorted(df["Status"].dropna().unique().astype(int))[:20]}')
display(df['Readable status'].value_counts().head(8).rename('minutes').to_frame())

# %% [markdown]
# ### The 9am-day convention, verified against the processed pickle
#
# `df_Jacros_9am_resampled.pkl` is what every daily analysis consumed. Whether its
# `day_9am` label is the *start* or the *end* of the 09:00–09:00 window is decided by
# which convention reproduces its `IR BCc` values from the minute file.

# %%
%%skip_unless_minute
pickle_daily = pd.read_pickle(PROCESSED_SITES_DIR / 'df_Jacros_9am_resampled.pkl')
pickle_daily['day_9am'] = pd.to_datetime(pickle_daily['day_9am'])
mine = df.groupby('day_9am')['IR BCc'].mean().rename('start_labelled')
cmp = pickle_daily[['day_9am', 'IR BCc']].merge(mine, on='day_9am', how='inner')
cmp['end_labelled'] = cmp['day_9am'].map(mine.shift(1, freq='D'))
conv = pd.DataFrame({
    'convention': ['label = window start (09:00 D -> 09:00 D+1)',
                   'label = window end   (09:00 D-1 -> 09:00 D)'],
    'median |rel diff| vs pickle (%)': [
        float((cmp['start_labelled'] / cmp['IR BCc'] - 1).abs().median() * 100),
        float((cmp['end_labelled'] / cmp['IR BCc'] - 1).abs().median() * 100)],
    'r with pickle': [float(cmp[['IR BCc', 'start_labelled']].corr().iloc[0, 1]),
                      float(cmp[['IR BCc', 'end_labelled']].corr().iloc[0, 1])],
})
conv.to_csv(OUT / 'day9am_convention_check.csv', index=False)
display(conv.round(3))
DAY_LABEL_IS_START = conv['median |rel diff| vs pickle (%)'].idxmin() == 0
print('pickle convention:', conv.loc[conv['median |rel diff| vs pickle (%)'].idxmin(), 'convention'])

# %% [markdown]
# ## Part A — dual-spot diagnostics
#
# ### A1. The instrument's K, the BC1/BC2/BCc closure, and the Poland K\*
#
# `K` reconciles the two spots by a running fit. The instantaneous value that would make
# the two spot-compensated concentrations equal at each minute is
# `K* = (BC1 − BC2)/(BC1·ATN2 − BC2·ATN1)`; the ratio of the spot-2-compensated
# concentration to the reported BCc, `BC2/(1 − K·ATN2) / BCc`, says how far the two
# spots disagree once the instrument's own K is applied. Both are ill-conditioned at
# low ATN (the denominator → 0), so they are reported by ATN1 bin and by spot age, on
# minutes with ATN1 ≥ 5 and both spot BCs positive.

# %%
%%skip_unless_minute
def kstar(bc1, bc2, atn1, atn2):
    den = bc1 * atn2 - bc2 * atn1
    return (bc1 - bc2) / den.where(den.abs() > 1e-9)


rows, closure_rows = [], []
for c in CHANNELS:
    bc1, bc2, bcc = df[f'{c} BC1'], df[f'{c} BC2'], df[f'{c} BCc']
    atn1, atn2, K = df[f'{c} ATN1'], df[f'{c} ATN2'], df[f'{c} K']
    ok = bc1.notna() & bc2.notna() & bcc.notna() & K.notna()
    identity = (bc1 / (1 - K * atn1))[ok]
    rel_identity = ((identity - bcc[ok]) / bcc[ok]).abs()
    spot2_comp = (bc2 / (1 - K * atn2))
    valid = ok & (atn1 >= 5) & (bc1 > 0) & (bc2 > 0)
    ks = kstar(bc1, bc2, atn1, atn2).where(valid)
    df[f'{c} Kstar'] = ks
    df[f'{c} spot2_over_bcc'] = (spot2_comp / bcc).where(valid)
    rows.append({
        'channel': c,
        'K p05': K.quantile(.05), 'K p25': K.quantile(.25), 'K median': K.median(),
        'K p75': K.quantile(.75), 'K p95': K.quantile(.95),
        'K == 0 (%)': 100 * K.eq(0).mean(),
        'K* median (ATN1>=5)': ks.median(), 'K* IQR lo': ks.quantile(.25), 'K* IQR hi': ks.quantile(.75),
        'BC2comp/BCc median': df[f'{c} spot2_over_bcc'].median(),
        'BC2comp/BCc IQR lo': df[f'{c} spot2_over_bcc'].quantile(.25),
        'BC2comp/BCc IQR hi': df[f'{c} spot2_over_bcc'].quantile(.75),
        'identity BC1/(1-K ATN1)=BCc: median |rel| (%)': 100 * rel_identity.median(),
        'identity within 1% (%)': 100 * (rel_identity < 0.01).mean(),
        'BC1/BCc median': (bc1 / bcc)[ok].median(),
        'BC2/BCc median': (bc2 / bcc)[ok].median(),
        'r(BC1,BC2)': float(pd.concat([bc1, bc2], axis=1)[ok].corr().iloc[0, 1]),
        'n minutes': int(ok.sum()),
    })
k_table = pd.DataFrame(rows).set_index('channel')
k_table.to_csv(OUT / 'k_distribution_and_closure.csv')
display(k_table.round(4))

# %%
%%skip_unless_minute
ATN_BINS = [5, 10, 20, 30, 50, 80, 200]
AGE_BINS = [0, 30, 60, 180, 360, 720, 1440, 1e6]
kbin_rows = []
for c in CHANNELS:
    atn_bin = pd.cut(df[f'{c} ATN1'], ATN_BINS, right=False)
    age_bin = pd.cut(df['age_min'], AGE_BINS, right=False,
                     labels=['<30 min', '30–60', '1–3 h', '3–6 h', '6–12 h', '12–24 h', '>24 h'])
    sub = df[[f'{c} Kstar', f'{c} K', f'{c} spot2_over_bcc', f'{c} BC1', f'{c} ATN1']].copy()
    sub['atn_bin'], sub['age_bin'] = atn_bin, age_bin
    sub = sub[sub[f'{c} Kstar'].notna()]
    # implied BCc if K* were used instead of K: BC1/(1 - K* ATN1)
    sub['bcc_star_over_bcc'] = (1 - sub[f'{c} K'] * sub[f'{c} ATN1']) / \
        (1 - sub[f'{c} Kstar'] * sub[f'{c} ATN1'])
    g = sub.groupby('atn_bin', observed=True)
    for b, part in g:
        kbin_rows.append({'channel': c, 'by': 'ATN1', 'bin': str(b), 'n': len(part),
                          'K median': part[f'{c} K'].median(),
                          'K* median': part[f'{c} Kstar'].median(),
                          'K* IQR lo': part[f'{c} Kstar'].quantile(.25),
                          'K* IQR hi': part[f'{c} Kstar'].quantile(.75),
                          'BC2comp/BCc median': part[f'{c} spot2_over_bcc'].median(),
                          'BCc(K*)/BCc median': part['bcc_star_over_bcc'].median()})
    g = sub.groupby('age_bin', observed=True)
    for b, part in g:
        kbin_rows.append({'channel': c, 'by': 'spot age', 'bin': str(b), 'n': len(part),
                          'K median': part[f'{c} K'].median(),
                          'K* median': part[f'{c} Kstar'].median(),
                          'K* IQR lo': part[f'{c} Kstar'].quantile(.25),
                          'K* IQR hi': part[f'{c} Kstar'].quantile(.75),
                          'BC2comp/BCc median': part[f'{c} spot2_over_bcc'].median(),
                          'BCc(K*)/BCc median': part['bcc_star_over_bcc'].median()})
kstar_bins = pd.DataFrame(kbin_rows)
kstar_bins.to_csv(OUT / 'kstar_vs_K_by_atn_and_age.csv', index=False)
print('K* vs instrument K by ATN1 bin (IR and Red shown; full table in kstar_vs_K_by_atn_and_age.csv)')
display(kstar_bins[kstar_bins['channel'].isin(['IR', 'Red']) & kstar_bins['by'].eq('ATN1')]
        .set_index(['channel', 'bin']).round(4))
print('by spot age (IR)')
display(kstar_bins[kstar_bins['channel'].eq('IR') & kstar_bins['by'].eq('spot age')]
        .set_index('bin').round(4))

# %%
%%skip_unless_minute
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
colors = {'UV': '#7D3C98', 'Blue': '#2E86C1', 'Green': '#28B463', 'Red': '#C0392B', 'IR': '#34495E'}
ax = axes[0]
for c in CHANNELS:
    K = df[f'{c} K'].dropna()
    K = K[(K > K.quantile(.005)) & (K < K.quantile(.995))]
    ax.hist(K, bins=80, histtype='step', lw=1.4, color=colors[c], label=c, density=True)
ax.set(xlabel='instrument K', ylabel='density', title='Distribution of the reported K')
ax.legend(frameon=False, fontsize=9)
ax = axes[1]
sub = kstar_bins[kstar_bins['by'].eq('ATN1')]
for c in CHANNELS:
    s = sub[sub['channel'].eq(c)]
    x = np.arange(len(s))
    ax.plot(x, s['K* median'], 'o-', color=colors[c], label=f'{c} K*')
    ax.plot(x, s['K median'], 's--', color=colors[c], alpha=0.5, mfc='none')
ax.set(xticks=x, xticklabels=s['bin'], xlabel='ATN1 bin',
       title='K* (closure, solid) vs reported K (dashed, open)')
ax.tick_params(axis='x', rotation=30)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax = axes[2]
for c in CHANNELS:
    s = sub[sub['channel'].eq(c)]
    x = np.arange(len(s))
    ax.plot(x, 100 * (s['BC2comp/BCc median'] - 1), 'o-', color=colors[c], label=c)
ax.axhline(0, color='k', lw=0.8)
ax.set(xticks=x, xticklabels=s['bin'], xlabel='ATN1 bin',
       ylabel='median BC2/(1−K·ATN2) ÷ BCc − 1 (%)',
       title='Spot-2 compensated vs reported BCc\n(residual spot disagreement under the instrument K)')
ax.tick_params(axis='x', rotation=30)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS / 'a1_k_and_closure.png', bbox_inches='tight')
plt.show()

# %%
# fallback: daily-pickle version of A1 when the minute record is unavailable
if not MINUTE_DATA:
    pk = pd.read_pickle(PROCESSED_SITES_DIR / 'df_Jacros_9am_resampled.pkl')
    pk['day_9am'] = pd.to_datetime(pk['day_9am'])
    print(f'FALLBACK (daily 9am pickle, {len(pk)} days): daily means of K, BC1, BC2, BCc, ATN1, ATN2. '
          'Ratios of daily means are not the daily means of minute ratios; the closure identity '
          'below is therefore only indicative.')
    rows = []
    for c in CHANNELS:
        K, bc1, bc2, bcc = pk[f'{c} K'], pk[f'{c} BC1'], pk[f'{c} BC2'], pk[f'{c} BCc']
        atn1, atn2 = pk[f'{c} ATN1'], pk[f'{c} ATN2']
        ks = (bc1 - bc2) / (bc1 * atn2 - bc2 * atn1)
        rows.append({'channel': c, 'K p05': K.quantile(.05), 'K median': K.median(), 'K p95': K.quantile(.95),
                     'BC1/BCc median': (bc1 / bcc).median(), 'BC2/BCc median': (bc2 / bcc).median(),
                     'BCc/BC1 median (loading factor)': (bcc / bc1).median(),
                     'ATN1 median': atn1.median(), 'K* (daily means) median': ks.median(),
                     'K* IQR lo': ks.quantile(.25), 'K* IQR hi': ks.quantile(.75),
                     'BC2/(1-K ATN2) / BCc median': (bc2 / (1 - K * atn2) / bcc).median(),
                     'n days': int(K.notna().sum())})
    k_table_daily = pd.DataFrame(rows).set_index('channel')
    k_table_daily.to_csv(OUT / 'k_distribution_daily_fallback.csv')
    display(k_table_daily.round(4))

    # How much of the reported BCc is loading compensation, and how wrong would K have to
    # be to move BCc by the 25 % yardstick?  BCc = BC1/(1 - K*ATN1), so
    # dBCc/BCc = (K*ATN1/(1 - K*ATN1)) * dK/K  with K*ATN1 = 1 - BC1/BCc.
    sens = []
    for c in CHANNELS:
        load = 1 - (pk[f'{c} BC1'] / pk[f'{c} BCc'])          # K*ATN1 per day (daily means)
        gain = load / (1 - load)
        sens.append({'channel': c,
                     'loading term K·ATN1, median': load.median(),
                     'compensation share of BCc (%) = 1 − BC1/BCc': 100 * load.median(),
                     'dBCc/BCc per +10% K (%)': 100 * 0.1 * gain.median(),
                     'K error needed for a 25% BCc shift (%)': 100 * 0.25 / gain.median() if gain.median() > 1e-3 else np.nan})
    k_sens_daily = pd.DataFrame(sens).set_index('channel')
    k_sens_daily.to_csv(OUT / 'k_sensitivity_daily_fallback.csv')
    print('Loading-compensation share of BCc and its K-sensitivity (daily means):')
    display(k_sens_daily.round(3))

# %% [markdown]
# ### A2. Discontinuities at tape advances
#
# For each advance, BCc is averaged over the 10 minutes ending 5 min before the advance
# and the 10 minutes starting 5 min after it (both windows need ≥ 8 minutes). The same
# statistic at random times far from any advance is the control: a real advance artefact
# must exceed the natural 20-minute variability.

# %%
%%skip_unless_minute
starts = df.groupby('segment')['t'].min()
starts = starts.iloc[1:]  # the first segment has no advance before it
series = df.set_index('t')
adv_times = starts.to_numpy()

# random control times at least 60 min from any advance
cand = df.loc[(df['age_min'] > 60) & (df['to_next_adv_min'] > 60), 't'].to_numpy()
ctrl_times = RNG.choice(cand, size=min(len(cand), 5 * len(adv_times)), replace=False)


def window_means(times, col):
    """Mean of `col` over [-15,-5) and [+5,+15) min around each time (vectorised via
    minute index positions)."""
    t_index = series.index
    values = series[col].to_numpy()
    pos = np.searchsorted(t_index.to_numpy(), times)
    pre, post = np.full(len(times), np.nan), np.full(len(times), np.nan)
    tv = t_index.to_numpy()
    for i, (p, t0) in enumerate(zip(pos, times)):
        lo, hi = max(0, p - 20), min(len(values), p + 20)
        tt = (tv[lo:hi] - t0) / np.timedelta64(1, 'm')
        vv = values[lo:hi]
        m_pre = (tt >= -15) & (tt < -5) & ~np.isnan(vv)
        m_post = (tt >= 5) & (tt < 15) & ~np.isnan(vv)
        if m_pre.sum() >= 8 and m_post.sum() >= 8:
            pre[i], post[i] = vv[m_pre].mean(), vv[m_post].mean()
    return pre, post


step_rows, step_detail = [], {}
for c in CHANNELS:
    pre, post = window_means(adv_times, f'{c} BCc')
    cpre, cpost = window_means(ctrl_times, f'{c} BCc')
    step = post - pre
    rel = step / pre
    cstep = cpost - cpre
    crel = cstep / cpre
    step_detail[c] = pd.DataFrame({'t': adv_times, 'pre': pre, 'post': post, 'step': step, 'rel': rel})
    s, cs = step[~np.isnan(step)], cstep[~np.isnan(cstep)]
    r, cr = rel[~np.isnan(rel)], crel[~np.isnan(crel)]
    step_rows.append({
        'channel': c, 'n advances scored': len(s), 'n control': len(cs),
        'step median (ng/m3)': np.median(s), 'step IQR lo': np.quantile(s, .25), 'step IQR hi': np.quantile(s, .75),
        'step MAD (ng/m3)': 1.4826 * np.median(np.abs(s - np.median(s))),
        'control MAD (ng/m3)': 1.4826 * np.median(np.abs(cs - np.median(cs))),
        'rel step median (%)': 100 * np.median(r), 'rel step IQR lo (%)': 100 * np.quantile(r, .25),
        'rel step IQR hi (%)': 100 * np.quantile(r, .75),
        '|rel step| > 25% (%)': 100 * np.mean(np.abs(r) > .25),
        'control |rel step| > 25% (%)': 100 * np.mean(np.abs(cr) > .25),
        'fraction positive': np.mean(s > 0), 'control fraction positive': np.mean(cs > 0),
    })
tape_step = pd.DataFrame(step_rows).set_index('channel')
tape_step.to_csv(OUT / 'tape_advance_step.csv')
display(tape_step.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
for c in CHANNELS:
    r = step_detail[c]['rel'].dropna() * 100
    ax.hist(r.clip(-100, 100), bins=60, histtype='step', lw=1.3, color=colors[c], label=c, density=True)
cr = (crel * 100)
ax.hist(cr[~np.isnan(cr)].clip(-100, 100), bins=60, histtype='stepfilled', alpha=0.2,
        color='grey', label='control (IR, random times)', density=True)
ax.axvline(0, color='k', lw=0.8)
ax.set(xlabel='BCc step across tape advance, (post − pre)/pre (%)', ylabel='density',
       title='Tape-advance step, per channel, vs random-time control')
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
d = step_detail['IR']
ax.scatter(d['pre'], d['step'], s=10, alpha=0.5, color=colors['IR'])
ax.axhline(0, color='k', lw=0.8)
ax.set(xlabel='IR BCc before advance (ng/m3)', ylabel='IR BCc step (ng/m3)',
       title='IR step vs level (no loading-dependent bias would be flat at 0)')
fig.tight_layout()
fig.savefig(PLOTS / 'a2_tape_advance_step.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### A3. Sensitivity to environment and loading; the clean mask
#
# Ratios and AAEs per bin are computed from the bin's *summed* absorption (the ratio of
# means), which is robust to the minute-level noise that makes an instantaneous AAE
# meaningless at 1-min resolution; the median of the per-minute BCc ratio is shown
# beside it.

# %%
%%skip_unless_minute
def bin_optics(frame, key, bins, labels=None, q=False):
    if q:
        b = pd.qcut(frame[key], bins, duplicates='drop')
    else:
        b = pd.cut(frame[key], bins, labels=labels, right=False)
    out = []
    for name, part in frame.groupby(b, observed=True):
        s = {c: part[f'{c} b'].sum() for c in CHANNELS}
        out.append({
            'bin': str(name), 'n minutes': len(part),
            'AAE(470,880)': float(aae(s['Blue'], s['IR'], WAVELENGTHS_NM['Blue'], WAVELENGTHS_NM['IR'])),
            'AAE(625,880)': float(aae(s['Red'], s['IR'], WAVELENGTHS_NM['Red'], WAVELENGTHS_NM['IR'])),
            'Green/IR BCc (ratio of means)': s['Green'] / SIGMA_ATN['Green'] / (s['IR'] / SIGMA_ATN['IR']),
            'Red/IR BCc (ratio of means)': s['Red'] / SIGMA_ATN['Red'] / (s['IR'] / SIGMA_ATN['IR']),
            'Red/IR BCc (median of minutes)': (part['Red BCc'] / part['IR BCc']).median(),
            'UV/IR BCc (ratio of means)': s['UV'] / SIGMA_ATN['UV'] / (s['IR'] / SIGMA_ATN['IR']),
            'mean IR BCc (ng/m3)': part['IR BCc'].mean(),
        })
    return pd.DataFrame(out)


optics_ok = df[[f'{c} BCc' for c in CHANNELS]].notna().all(axis=1)
base = df[optics_ok]
env_tables = {}
env_tables['|dT/dt| (10-min mean, °C/min)'] = bin_optics(base[base['dT_roll10_abs'].notna()], 'dT_roll10_abs', 5, q=True)
env_tables['|dRH/dt| (10-min mean, %/min)'] = bin_optics(base[base['dRH_roll10_abs'].notna()], 'dRH_roll10_abs', 5, q=True)
env_tables['Flow1/Flow2 deviation from median ratio'] = bin_optics(
    base.assign(fr_dev=(base['flow_ratio'] / ratio_nominal - 1).abs()),
    'fr_dev', [0, .02, .05, .10, .25, 10], labels=['<2%', '2–5%', '5–10%', '10–25%', '>25%'])
env_tables['Total flow deviation from setpoint'] = bin_optics(
    base.assign(fd=base['flow_dev'].abs()), 'fd', [0, .02, .05, .10, 10], labels=['<2%', '2–5%', '5–10%', '>10%'])
env_tables['IR ATN1 loading'] = bin_optics(base, 'IR ATN1', [0, 5, 10, 20, 30, 50, 80, 300])
env_tables['UV ATN1 loading'] = bin_optics(base, 'UV ATN1', [0, 5, 10, 20, 30, 50, 80, 300])
env_tables['spot age'] = bin_optics(base, 'age_min', AGE_BINS,
                                    labels=['<30 min', '30–60', '1–3 h', '3–6 h', '6–12 h', '12–24 h', '>24 h'])
env_all = pd.concat({k: v for k, v in env_tables.items()}, names=['binned by']).reset_index(level=0)
env_all.to_csv(OUT / 'optics_by_environment_bins.csv', index=False)
for k, v in env_tables.items():
    print(f'\n— binned by {k}')
    display(v.set_index('bin').round(4))

# %%
%%skip_unless_minute
median_dT = float(base['dT_roll10_abs'].median())
median_dRH = float(base['dRH_roll10_abs'].median())
clean = (base['flow_nominal']
         & (base['dT_roll10_abs'] < median_dT)
         & (base['dRH_roll10_abs'] < median_dRH)
         & (base['UV ATN1'] < 30)
         & ~base['near_advance'])
base = base.assign(clean=clean)
clean_fraction = float(clean.mean())
print(f'median |dT/dt| = {median_dT:.4f} °C/min, median |dRH/dt| = {median_dRH:.4f} %/min')
print(f'clean-mask fraction of minutes (all four conditions): {100 * clean_fraction:.1f} %  '
      f'(n = {int(clean.sum()):,} of {len(base):,})')
parts = {'nominal flow': base['flow_nominal'], '|dT/dt| < median': base['dT_roll10_abs'] < median_dT,
         '|dRH/dt| < median': base['dRH_roll10_abs'] < median_dRH, 'UV ATN1 < 30': base['UV ATN1'] < 30,
         '>30 min from advance': ~base['near_advance']}
print('individual condition pass rates: ' + ', '.join(f'{k} {100 * v.mean():.1f}%' for k, v in parts.items()))

# per 9am-day AAE, all minutes vs clean minutes, on days with enough of both
daily_rows = []
for label, sel in (('all minutes', pd.Series(True, index=base.index)), ('clean mask', base['clean'])):
    g = base[sel].groupby('day_9am')
    s = g[[f'{c} b' for c in CHANNELS]].mean()
    n = g.size()
    d = pd.DataFrame({
        'n': n,
        'AAE(470,880)': aae(s['Blue b'], s['IR b'], WAVELENGTHS_NM['Blue'], WAVELENGTHS_NM['IR']),
        'AAE(625,880)': aae(s['Red b'], s['IR b'], WAVELENGTHS_NM['Red'], WAVELENGTHS_NM['IR']),
        'AAE(528,880)': aae(s['Green b'], s['IR b'], WAVELENGTHS_NM['Green'], WAVELENGTHS_NM['IR']),
        'Red/IR BCc': (s['Red b'] / SIGMA_ATN['Red']) / (s['IR b'] / SIGMA_ATN['IR']),
        'Green/IR BCc': (s['Green b'] / SIGMA_ATN['Green']) / (s['IR b'] / SIGMA_ATN['IR']),
    })
    d['which'] = label
    daily_rows.append(d)
daily_all, daily_clean = daily_rows
common = daily_all.index[(daily_all['n'] >= 720)].intersection(daily_clean.index[daily_clean['n'] >= 180])
clean_rows = []
for label, d in (('all minutes', daily_all), ('clean mask', daily_clean)):
    dd = d.loc[common]
    clean_rows.append({
        'minutes used': label, 'n days': len(dd),
        'AAE(625,880) mean': dd['AAE(625,880)'].mean(), 'AAE(625,880) sd': dd['AAE(625,880)'].std(),
        'AAE(625,880) % days > 1': 100 * (dd['AAE(625,880)'] > 1).mean(),
        'AAE(528,880) mean': dd['AAE(528,880)'].mean(), 'AAE(528,880) sd': dd['AAE(528,880)'].std(),
        'AAE(470,880) mean': dd['AAE(470,880)'].mean(), 'AAE(470,880) sd': dd['AAE(470,880)'].std(),
        'Red/IR BCc median': dd['Red/IR BCc'].median(), 'Green/IR BCc median': dd['Green/IR BCc'].median(),
    })
paired = (daily_clean.loc[common, 'AAE(625,880)'] - daily_all.loc[common, 'AAE(625,880)'])
clean_rows.append({'minutes used': 'paired difference (clean − all)', 'n days': len(common),
                   'AAE(625,880) mean': paired.mean(), 'AAE(625,880) sd': paired.std(),
                   'AAE(625,880) % days > 1': np.nan,
                   'AAE(528,880) mean': (daily_clean.loc[common, 'AAE(528,880)'] - daily_all.loc[common, 'AAE(528,880)']).mean(),
                   'AAE(528,880) sd': np.nan,
                   'AAE(470,880) mean': (daily_clean.loc[common, 'AAE(470,880)'] - daily_all.loc[common, 'AAE(470,880)']).mean(),
                   'AAE(470,880) sd': np.nan,
                   'Red/IR BCc median': np.nan, 'Green/IR BCc median': np.nan})
clean_aae = pd.DataFrame(clean_rows).set_index('minutes used')
clean_aae.to_csv(OUT / 'clean_mask_daily_aae.csv')
display(clean_aae.round(4))
print('ftir_28 reference on the processed filter-days: AAE(625,880) = 0.944 ± 0.060')

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
ax = axes[0]
ax.hist(daily_all.loc[common, 'AAE(625,880)'], bins=40, alpha=0.5, color='#7F8C8D', label='all minutes')
ax.hist(daily_clean.loc[common, 'AAE(625,880)'], bins=40, alpha=0.6, color='#C0392B', label='clean mask')
ax.axvline(1.0, color='k', ls='--', lw=1, label='AAE_BC ≈ 1 anchor')
ax.set(xlabel='daily AAE(625, 880)', ylabel='days', title='Daily AAE(625,880): all vs clean minutes')
ax.legend(frameon=False, fontsize=9)
for ax, key, xlabel in ((axes[1], '|dT/dt| (10-min mean, °C/min)', '|dT/dt| quintile'),
                        (axes[2], 'IR ATN1 loading', 'IR ATN1 bin')):
    t = env_tables[key]
    x = np.arange(len(t))
    ax.plot(x, t['AAE(625,880)'], 'o-', color='#C0392B', label='AAE(625,880)')
    ax.plot(x, t['AAE(470,880)'], 's-', color='#2E86C1', label='AAE(470,880)')
    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.set(xticks=x, xticklabels=t['bin'], xlabel=xlabel, ylabel='AAE (ratio of bin means)',
           title=f'AAE by {xlabel}')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(PLOTS / 'a3_clean_mask_aae.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Part B — the missingness benchmark
#
# ### B4. Which 20 % is missing matters: masked daily means on complete days
#
# Days with ≥ 95 % of their 1440 minutes are made complete by linear interpolation of the
# ≤ 5 % gaps (so every mask removes an exact fraction). Five loss patterns at 10/20/40 %:
# random minutes; one contiguous outage at a random start; rush-hour-only loss (07–10 and
# 17–20 local — 360 min, so the 40 % level is all rush-hour minutes plus random fill);
# high-pollution loss (the day's highest minutes, i.e. the top decile at 10 %, top 40 % at
# 40 %); and low-pollution loss (the lowest minutes). Random and contiguous masks are
# drawn 20 times per day.

# %%
%%skip_unless_minute
def day_matrix(col):
    sub = df[['day_9am', 'min_9am', col]].dropna()
    m = sub.pivot_table(index='day_9am', columns='min_9am', values=col, aggfunc='mean')
    m = m.reindex(columns=np.arange(1440))
    return m


COVERAGE = 0.95
mats, complete_days = {}, None
for c in ('IR', 'Red'):
    m = day_matrix(f'{c} BCc')
    cov = m.notna().mean(axis=1)
    m = m[cov >= COVERAGE]
    m = m.interpolate(axis=1, limit_direction='both')
    mats[c] = m
    complete_days = m.index if complete_days is None else complete_days.intersection(m.index)
for c in mats:
    mats[c] = mats[c].loc[complete_days]
n_days = len(complete_days)
print(f'9am–9am days with ≥ {COVERAGE:.0%} minute coverage: {n_days} of {df["day_9am"].nunique()} days '
      f'({complete_days.min().date()} to {complete_days.max().date()})')
print(f'mean daily IR BCc on those days: {mats["IR"].mean(axis=1).mean():.0f} ng/m3, '
      f'Red BCc: {mats["Red"].mean(axis=1).mean():.0f} ng/m3')

minute_hour = ((9 * 60 + np.arange(1440)) // 60) % 24
RUSH = np.isin(minute_hour, [7, 8, 9, 17, 18, 19])
N_DRAWS = 20
LEVELS = [0.10, 0.20, 0.40]


def masked_errors(M, kind, frac):
    """Percent error of the masked daily mean, one value per (day, draw)."""
    D, N = M.shape
    L = int(round(frac * N))
    true = M.mean(axis=1)
    errs = []
    draws = N_DRAWS if kind in ('random', 'contiguous', 'rush-hour') else 1
    for _ in range(draws):
        keep = np.ones((D, N), dtype=bool)
        if kind == 'random':
            ranks = RNG.random((D, N)).argsort(axis=1)
            keep[ranks < L] = False
        elif kind == 'contiguous':
            s = RNG.integers(0, N - L + 1, size=D)
            idx = np.arange(N)[None, :]
            keep = ~((idx >= s[:, None]) & (idx < (s + L)[:, None]))
        elif kind == 'rush-hour':
            rush_idx = np.flatnonzero(RUSH)
            other_idx = np.flatnonzero(~RUSH)
            for d in range(D):
                if L <= len(rush_idx):
                    drop = RNG.choice(rush_idx, L, replace=False)
                else:
                    drop = np.concatenate([rush_idx, RNG.choice(other_idx, L - len(rush_idx), replace=False)])
                keep[d, drop] = False
        elif kind == 'high-pollution':
            order = np.argsort(-M.values, axis=1)
            keep[np.arange(D)[:, None], order[:, :L]] = False
        elif kind == 'low-pollution':
            order = np.argsort(M.values, axis=1)
            keep[np.arange(D)[:, None], order[:, :L]] = False
        rec = np.where(keep, M.values, np.nan)
        rec_mean = np.nanmean(rec, axis=1)
        errs.append(100 * (rec_mean / true.values - 1))
    e = np.concatenate(errs)
    abs_err = np.concatenate([(np.nanmean(np.where(k, M.values, np.nan), axis=1) - true.values)
                              for k in [keep]])  # last draw, absolute ng/m3 (for the 2 µg/m³ yardstick)
    return e, abs_err


KINDS = ['random', 'contiguous', 'rush-hour', 'high-pollution', 'low-pollution']
mask_rows, mask_err = [], {}
for c in ('IR', 'Red'):
    for kind in KINDS:
        for frac in LEVELS:
            e, abs_err = masked_errors(mats[c], kind, frac)
            mask_err[(c, kind, frac)] = e
            mask_rows.append({
                'channel': c, 'mask': kind, 'loss': f'{int(frac * 100)}%',
                'bias (mean err, %)': e.mean(), 'median err (%)': np.median(e),
                'IQR lo (%)': np.quantile(e, .25), 'IQR hi (%)': np.quantile(e, .75),
                'p2.5 (%)': np.quantile(e, .025), 'p97.5 (%)': np.quantile(e, .975),
                '|err| > 25% (%)': 100 * np.mean(np.abs(e) > 25),
                'median |abs err| (ng/m3)': np.median(np.abs(abs_err)),
                '|abs err| > 2000 ng/m3 (%)': 100 * np.mean(np.abs(abs_err) > 2000),
            })
mask_table = pd.DataFrame(mask_rows)
mask_table.to_csv(OUT / 'missingness_benchmark.csv', index=False)
print('IR BCc')
display(mask_table[mask_table['channel'].eq('IR')].drop(columns='channel').set_index(['mask', 'loss']).round(2))
print('Red BCc')
display(mask_table[mask_table['channel'].eq('Red')].drop(columns='channel').set_index(['mask', 'loss']).round(2))

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
level_colors = {0.10: '#AED6F1', 0.20: '#5DADE2', 0.40: '#1B4F72'}
for ax, c in zip(axes, ('IR', 'Red')):
    positions, data, cols = [], [], []
    for i, kind in enumerate(KINDS):
        for j, frac in enumerate(LEVELS):
            positions.append(i * 4 + j)
            data.append(np.clip(mask_err[(c, kind, frac)], -60, 60))
            cols.append(level_colors[frac])
    bp = ax.boxplot(data, positions=positions, widths=0.8, showfliers=False, patch_artist=True,
                    whis=(2.5, 97.5), medianprops={'color': 'k'})
    for patch, col in zip(bp['boxes'], cols):
        patch.set_facecolor(col)
    ax.axhline(0, color='k', lw=0.8)
    ax.axhspan(-25, 25, color='#F5B7B1', alpha=0.25, zorder=0)
    ax.set(xticks=[i * 4 + 1 for i in range(len(KINDS))], xticklabels=KINDS,
           title=f'{c} BCc: error of the recovered daily mean (n = {n_days} days)',
           ylabel='(masked mean − true mean)/true (%)' if c == 'IR' else '')
    ax.tick_params(axis='x', rotation=15)
for frac, col in level_colors.items():
    axes[0].bar([0], [0], color=col, label=f'{int(frac * 100)}% of minutes lost')
axes[0].legend(frameon=False, fontsize=9, loc='upper left')
axes[0].annotate('pink band: ±25 % ≈ the 2 µg/m³ FTIR–HIPS–MA350 offset', (0.02, 0.02),
                 xycoords='axes fraction', fontsize=8, color='#922B21')
fig.tight_layout()
fig.savefig(PLOTS / 'b4_missingness_benchmark.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### B5. AAE from daily-aggregated absorption vs the daily mean of instantaneous AAE

# %%
%%skip_unless_minute
aae_rows, aae_diff = [], {}
sel = df['day_9am'].isin(complete_days)
day = df[sel]
for short in ('Blue', 'Red'):
    pair = f'AAE({WAVELENGTHS_NM[short]},880)'
    b_s, b_l = day[f'{short} b'], day['IR b']
    inst = aae(b_s.where((b_s > 0) & (b_l > 0)), b_l.where((b_s > 0) & (b_l > 0)),
               WAVELENGTHS_NM[short], WAVELENGTHS_NM['IR'])
    g = day.assign(inst=inst).groupby('day_9am')
    agg = aae(g[f'{short} b'].mean(), g['IR b'].mean(), WAVELENGTHS_NM[short], WAVELENGTHS_NM['IR'])
    inst_mean, inst_median = g['inst'].mean(), g['inst'].median()
    invalid = 100 * inst.isna().mean()
    diff_mean, diff_median = inst_mean - agg, inst_median - agg
    aae_diff[pair] = diff_mean
    aae_rows.append({
        'pair': pair, 'n days': len(agg), 'minutes with undefined instantaneous AAE (%)': invalid,
        'aggregated AAE mean': agg.mean(), 'aggregated AAE sd': agg.std(),
        'mean-of-instantaneous AAE mean': inst_mean.mean(), 'mean-of-instantaneous sd': inst_mean.std(),
        'diff (mean-of-inst − agg) median': diff_mean.median(),
        'diff IQR lo': diff_mean.quantile(.25), 'diff IQR hi': diff_mean.quantile(.75),
        'diff p2.5': diff_mean.quantile(.025), 'diff p97.5': diff_mean.quantile(.975),
        '|diff| > 0.1 (% days)': 100 * (diff_mean.abs() > 0.1).mean(),
        'median-of-instantaneous − agg, median': diff_median.median(),
        'median-of-instantaneous − agg, IQR lo': diff_median.quantile(.25),
        'median-of-instantaneous − agg, IQR hi': diff_median.quantile(.75),
    })
aae_table = pd.DataFrame(aae_rows).set_index('pair')
aae_table.to_csv(OUT / 'aae_aggregation_order.csv')
display(aae_table.round(3))

fig, ax = plt.subplots(figsize=(7, 4.2))
for pair, col in zip(aae_diff, ('#2E86C1', '#C0392B')):
    ax.hist(aae_diff[pair].clip(-1.5, 1.5), bins=60, histtype='step', lw=1.5, color=col, label=pair)
ax.axvline(0, color='k', lw=0.8)
ax.set(xlabel='daily mean of instantaneous AAE − AAE of daily-aggregated absorption',
       ylabel='days', title='Order of operations matters for AAE')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PLOTS / 'b5_aae_aggregation_order.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### B6. The real ETAD filter dates: calendar day vs 9am–9am vs the ±1-day tolerant mean
#
# `match_aeth_filter_data` (`research/ftir_hips_chem/scripts/data_matching.py`) averages
# every daily `day_9am` value within ±1 day of the filter's `SampleDate` — i.e. the mean
# of three daily means. The three candidate "filter-day" numbers are compared on every
# ETAD evaluation filter whose start date falls inside the minute record.

# %%
%%skip_unless_minute
etad_eval, _, _ = load_addis_evaluation()
filter_dates = pd.DatetimeIndex(etad_eval['SamplingStartDate'].dropna().dt.normalize().unique()).sort_values()
in_range = filter_dates[(filter_dates >= df['t'].min().normalize()) & (filter_dates <= df['t'].max().normalize())]
print(f'ETAD evaluation filters: {len(etad_eval)} with {len(filter_dates)} distinct start dates; '
      f'{len(in_range)} fall inside the minute record')

daily_9am = df.groupby('day_9am')[['IR BCc', 'Red BCc']].mean()
daily_9am_n = df.groupby('day_9am')['IR BCc'].count()
daily_cal = df.groupby('cal_day')[['IR BCc', 'Red BCc']].mean()
daily_cal_n = df.groupby('cal_day')['IR BCc'].count()

fw_rows = []
for d in in_range:
    row = {'filter_date': d}
    for c in ('IR', 'Red'):
        col = f'{c} BCc'
        nine = daily_9am[col].get(d, np.nan) if daily_9am_n.get(d, 0) >= 720 else np.nan
        cal = daily_cal[col].get(d, np.nan) if daily_cal_n.get(d, 0) >= 720 else np.nan
        tol_days = [d + pd.Timedelta(days=k) for k in (-1, 0, 1)]
        tol_vals = [daily_9am[col].get(x, np.nan) for x in tol_days if daily_9am_n.get(x, 0) >= 720]
        tol = np.mean(tol_vals) if len(tol_vals) > 0 else np.nan
        row.update({f'{c} 9am': nine, f'{c} calendar': cal, f'{c} tolerant': tol,
                    f'{c} n tolerant days': len(tol_vals)})
    fw_rows.append(row)
fw = pd.DataFrame(fw_rows)
for c in ('IR', 'Red'):
    fw[f'{c} cal − 9am (%)'] = 100 * (fw[f'{c} calendar'] / fw[f'{c} 9am'] - 1)
    fw[f'{c} tol − 9am (%)'] = 100 * (fw[f'{c} tolerant'] / fw[f'{c} 9am'] - 1)
    trio = fw[[f'{c} 9am', f'{c} calendar', f'{c} tolerant']]
    fw[f'{c} spread (max−min)/9am (%)'] = 100 * (trio.max(axis=1) - trio.min(axis=1)) / fw[f'{c} 9am']
fw.to_csv(OUT / 'filter_day_window_comparison.csv', index=False)

fw_rows = []
for c in ('IR', 'Red'):
    ok = fw[[f'{c} 9am', f'{c} calendar', f'{c} tolerant']].notna().all(axis=1)
    f = fw[ok]
    fw_rows.append({
        'channel': c, 'filter days with all three': int(ok.sum()),
        'cal − 9am median (%)': f[f'{c} cal − 9am (%)'].median(),
        'cal − 9am IQR lo': f[f'{c} cal − 9am (%)'].quantile(.25), 'cal − 9am IQR hi': f[f'{c} cal − 9am (%)'].quantile(.75),
        '|cal − 9am| > 10% (days)': int((f[f'{c} cal − 9am (%)'].abs() > 10).sum()),
        'tol − 9am median (%)': f[f'{c} tol − 9am (%)'].median(),
        'tol − 9am IQR lo': f[f'{c} tol − 9am (%)'].quantile(.25), 'tol − 9am IQR hi': f[f'{c} tol − 9am (%)'].quantile(.75),
        '|tol − 9am| > 10% (days)': int((f[f'{c} tol − 9am (%)'].abs() > 10).sum()),
        '|tol − 9am| > 25% (days)': int((f[f'{c} tol − 9am (%)'].abs() > 25).sum()),
        'spread median (%)': f[f'{c} spread (max−min)/9am (%)'].median(),
        'spread p90 (%)': f[f'{c} spread (max−min)/9am (%)'].quantile(.9),
        'spread > 10% (days)': int((f[f'{c} spread (max−min)/9am (%)'] > 10).sum()),
        'spread > 25% (days)': int((f[f'{c} spread (max−min)/9am (%)'] > 25).sum()),
        'median |tol − 9am| (ng/m3)': (f[f'{c} tolerant'] - f[f'{c} 9am']).abs().median(),
        '|tol − 9am| > 2000 ng/m3 (days)': int(((f[f'{c} tolerant'] - f[f'{c} 9am']).abs() > 2000).sum()),
    })
filter_window = pd.DataFrame(fw_rows).set_index('channel')
filter_window.to_csv(OUT / 'filter_day_window_summary.csv')
display(filter_window.round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ok = fw[['IR 9am', 'IR calendar', 'IR tolerant']].notna().all(axis=1)
ax = axes[0]
ax.hist(fw.loc[ok, 'IR cal − 9am (%)'].clip(-60, 60), bins=40, alpha=0.55, color='#7F8C8D', label='calendar day − 9am day')
ax.hist(fw.loc[ok, 'IR tol − 9am (%)'].clip(-60, 60), bins=40, alpha=0.55, color='#C0392B', label='±1-day tolerant − 9am day')
ax.axvline(0, color='k', lw=0.8)
ax.axvspan(-10, 10, color='#F5B7B1', alpha=0.25, zorder=0)
ax.set(xlabel='difference relative to the 9am–9am filter-day mean (%)', ylabel='filter days',
       title=f'IR BCc on {int(ok.sum())} ETAD filter dates')
ax.legend(frameon=False, fontsize=9)
ax = axes[1]
ax.scatter(fw.loc[ok, 'IR 9am'] / 1000, fw.loc[ok, 'IR tolerant'] / 1000, s=16, alpha=0.6, color='#C0392B',
           label='±1-day tolerant')
ax.scatter(fw.loc[ok, 'IR 9am'] / 1000, fw.loc[ok, 'IR calendar'] / 1000, s=16, alpha=0.6, color='#7F8C8D',
           label='calendar day')
lim = [0, np.nanmax(fw.loc[ok, ['IR 9am', 'IR tolerant', 'IR calendar']].to_numpy()) / 1000 * 1.05]
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set(xlabel='9am–9am mean IR BCc (µg/m³)', ylabel='alternative window mean (µg/m³)',
       title='Alternative filter-day windows vs the 9am day', xlim=lim, ylim=lim)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(PLOTS / 'b6_filter_day_windows.png', bbox_inches='tight')
plt.show()

# %%
# fallback: daily-pickle version of B6 (9am day vs ±1-day tolerant; no calendar day possible)
if not MINUTE_DATA:
    etad_eval, _, _ = load_addis_evaluation()
    pk = pd.read_pickle(PROCESSED_SITES_DIR / 'df_Jacros_9am_resampled.pkl')
    pk['day_9am'] = pd.to_datetime(pk['day_9am'])
    daily = pk.set_index('day_9am')[['IR BCc', 'Red BCc']]
    filter_dates = pd.DatetimeIndex(etad_eval['SamplingStartDate'].dropna().dt.normalize().unique()).sort_values()
    in_range = filter_dates[(filter_dates >= daily.index.min()) & (filter_dates <= daily.index.max())]
    print(f'FALLBACK (daily 9am pickle): {len(in_range)} ETAD filter start dates inside the daily record; '
          'the calendar-day mean needs minutes and is not computed.')
    rows = []
    for d in in_range:
        row = {'filter_date': d}
        for c in ('IR', 'Red'):
            nine = daily[f'{c} BCc'].get(d, np.nan)
            tol_vals = [daily[f'{c} BCc'].get(d + pd.Timedelta(days=k), np.nan) for k in (-1, 0, 1)]
            tol_vals = [v for v in tol_vals if not np.isnan(v)]
            row[f'{c} 9am'] = nine
            row[f'{c} tolerant'] = np.mean(tol_vals) if tol_vals else np.nan
            row[f'{c} n tolerant days'] = len(tol_vals)
        rows.append(row)
    fw = pd.DataFrame(rows)
    frows = []
    for c in ('IR', 'Red'):
        fw[f'{c} tol − 9am (%)'] = 100 * (fw[f'{c} tolerant'] / fw[f'{c} 9am'] - 1)
        ok = fw[[f'{c} 9am', f'{c} tolerant']].notna().all(axis=1)
        f = fw[ok]
        frows.append({'channel': c, 'filter days with both': int(ok.sum()),
                      'tol − 9am median (%)': f[f'{c} tol − 9am (%)'].median(),
                      'tol − 9am IQR lo': f[f'{c} tol − 9am (%)'].quantile(.25),
                      'tol − 9am IQR hi': f[f'{c} tol − 9am (%)'].quantile(.75),
                      'tol − 9am p2.5': f[f'{c} tol − 9am (%)'].quantile(.025),
                      'tol − 9am p97.5': f[f'{c} tol − 9am (%)'].quantile(.975),
                      '|tol − 9am| > 10% (days)': int((f[f'{c} tol − 9am (%)'].abs() > 10).sum()),
                      '|tol − 9am| > 25% (days)': int((f[f'{c} tol − 9am (%)'].abs() > 25).sum()),
                      'median |tol − 9am| (ng/m3)': (f[f'{c} tolerant'] - f[f'{c} 9am']).abs().median(),
                      '|tol − 9am| > 2000 ng/m3 (days)': int(((f[f'{c} tolerant'] - f[f'{c} 9am']).abs() > 2000).sum())})
    filter_window_daily = pd.DataFrame(frows).set_index('channel')
    fw.to_csv(OUT / 'filter_day_window_comparison_daily_fallback.csv', index=False)
    filter_window_daily.to_csv(OUT / 'filter_day_window_summary_daily_fallback.csv')
    display(filter_window_daily.round(2))
    RUNTIME = time.time() - T0
    print(f'runtime {RUNTIME / 60:.1f} min')

# %% [markdown]
# ## Scale against the FTIR–HIPS–MA350 discrepancy
#
# One table putting every effect on the same yardstick: the ≈ 2 µg/m³ (≈ 25 % of the
# ≈ 8 µg/m³ daily IR BCc) offset between FTIR EC, HIPS/MAC and MA350 BCc at Addis.

# %%
%%skip_unless_minute
ir_daily_mean = float(mats['IR'].mean(axis=1).mean())
yard = 2000.0
scale_rows = [
    {'effect': 'DualSpot: spot-2 compensated vs reported BCc (IR, median, ATN1 ≥ 5)',
     'typical size (% of level)': 100 * (k_table.loc['IR', 'BC2comp/BCc median'] - 1),
     'tail (IQR or p97.5, %)': 100 * (k_table.loc['IR', 'BC2comp/BCc IQR hi'] - 1)},
    {'effect': 'DualSpot: BCc under K* instead of K (IR, ATN1 30–50 bin, median)',
     'typical size (% of level)': 100 * (kstar_bins[(kstar_bins.channel == 'IR') & (kstar_bins.by == 'ATN1')
                                                    & kstar_bins.bin.str.startswith('[30')]['BCc(K*)/BCc median'].iloc[0] - 1),
     'tail (IQR or p97.5, %)': np.nan},
    {'effect': 'tape-advance step (IR, median relative step)',
     'typical size (% of level)': tape_step.loc['IR', 'rel step median (%)'],
     'tail (IQR or p97.5, %)': tape_step.loc['IR', 'rel step IQR hi (%)']},
    {'effect': 'random 20 % minute loss (IR daily mean)',
     'typical size (% of level)': mask_table.query("channel=='IR' and mask=='random' and loss=='20%'")['bias (mean err, %)'].iloc[0],
     'tail (IQR or p97.5, %)': mask_table.query("channel=='IR' and mask=='random' and loss=='20%'")['p97.5 (%)'].iloc[0]},
    {'effect': 'contiguous 20 % outage (IR daily mean)',
     'typical size (% of level)': mask_table.query("channel=='IR' and mask=='contiguous' and loss=='20%'")['bias (mean err, %)'].iloc[0],
     'tail (IQR or p97.5, %)': mask_table.query("channel=='IR' and mask=='contiguous' and loss=='20%'")['p97.5 (%)'].iloc[0]},
    {'effect': 'rush-hour 20 % loss (IR daily mean)',
     'typical size (% of level)': mask_table.query("channel=='IR' and mask=='rush-hour' and loss=='20%'")['bias (mean err, %)'].iloc[0],
     'tail (IQR or p97.5, %)': mask_table.query("channel=='IR' and mask=='rush-hour' and loss=='20%'")['p2.5 (%)'].iloc[0]},
    {'effect': 'high-pollution 20 % loss (IR daily mean)',
     'typical size (% of level)': mask_table.query("channel=='IR' and mask=='high-pollution' and loss=='20%'")['bias (mean err, %)'].iloc[0],
     'tail (IQR or p97.5, %)': mask_table.query("channel=='IR' and mask=='high-pollution' and loss=='20%'")['p2.5 (%)'].iloc[0]},
    {'effect': 'low-pollution 20 % loss (IR daily mean)',
     'typical size (% of level)': mask_table.query("channel=='IR' and mask=='low-pollution' and loss=='20%'")['bias (mean err, %)'].iloc[0],
     'tail (IQR or p97.5, %)': mask_table.query("channel=='IR' and mask=='low-pollution' and loss=='20%'")['p97.5 (%)'].iloc[0]},
    {'effect': '±1-day tolerant window vs 9am day on ETAD filter dates (IR)',
     'typical size (% of level)': filter_window.loc['IR', 'tol − 9am median (%)'],
     'tail (IQR or p97.5, %)': filter_window.loc['IR', 'spread p90 (%)']},
    {'effect': 'calendar day vs 9am day on ETAD filter dates (IR)',
     'typical size (% of level)': filter_window.loc['IR', 'cal − 9am median (%)'],
     'tail (IQR or p97.5, %)': filter_window.loc['IR', 'cal − 9am IQR hi']},
]
scale = pd.DataFrame(scale_rows)
scale['yardstick: 2 µg/m³ as % of daily IR BCc'] = 100 * yard / ir_daily_mean
scale.to_csv(OUT / 'scale_against_discrepancy.csv', index=False)
display(scale.round(2))
RUNTIME = time.time() - T0
print(f'runtime {RUNTIME / 60:.1f} min')

# %% [markdown]
# ## Takeaways
#
# - **Two of the audited steps are already yardstick-sized from daily data alone.** The
#   loading compensation is 37–48 % of the reported BCc at the channels that matter, so the
#   MA350 "measurement" the calibrations consume is roughly 60 % raw attenuation and 40 %
#   firmware model; and the ±1-day matching window moves one filter day in two by > 10 %
#   and one in six by > 2 µg/m³. Neither can be dismissed as small relative to the ≈ 2 µg/m³
#   FTIR–HIPS–MA350 offset.
# - **The K question is answerable.** The spot-2 residual (BC2comp/BCc ≈ 0.93 at IR) and the
#   closure K\* (+16 % over the reported K at IR) say the compensation is not self-consistent
#   at the few-percent level, but the direction (BCc high vs spot 2) is also what an
#   under-compensated spot 2 leak would produce (`calibration.py` carries the leak-factor
#   machinery for exactly this). The minute-level A1 cell, binned by ATN1 and spot age,
#   separates those readings; it is the first thing to run when the CSV hydrates.
# - **The matching window should be reported alongside every MA350 number.** The
#   repo's default (`date_tolerance_days=1`) averages three 9am days; a 3-h-staggered SPARTAN
#   filter is neither of those windows. Re-running the locked calibrations with the 9am-day
#   mean instead of the tolerant mean is a cheap sensitivity test that ftir_49's B6 table now
#   motivates in numbers (median shift +3.7 %, but tails of ±25–55 %).
# - **The missingness benchmark (B4) remains the reviews' open item** and is only possible
#   from the minute record; the script is written and cached so that it runs unattended
#   (≈ minutes of compute) once the file is local.
#
# ## Limits
#
# - This run used daily means from the processed pickle, not the minute record. Ratios of
#   daily means are not daily means of minute ratios; the closure identity, K\* and the
#   spot-2 residual are indicative until the minute cells run. The UV spot-2 residual (0.33)
#   is a daily-mean artefact — (1 − K·ATN2) approaches zero at UV loadings of ~60 — and is
#   not interpreted.
# - The pickle's Green channel carries ATN1 ≈ 0 and K ≈ 0 throughout (BC1 = BCc), while the
#   raw CSV header rows show Green ATN increasing normally: the Green DualSpot columns in the
#   processed pickle are not trustworthy and Green is excluded from the daily-level claims.
# - The K-sensitivity rows are a first-order propagation (dBCc/BCc = K·ATN1/(1 − K·ATN1) ·
#   dK/K) on the median day; the tape-advance/ATN dependence of K is exactly what the
#   skipped minute cells would resolve.
# - ETAD filter windows are treated as 24-h days from `SamplingStartDate`, as the repo does;
#   the real SPARTAN schedule (~8 × 3-h sub-windows over ~9 days) means all three "filter-day"
#   windows compared here are approximations of the sampled air, and the ±1-day tolerant
#   mean is not obviously the worse one — the point is the spread, not which is right.
# - Absorption here is on the attenuation scale (BCc·σ_ATN); AAE cells did not run.
