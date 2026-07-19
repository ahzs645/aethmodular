"""Post-meeting context addenda: Adama TOR OC/EC and the ETBI (Bishoftu) site.

Produces tables/plots under output/{tables,plots}/context/. Run from
research/ftir_ec_phase3/. These are context for PHASE3_SUMMARY.md, not new
calibrations: the Adama filters are quartz (no FTIR/HIPS on the same filters),
and ETBI has no locally available spectra yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path('scripts').resolve()))
sys.path.insert(0, str((Path('..') / 'ftir_hips_chem' / 'scripts').resolve()))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase3_common import PATHS, load_pool_metadata, load_tor_loadings

TABLE_DIR = Path('output/tables/context')
PLOT_DIR = Path('output/plots/context')
for directory in (TABLE_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DATA = PATHS.etad_dir.parent.parent  # .../NASA MAIA/Data

# ---- Adama TOR (Batch 54, 5 quartz filters, July 2024) ----------------------
adama = pd.read_csv(DATA / 'DAVIS/Adama TOR/OC_EC_concs_Batch54.csv')
wide = (adama.pivot_table(index=['FilterId', 'SampleDate'], columns='Parameter',
                          values='Concentration_ug_m3')
        .reset_index())
wide['OC_EC_ratio_TR'] = wide['OCTR'] / wide['ECTR']
wide['OC_EC_ratio_TT'] = wide['OCTT'] / wide['ECTT']
wide.to_csv(TABLE_DIR / 'adama_tor_ocec_summary.csv', index=False)
print(wide[['FilterId', 'SampleDate', 'ECTR', 'OCTR', 'OC_EC_ratio_TR',
            'OC_EC_ratio_TT']].to_string(index=False))

# ---- IMPROVE pool OC/EC context ---------------------------------------------
pool = load_pool_metadata().merge(load_tor_loadings(), on=['Site', 'date'],
                                  how='left', validate='many_to_one')
eligible = (pool['TOR_EC_ugm3'].gt(0) & pool['TOR_OC_ugm3'].gt(0)
            & pool['OC_EC_ratio'].notna())
pool_ratio = (pool[eligible].drop_duplicates('FilterId')['OC_EC_ratio'])
cohort_cut_800 = float(pd.read_csv(
    'output/tables/ftir11/cohort_composition.csv')
    .set_index('cohort').loc['lowest-OCEC 800', 'OCEC_max'])

# ---- ETBI vs ETAD HIPS ------------------------------------------------------
hips = pd.read_csv(PATHS.spartan_hips_primary, encoding='cp1252')
site_rows = []
for site, label in (('ETAD', 'Addis Ababa'), ('ETBI', 'Bishoftu')):
    d = hips[hips['Site'].eq(site)].copy()
    d['date'] = pd.to_datetime(d['SampleDate'], errors='coerce')
    f = d['Fabs'].dropna()
    site_rows.append({
        'site': site, 'name': label, 'filters': int(d['FilterId'].nunique()),
        'with_Fabs': int(len(f)),
        'date_min': str(d['date'].min().date()), 'date_max': str(d['date'].max().date()),
        'Fabs_p25': float(f.quantile(.25)), 'Fabs_median': float(f.median()),
        'Fabs_p75': float(f.quantile(.75)),
        'HIPS_EC_equiv_MAC10_median_ugm3': float(f.median() / 10),
    })
etbi_summary = pd.DataFrame(site_rows)
etbi_summary.to_csv(TABLE_DIR / 'etad_etbi_hips_summary.csv', index=False)
print()
print(etbi_summary.to_string(index=False))

# ---- Figure -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
axes[0].hist(pool_ratio.clip(upper=25), bins=120, color='#7F8C8D', alpha=.8,
             label='IMPROVE lot-248/251 pool')
axes[0].axvline(cohort_cut_800, color='#2980B9', lw=1.6,
                label=f'lowest-OCEC 800 cut (≤{cohort_cut_800:.2f})')
for _, row in wide.iterrows():
    axes[0].axvline(row['OC_EC_ratio_TR'], color='#C0392B', lw=1.4, alpha=.85)
axes[0].axvline(np.nan, color='#C0392B', lw=1.4, label='Adama TOR filters (TR basis, n=5)')
axes[0].set(xlim=(0, 25.5), ylim=(0, None),
            xlabel='TOR OC/EC ratio (values >25 stacked at 25)', ylabel='Filters',
            title='Adama sits at the IMPROVE pool median,\nnot in the low-OC/EC tail')
axes[0].legend(fontsize=8)

for (site, color) in (('ETAD', '#C0392B'), ('ETBI', '#2980B9')):
    f = hips.loc[hips['Site'].eq(site), 'Fabs'].dropna()
    axes[1].hist(f, bins=np.arange(0, 90, 4), color=color, alpha=.55,
                 label=f'{site} (n={len(f)})')
axes[1].set(xlim=(0, 90), ylim=(0, None), xlabel='HIPS Fabs (Mm⁻¹)', ylabel='Filters',
            title='Bishoftu (ETBI) absorbs less than Addis\nbut is far above IMPROVE levels')
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'adama_etbi_context.png', dpi=180, bbox_inches='tight')
print(f"\nwrote {PLOT_DIR / 'adama_etbi_context.png'}")
