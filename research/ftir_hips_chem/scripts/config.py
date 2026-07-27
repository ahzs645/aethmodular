"""
Central configuration for multi-site aethalometer analysis.
Contains site definitions, paths, and analysis parameters.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("AETHMODULAR_DATA_ROOT", _DEFAULT_DATA_ROOT)).expanduser().resolve()

PROCESSED_SITES_DIR = DATA_ROOT / "processed_sites"
FILTER_DATA_PATH = DATA_ROOT / "Filter Data" / "unified_filter_dataset.pkl"
ETAD_FACTOR_CONTRIBUTIONS_PATH = DATA_ROOT / "Filter Data" / "ETAD Factor Contributions .csv"
ETAD_FILTER_ID_PATH = DATA_ROOT / "Filter Data" / "ETAD Filter ID.csv"
AERONET_DATA_DIR = DATA_ROOT / "AERONET"
WEATHER_DATA_DIR = DATA_ROOT / "Weather Data"

# =============================================================================
# SITE CONFIGURATIONS
# =============================================================================
SITES = {
    'Beijing': {
        'file': 'df_Beijing_9am_resampled.pkl',
        'code': 'CHTS',
        'color': '#E74C3C',  # Red
        'location': 'Beijing, China',
        'timezone': 'Asia/Shanghai'
    },
    'Delhi': {
        'file': 'df_Delhi_9am_resampled.pkl',
        'code': 'INDH',
        'color': '#3498DB',  # Blue
        'location': 'Delhi, India',
        'timezone': 'Asia/Kolkata'
    },
    'JPL': {
        'file': 'df_JPL_9am_resampled.pkl',
        'code': 'USPA',
        'color': '#2ECC71',  # Green
        'location': 'Pasadena, USA',
        'timezone': 'America/Los_Angeles'
    },
    'Addis_Ababa': {
        'file': 'df_Addis_Ababa_9am_resampled.pkl',
        'code': 'ETAD',
        'color': '#F39C12',  # Orange
        'location': 'Addis Ababa, Ethiopia',
        'timezone': 'Africa/Addis_Ababa'
    }
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Mass Absorption Cross-section (m^2/g) for HIPS conversion
MAC_VALUE = 10

# Flow fix periods (when aethalometer flow was corrected)
# Updated based on FlowFix_BeforeAfter_Analysis.ipynb findings
# NOTE: Use flow_periods.py for the comprehensive version with has_before_data flags
FLOW_FIX_PERIODS = {
    'Beijing': {
        'description': 'NO BEFORE DATA - Filter sampling started Sep 2023 (degraded period only)',
        'before_end': '2022-07-31',
        'after_start': '2023-09-01',
        'has_before_data': False,
        'notes': 'Flow ratio ~1.8-2.2 in available data period'
    },
    'Delhi': {
        'description': 'NO BEFORE DATA - Filter sampling started Feb 2024 (degraded period only)',
        'before_end': '2023-12-31',
        'after_start': '2024-02-01',
        'has_before_data': False,
        'notes': 'Flow ratio ~2.5-3.2 in available data period'
    },
    'JPL': {
        'description': 'Has data in both periods - suitable for before/after analysis',
        'before_end': '2022-09-30',
        'after_start': '2023-05-01',
        'has_before_data': True,
        'notes': 'Good flow ratio throughout'
    },
    'Addis_Ababa': {
        'description': 'No flow fix periods defined',
        'before_end': None,
        'after_start': None,
        'has_before_data': False,
        'notes': 'Consistently low flow ratio (~1.2 throughout)'
    }
}

# Smooth vs raw thresholds to test (% difference)
SMOOTH_RAW_THRESHOLDS = [1, 2.5, 4, 5]

# Default BC wavelength
DEFAULT_BC_WAVELENGTH = 'IR'

# Minimum EC value (ug/m3) - below this is considered below MDL
MIN_EC_THRESHOLD = 0.5

# =============================================================================
# FILTER DATA PARAMETER CATEGORIES
# =============================================================================
FILTER_CATEGORIES = {
    'ChemSpec EC/OC': [
        'ChemSpec_EC_PM2.5', 'ChemSpec_OC_PM2.5',
        'ChemSpec_OM_PM2.5', 'ChemSpec_BC_PM2.5'
    ],
    'FTIR EC/OC': ['EC_ftir', 'OC_ftir', 'OM'],
    'FTIR Functional Groups': ['alcoholCOH', 'alkaneCH', 'carboxylicCOOH', 'naCO'],
    'HIPS': [
        'HIPS_T1', 'HIPS_Slope', 'HIPS_Intercept', 'HIPS_R1',
        'HIPS_t', 'HIPS_tau', 'HIPS_r', 'HIPS_Fabs',
        'HIPS_Uncertainty', 'HIPS_MDL'
    ],
    'ChemSpec Ions': [
        'ChemSpec_Sulfate_Ion_PM2.5', 'ChemSpec_Nitrate_Ion_PM2.5',
        'ChemSpec_Ammonium_Ion_PM2.5', 'ChemSpec_Chloride_Ion_PM2.5',
        'ChemSpec_Sodium_Ion_PM2.5', 'ChemSpec_Potassium_Ion_PM2.5',
        'ChemSpec_Magnesium_Ion_PM2.5', 'ChemSpec_Calcium_Ion_PM2.5'
    ],
    'ChemSpec Metals': [
        'ChemSpec_Iron_PM2.5', 'ChemSpec_Aluminum_PM2.5',
        'ChemSpec_Silicon_PM2.5', 'ChemSpec_Sulfur_PM2.5',
        'ChemSpec_Calcium_PM2.5', 'ChemSpec_Potassium_PM2.5',
        'ChemSpec_Zinc_PM2.5', 'ChemSpec_Lead_PM2.5',
        'ChemSpec_Copper_PM2.5', 'ChemSpec_Manganese_PM2.5'
    ]
}

# =============================================================================
# CROSS-COMPARISON DEFINITIONS
# =============================================================================
CROSS_COMPARISONS = [
    {
        'name': 'HIPS Fabs vs FTIR EC',
        'x_col': 'hips_fabs',
        'y_col': 'ftir_ec',
        'x_label': 'HIPS Fabs / MAC (ug/m3)',
        'y_label': 'FTIR EC (ug/m3)',
        'show_mac': True,
        'equal_axes': True
    },
    {
        'name': 'HIPS Fabs vs Iron',
        'x_col': 'hips_fabs',
        'y_col': 'iron',
        'x_label': 'HIPS Fabs / MAC (ug/m3)',
        'y_label': 'Iron (ug/m3)',
        'show_mac': True,
        'equal_axes': False
    },
    {
        'name': 'FTIR EC vs Iron',
        'x_col': 'ftir_ec',
        'y_col': 'iron',
        'x_label': 'FTIR EC (ug/m3)',
        'y_label': 'Iron (ug/m3)',
        'show_mac': False,
        'equal_axes': False
    },
    {
        'name': 'Aethalometer IR BCc vs Iron',
        'x_col': 'ir_bcc',
        'y_col': 'iron',
        'x_label': 'IR BCc (ug/m3)',
        'y_label': 'Iron (ug/m3)',
        'show_mac': False,
        'equal_axes': False
    },
    {
        'name': 'HIPS Fabs vs Aethalometer IR BCc',
        'x_col': 'hips_fabs',
        'y_col': 'ir_bcc',
        'x_label': 'HIPS Fabs / MAC (ug/m3)',
        'y_label': 'IR BCc (ug/m3)',
        'show_mac': True,
        'equal_axes': True
    }
]

# =============================================================================
# ETHIOPIA SEASONS (Addis Ababa)
# =============================================================================
# Canonical 3-season calendar. Replaces the per-notebook map_ethiopian_seasons /
# get_season_3 helpers and their divergent inline copies. Colors match
# SITES['Addis_Ababa'] usage and src season definitions.
#
# ON THE FEBRUARY BOUNDARY -- this is genuinely ambiguous, not a bug.
# Two conventions appear in the Ethiopian climate literature:
#
#   (a) Bega/Dry Oct-Feb, Belg Mar-May, Kiremt Jun-Sep   <- used here
#   (b) Bega/Dry Oct-Jan, Belg Feb-May,  Kiremt Jun-Sep
#
# They differ only in which season owns February, and both are defensible --
# Belg onset varies year to year and by altitude. (a) is canonical for this
# project so that seasonal means are comparable across notebooks.
#
# research/ftir_hips_chem/ETAD_Factor_Analysis.ipynb deliberately uses (b) and
# labels its own ranges in the season names ("Bega (Oct-Jan, dry)"). Its
# seasonal means are therefore NOT directly comparable with the rest of the
# project -- February is a high-BC month, so the difference is not negligible.
# If you compare that notebook's numbers against others, say which convention
# each used.
ETHIOPIA_SEASONS = {
    'Dry (Oct-Feb)':    {'months': [10, 11, 12, 1, 2], 'color': '#E67E22'},
    'Belg (Mar-May)':   {'months': [3, 4, 5],          'color': '#27AE60'},
    'Kiremt (Jun-Sep)': {'months': [6, 7, 8, 9],       'color': '#3498DB'},
}


def season_for_month(month):
    """Return the Ethiopian season name for a calendar month (1-12).

    Replaces inline map_ethiopian_seasons / get_season_3. Returns None if the
    month is out of range.
    """
    for name, info in ETHIOPIA_SEASONS.items():
        if month in info['months']:
            return name
    return None


# =============================================================================
# FILTER ID FORMAT
# =============================================================================

# Canonical pattern for stripping a replicate suffix from a FilterId:
#   'ETAD-0035-3'  -> 'ETAD-0035'
#   'ZAJB-0041-12' -> 'ZAJB-0041'   (two-digit replicates included)
#   'ETAD-0035'    -> 'ETAD-0035'   (already base: left alone)
#
# Anchoring on the full SITE-NNNN prefix matters in both directions. A bare
# r'-\d+$' also strips the 4-digit sample number from ids already in base form,
# collapsing every sample at a site onto one join key; a bare r'-(\d)$' misses
# two-digit replicates so those rows never join at all. Both bugs existed here.
#
# Lives in config.py because it is needed by data_matching and etad_factors, and
# data_matching already imports etad_factors -- so etad_factors cannot import it
# back without a cycle. config.py imports nothing local.
BASE_FILTER_ID_PATTERN = r'^([A-Za-z]+-\d{4})-\d+$'
BASE_FILTER_ID_REPL = r'\1'


# =============================================================================
# INSTRUMENT CHANNEL WAVELENGTHS
# =============================================================================

# MA350 / MA200 microAeth channel centres (nm). These are the wavelengths that
# go with the ``'<Name> BCc'`` / ``'<Name>.BCc'`` columns in every processed
# pickle in this project, and they match the firmware constants vendored in
# src/external/calibration.py.
#
# Do not substitute the AE33 set below. Several src/ modules keyed AE33 values
# onto MA350 column names, which inflates AAE(Red,IR) by ~16 % because
# ln(660/880) = -0.288 while ln(625/880) = -0.342 -- and any biomass fraction
# derived from that AAE inherits the error.
WAVELENGTHS_NM = {
    'UV':    375,
    'Blue':  470,
    'Green': 528,
    'Red':   625,
    'IR':    880,
}

# Magee AE33 seven-channel set, for reference when reading AE33 exports whose
# columns are named BC1..BC7. Not interchangeable with WAVELENGTHS_NM.
AE33_WAVELENGTHS_NM = {
    'BC1': 370, 'BC2': 470, 'BC3': 520, 'BC4': 590,
    'BC5': 660, 'BC6': 880, 'BC7': 950,
}


# =============================================================================
# AAE SOURCE-REGION BOUNDARIES
# =============================================================================

# Absorption Angstrom Exponent cut points used to shade fossil / mixed / biomass
# regions on AAE plots. 1.5 is the value the existing addis_01 figures were
# produced with; plotting_gaps_scenarios.ipynb once proposed 1.4 for a helper
# that was never written, and notebooks/archive used 1.0/1.2.
#
# NOT the same thing as the endmember AAEs in
# src/analysis/bc/source_apportionment.py (aae_fossil=1.0, aae_biomass=2.0),
# which parameterize a linear mixing model rather than classify samples.
AAE_REGIONS = {
    'fossil_max':  0.9,
    'biomass_min': 1.5,
}
