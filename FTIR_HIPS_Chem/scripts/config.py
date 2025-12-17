"""
Central configuration for multi-site aethalometer analysis.
Contains site definitions, paths, and analysis parameters.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROCESSED_SITES_DIR = Path('/Users/ahmadjalil/Github/aethmodular/FTIR_HIPS_Chem/processed_sites')
FILTER_DATA_PATH = Path('/Users/ahmadjalil/Github/aethmodular/FTIR_HIPS_Chem/Filter Data/unified_filter_dataset.pkl')

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
FLOW_FIX_PERIODS = {
    'Beijing': {
        'description': 'Gap in data where instrument was likely serviced',
        'before_end': '2022-06-01',
        'after_start': '2022-09-01',
        'notes': 'May 2022 data has very low flow ratios pulling down the mean'
    },
    'JPL': {
        'description': 'Similar gap where flow ratio improved',
        'before_end': '2022-06-01',
        'after_start': '2022-09-01',
        'notes': 'Early days have flow ratio near 1.2, later improved to ~2.5'
    },
    'Delhi': {
        'description': 'Started after flow issues were resolved',
        'before_end': None,
        'after_start': None,
        'notes': 'Best flow ratios (~2.4 mean), no problematic early period'
    },
    'Addis_Ababa': {
        'description': 'Consistently low flow ratio throughout',
        'before_end': None,
        'after_start': None,
        'notes': 'Flow ratio ~1.2 throughout - never fixed'
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
