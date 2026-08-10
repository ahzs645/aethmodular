"""Global configuration settings"""

from pathlib import Path
from typing import Dict, Any

# File paths
DEFAULT_DB_PATH = Path("data/spartan_ftir_hips.db")
DEFAULT_OUTPUT_DIR = Path("outputs")

# Analysis parameters
QUALITY_THRESHOLDS = {
    'ec_ftir': {'min_value': 0.1, 'max_value': 50.0},
    'oc_ftir': {'min_value': 0.1, 'max_value': 100.0},
    'fabs': {'min_value': 1.0, 'max_value': 1000.0},
    'mac': {'min_value': 1.0, 'max_value': 50.0},
}

# Ethiopian seasons
ETHIOPIAN_SEASONS = {
    'Dry Season': [10, 11, 12, 1, 2],
    'Belg Rainy Season': [3, 4, 5],
    'Kiremt Rainy Season': [6, 7, 8, 9]
}

# Statistical settings
MIN_SAMPLES_FOR_ANALYSIS = 5
DEFAULT_CONFIDENCE_LEVEL = 0.95
