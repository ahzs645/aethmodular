"""Plotting configuration"""

import matplotlib.pyplot as plt

FIGURE_SETTINGS = {
    'dpi': 150,
    'savefig_dpi': 300,
    'figsize_default': (12, 8),
}

COLOR_SCHEMES = {
    'seasons': {
        'Dry Season': '#FF6B6B',
        'Belg Rainy Season': '#4ECDC4',
        'Kiremt Rainy Season': '#45B7D1'
    }
}

def setup_plotting_style():
    """Apply default plotting style"""
    plt.rcParams.update({
        'figure.figsize': FIGURE_SETTINGS['figsize_default'],
        'figure.dpi': FIGURE_SETTINGS['dpi'],
        'savefig.dpi': FIGURE_SETTINGS['savefig_dpi'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
    })
