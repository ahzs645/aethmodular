"""
Multi-site seasonal definitions for aethalometer analysis.

Provides seasonal classifications for all SPARTAN/ETAD sites:
- Addis Ababa (Ethiopia): Dry/Belg/Kiremt
- Beijing (China): Standard 4-season
- Delhi (India): Winter/Pre-monsoon/Monsoon/Post-monsoon (provisional)
- JPL/Pasadena (USA): Standard 4-season
"""

SITE_SEASONS = {
    'Addis_Ababa': {
        'Dry (Oct-Feb)':    {'months': [10, 11, 12, 1, 2], 'color': '#E67E22'},
        'Belg (Mar-May)':   {'months': [3, 4, 5],          'color': '#27AE60'},
        'Kiremt (Jun-Sep)': {'months': [6, 7, 8, 9],       'color': '#3498DB'},
    },
    'Beijing': {
        'Spring (Mar-May)':  {'months': [3, 4, 5],    'color': '#27AE60'},
        'Summer (Jun-Aug)':  {'months': [6, 7, 8],    'color': '#E74C3C'},
        'Autumn (Sep-Nov)':  {'months': [9, 10, 11],  'color': '#F39C12'},
        'Winter (Dec-Feb)':  {'months': [12, 1, 2],   'color': '#3498DB'},
    },
    'Delhi': {
        # Provisional — confirm with Amina (Navid's student)
        'Winter (Nov-Feb)':      {'months': [11, 12, 1, 2],  'color': '#3498DB'},
        'Pre-monsoon (Mar-Jun)': {'months': [3, 4, 5, 6],    'color': '#E74C3C'},
        'Monsoon (Jul-Sep)':     {'months': [7, 8, 9],       'color': '#27AE60'},
        'Post-monsoon (Oct)':    {'months': [10],             'color': '#F39C12'},
    },
    'JPL': {
        'Winter (Dec-Feb)':  {'months': [12, 1, 2],   'color': '#3498DB'},
        'Spring (Mar-May)':  {'months': [3, 4, 5],    'color': '#27AE60'},
        'Summer (Jun-Aug)':  {'months': [6, 7, 8],    'color': '#E74C3C'},
        'Autumn (Sep-Nov)':  {'months': [9, 10, 11],  'color': '#F39C12'},
    },
}

# Site display colors (from scripts/config.py)
SITE_COLORS = {
    'Beijing':     '#E74C3C',
    'Delhi':       '#3498DB',
    'JPL':         '#2ECC71',
    'Addis_Ababa': '#F39C12',
}


def assign_season(df, site_name):
    """Add a 'Season' column to df based on the month of its DatetimeIndex."""
    seasons = SITE_SEASONS[site_name]
    month_to_season = {}
    for season_name, info in seasons.items():
        for m in info['months']:
            month_to_season[m] = season_name
    df['Season'] = df.index.month.map(month_to_season)
    return df


def get_season_list(site_name):
    """Return ordered list of season names for a site."""
    return list(SITE_SEASONS[site_name].keys())


def get_season_colors(site_name):
    """Return {season_name: color} mapping for a site."""
    return {name: info['color'] for name, info in SITE_SEASONS[site_name].items()}
