"""
Multi-site seasonal definitions for aethalometer analysis.

Two named, versioned calendars for the four focus sites:

- ``research_2026_04`` (the default, ``SITE_SEASONS``): the calendar the April 2026
  board assets, ``multisite_diurnal_wavelength_analysis`` and ``warren_meeting_slides``
  were made with. Addis puts February in Dry (Oct-Feb), Delhi uses a provisional
  Nov-Feb / Mar-Jun / Jul-Sep / Oct split, and JPL uses meteorological quarters. Kept
  unchanged so those outputs reproduce.
- ``local_2026_10``: the gallery's per-site calendar (docs/site-seasonality.md) for new
  work. Addis Bega Oct-Jan / Belg Feb-May / Kiremt Jun-Sep (Ethiopian Meteorological
  Institute), Delhi the IMD national seasons, JPL the California wet (Oct-Apr) and dry
  (May-Sep) windows, Beijing quarters.

The two differ at Addis in February, at Delhi in June, November and December, and at
JPL everywhere, so they do not give interchangeable seasonal means. Pick one by name and
label the output with it; ``assign_season`` records the id in ``df.attrs``.
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
        # Provisional when written (to confirm with Amina, Navid's student); not the IMD
        # national seasons. local_2026_10 has those.
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

# The gallery's local calendar (gallery/app/src/siteSeasons.ts LOCAL_SEASONS and the
# belg_feb Addis bins); sources in docs/site-seasonality.md.
LOCAL_SEASONS_2026_10 = {
    'Addis_Ababa': {
        'Bega (Oct-Jan)':   {'months': [10, 11, 12, 1], 'color': '#E67E22'},
        'Belg (Feb-May)':   {'months': [2, 3, 4, 5],    'color': '#27AE60'},
        'Kiremt (Jun-Sep)': {'months': [6, 7, 8, 9],    'color': '#3498DB'},
    },
    'Beijing': SITE_SEASONS['Beijing'],
    'Delhi': {
        'Winter (Jan-Feb)':       {'months': [1, 2],        'color': '#3498DB'},
        'Pre-monsoon (Mar-May)':  {'months': [3, 4, 5],     'color': '#E74C3C'},
        'Monsoon (Jun-Sep)':      {'months': [6, 7, 8, 9],  'color': '#27AE60'},
        'Post-monsoon (Oct-Dec)': {'months': [10, 11, 12],  'color': '#F39C12'},
    },
    'JPL': {
        'Wet window (Oct-Apr)': {'months': [10, 11, 12, 1, 2, 3, 4], 'color': '#3498DB'},
        'Dry window (May-Sep)': {'months': [5, 6, 7, 8, 9],          'color': '#E74C3C'},
    },
}

CALENDARS = {
    'research_2026_04': SITE_SEASONS,
    'local_2026_10': LOCAL_SEASONS_2026_10,
}
DEFAULT_CALENDAR = 'research_2026_04'

# Site display colors (from scripts/config.py)
SITE_COLORS = {
    'Beijing':     '#E74C3C',
    'Delhi':       '#3498DB',
    'JPL':         '#2ECC71',
    'Addis_Ababa': '#F39C12',
}


def resolve_calendar(calendar=None):
    """Return a calendar's {site: seasons} mapping from its id (None = the default)."""
    key = DEFAULT_CALENDAR if calendar is None else calendar
    if isinstance(key, dict):
        return key
    try:
        return CALENDARS[key]
    except KeyError:
        raise KeyError(f"unknown season calendar {key!r}; expected one of {sorted(CALENDARS)}") from None


def calendar_id(calendar=None):
    """The registered id of a calendar, for labelling output ('custom' if unregistered)."""
    mapping = resolve_calendar(calendar)
    return next((k for k, v in CALENDARS.items() if v is mapping), 'custom')


def assign_season(df, site_name, calendar=None):
    """Add a 'Season' column to df from the month of its DatetimeIndex.

    The calendar used is recorded as ``df.attrs['season_calendar']``.
    """
    seasons = resolve_calendar(calendar)[site_name]
    month_to_season = {}
    for season_name, info in seasons.items():
        for m in info['months']:
            month_to_season[m] = season_name
    df['Season'] = df.index.month.map(month_to_season)
    df.attrs['season_calendar'] = calendar_id(calendar)
    return df


def get_season_list(site_name, calendar=None):
    """Return ordered list of season names for a site."""
    return list(resolve_calendar(calendar)[site_name].keys())


def get_season_colors(site_name, calendar=None):
    """Return {season_name: color} mapping for a site."""
    return {name: info['color'] for name, info in resolve_calendar(calendar)[site_name].items()}
