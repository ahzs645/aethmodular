"""Season assignment, anchored to the one authoritative site-season table.

The Ethiopian season calendar was previously redefined inline in many
notebooks (`map_ethiopian_seasons`, `get_season_3`), and at least one copy had
scrambled month assignments. All of them are superseded by
`src.config.multi_site_seasons.SITE_SEASONS`, which holds the correct calendar
(Dry Oct-Feb, Belg Mar-May, Kiremt Jun-Sep) and is already imported by active
code. This module exposes thin, scalar-friendly helpers over that table.
"""

from __future__ import annotations

from ..config.multi_site_seasons import SITE_SEASONS, assign_season

__all__ = ["SITE_SEASONS", "assign_season", "season_for_month"]


def season_for_month(month: int, site: str = "Addis_Ababa") -> str | None:
    """Return the season name for a calendar month (1-12) at a given site.

    Replaces the per-notebook `map_ethiopian_seasons` / `get_season_3` helpers.
    Returns None if the month is not covered by the site's definition.
    """
    seasons = SITE_SEASONS.get(site)
    if seasons is None:
        raise KeyError(
            f"Unknown site {site!r}; known sites: {sorted(SITE_SEASONS)}"
        )
    for name, info in seasons.items():
        if month in info["months"]:
            return name
    return None
