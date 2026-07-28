"""Tests for the two Ethiopian season calendars and the convention registry.

The February boundary is a genuine choice between two published calendars, not a
bug. These tests pin the resolution: both are registered, the default is
unchanged, and a caller's choice is always recoverable for labelling.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "ftir_hips_chem"
    / "scripts"
)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (  # noqa: E402
    DEFAULT_SEASON_CONVENTION,
    ETHIOPIA_SEASONS,
    ETHIOPIA_SEASONS_BELG_FEB,
    SEASON_CONVENTIONS,
    resolve_seasons,
    season_convention_name,
    season_for_month,
)


@pytest.mark.parametrize("mapping", [ETHIOPIA_SEASONS, ETHIOPIA_SEASONS_BELG_FEB])
def test_every_month_belongs_to_exactly_one_season(mapping):
    months = [m for info in mapping.values() for m in info["months"]]
    assert sorted(months) == list(range(1, 13))


def test_february_is_the_only_difference_between_the_conventions():
    differing = {
        month
        for month in range(1, 13)
        if season_for_month(month, "dry_feb")[:4]
        != season_for_month(month, "belg_feb")[:4]
    }
    # Names differ in wording, so compare which *group* each month lands in.
    dry = {m: season_for_month(m, "dry_feb") for m in range(1, 13)}
    belg = {m: season_for_month(m, "belg_feb") for m in range(1, 13)}
    regrouped = {
        m for m in range(1, 13) if (dry[m].startswith("Kiremt")) != (belg[m].startswith("Kiremt"))
    }
    assert regrouped == set()
    assert 2 in differing


def test_february_lands_in_dry_by_default_and_belg_under_the_alternative():
    assert season_for_month(2) == "Dry (Oct-Feb)"
    assert season_for_month(2, "belg_feb") == "Belg (Feb-May, short rains)"


def test_default_convention_is_unchanged():
    assert DEFAULT_SEASON_CONVENTION == "dry_feb"
    assert resolve_seasons() is ETHIOPIA_SEASONS
    assert resolve_seasons(None) is ETHIOPIA_SEASONS


def test_belg_feb_reproduces_the_notebooks_former_inline_mapping():
    """ETAD_Factor_Analysis.ipynb declared this calendar inline; migrating it to
    the named convention must not move a single month."""

    def former_inline(month):
        if month in [10, 11, 12, 1]:
            return "Bega (Oct-Jan, dry)"
        if month in [2, 3, 4, 5]:
            return "Belg (Feb-May, short rains)"
        if month in [6, 7, 8, 9]:
            return "Kiremt (Jun-Sep, long rains)"
        return None

    for month in range(1, 13):
        assert season_for_month(month, "belg_feb") == former_inline(month)


def test_out_of_range_month_returns_none():
    assert season_for_month(0) is None
    assert season_for_month(13, "belg_feb") is None


def test_resolve_seasons_accepts_a_name_or_a_mapping():
    assert resolve_seasons("belg_feb") is ETHIOPIA_SEASONS_BELG_FEB
    custom = {"All year": {"months": list(range(1, 13)), "color": "#000000"}}
    assert resolve_seasons(custom) is custom


def test_unknown_convention_name_raises_rather_than_falling_back():
    with pytest.raises(KeyError, match="unknown season convention"):
        resolve_seasons("wet_feb")


def test_convention_name_is_recoverable_for_labelling_output():
    assert season_convention_name() == "dry_feb"
    assert season_convention_name("belg_feb") == "belg_feb"
    assert season_convention_name(ETHIOPIA_SEASONS_BELG_FEB) == "belg_feb"


def test_custom_calendar_is_reported_as_custom_not_canonical():
    custom = {"Half": {"months": list(range(1, 7))}, "Rest": {"months": list(range(7, 13))}}
    assert season_convention_name(custom) == "custom"


def test_both_conventions_share_one_color_per_season_role():
    """Color encodes which season, not which convention, so the two calendars
    must not introduce a second palette."""
    assert sorted(i["color"] for i in ETHIOPIA_SEASONS.values()) == sorted(
        i["color"] for i in ETHIOPIA_SEASONS_BELG_FEB.values()
    )


def test_registry_covers_both_calendars():
    assert set(SEASON_CONVENTIONS) == {"dry_feb", "belg_feb"}
