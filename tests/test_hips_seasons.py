"""Seasons and the HIPS bend (research/ftir_hips_chem/workflows/run_hips_seasons_20261111.py) and its committed summary."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research" / "ftir_hips_chem" / "workflows"), str(ROOT / "research" / "ftir_hips_chem" / "scripts")]

import hips_loading as hl  # noqa: E402
import run_hips_seasons_20261111 as seasons  # noqa: E402

SUMMARY = json.loads((ROOT / "docs" / "hips-seasons-2026-11-11.json").read_text())
YEAR = pd.DataFrame({"rain": [0, 0, 0, 1, 2, 9, 9, 9, 9, 1, 0, 0], "temperature": [10, 11, 14, 18, 22, 24, 25, 25, 22, 18, 13, 10]}, index=range(1, 13), dtype=float)


def test_every_calendar_covers_each_month_once_and_every_hips_site_has_one():
    calendars = seasons.load_calendars()
    sites = {code: entry for code, entry in calendars.items() if not code.startswith("_")}
    assert len(sites) == SUMMARY["calendar_structure"][0]["sites_with_a_calendar"]
    for code, entry in sites.items():
        for key, calendar in entry["calendars"].items():
            assert sorted(calendar) == list(range(1, 13)), f"{code} {key}"
    filters = seasons.network.load(seasons.network.ceiling())
    assert set(filters["code"]) <= set(sites)
    assert set(seasons.season_of(filters, calendars).loc[filters["code"] == "ETAD"]) == {"Bega (Oct–Jan)", "Belg (Feb–May)", "Kiremt (Jun–Sep)"}
    assert all(row["same_months"] for row in SUMMARY["calendar_consistency"])            # the Python calendars and the gallery's agree


def test_a_calendar_that_follows_the_rain_scores_higher_and_the_best_split_finds_it():
    wet = {m: "Wet" if 6 <= m <= 9 else "Dry" for m in range(1, 13)}
    off = {m: "Wet" if 3 <= m <= 6 else "Dry" for m in range(1, 13)}
    good, poor = (seasons._separation(YEAR[["rain"]], [c[m] for m in range(1, 13)])["rain"] for c in (wet, off))
    assert good > 0.9 > poor
    best, groups = seasons._best_partition(YEAR[["rain"]], 2)
    assert best >= good - 1e-9 and len(set(groups[5:9])) == 1 and groups[5] != groups[0]
    assert seasons._moved(wet, groups) == []
    assert "Jun to Dry" in seasons._moved(off, np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])) or seasons._moved(off, groups)


def test_wet_dry_and_cold_warm_names_are_checked_against_the_values():
    calendar = {m: "Rainy (Jun–Sep)" if 6 <= m <= 9 else "Dry (Oct–May)" for m in range(1, 13)}
    assert seasons._labels_hold(calendar, YEAR["rain"], seasons.DRY, seasons.WET) is True
    wrong = {m: "Dry (Jun–Sep)" if 6 <= m <= 9 else "Rainy (Oct–May)" for m in range(1, 13)}
    assert seasons._labels_hold(wrong, YEAR["rain"], seasons.DRY, seasons.WET) is False
    quarters = {m: name for name, months in (("Winter", (12, 1, 2)), ("Spring", (3, 4, 5)), ("Summer", (6, 7, 8)), ("Autumn", (9, 10, 11))) for m in months}
    assert seasons._labels_hold(quarters, YEAR["rain"], seasons.DRY, seasons.WET) is None      # no wet or dry names to check
    assert seasons._labels_hold(quarters, YEAR["temperature"], seasons.COLD, seasons.WARM) is True
    assert not seasons._kind("Pre-monsoon (Mar–May)", seasons.WET) and seasons._kind("Monsoon (Jun–Sep)", seasons.WET)


def test_finer_groups_are_closer_to_each_filters_own_factor():
    levels = pd.DataFrame(SUMMARY["levels"])
    for _, part in levels.groupby("scope"):
        m = part.set_index("level")["median"]
        assert m["One factor for the network"] > m["One per site"] > m["One per site and season"] > m["One per site and month"] > 0
    shares = pd.DataFrame(SUMMARY["variance"]).groupby("scope")["share"].sum()
    assert np.allclose(shares, 1.0)


def test_check_by_runs_the_check_in_each_season_and_a_season_offset_fails_only_pooled():
    rng = np.random.default_rng(4)
    n = 800
    season = np.where(np.arange(n) % 2 == 0, "A", "B")
    # season B is the heavier season and has twice the reference per unit of absorption: an offset between seasons, not a bend
    air = 12.0 * rng.lognormal(0.0, 0.5, n) * np.where(season == "B", 2.0, 1.0)
    loading = air * 7.2 / 353.0
    frame = pd.DataFrame({"fabs": loading / (1 + loading / 2.26) * 353.0 / 7.2, "volume": 7.2,
                          "ftir_ec": air / 10.0 * np.where(season == "B", 2.0, 1.0) * rng.lognormal(0.0, 0.1, n),
                          "pm25": air * rng.lognormal(0.0, 0.2, n), "season": season})
    screened = hl.screen(frame, ceiling=hl.Ceiling(2.26, 1.82, 2.70, 1.18))
    each = hl.check_by(screened, n_boot=300)
    assert set(each["season"]) == {"A", "B"} and (each["verdict"] == "corrected").all()
    assert hl.check(screened, n_boot=300)["verdict"] == "under-corrected"               # pooled, the season offset reads as a failed correction


def test_summary_and_normals_are_complete_and_use_no_aethalometer():
    assert SUMMARY["aethalometer_used"] is False
    normals = json.loads(seasons.NORMALS.read_text())
    assert normals["period"] == ["1991-01-01", "2020-12-31"] and len(normals["sites"]) == SUMMARY["calendar_structure"][0]["sites_with_normals"]
    assert all(len(site[key]) == 12 and all(v is not None for v in site[key]) for site in normals["sites"].values() for key in ("temperature", "precipitation"))
    addis = next(row for row in SUMMARY["calendar_alternatives"] if row["code"] == "ETAD" and row["in_use"])
    assert addis["headroom"] < 0.01                                                      # the calendar in use is the best three-season split
    within = pd.DataFrame(SUMMARY["within"]).set_index("code")
    assert within.loc["ETAD", "reading"].startswith("holds within seasons") and within.loc["ETAD", "pooled_verdict"] == "under-corrected"
