"""ETAD PMF source apportionment as an evaluation-grouping axis.

Navid's ETAD PMF run (``research/ftir_hips_chem/Filter Data/ETAD Factor
Contributions .csv``) gives each sampled day a mix of five sources. The
calibration explorer already reads its target crossplot out one group at a time
(season); turning those daily mixes into one label per evaluation filter adds
the second axis the group asked for on 2026-08-27 - does a calibration sit
differently on marine days than on combustion days?

Two schemes come out of here:

``pmf_source``  the dominant factor, named the way Navid names it (five
                classes). Per-class n on the 239-filter Addis evaluation set is
                11-25, which is thin for a regression - hence the second scheme.
``pmf_class``   marine (Sea Salt Mixed + Polluted Marine, n=44) against
                combustion (Wood Burning + Charcoal + Fossil Fuel Combustion,
                n=58). This is the contrast that was actually asked about.

TRAP, and the reason this is a module rather than a few inline lines: the raw
``GF1``-``GF5`` columns are PM2.5 *mass* fractions summing to only 0.03-0.46 per
row, not relative source contributions. ``normalize_gf_fractions`` MUST run
before ``add_dominant_source`` or every dominant-source label is wrong -
unnormalized, the dominant fraction tops out near 0.24 and the winner is
whichever source happens to carry the most PM2.5 mass. See "Data join quirks" in
AGENTS.md.

Second trap: the PMF table covers 102 days of calendar 2023 while the Addis
evaluation set is 239 filters, so most filters have no PMF day at all. Those are
labelled ``unmatched`` rather than dropped - the label list has to stay aligned
1:1 with the evaluation filters, and quietly folding 137 unknown-source filters
in with everything else is exactly the wrong default for an axis whose whole
point is a clean contrast. ``unmatched`` is a selectable group like any other.

Standalone (prints the PMF-day distribution, no evaluation set needed):

    python research/ftir_ec_phase3/scripts/pmf_source_groups.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_SCRIPTS = Path(__file__).resolve().parents[2] / "ftir_hips_chem" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

import pandas as pd

from etad_factors import (  # noqa: E402  (needs the sys.path insert above)
    ETAD_PMF_SOURCE_NAMES, add_dominant_source, load_etad_factor_contributions,
    normalize_gf_fractions,
)

# Filters whose sampling date has no PMF day. Selectable, never silently pooled.
UNMATCHED = "unmatched"

# add_dominant_source names the winner after the *_frac column it came from
# (``sea_salt_frac`` -> ``sea_salt``), so map those short names onto the factor
# names Navid publishes instead of inventing a second vocabulary for the same
# five sources.
SOURCE_LABEL = {
    "sea_salt": ETAD_PMF_SOURCE_NAMES["1"],           # Sea Salt Mixed
    "wood": ETAD_PMF_SOURCE_NAMES["2"],               # Wood Burning
    "charcoal": ETAD_PMF_SOURCE_NAMES["3"],           # Charcoal
    "polluted_marine": ETAD_PMF_SOURCE_NAMES["4"],    # Polluted Marine
    "fossil_fuel": ETAD_PMF_SOURCE_NAMES["5"],        # Fossil Fuel Combustion
}

MARINE = "Marine"
COMBUSTION = "Combustion"
SOURCE_CLASS = {
    ETAD_PMF_SOURCE_NAMES["1"]: MARINE,
    ETAD_PMF_SOURCE_NAMES["4"]: MARINE,
    ETAD_PMF_SOURCE_NAMES["2"]: COMBUSTION,
    ETAD_PMF_SOURCE_NAMES["3"]: COMBUSTION,
    ETAD_PMF_SOURCE_NAMES["5"]: COMBUSTION,
}

SCHEME_LABELS = {
    "pmf_source": "PMF dominant source",
    "pmf_class": "PMF marine vs combustion",
}


def dominant_source_by_date(csv_path=None) -> dict[str, str]:
    """``{'2023-01-03': 'Charcoal', ...}`` - the dominant source of each PMF day.

    Normalization runs first (see the module docstring); days whose fractions are
    all zero have no dominant source and are simply left out of the mapping, so
    they read as unmatched downstream rather than as a made-up winner.
    """
    factors = add_dominant_source(
        normalize_gf_fractions(load_etad_factor_contributions(csv_path)))
    out: dict[str, str] = {}
    for day, source in zip(factors["date"], factors["dominant_source"]):
        if pd.isna(day) or source is None or pd.isna(source):
            continue
        out[pd.Timestamp(day).date().isoformat()] = SOURCE_LABEL.get(
            str(source), str(source))
    return out


def pmf_group_schemes(dates, csv_path=None) -> dict[str, list[str]]:
    """``{"pmf_source": [...], "pmf_class": [...]}``, aligned 1:1 with `dates`.

    `dates` are the evaluation target's per-filter ISO dates (``None`` where the
    date is unknown). The join is an EXACT calendar-date match with no tolerance:
    measured 2026-08-27, all 102 PMF days land on an Addis evaluation date
    exactly, so a tolerance window could only invent matches that are not there.
    """
    by_date = dominant_source_by_date(csv_path)
    source: list[str] = []
    klass: list[str] = []
    for value in dates:
        label = by_date.get(str(value)[:10]) if value else None
        if label is None:
            source.append(UNMATCHED)
            klass.append(UNMATCHED)
            continue
        if label not in SOURCE_CLASS:
            # A factor name that is not one of the five: fail loudly rather than
            # dropping it into `unmatched`, where it would look like a filter
            # with no PMF day at all.
            raise ValueError(
                f"PMF source {label!r} has no marine/combustion class; "
                f"known sources: {', '.join(sorted(SOURCE_CLASS))}")
        source.append(label)
        klass.append(SOURCE_CLASS[label])
    return {"pmf_source": source, "pmf_class": klass}


def _main() -> None:
    factors = add_dominant_source(
        normalize_gf_fractions(load_etad_factor_contributions()))
    labelled = factors["dominant_source"].map(lambda s: SOURCE_LABEL.get(str(s), s))
    print("\ndominant source per PMF day")
    for name, count in labelled.value_counts().items():
        print(f"  {name:<24s} {count:>4d}   ({SOURCE_CLASS.get(name, '?')})")
    print("\nmarine vs combustion")
    for name, count in labelled.map(SOURCE_CLASS).value_counts().items():
        print(f"  {name:<24s} {count:>4d}")
    print(f"\nmedian dominant fraction: {factors['dominant_fraction'].median():.3f}"
          "   (about 0.24 if you forgot to normalize)")
    print("coverage against an evaluation set is reported by the explorer's "
          "/api/eval_view_options")


if __name__ == "__main__":
    _main()
