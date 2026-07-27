"""Tests for the exclusion API in research/ftir_hips_chem/scripts/outliers.py.

AGENTS.md makes this API mandatory for every analysis notebook ("Never drop rows
by hand ... The exclusion system is *flagging-based* (non-destructive)"), yet it
had no test coverage at all. These lock in the two properties the whole audit
trail depends on:

* flagging never removes rows, and
* every registry entry carries a human-readable reason.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..",
                 "research", "ftir_hips_chem", "scripts")
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from outliers import (                                       # noqa: E402
    EXCLUDED_SAMPLES, MANUAL_OUTLIERS,
    apply_exclusion_flags, apply_threshold_flags,
    get_clean_data, get_excluded_data, get_outlier_data,
)

SITES = ["Beijing", "Delhi", "JPL", "Addis_Ababa"]


def _frame(dates, **cols):
    return pd.DataFrame({"date": pd.to_datetime(dates), **cols})


# ---------------------------------------------------------------------------
# Registry integrity — these guard the audit trail, not the code
# ---------------------------------------------------------------------------

class TestRegistryIntegrity:
    @pytest.mark.parametrize("site", SITES)
    def test_every_exclusion_has_a_nonempty_reason(self, site):
        """AGENTS.md: 'Every exclusion needs a reason. Reviewers read these.'"""
        for entry in EXCLUDED_SAMPLES[site]:
            reason = entry.get("reason", "")
            assert reason and reason.strip(), f"{site} {entry.get('date')} has no reason"

    @pytest.mark.parametrize("site", SITES)
    def test_every_exclusion_date_parses(self, site):
        for entry in EXCLUDED_SAMPLES[site]:
            assert not pd.isna(pd.to_datetime(entry["date"])), entry

    @pytest.mark.parametrize("site", SITES)
    def test_threshold_criteria_declare_a_known_type(self, site):
        known = {
            "high_aeth", "low_aeth", "high_aeth_low_ec", "low_aeth_high_ec",
            "high_both", "high_ec", "negative_ec", "high_either",
        }
        for crit in MANUAL_OUTLIERS[site].get("remove_criteria", []):
            assert crit.get("type") in known, f"{site}: unknown criteria type {crit}"


# ---------------------------------------------------------------------------
# apply_exclusion_flags
# ---------------------------------------------------------------------------

class TestApplyExclusionFlags:
    def test_adds_both_columns_and_keeps_every_row(self):
        df = _frame(["2024-01-01", "2024-01-02"], aeth_bc=[1.0, 2.0])
        out = apply_exclusion_flags(df, "Beijing")

        assert len(out) == len(df), "flagging must be non-destructive"
        assert {"is_excluded", "exclusion_reason"} <= set(out.columns)
        assert out["is_excluded"].dtype == bool

    def test_does_not_mutate_the_caller_frame(self):
        df = _frame(["2024-01-01"], aeth_bc=[1.0])
        apply_exclusion_flags(df, "Beijing")
        assert "is_excluded" not in df.columns

    def test_unknown_site_flags_nothing(self):
        df = _frame(["2024-01-01", "2024-01-02"], aeth_bc=[1.0, 2.0])
        out = apply_exclusion_flags(df, "NotASite")
        assert not out["is_excluded"].any()
        assert (out["exclusion_reason"] == "").all()

    def test_registered_date_is_flagged_with_its_reason(self):
        site = next(s for s in SITES if EXCLUDED_SAMPLES[s])
        entry = EXCLUDED_SAMPLES[site][0]
        df = _frame([entry["date"], "1990-01-01"], aeth_bc=[1.0, 2.0])

        out = apply_exclusion_flags(df, site)

        assert bool(out.loc[0, "is_excluded"]) is True
        assert entry["reason"] in out.loc[0, "exclusion_reason"]
        assert bool(out.loc[1, "is_excluded"]) is False

    def test_date_tolerance_window_is_inclusive_and_bounded(self):
        site = next(s for s in SITES if EXCLUDED_SAMPLES[s])
        d = pd.to_datetime(EXCLUDED_SAMPLES[site][0]["date"])
        df = _frame(
            [d - pd.Timedelta(days=1), d, d + pd.Timedelta(days=1),
             d + pd.Timedelta(days=5)],
            aeth_bc=[1.0, 2.0, 3.0, 4.0],
        )

        out = apply_exclusion_flags(df, site)          # default tolerance = 1 day
        assert list(out["is_excluded"]) == [True, True, True, False]

        strict = apply_exclusion_flags(df, site, date_tolerance_days=0)
        assert list(strict["is_excluded"]) == [False, True, False, False]

    def test_filter_id_narrows_the_match_when_present(self):
        site = next((s for s in SITES
                     if any(e.get("filter_id") for e in EXCLUDED_SAMPLES[s])), None)
        if site is None:
            pytest.skip("no registry entry carries a filter_id")
        entry = next(e for e in EXCLUDED_SAMPLES[site] if e.get("filter_id"))

        df = _frame([entry["date"], entry["date"]],
                    filter_id=[entry["filter_id"], "SOME-OTHER-0001"],
                    aeth_bc=[1.0, 2.0])
        out = apply_exclusion_flags(df, site)

        assert bool(out.loc[0, "is_excluded"]) is True
        assert bool(out.loc[1, "is_excluded"]) is False, "filter_id must narrow the date match"


# ---------------------------------------------------------------------------
# apply_threshold_flags
# ---------------------------------------------------------------------------

class TestApplyThresholdFlags:
    def test_adds_columns_and_keeps_every_row(self):
        df = _frame(["2024-01-01"] * 3, aeth_bc=[1.0, 2.0, 3.0], filter_ec=[1.0, 2.0, 3.0])
        out = apply_threshold_flags(df, "Beijing")

        assert len(out) == 3
        assert {"is_outlier", "outlier_reason"} <= set(out.columns)

    def test_unknown_site_flags_nothing(self):
        df = _frame(["2024-01-01"] * 3, aeth_bc=[1.0, 1e9, 3.0], filter_ec=[1.0, 2.0, -5.0])
        out = apply_threshold_flags(df, "NotASite")
        assert not out["is_outlier"].any()

    def test_high_aeth_rule_flags_above_threshold_only(self):
        # exercise the rule directly rather than depending on registry contents
        import outliers as O
        saved = O.MANUAL_OUTLIERS.get("Beijing")
        O.MANUAL_OUTLIERS["Beijing"] = {
            "remove_criteria": [{"type": "high_aeth", "aeth_bc_min": 10.0}]
        }
        try:
            df = _frame(["2024-01-01"] * 3, aeth_bc=[9.9, 10.0, 10.1],
                        filter_ec=[1.0, 1.0, 1.0])
            out = O.apply_threshold_flags(df, "Beijing")
            assert list(out["is_outlier"]) == [False, False, True]
            assert "high_aeth" in out.loc[2, "outlier_reason"]
        finally:
            O.MANUAL_OUTLIERS["Beijing"] = saved

    def test_negative_ec_rule(self):
        import outliers as O
        saved = O.MANUAL_OUTLIERS.get("Delhi")
        O.MANUAL_OUTLIERS["Delhi"] = {"remove_criteria": [{"type": "negative_ec"}]}
        try:
            df = _frame(["2024-01-01"] * 3, aeth_bc=[1.0, 1.0, 1.0],
                        filter_ec=[0.5, -0.1, 0.0])
            out = O.apply_threshold_flags(df, "Delhi")
            assert list(out["is_outlier"]) == [False, True, False]
        finally:
            O.MANUAL_OUTLIERS["Delhi"] = saved


# ---------------------------------------------------------------------------
# get_clean_data / get_excluded_data / get_outlier_data
# ---------------------------------------------------------------------------

class TestSubsetHelpers:
    @staticmethod
    def _flagged():
        df = _frame(["2024-01-01"] * 4, aeth_bc=[1.0, 2.0, 3.0, 4.0],
                    filter_ec=[1.0, 2.0, 3.0, 4.0])
        df["is_excluded"] = [True, False, False, False]
        df["is_outlier"] = [False, True, False, False]
        return df

    def test_clean_drops_both_excluded_and_outliers(self):
        clean = get_clean_data(self._flagged())
        assert len(clean) == 2
        assert list(clean["aeth_bc"]) == [3.0, 4.0]

    def test_subsets_partition_the_frame(self):
        df = self._flagged()
        total = len(get_clean_data(df)) + len(get_excluded_data(df)) + len(get_outlier_data(df))
        assert total == len(df), "clean/excluded/outlier should account for every row"

    def test_helpers_are_safe_when_flags_are_absent(self):
        bare = _frame(["2024-01-01"] * 2, aeth_bc=[1.0, 2.0])
        assert len(get_clean_data(bare)) == 2
        assert get_excluded_data(bare).empty
        assert get_outlier_data(bare).empty

    def test_end_to_end_flag_then_clean_is_non_destructive(self):
        site = next(s for s in SITES if EXCLUDED_SAMPLES[s])
        entry = EXCLUDED_SAMPLES[site][0]
        df = _frame([entry["date"], "1990-01-01", "1990-01-02"],
                    aeth_bc=[1.0, 2.0, 3.0], filter_ec=[1.0, 2.0, 3.0])

        flagged = apply_threshold_flags(apply_exclusion_flags(df, site), site)
        clean = get_clean_data(flagged)

        assert len(flagged) == 3, "the flagged frame keeps every row"
        assert len(clean) < len(flagged), "the registered sample should be removed"
        # the audit trail survives on the flagged frame
        assert flagged.loc[0, "exclusion_reason"]
        assert np.isin(clean.index, flagged.index).all()


# ---------------------------------------------------------------------------
# ETAD GF fraction normalization
# ---------------------------------------------------------------------------

from etad_factors import (                                   # noqa: E402
    FACTOR_TO_FRAC, GF_FRACTION_COLUMNS,
    add_dominant_source, normalize_gf_fractions,
)


def _raw_gf(rows):
    """rows: list of 5-tuples in GF1..GF5 order."""
    cols = list(FACTOR_TO_FRAC)
    return pd.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})


class TestGfFractionNormalization:
    """AGENTS.md 'Data join quirks': raw GF1-GF5 are PM2.5 mass fractions
    summing to ~0.03-0.46 per row, not relative source contributions. Skipping
    the row-sum normalization makes every threshold comparison wrong.
    """

    def test_rows_sum_to_one_after_normalization(self):
        out = normalize_gf_fractions(_raw_gf([(0.05, 0.10, 0.20, 0.05, 0.10)]))
        assert out[GF_FRACTION_COLUMNS].sum(axis=1).iloc[0] == pytest.approx(1.0)

    def test_relative_proportions_are_preserved(self):
        raw = _raw_gf([(0.05, 0.10, 0.20, 0.05, 0.10)])
        out = normalize_gf_fractions(raw)
        # charcoal was 0.20 of a 0.50 total -> 40 %
        assert out["charcoal_frac"].iloc[0] == pytest.approx(0.4)

    def test_all_zero_row_yields_nan_not_inf(self):
        out = normalize_gf_fractions(_raw_gf([(0.0, 0.0, 0.0, 0.0, 0.0)]))
        vals = out[GF_FRACTION_COLUMNS].iloc[0]
        assert vals.isna().all()
        assert not np.isinf(vals.astype(float).fillna(0)).any()

    def test_columns_stay_float(self):
        """pd.NA would upcast to object and break downstream .max()/idxmax()."""
        out = normalize_gf_fractions(_raw_gf([(0.05, 0.1, 0.2, 0.05, 0.1),
                                              (0.0, 0.0, 0.0, 0.0, 0.0)]))
        assert (out[GF_FRACTION_COLUMNS].dtypes == "float64").all()

    def test_missing_fraction_columns_raise(self):
        with pytest.raises(KeyError):
            normalize_gf_fractions(pd.DataFrame({"nope": [1.0]}), rename=False)

    def test_normalization_changes_threshold_outcomes(self):
        """The concrete failure AGENTS.md warns about: unnormalized fractions
        never cross a 30 % threshold."""
        raw = _raw_gf([(0.01, 0.02, 0.03, 0.01, 0.20)])
        unnormalized = add_dominant_source(raw.rename(columns=FACTOR_TO_FRAC))
        normalized = add_dominant_source(normalize_gf_fractions(raw))

        assert unnormalized["dominant_fraction"].iloc[0] < 0.30
        assert normalized["dominant_fraction"].iloc[0] >= 0.30
        # the winner itself is unchanged - only the scale was wrong
        assert unnormalized["dominant_source"].iloc[0] == normalized["dominant_source"].iloc[0]


class TestDominantSource:
    def test_picks_the_largest_fraction(self):
        out = add_dominant_source(normalize_gf_fractions(
            _raw_gf([(0.01, 0.02, 0.03, 0.01, 0.40)])))
        assert out["dominant_source"].iloc[0] == "fossil_fuel"

    def test_all_na_row_does_not_raise(self):
        """idxmax on an all-NA row warns today and raises in a later pandas."""
        out = add_dominant_source(normalize_gf_fractions(
            _raw_gf([(0.0, 0.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3, 0.1, 0.3)])))
        assert pd.isna(out["dominant_source"].iloc[0])
        assert out["dominant_source"].iloc[1] == "charcoal"

    def test_raises_when_no_fraction_columns_present(self):
        with pytest.raises(KeyError):
            add_dominant_source(pd.DataFrame({"x": [1.0]}))


# ---------------------------------------------------------------------------
# Instrument channel wavelengths
# ---------------------------------------------------------------------------

from config import AE33_WAVELENGTHS_NM, WAVELENGTHS_NM   # noqa: E402


class TestChannelWavelengths:
    """The MA350 channel centres must not be confused with the AE33 set.

    src/analysis/bc/* keyed AE33 values (370/520/660) onto MA350 column names
    ('UV BCc', 'Green BCc', 'Red BCc'). Because AAE divides by ln(w1/w2), that
    inflates AAE(Red,IR) by ~16 % -- and source_apportionment turns AAE straight
    into a biomass fraction, so a true AAE of 1.2 reported as 1.43 doubles the
    biomass share.
    """

    def test_ma350_channels_are_the_microaeth_values(self):
        assert WAVELENGTHS_NM == {
            "UV": 375, "Blue": 470, "Green": 528, "Red": 625, "IR": 880
        }

    def test_ma350_and_ae33_sets_are_distinct(self):
        assert WAVELENGTHS_NM["Red"] != AE33_WAVELENGTHS_NM["BC5"]
        assert WAVELENGTHS_NM["Green"] != AE33_WAVELENGTHS_NM["BC3"]
        assert WAVELENGTHS_NM["UV"] != AE33_WAVELENGTHS_NM["BC1"]

    @pytest.mark.parametrize(
        "module_name,attr_getter",
        [
            ("src.analysis.bc.source_apportionment", "UV BCc"),
            ("src.analysis.bc.black_carbon_analyzer", "UV.BCc"),
        ],
    )
    def test_src_bc_modules_use_microaeth_values(self, module_name, attr_getter):
        """Guard against the AE33 values creeping back into BCc-keyed maps."""
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(module_name))
        # the giveaway trio, keyed on microAeth column names
        assert "'UV BCc': 370" not in src and "'UV.BCc': 370" not in src
        assert "'Green BCc': 520" not in src and "'Green.BCc': 520" not in src
        assert "'Red BCc': 660" not in src and "'Red.BCc': 660" not in src

    def test_ae33_bc1_bc7_values_are_preserved(self):
        """BC1..BC7 are genuine AE33 columns and must keep AE33 wavelengths."""
        import inspect
        import src.analysis.bc.source_apportionment as sa

        source = inspect.getsource(sa)
        for key, nm in AE33_WAVELENGTHS_NM.items():
            assert f"'{key}': {nm}" in source, f"{key} should stay at {nm} nm"
