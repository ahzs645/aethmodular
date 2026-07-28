"""Regression tests for four defects in the src/ completeness/quality code.

All four were found by audit on 2026-07-27 and are fixed. None of this code has
an active consumer today, so these tests exist to stop the bugs coming back if
the pipeline is ever revived rather than to guard a live path.
"""

import pathlib
import re

import pandas as pd
import pytest

from src.analysis.aethalometer.period_processor import NineAMPeriodProcessor
from src.analysis.quality.completeness_analyzer import CompletenessAnalyzer
from src.analysis.quality.period_classifier import PeriodClassifier
from src.data.processors.aethalometer_filter_merger import identify_excellent_periods


def _minute_data(start, end, drop=None):
    idx = pd.date_range(start, end, freq="min", inclusive="left")
    if drop is not None:
        idx = idx.difference(drop)
    return pd.DataFrame({"v": 1.0}, index=idx)


class TestIdentifyExcellentPeriods:
    """It grouped only the MISSING timestamps, so a period with perfect
    coverage never entered the index and could never be returned. The function
    returned exactly the periods that had gaps -- the inverse of its purpose.
    """

    @staticmethod
    def _four_periods_one_small_gap():
        return _minute_data(
            "2024-01-01 09:00", "2024-01-05 09:00",
            drop=pd.date_range("2024-01-02 14:00", periods=5, freq="min"),
        )

    def test_perfect_periods_are_returned(self):
        out = identify_excellent_periods(self._four_periods_one_small_gap(), quality_threshold=10)
        assert len(out) == 4, "all four periods are within threshold"
        assert (out["missing_minutes"] == 0).sum() == 3, "three periods had no gaps at all"

    def test_threshold_actually_excludes(self):
        out = identify_excellent_periods(self._four_periods_one_small_gap(), quality_threshold=3)
        assert len(out) == 3
        assert 5 not in set(out["missing_minutes"]), "the 5-minute gap must not qualify"

    def test_large_gap_period_is_dropped_but_others_kept(self):
        df = _minute_data(
            "2024-01-01 09:00", "2024-01-05 09:00",
            drop=pd.date_range("2024-01-03 10:00", periods=200, freq="min"),
        )
        out = identify_excellent_periods(df, quality_threshold=10)
        starts = set(out["start_time"])
        assert pd.Timestamp("2024-01-03 09:00") not in starts
        assert len(out) == 3

    def test_end_time_is_one_day_after_start(self):
        out = identify_excellent_periods(self._four_periods_one_small_gap(), quality_threshold=10)
        assert ((out["end_time"] - out["start_time"]) == pd.Timedelta(days=1)).all()


class TestThresholdOrderIndependence:
    """Classification iterated the threshold dict in insertion order, so a
    caller-supplied mapping returned whichever tier happened to be listed first
    rather than the tightest matching one.
    """

    SCRAMBLED = {"Good": 60, "Excellent": 10, "Moderate": 240, "Poor": float("inf")}
    EXPECTED = [(5, "Excellent"), (30, "Good"), (100, "Moderate"), (500, "Poor")]

    @pytest.mark.parametrize("missing,expected", EXPECTED)
    def test_period_classifier(self, missing, expected):
        pc = PeriodClassifier(custom_thresholds=dict(self.SCRAMBLED))
        assert pc._get_base_quality(missing) == expected

    @pytest.mark.parametrize("missing,expected", EXPECTED)
    def test_completeness_analyzer(self, missing, expected):
        ca = CompletenessAnalyzer()
        ca.quality_thresholds = dict(self.SCRAMBLED)
        assert ca._classify_period_quality(missing) == expected

    def test_default_ordering_is_unchanged(self):
        pc = PeriodClassifier()
        assert pc._get_base_quality(5) == "Excellent"
        assert pc._get_base_quality(500) == "Poor"


class TestQualityTierParity:
    """period_processor jumped from 'good' (60) straight to 'poor', so a period
    with 100 missing minutes was 'poor' there and 'moderate' in every other
    classifier in the repo.
    """

    def test_period_processor_has_the_moderate_tier(self):
        tiers = NineAMPeriodProcessor().quality_thresholds
        assert "moderate" in tiers
        assert tiers["moderate"] == 240

    def test_tier_boundaries_match_the_other_classifiers(self):
        tiers = NineAMPeriodProcessor().quality_thresholds
        reference = PeriodClassifier().thresholds
        # compare case-insensitively; the two modules differ only in key casing
        got = {k.lower(): v for k, v in tiers.items()}
        want = {k.lower(): v for k, v in reference.items()}
        assert got == want


class TestSingleSourceThresholds:
    """The 10/60/240 tiers were hardcoded in four places. They now all derive
    from src/config/quality_thresholds.CompletenessThresholds so they cannot
    drift apart again.
    """

    def test_all_classifiers_share_one_definition(self):
        from src.config.quality_thresholds import completeness_tiers

        canonical = completeness_tiers()
        assert PeriodClassifier().thresholds == canonical
        assert CompletenessAnalyzer().quality_thresholds == canonical

    def test_processor_uses_the_lowercase_view(self):
        from src.config.quality_thresholds import completeness_tiers

        assert NineAMPeriodProcessor().quality_thresholds == completeness_tiers(lowercase=True)

    def test_tiers_match_the_dataclass(self):
        from src.config.quality_thresholds import CompletenessThresholds, completeness_tiers

        t = CompletenessThresholds()
        tiers = completeness_tiers()
        assert tiers["Excellent"] == t.excellent_max_missing
        assert tiers["Good"] == t.good_max_missing
        assert tiers["Moderate"] == t.moderate_max_missing

    def test_editing_the_dataclass_would_move_every_classifier(self):
        """Guards the wiring itself: the tiers must not be re-literalled."""
        import inspect

        for mod in (PeriodClassifier, CompletenessAnalyzer, NineAMPeriodProcessor):
            src = inspect.getsource(mod.__init__)
            assert "completeness_tiers" in src, f"{mod.__name__} no longer reads the shared source"
            assert "240" not in src, f"{mod.__name__} re-hardcodes a threshold"


class TestCompletenessDenominator:
    """_analyze_daily_missing / _analyze_9am_missing grouped only the MISSING
    timestamps, so a period with complete coverage never entered the index. The
    completeness percentages were therefore computed over "periods that had at
    least one gap" -- a dataset that was 75% perfect reported 100% Excellent.
    """

    @staticmethod
    def _four_periods_one_gap():
        return _minute_data(
            "2024-01-01 09:00", "2024-01-05 09:00",
            drop=pd.date_range("2024-01-02 14:00", periods=5, freq="min"),
        )

    def test_9am_periods_include_the_gapless_ones(self):
        r = CompletenessAnalyzer().analyze_completeness(
            self._four_periods_one_gap(), period_type="9am_to_9am")
        mp = r["period_analysis"]["missing_per_period"]
        assert len(mp) == 4
        assert (mp == 0).sum() == 3

    def test_daily_periods_include_the_gapless_ones(self):
        r = CompletenessAnalyzer().analyze_completeness(
            self._four_periods_one_gap(), period_type="daily")
        mp = r["period_analysis"]["missing_per_period"]
        assert (mp == 0).sum() >= 3

    def test_period_count_matches_period_classifier(self):
        """The two stacks previously disagreed on how many periods exist."""
        df = self._four_periods_one_gap()
        ca = CompletenessAnalyzer().analyze_completeness(df, period_type="9am_to_9am")
        pc = PeriodClassifier().classify_periods(df, period_type="9am_to_9am")
        assert len(ca["period_analysis"]["missing_per_period"]) == len(pc["classifications"])

    def test_perfect_dataset_reports_every_period(self):
        df = _minute_data("2024-01-01 09:00", "2024-01-04 09:00")
        mp = CompletenessAnalyzer().analyze_completeness(
            df, period_type="9am_to_9am")["period_analysis"]["missing_per_period"]
        assert len(mp) == 3 and (mp == 0).all()


class TestNineAmPeriodConvention:
    """src/data/qc labelled each 9am period by its END while
    src/analysis/quality and filter_mapping use the START, so the filter/quality
    join in FilterSampleMapper was a day out.
    """

    TIMESTAMPS = [
        ("2024-01-02 14:00", "2024-01-02 09:00"),   # after 9am -> same-day start
        ("2024-01-02 03:00", "2024-01-01 09:00"),   # before 9am -> previous-day start
        ("2024-01-02 09:00", "2024-01-02 09:00"),   # exactly 9am -> starts its own period
    ]

    @pytest.mark.parametrize("ts,expected", TIMESTAMPS)
    def test_completeness_analyzer_uses_period_start(self, ts, expected):
        ca = CompletenessAnalyzer()
        got = ca._analyze_9am_missing(pd.DatetimeIndex([ts]))
        assert got.index[0] == pd.Timestamp(expected)

    @pytest.mark.parametrize("ts,expected", TIMESTAMPS)
    def test_qc_classifier_agrees(self, ts, expected):
        """Replicates the mapping in src/data/qc/quality_classifier."""
        import inspect
        import src.data.qc.quality_classifier as qc

        source = inspect.getsource(qc)
        # the end-convention form must not reappear
        assert "pd.Timedelta(hours=9) + pd.Timedelta(days=1)" not in source

        t = pd.Timestamp(ts)
        base = t.normalize() + pd.Timedelta(hours=9)
        got = base if t.hour >= 9 else base - pd.Timedelta(days=1)
        assert got == pd.Timestamp(expected)


class TestFtirJoinProjectionParity:
    """The fallback FTIR loader selected 6 columns where FTIRHIPSLoader selects
    12, silently dropping volume_m3, all three MDLs, fabs_uncertainty and
    ftir_batch_id. Which schema a caller received depended on an ImportError it
    never saw.
    """

    @staticmethod
    def _projection(path):
        import pathlib
        import re

        txt = pathlib.Path(path).read_text()
        m = re.search(
            r"SELECT\s*\n(.*?)\n\s*FROM filters f\s*\n\s*JOIN ftir_sample_measurements",
            txt, re.S,
        )
        assert m, f"no FTIR join found in {path}"
        return sorted(c.strip() for c in m.group(1).replace("\n", " ").split(",") if c.strip())

    def test_fallback_matches_the_canonical_loader(self):
        canonical = self._projection("src/data/loaders/database.py")
        fallback = self._projection("src/data/processors/aethalometer_filter_merger.py")
        assert fallback == canonical

    def test_uncertainty_columns_are_not_dropped(self):
        fallback = self._projection("src/data/processors/aethalometer_filter_merger.py")
        flat = {c.split(".")[-1] for c in fallback}
        assert {"ec_ftir_mdl", "oc_ftir_mdl", "fabs_mdl", "fabs_uncertainty"} <= flat


class TestAethalometerFilterMatcherConstruction:
    """AethalometerFilterMatcher could not be constructed at all.

    _setup_filter_loader side-loaded data_loader_module.py from a path derived
    as dirname(dirname(filter_db_path)) -- which resolved to
    research/ftir_hips_chem/data_loader_module.py, a file that never existed
    there (the only copy was in notebooks/archive/scratch/). The resulting
    FileNotFoundError escaped the surrounding `except ImportError`, so the four
    notebooks under notebooks/analysis/absorption/ were broken.

    The loader now lives at src/data/loaders/filter_data_loader.py.
    """

    def test_loader_is_a_normal_import(self):
        from src.data.loaders import FilterDataLoader, load_filter_database

        assert FilterDataLoader is not None and load_filter_database is not None

    def test_matcher_no_longer_side_loads_by_path(self):
        """Checks executable code, not the docstring that records the history."""
        import ast
        import inspect

        from src.data.loaders import aethalometer_filter_matcher as m

        tree = ast.parse(inspect.getsource(m))
        # strip every docstring, then look at what is left
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                        and isinstance(first.value.value, str):
                    node.body = node.body[1:]
        code = ast.unparse(tree)

        assert "spec_from_file_location" not in code
        assert "data_loader_module" not in code

    def test_matcher_constructs_against_the_real_filter_db(self):
        import pathlib

        from src.data.loaders import AethalometerFilterMatcher

        db = pathlib.Path("research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl")
        aeth = pathlib.Path(
            "research/ftir_hips_chem/processed_sites/df_Jacros_9am_resampled.pkl")
        if not (db.exists() and aeth.exists()):
            pytest.skip("filter database / aethalometer pickle not present")

        matcher = AethalometerFilterMatcher(str(aeth), str(db))
        assert matcher.filter_loader is not None
        assert matcher.get_available_sites(), "expected at least one site"

    def test_missing_filter_db_raises_a_clear_error(self):
        from src.data.loaders import AethalometerFilterMatcher

        with pytest.raises(FileNotFoundError, match="Filter database not found"):
            AethalometerFilterMatcher("nope.pkl", "also-nope.pkl")


def test_qc_classifier_reads_the_canonical_thresholds_not_a_hardcoded_copy():
    """src/data/qc/quality_classifier.py used to hardcode 10/60/240 while
    src/analysis/quality/period_classifier.py read them from config. Two rival
    copies of the same tiers can silently diverge -- which already happened once,
    leaving period_processor without the 240 tier so a 100-minute gap classified
    as 'poor' there and 'moderate' everywhere else."""
    from src.config.quality_thresholds import completeness_tiers
    from src.data.qc.quality_classifier import QualityClassifier

    assert QualityClassifier().quality_thresholds == completeness_tiers(lowercase=True)

    # Assert on the *assignment*, not on any occurrence: the docstring
    # legitimately shows {'excellent': 10, ...} as the parameter format.
    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "data" / "qc" / "quality_classifier.py"
    ).read_text()
    assigned = re.search(
        r"self\.quality_thresholds\s*=\s*(.+?)(?:\n\s*\n|\n\s{4}\w)",
        source, re.S,
    ).group(1)
    assert "completeness_tiers" in assigned
    assert "10" not in assigned and "240" not in assigned


def test_qc_classifier_tier_boundaries_are_unchanged():
    """Behaviour-preservation guard for the swap above."""
    from src.data.qc.quality_classifier import QualityClassifier

    classifier = QualityClassifier()
    for missing, expected in [
        (0, "Excellent"), (10, "Excellent"),
        (11, "Good"), (60, "Good"),
        (61, "Moderate"), (240, "Moderate"),
        (241, "Poor"), (10_000, "Poor"),
    ]:
        assert classifier._classify_quality_by_missing(missing) == expected, missing


def test_qc_classifier_still_honours_an_explicit_override():
    from src.data.qc.quality_classifier import QualityClassifier

    custom = {"excellent": 1, "good": 2, "moderate": 3}
    classifier = QualityClassifier(custom)
    assert classifier.quality_thresholds == custom
    assert classifier._classify_quality_by_missing(2) == "Good"
