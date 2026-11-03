"""Check matching ownership, measured calipers, and outcome isolation."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from existing_data_cohorts import match_existing_ethiopian_filters  # noqa: E402


def frame():
    return pd.DataFrame([
        dict(id=f"{code}-{i}", code=code, date=f"2025-11-{day:02d}", lot_group=lot,
             optical_bc_ugm3=x, pm25_ugm3=20, volume_m3=7,
             optical_disagreement_ugm3=y, spectral_pair_eligible=True,
             optical_source="direct_hips_fabs")
        for code, i, day, x, y, lot in [
            ("ETAD", 1, 2, 3, -2, "unknown"), ("ETAD", 2, 3, 4, 4, "251"),
            ("ETBI", 1, 2, 3.1, -1, "251"), ("ETBI", 2, 3, 4.1, -1.5, "251"),
        ]
    ])


def test_matching_never_reuses_a_physical_filter_and_preserves_unknown_lot():
    pairs = match_existing_ethiopian_filters(frame())
    assert len(pairs) == 2
    assert pairs.addis_id.is_unique and pairs.bishoftu_id.is_unique
    assert pairs.same_known_lot.sum() == 1
    strict = match_existing_ethiopian_filters(frame(), same_lot=True)
    assert strict.addis_id.tolist() == ["ETAD-2"]


def test_matching_cannot_select_using_the_ftir_outcome():
    f = frame()
    before = match_existing_ethiopian_filters(f)[["addis_id", "bishoftu_id"]]
    f["optical_disagreement_ugm3"] = [1000, -1000, 8000, -8000]
    after = match_existing_ethiopian_filters(f)[["addis_id", "bishoftu_id"]]
    pd.testing.assert_frame_equal(before, after)


def test_outside_calipers_is_unmatched_and_duplicates_are_rejected():
    f = frame()
    f.loc[f.code.eq("ETBI"), "volume_m3"] = 1
    assert match_existing_ethiopian_filters(f).empty
    with pytest.raises(ValueError, match="physical filter"):
        match_existing_ethiopian_filters(pd.concat([frame(), frame().iloc[[0]]]))
