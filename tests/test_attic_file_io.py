"""Tests for attic/utils/file_io.py JSON serialization.

Regression cover for the numpy/pandas coercion order: ndarray and Series expose
BOTH .item() and .tolist(), and .item() raises ValueError on anything with more
than one element. Checking .item() first made the .tolist() branch unreachable,
so saving an array -- the case the helper exists for -- crashed.

The module moved from src/utils/ to attic/ on 2026-07-26 (no consumer), but
the bug was real and the fix is worth holding in place.
"""

import json

import numpy as np
import pandas as pd
import pytest

from attic.utils.file_io import load_results_from_json, save_results_to_json


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.array([1, 2, 3]), [1, 2, 3]),
        (np.array([1.5, 2.5]), [1.5, 2.5]),
        (pd.Series([1, 2, 3]), [1, 2, 3]),
        (np.int64(5), 5),
        (np.float64(1.25), 1.25),
        ("plain", "plain"),
        (42, 42),
    ],
)
def test_roundtrips_numpy_and_pandas_values(tmp_path, value, expected):
    path = tmp_path / "results.json"
    save_results_to_json({"k": value}, path)
    assert load_results_from_json(path)["k"] == expected


def test_converts_nested_structures(tmp_path):
    path = tmp_path / "nested.json"
    save_results_to_json(
        {"outer": {"inner": np.array([7, 8])}, "listed": [np.int64(1), np.int64(2)]},
        path,
    )
    loaded = load_results_from_json(path)
    assert loaded["outer"]["inner"] == [7, 8]
    assert loaded["listed"] == [1, 2]


def test_written_file_is_valid_json(tmp_path):
    path = tmp_path / "valid.json"
    save_results_to_json({"arr": np.array([1, 2])}, path)
    assert json.loads(path.read_text()) == {"arr": [1, 2]}
