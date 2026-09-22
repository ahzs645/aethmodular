"""Failure-mode checks for the run's evidence gate; no scientific fit is run."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from run import compare_table, safe_relative, sha, verify_file


class ContractTests(unittest.TestCase):
    def test_changed_identity_fails(self):
        a = pd.DataFrame({"id": ["A", "B"], "value": [1., 2.]})
        b = a.copy(); b.loc[0, "id"] = "C"
        self.assertFalse(all(x["passed"] for x in compare_table(a, b, 1e-6, "test")))

    def test_missing_column_fails(self):
        a = pd.DataFrame({"id": [1], "value": [2.]})
        self.assertFalse(compare_table(a, a[["id"]], 1e-6, "test")[0]["passed"])

    def test_nan_cannot_replace_result(self):
        a = pd.DataFrame({"value": [2.]})
        b = pd.DataFrame({"value": [float("nan")]})
        self.assertFalse(compare_table(a, b, 1e-6, "test")[0]["passed"])

    def test_tolerance_is_absolute(self):
        a = pd.DataFrame({"value": [1e6]})
        b = pd.DataFrame({"value": [1e6 + .01]})
        self.assertFalse(compare_table(a, b, 1e-6, "test")[0]["passed"])

    def test_changed_input_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "data"; p.write_bytes(b"original")
            digest = sha(p); verify_file(p, digest)
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError): verify_file(p, digest)

    def test_manifest_cannot_escape_root(self):
        for name in ["../outside", "/absolute"]:
            with self.assertRaises(ValueError): safe_relative(name)


if __name__ == "__main__":
    unittest.main()
