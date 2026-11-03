"""Physical identities and comparator provenance in the portable cohort screen."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from config import MAC_VALUE
from portable_cohort_screen import (
    COMBINED_HIPS, DIRECT_HIPS, PUBLIC_EXTENSION,
    load_public_chemistry, optical_pair_ledger, site_optical_summary,
)


def sample_rows():
    def row(i, ec, fabs=None, public=None, excluded=False):
        return {"id": f"TEST-{i:04d}", "code": "TEST", "date": f"2024-01-{i:02d}",
                "excluded": excluded, "exclusion_reason": "documented sampling defect" if excluded else "",
                "EC (FTIR)": ec, "EC (ChemSpec FTIR)": ec,
                "HIPS Fabs": fabs, "HIPS Fabs uncertainty": 1.5 if fabs is not None else None,
                "BC (public HIPS)": public, "BC (public SSR estimate)": 99.0,
                "Equivalent BC (SSR)": 99.0, "Sampled volume": 7.2}
    return [row(1, 3.0, MAC_VALUE, 99.0), row(2, 5.0, 2 * MAC_VALUE),
            row(3, 7.0, 3 * MAC_VALUE), row(4, 9.0, None, 4.0),
            row(5, 11.0, 5 * MAC_VALUE, excluded=True), row(6, 13.0),
            row(7, None, 7 * MAC_VALUE), row(8, -1.0, -MAC_VALUE)]


class PortableCohortScreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "gallery/app/public/data/spartan_reference.json"
        self.path.parent.mkdir(parents=True)
        self.write(sample_rows())

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows):
        self.path.write_text(json.dumps({"schema_version": 1, "n": len(rows), "rows": rows}))

    def test_no_ssr_fallback_and_no_duplicate_product_count(self):
        rows = sample_rows()
        rows[6]["EC (ChemSpec FTIR)"] = 999.0
        self.write(rows)
        frame = load_public_chemistry(self.root)
        ledger = optical_pair_ledger(frame).set_index("id")
        self.assertEqual(len(ledger), 8)
        self.assertEqual(ledger.loc["TEST-0001", "optical_source"], DIRECT_HIPS)
        self.assertEqual(ledger.loc["TEST-0001", "optical_bc_ugm3"], 1.0)
        self.assertEqual(ledger.loc["TEST-0004", "optical_source"], PUBLIC_EXTENSION)
        self.assertFalse(ledger.loc["TEST-0006", "pair_eligible"])
        self.assertEqual(ledger.loc["TEST-0006", "pair_reason"], "missing_hips")
        self.assertEqual(ledger.loc["TEST-0007", "pair_reason"], "missing_ftir_ec")
        self.assertEqual(ledger.pair_eligible.sum(), 5)

    def test_published_exclusions_survive_and_nonpositive_values_are_retained(self):
        ledger = optical_pair_ledger(load_public_chemistry(self.root)).set_index("id")
        self.assertTrue(ledger.loc["TEST-0005", "is_excluded"])
        self.assertFalse(ledger.loc["TEST-0005", "pair_eligible"])
        self.assertTrue(ledger.loc["TEST-0008", "pair_eligible"])
        self.assertTrue(ledger.loc["TEST-0008", "nonpositive_pair"])
        self.assertEqual(ledger.loc["TEST-0008", "optical_disagreement_ugm3"], 0.0)

    def test_summary_uses_one_physical_filter_and_known_linear_relation(self):
        frame = load_public_chemistry(self.root)
        summary = site_optical_summary(frame).set_index("optical_source")
        self.assertEqual(summary.loc[DIRECT_HIPS, "n_pairs"], 4)
        self.assertEqual(summary.loc[PUBLIC_EXTENSION, "n_pairs"], 1)
        self.assertEqual(summary.loc[COMBINED_HIPS, "n_pairs"], 5)
        self.assertEqual(summary.loc[COMBINED_HIPS, "n_excluded"], 1)
        self.assertEqual(summary.loc[DIRECT_HIPS, "n_nonpositive_pairs"], 1)
        for lam in ("0p25", "1p0", "4p0"):
            self.assertAlmostEqual(summary.loc[DIRECT_HIPS, f"deming_lambda_{lam}_slope"], 2.0)
            self.assertAlmostEqual(summary.loc[DIRECT_HIPS, f"deming_lambda_{lam}_intercept_ugm3"], 1.0)
        self.assertAlmostEqual(summary.loc[DIRECT_HIPS, "pearson_r_squared"], 1.0)

    def test_duplicate_ids_and_unreasoned_flags_are_rejected(self):
        rows = sample_rows()
        self.write(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate physical-filter"):
            load_public_chemistry(self.root)
        rows[4]["exclusion_reason"] = ""
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "require a reason"):
            load_public_chemistry(self.root)

    def test_constancy_has_no_invented_correlation_or_slope(self):
        rows = sample_rows()[:3]
        for row in rows:
            row["HIPS Fabs"] = MAC_VALUE
        self.write(rows)
        summary = site_optical_summary(load_public_chemistry(self.root)).set_index("optical_source")
        self.assertTrue(np.isnan(summary.loc[DIRECT_HIPS, "pearson_r_squared"]))
        self.assertTrue(np.isnan(summary.loc[DIRECT_HIPS, "deming_lambda_1p0_slope"]))


if __name__ == "__main__":
    unittest.main()
