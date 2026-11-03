"""Numerical and provenance checks for the portable exploratory screen."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "portable_baseline_screen",
    ROOT / "research/ftir_hips_chem/scripts/portable_baseline_screen.py",
)
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


class PortableBaselineScreenTests(unittest.TestCase):
    def setUp(self):
        self.wn = np.linspace(500, 3998, 454)

    def test_signed_contrast_is_invariant_to_affine_background_and_grid_order(self):
        peak = -0.02 * np.exp(-0.5 * ((self.wn - 1617) / 20) ** 2)
        plain = SCREEN.local_band_contrast(self.wn, peak)
        affine = SCREEN.local_band_contrast(self.wn[::-1], (peak + 0.3 + self.wn * 0.0002)[::-1])
        np.testing.assert_allclose(plain["height"], affine["height"], atol=1e-14)
        np.testing.assert_allclose(plain["area"], affine["area"], atol=1e-12)
        self.assertLess(plain["height"][0], 0)

    def test_corrected_values_remain_signed_and_unsupported_channels_are_nan(self):
        x = 0.2 + 0.03*np.sin(self.wn/25)
        r = SCREEN.asymmetric_baseline(self.wn, x)
        self.assertTrue(np.any(r["corrected"] < 0))
        self.assertTrue(np.isnan(r["corrected"][:, ~r["reported_mask"]]).all())
        keep = r["reported_mask"]
        np.testing.assert_allclose((r["baseline"] + r["corrected"])[:, keep], x[None, keep], atol=1e-12)

    def test_reverse_grid_and_masked_contamination_do_not_change_fit(self):
        x = 0.2 + self.wn * 0.0001 + 0.02*np.exp(-0.5*((self.wn-1617)/25)**2)
        r = SCREEN.asymmetric_baseline(self.wn, x)
        contaminated = x.copy()
        contaminated[~r["reported_mask"]] = 1e8
        c = SCREEN.asymmetric_baseline(self.wn[::-1], contaminated[::-1])
        np.testing.assert_allclose(r["corrected"], c["corrected"][:, ::-1], atol=1e-11, equal_nan=True)

    def test_protected_band_cannot_pull_the_baseline(self):
        parent = 0.2 + self.wn*0.0001
        protected = (self.wn >= 1500) & (self.wn <= 1760)
        spike = parent.copy()
        spike[protected] += 0.05
        a = SCREEN.asymmetric_baseline(self.wn, parent, protected_windows=((1500,1760),))
        b = SCREEN.asymmetric_baseline(self.wn, spike, protected_windows=((1500,1760),))
        np.testing.assert_allclose(a["baseline"], b["baseline"], atol=1e-9, equal_nan=True)
        np.testing.assert_allclose((b["corrected"]-a["corrected"])[0, protected], 0.05, atol=1e-9)

    def test_invalid_grid_and_hyperparameters_are_rejected(self):
        for wn in ([1, 1, 2], [1, 3, 2], [1, np.nan, 3]):
            with self.assertRaises(ValueError):
                SCREEN.asymmetric_baseline(wn, [0,0,0])
        for kw in ({"lam":0}, {"p":0.5}, {"max_iter":0}):
            with self.assertRaises(ValueError):
                SCREEN.asymmetric_baseline(self.wn, np.ones(454), **kw)

    def test_packed_reader_preserves_sign_and_rejects_bad_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            q = np.array([[1, -2, 3], [4, -5, 6]], dtype="<i2")
            scales = np.array([0.1, 0.2], dtype="<f4")
            data = scales.tobytes() + q.tobytes()
            (path/"raw.bin").write_bytes(data)
            d = {"schema_version":1,"filters":{"id":["A","B"]},
                 "views":{"raw":{"wn":[500,1000,1500],"points":3,"bytes":len(data),"file":"raw.bin"}}}
            (path/"lot_spectra.json").write_text(json.dumps(d))
            loaded = SCREEN.load_packed_spectra(path,views=("raw",))
            np.testing.assert_allclose(loaded["views"]["raw"]["values"], q*scales[:,None])
            self.assertFalse(loaded["provenance"]["native_spectra"])
            (path/"raw.bin").write_bytes(data[:-1])
            with self.assertRaises(ValueError):
                SCREEN.load_packed_spectra(path,views=("raw",))

    def test_synthetic_controls_have_known_inputs_and_no_false_replication(self):
        rows = SCREEN.run_synthetic_controls(self.wn)
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(np.isfinite(r["area_recovery_fraction"]) for r in rows))
        self.assertTrue(all(r["independent_parent_count"] == 1 for r in rows))
        # Broad real absorbance can be removed as background: retain this
        # deliberate failure control instead of certifying all smoothers.
        broad = [r for r in rows if r["control"].startswith("broad_")]
        self.assertTrue(any(r["area_recovery_fraction"] < 0.8 for r in broad))


if __name__ == "__main__":
    unittest.main()
