"""Physical-mixture and input-contract regressions for the optical source model."""

import json
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from src.analysis.bc.source_apportionment import SourceApportionmentAnalyzer
from src.analysis.bc.source_model import (
    MA350_ATTENUATION_CROSS_SECTIONS,
    MA350_WAVELENGTHS_NM,
    two_source_absorption,
)


def absorption_mixture(fractions, short_nm=470, long_nm=880, ff=1.0, bb=2.0, reference=10.0):
    """Construct independent source amplitudes, then sum their power-law spectra."""
    fraction = np.asarray(fractions, dtype=float)
    fossil_long = reference * (1 - fraction)
    biomass_long = reference * fraction
    short = (fossil_long * (short_nm / long_nm) ** -ff
             + biomass_long * (short_nm / long_nm) ** -bb)
    return short, np.broadcast_to(reference, fraction.shape)


def ma350_frame(fractions):
    return pd.DataFrame({
        f"{channel} BCc": absorption_mixture(fractions, nm)[0] / MA350_ATTENUATION_CROSS_SECTIONS[channel]
        for channel, nm in MA350_WAVELENGTHS_NM.items()
    })


@pytest.mark.parametrize("short_nm", [375, 470, 625])
@pytest.mark.parametrize("exponents", [(1.0, 2.0), (0.9, 1.8)])
def test_recovers_constructed_pure_and_half_mixtures(short_nm, exponents):
    short, long = absorption_mixture([0, 0.5, 1], short_nm, ff=exponents[0], bb=exponents[1])
    result = two_source_absorption(
        short, long, wavelength_short_nm=short_nm, wavelength_long_nm=880,
        alpha_fossil=exponents[0], alpha_biomass=exponents[1],
    )
    np.testing.assert_allclose(result.biomass_fraction_unbounded, [0, 0.5, 1], atol=1e-14)
    np.testing.assert_allclose(result.biomass_fraction + result.fossil_fraction, 1)
    assert result.valid.all()
    assert not result.clipped_low.any() and not result.clipped_high.any()


def test_mixing_is_not_linear_interpolation_of_aae():
    ratio = 880 / 470
    result = two_source_absorption(ratio ** 1.5, 1, wavelength_short_nm=470, wavelength_long_nm=880)
    expected = (ratio ** 1.5 - ratio) / (ratio ** 2 - ratio)
    assert float(result.aae) == pytest.approx(1.5)
    assert float(result.biomass_fraction) == pytest.approx(expected)
    assert abs(float(result.biomass_fraction) - 0.5) > 0.05


def test_common_optical_scale_cancels_and_broadcast_shape_is_retained():
    short, long = absorption_mixture([0.2, 0.7])
    base = two_source_absorption(short, long, wavelength_short_nm=470, wavelength_long_nm=880)
    scaled = two_source_absorption(short[:, None] / 1.3, long[:, None] / 1.3,
                                   wavelength_short_nm=470, wavelength_long_nm=880)
    assert scaled.biomass_fraction.shape == (2, 1)
    np.testing.assert_allclose(base.biomass_fraction, scaled.biomass_fraction[:, 0])


def test_clipping_retains_raw_fractions_and_distinguishes_invalid_signals():
    short, long = absorption_mixture([-0.5, 0.5, 1.5])
    short = np.r_[short, np.nan, 0, -1, np.inf]
    long = np.r_[long, 1, 1, 1, 1]
    result = two_source_absorption(short, long, wavelength_short_nm=470, wavelength_long_nm=880)
    np.testing.assert_allclose(result.biomass_fraction_unbounded[:3], [-0.5, 0.5, 1.5])
    np.testing.assert_allclose(result.biomass_fraction[:3], [0, 0.5, 1])
    assert result.clipped_low.tolist() == [True, False, False, False, False, False, False]
    assert result.clipped_high.tolist() == [False, False, True, False, False, False, False]
    assert result.invalid_reason.tolist() == ["", "", "", "nonfinite_signal", "nonpositive_signal",
                                               "nonpositive_signal", "nonfinite_signal"]
    assert np.isnan(result.biomass_fraction[3:]).all()
    unbounded = two_source_absorption(short, long, wavelength_short_nm=470,
                                      wavelength_long_nm=880, clip=False)
    np.testing.assert_allclose(unbounded.biomass_fraction[:3], [-0.5, 0.5, 1.5])
    assert not unbounded.clipped_low.any() and not unbounded.clipped_high.any()


@pytest.mark.parametrize("parameters", [
    {"wavelength_short_nm": 880}, {"wavelength_short_nm": 0},
    {"wavelength_long_nm": np.nan}, {"alpha_fossil": 2.0},
    {"alpha_biomass": 0.5}, {"alpha_biomass": np.inf},
    {"alpha_fossil": 1e5, "alpha_biomass": 1e5 + 1}, {"clip": "yes"},
])
def test_invalid_or_degenerate_model_settings_raise(parameters):
    settings = {"wavelength_short_nm": 470, "wavelength_long_nm": 880, **parameters}
    with pytest.raises(ValueError):
        two_source_absorption([2], [1], **settings)


def test_legacy_analyzer_recovers_half_mixture_and_uses_ir_mass_reference():
    result = SourceApportionmentAnalyzer().analyze(ma350_frame([0.5]))
    assert "error" not in result
    assert result["wavelength_pair"] == {"bc1": "Blue BCc", "bc2": "IR BCc"}
    source = result["source_contributions"]
    assert source["biomass_fraction"]["mean"] == pytest.approx(0.5)
    assert source["biomass_bc"]["mean"] == pytest.approx(5 / MA350_ATTENUATION_CROSS_SECTIONS["IR"])
    assert result["source_model"]["calculation_basis"] == "attenuation_proxy"
    assert "common" in result["source_model"]["scattering_assumption"]
    assert "common reference-wavelength" in result["source_model"]["bc_partition_assumption"]


def test_column_and_row_order_do_not_change_selected_pair_or_summary():
    data = ma350_frame([0.1, 0.5, 0.9])
    analyzer = SourceApportionmentAnalyzer()
    baseline = analyzer.analyze(data)
    # Cover arbitrary input-column ordering without depending on dict ordering.
    for columns in list(permutations(data.columns))[:12]:
        result = analyzer.analyze(data.loc[::-1, columns])
        assert result["wavelength_pair"] == baseline["wavelength_pair"]
        assert result["source_contributions"]["biomass_fraction"]["mean"] == pytest.approx(0.5)


def test_uv_fallback_is_named_and_other_pairs_require_explicit_choice():
    data = ma350_frame([0.5])
    result = SourceApportionmentAnalyzer().analyze(data[["UV BCc", "IR BCc"]])
    assert result["source_model"]["pair_selection"] == "UV/IR fallback (Blue absent)"
    assert result["source_contributions"]["biomass_fraction"]["mean"] == pytest.approx(0.5)
    red_ir = data[["IR BCc", "Red BCc"]]
    assert "error" in SourceApportionmentAnalyzer().analyze(red_ir)
    explicit = SourceApportionmentAnalyzer(wavelength_pair=("Red BCc", "IR BCc")).analyze(red_ir)
    assert explicit["source_contributions"]["biomass_fraction"]["mean"] == pytest.approx(0.5)
    assert "error" in SourceApportionmentAnalyzer(wavelength_pair=("IR BCc", "Red BCc")).analyze(red_ir)


def test_ae33_never_borrows_ma350_cross_sections():
    short, long = absorption_mixture([0.5])
    declared = {"BC2": 20.0, "BC6": 10.0}
    data = pd.DataFrame({"BC2": short / declared["BC2"], "BC6": long / declared["BC6"]})
    refused = SourceApportionmentAnalyzer().analyze(data)
    assert "explicitly supplied cross_sections" in refused["error"]
    result = SourceApportionmentAnalyzer(instrument="AE33", cross_sections=declared).analyze(data)
    assert result["source_contributions"]["biomass_fraction"]["mean"] == pytest.approx(0.5)
    assert result["source_model"]["cross_section_provenance"] == "caller supplied"
    missing = SourceApportionmentAnalyzer(instrument="AE33", cross_sections={"BC2": 20}).analyze(data)
    assert "Missing declared cross-sections" in missing["error"]


def test_numeric_and_mixed_families_need_declared_instrument():
    data = ma350_frame([0.5])
    numeric = data.rename(columns={f"{c} BCc": f"BC_{nm}" for c, nm in MA350_WAVELENGTHS_NM.items()})
    assert "error" in SourceApportionmentAnalyzer().analyze(numeric)
    result = SourceApportionmentAnalyzer(instrument="MA350").analyze(numeric)
    assert result["source_contributions"]["biomass_fraction"]["mean"] == pytest.approx(0.5)
    data["BC2"], data["BC6"] = 1.0, 1.0
    assert "error" in SourceApportionmentAnalyzer().analyze(data)
    assert "error" not in SourceApportionmentAnalyzer(instrument="MA350").analyze(data)


def test_missing_invalid_and_clipped_inputs_keep_honest_coverage_and_json():
    data = ma350_frame([-0.5, 0.5, 1.5, 0.5, 0.5])[["Blue BCc", "IR BCc"]]
    data.loc[3, "Blue BCc"] = np.nan
    data.loc[4, "IR BCc"] = 0
    result = SourceApportionmentAnalyzer().analyze(data)
    assert result["signal_validity"]["valid"] == 3
    assert result["signal_validity"]["invalid"] == 2
    assert result["signal_validity"]["clipped_low"] == 1
    assert result["signal_validity"]["clipped_high"] == 1
    assert result["source_contributions"]["biomass_fraction_unbounded"]["max"] == pytest.approx(1.5)
    json.dumps(result, allow_nan=False)
    empty = SourceApportionmentAnalyzer().analyze(data.iloc[3:])
    assert empty["source_contributions"]["biomass_fraction"]["mean"] is None
    assert empty["source_contributions"]["fossil_bc"]["total"] is None
    assert empty["signal_validity"]["valid"] == 0
    json.dumps(empty, allow_nan=False)


def test_absorption_input_does_not_get_mislabeled_as_bc_mass():
    short, long = absorption_mixture([0.5])
    # Existing column family retained, but the caller explicitly declares its basis.
    data = pd.DataFrame({"Blue BCc": short, "IR BCc": long})
    result = SourceApportionmentAnalyzer(signal_basis="absorption").analyze(data)
    assert result["source_model"]["calculation_basis"] == "absorption"
    assert result["source_model"]["cross_sections"] is None
    assert "biomass_bc" not in result["source_contributions"]
    assert result["source_contributions"]["biomass_reference_optical"]["mean"] == pytest.approx(5)


def test_alias_ambiguity_and_duplicate_selected_columns_are_not_silently_resolved():
    data = ma350_frame([0.5])
    data["BC_470"] = data["Blue BCc"]
    assert "Ambiguous wavelength aliases" in SourceApportionmentAnalyzer().analyze(data)["error"]
    assert "error" not in SourceApportionmentAnalyzer(wavelength_pair=("Blue BCc", "IR BCc")).analyze(data)
    duplicate = pd.concat([data[["Blue BCc"]], data[["Blue BCc", "IR BCc"]]], axis=1)
    assert "unique names" in SourceApportionmentAnalyzer().analyze(duplicate)["error"]
