"""Two-component optical mixing, with an explicit long-wavelength reference.

This numerical model does not establish that its components identify unique
emission sources. It estimates absorption fractions conditional on two fixed
power-law exponents. Converting those fractions to equivalent BC assumes the
same reference-wavelength mass absorption cross-section for both components.
"""

from dataclasses import dataclass

import numpy as np


# MA350 firmware attenuation cross-sections, not atmospheric mass absorption
# cross-sections. Values match src/external/calibration.py. A common scattering
# correction cancels in a ratio; wavelength-dependent corrections do not.
MA350_ATTENUATION_CROSS_SECTIONS = {
    "UV": 24.069, "Blue": 19.070, "Green": 17.028, "Red": 14.091, "IR": 10.120,
}
MA350_WAVELENGTHS_NM = {"UV": 375, "Blue": 470, "Green": 528, "Red": 625, "IR": 880}


@dataclass(frozen=True)
class TwoSourceResult:
    """Arrays retain input order and broadcast shape; invalid results are NaN.

    ``valid`` describes numerical signal eligibility, not instrument QA/QC or
    demonstrated source identity. Out-of-end-member observations remain valid
    and retain their unbounded fraction even when ``clip`` was requested.
    ``clipped_low`` / ``clipped_high`` describe clipping beyond machine roundoff
    (eight floating-point epsilons at either endpoint). When clipping is disabled
    they are false. The unbounded fraction is always preserved, including roundoff.
    """

    aae: np.ndarray
    biomass_fraction_unbounded: np.ndarray
    biomass_fraction: np.ndarray
    fossil_fraction: np.ndarray
    valid: np.ndarray
    clipped_low: np.ndarray
    clipped_high: np.ndarray
    invalid_reason: np.ndarray


def two_source_absorption(
    short_absorption,
    long_absorption,
    *,
    wavelength_short_nm,
    wavelength_long_nm,
    alpha_fossil=1.0,
    alpha_biomass=2.0,
    clip=True,
):
    """Solve two power laws for their fraction of long-wavelength absorption.

    For ``r = wavelength_long_nm / wavelength_short_nm``, the unbounded biomass
    fraction is ``(b_short/b_long - r**alpha_fossil) /
    (r**alpha_biomass - r**alpha_fossil)``. Interpolating the observed AAE
    linearly between the end members is not equivalent to this mixing model.

    Inputs must describe the same optical quantity and units at both wavelengths.
    A common positive multiplicative scale cancels. If inputs are attenuation
    proxies, the caller must disclose the assumption that the wavelength-dependent
    scattering correction does not alter their ratio; this helper cannot verify it.

    Nonfinite or nonpositive signals remain in place with a reason and NaN results.
    Clipping never deletes the unbounded fraction. No uncertainties, firmware
    smoothing, time aggregation, or instrument QC are inferred here.
    """
    short_nm, long_nm = float(wavelength_short_nm), float(wavelength_long_nm)
    alpha_ff, alpha_bb = float(alpha_fossil), float(alpha_biomass)
    if not np.isfinite([short_nm, long_nm]).all() or not 0 < short_nm < long_nm:
        raise ValueError("Wavelengths must be finite, positive and ordered short < long")
    if not np.isfinite([alpha_ff, alpha_bb]).all() or not alpha_ff < alpha_bb:
        raise ValueError("End-member exponents must be finite with alpha_fossil < alpha_biomass")
    if not isinstance(clip, (bool, np.bool_)):
        raise ValueError("clip must be a boolean")

    log_r = np.log(long_nm / short_nm)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ff_ratio = np.exp(alpha_ff * log_r)
        # expm1 avoids cancellation when the exponents are close but distinct.
        denominator = ff_ratio * np.expm1((alpha_bb - alpha_ff) * log_r)
    if not np.isfinite([ff_ratio, denominator]).all() or denominator <= 0:
        raise ValueError("End-member ratios are numerically degenerate at these wavelengths")

    short, long = np.broadcast_arrays(
        np.asarray(short_absorption, dtype=float), np.asarray(long_absorption, dtype=float)
    )
    finite = np.isfinite(short) & np.isfinite(long)
    positive = (short > 0) & (long > 0)
    eligible = finite & positive
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        aae = (np.log(short) - np.log(long)) / log_r
        unbounded = (short / long - ff_ratio) / denominator
    numerical = np.isfinite(aae) & np.isfinite(unbounded)
    valid = eligible & numerical
    reason = np.where(
        ~finite, "nonfinite_signal",
        np.where(~positive, "nonpositive_signal", np.where(~numerical, "numerical_failure", "")),
    )
    aae = np.where(valid, aae, np.nan)
    unbounded = np.where(valid, unbounded, np.nan)
    # A mathematically pure end member can evaluate to 1 + 2e-16. Preserve that
    # raw value without counting its machine-roundoff adjustment as source clipping.
    tolerance = 8.0 * np.finfo(float).eps
    low, high = valid & (unbounded < -tolerance), valid & (unbounded > 1 + tolerance)
    fraction = np.clip(unbounded, 0, 1) if clip else unbounded.copy()
    return TwoSourceResult(
        aae=aae,
        biomass_fraction_unbounded=unbounded,
        biomass_fraction=fraction,
        fossil_fraction=1.0 - fraction,
        valid=valid,
        clipped_low=low & clip,
        clipped_high=high & clip,
        invalid_reason=reason,
    )
