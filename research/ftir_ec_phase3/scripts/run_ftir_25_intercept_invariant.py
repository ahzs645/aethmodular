"""Restate the Addis intercept as an absorption target and report what is invariant.

C = |intercept| * MAC / slope is the Fabs value (Mm^-1) at which a calibration predicts
zero EC -- the constant absorption excess carried by Fabs with no FTIR-EC behind it.

Two things are worth checking, and this script checks both rather than asserting them:

1. C is MAC-invariant. ftir_19 establishes that the intercept does not move with MAC and
   that slope@MAC6 = 0.6 * slope@MAC10; together those force |b|*6/(0.6a) == |b|*10/a. So
   the target is well posed whether or not the MAC fork is ever settled.
2. C is far more stable across training cohorts than either slope or intercept alone.

See ftir_25_intercept_invariant.md for what that stability does and does not license --
in short, C is invariant to any multiplicative rescaling of the predicted-EC axis, so the
setups are not independent estimates converging.

Inputs are constants committed in ftir_19 (cell 5, 2 dp) and ftir_13 (cell 11, 6 dp).
No spectra, no Drive access, no refit.
"""

from __future__ import annotations

import statistics

# name, intercept, slope @ MAC 10, slope @ MAC 6  -- ftir_19 cell 5
SETUPS = (
    ("Deployed SPARTAN", -4.17, 1.90, 1.14),
    ("Biomass-smoke (906)", -6.91, 2.65, 1.59),
    ("Ethiopia-shaped smoke (300)", -3.69, 1.75, 1.05),
    ("Spectral analogs (400)", -6.43, 2.91, 1.75),
    ("Lowest-OC/EC (800)", -3.22, 1.59, 0.95),
    ("Lowest-OC/EC + AIRSpec", -1.62, 0.86, 0.51),
)

# name, intercept, slope @ MAC 10 -- ftir_13 cell 11, full precision
EXACT = (
    ("Lowest-OC/EC (800) raw", -3.221502, 1.585381),
    ("Lowest-OC/EC + AIRSpec df1=6", -1.615099, 0.857004),
    ("Deployed SPARTAN", -4.170345, 1.898281),
    ("Smoke 906 + AIRSpec df1=6", -0.674894, 0.371744),
)

MEDIAN_ADDIS_FABS = 47.11  # Mm^-1, ftir_13 cell 13


def absorption_target(intercept: float, slope: float, mac: float) -> float:
    """Return C in Mm^-1: the Fabs at which this calibration predicts zero EC."""
    return abs(intercept) * mac / slope


def main() -> int:
    print(f"{'setup':30} {'b':>7} {'a@10':>6} {'a@6':>6} {'C@10':>7} {'C@6':>7}  Δ")
    print("-" * 78)

    targets = []
    for name, intercept, slope10, slope6 in SETUPS:
        c10 = absorption_target(intercept, slope10, 10.0)
        c6 = absorption_target(intercept, slope6, 6.0)
        targets.append(c10)
        print(
            f"{name:30} {intercept:7.2f} {slope10:6.2f} {slope6:6.2f} "
            f"{c10:7.2f} {c6:7.2f}  {abs(c10 - c6):.2f}"
        )

    # The published slopes are rounded to 2 dp, so the identity only holds to that
    # precision here; the exact values below close the gap.
    worst = max(abs(absorption_target(b, a10, 10.0) - absorption_target(b, a6, 6.0))
                for _, b, a10, a6 in SETUPS)
    print(f"\nMAC-invariance: worst |C@10 - C@6| = {worst:.3f} Mm^-1 (2-dp rounding)")

    print("\nfull precision (ftir_13 cell 11):")
    for name, intercept, slope in EXACT:
        print(f"   {name:32} C = {absorption_target(intercept, slope, 10.0):6.2f} Mm^-1")

    slopes = [s for _, _, s, _ in SETUPS]
    intercepts = [b for _, b, _, _ in SETUPS]
    print("\nspread across the six setups:")
    print(f"   slope     {min(slopes):.2f} - {max(slopes):.2f}"
          f"   ({max(slopes) / min(slopes):.1f}x)")
    print(f"   intercept {max(intercepts):.2f} - {min(intercepts):.2f}"
          f"   ({min(intercepts) / max(intercepts):.1f}x)")
    print(f"   C         {min(targets):.1f} - {max(targets):.1f} Mm^-1"
          f"   ({max(targets) / min(targets):.1f}x),  median {statistics.median(targets):.1f}")

    share = statistics.median(targets) / MEDIAN_ADDIS_FABS * 100
    print(f"\nmedian C is {share:.0f}% of median Addis Fabs ({MEDIAN_ADDIS_FABS} Mm^-1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
