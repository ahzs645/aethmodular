# Raw HIPS pull from Networks_1_0 — what it settled, 2026-08-23

`scripts/get_hips_internals.ps1` against `hips.Results`, all 27 SPARTAN sites:
**6,876 rows, 4,115 filters, 4,122 analyses, every one at 633 nm.**

## The telemetry columns are empty — but the epoch structure is recoverable

I pitched `LaserPower`, `LaserDiodeTemperature` and the two sensor-temperature
columns as the most promising thing in the database — a way to test whether the
Addis intercept tracks instrument drift. **All four are 100% NULL for every one
of the 6,876 rows.** The columns exist in the schema; SPARTAN never populated
them. There is no instrument-state telemetry to correlate against, and the
instrument-shift hypothesis cannot be tested this way.

**But the pull carries two columns the shipped CSV does not**, and they recover
the epoch structure anyway: `TransmittanceRaw` / `ReflectanceRaw`. They are not
derivable from T1/R1 (median |TRaw - T1/(T1+R1)| = 6.6), and the ratio
`T1 / TransmittanceRaw` is flat within an instrument configuration and steps
sharply between them:

| epoch | sessions | gain T1/TRaw | filters | sites |
|---|---|---|---|---|
| **E1** | 2022-10-20 → 2023-03-17 | ~2,420 | 728 | 17 |
| **E2** | 2023-05-03 → 2023-08-22 | ~1,415 | 640 | 20 |
| **E3** | 2023-09-22 → 2026-07-31 | **77–80** | 2,747 | 28 |

Two sharp, network-wide breakpoints: **2023-05-03** (x1.7) and **2023-09-22**
(x18). Within E3 the gain holds at 77–80 for three years and 25 sessions. These
line up with the instrument events named in `hips.CalibrationSets` — collimator
replacement, fibre-optic reconfiguration, the lab move.

**Read this carefully: it is an epoch _marker_, not proof of drift in the
measured quantity.** `Transmittance` (T1) itself stays in the 674–818 band across
all three epochs, and T1 is what enters tau. The jump is in TransmittanceRaw's
normalisation. So the epochs tell us *when the instrument configuration changed*;
they do not by themselves say Fabs moved.

What that buys: a clean, data-derived segmentation to test the shift hypothesis
against. **ETAD straddles all three epochs — 8 filters in E1, 48 in E2, 240 in
E3** — so the Addis offset can be re-fitted per epoch. If it is constant across
the 2023-09-22 boundary, the instrument-shift explanation is dead for real; if it
steps, that is a measurement-side account of the intercept.

## What the pull did settle

**1. The shipped T1/R1 are the raw instrument values.** Joining 3,859 filters,
raw `Transmittance`/`Reflectance` are **bit-identical** to shipped `T1`/`R1`
(max |diff| = 0). No hidden preprocessing sits between the instrument and the CSV.

**2. The tau formula is confirmed exactly.**
`tau = ln((Intercept + Slope*R1)/T1)` reproduces the shipped `tau` for **100%**
of filters to within 1e-4. This retroactively validates the lot-253 blank-line
swap in `SPARTAN_LOT_INVENTORY_2026-08-23.md` (+0.57 Mm-1) as using the right
formula.

**3. `ResultTypeId = 1` is not a per-filter reference beam.** I hoped it might
let us compute tau without the lot blank line. It does not: `ln(T_ref/T1)`
misses the shipped tau by a median of **1.33**, and `T_ref` (median 249) sits
nowhere near the blank-line-predicted transmittance (median 937). The blank line
remains unavoidable.

## The payoff: 54 recovered filters

`hips.Results` holds **256 filters that are not in
`SPARTAN_HIPS_Batch1-51.v2.csv`**, across 15 sites. Because (1) and (2) hold, their
Fabs can be reconstructed from raw T1/R1 with the lot's blank line
(`scripts/reconstruct_hips_fabs.py`):

| site | recovered | lot | median Fabs | shipped site median | have spectra |
|---|---|---|---|---|---|
| **ETAD** | **14** | **253** | 44.85 (31.2–63.1) | 46.2 | yes (all 296 in ETAD_FTIR_spectra) |
| **ETBI** | **14** | 251 | 18.32 (11.3–30.9) | 19.0–42.1 range | yes (staged pull) |
| **INDH** | **26** | **253** | 74.42 (29.4–143.3) | 75.0 | yes (staged pull) |

Every recovered median lands on its site's shipped median — INDH 74.4 vs 75.0,
ETAD 44.9 vs 46.2. A second check: **42 of the 256 come out at tau < 0.05 with no
sample volume**, exactly the signature of a field/lab blank, and the two ETAD
blanks land at tau = 0.004 and 0.007. The reconstruction reproduces both the
loaded filters and the blanks correctly.

Effect on the targets, all 54 confirmed to have an FTIR analysis:

| target | now | with recovered | gain |
|---|---|---|---|
| **etbi** | 26 | **40** | **+54%** |
| indh | 152 | 178 | +17% |
| Addis (lot 253) | 14 w/Fabs | 28 | +100% |

The ETBI gain matters most — n = 26 was the binding constraint on the
Bishoftu-vs-Addis offset comparison, the strongest evidence that the intercept is
Addis-specific.

The remaining 202 filters (TWTA 40, TWKA 24, USSL 24, AUMN/INJA/KRUL/USNO 16
each, …) have tau but no sample volume locally. Volume lives in each site's
`<SITE>_filters.csv`, so pulling those sites with `get_spartan_spectra.ps1`
converts them too.

## Caveat: reconstructed, not official

The production pipeline may apply QC this does not replicate — MDL flags,
uncertainty propagation, comment-based rejection — and `DepositArea` is taken as
the site median rather than per-filter. These values should carry a provisional
flag until they appear in a shipped batch, and any headline result should be
checked with and without them.
