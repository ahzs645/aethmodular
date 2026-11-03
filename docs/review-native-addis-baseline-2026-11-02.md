# Native Addis spectra and baseline audit — 2 November 2026

## Conclusion

The native data reproduce the saved spline-baseline calculation very closely.
They also confirm two already documented acquisition problems that baseline
adjustments cannot resolve: the nearly flat ETAD-0037 spectrum and one nearly
flat scan of ETAD-0139. Keeping that nearly flat scan in the latter filter's
average approximately halves several spectral features. These observations
support laboratory review of the acquisitions and a recorded scan-selection
decision before a new calibration comparison.

The broader Addis discrepancy remains scientifically unresolved. The removed
spectral component carries substantial information associated with the measured
HIPS–FTIR difference, but this does not prove that the removed component is an
artifact. It can include real broad aerosol absorption, and the discrepancy
itself includes an FTIR-derived quantity. Neither an attractive baseline nor
better agreement with HIPS establishes accurate EC.

This audit is separate from the coarsened multisite screen. It uses native,
signed absorbance without clipping or area normalization. No source spectrum,
existing EC product, or canonical exclusion was changed.

## Reproduction and provenance

Workflow:
[`run_native_addis_baseline_audit.py`](../research/ftir_hips_chem/workflows/run_native_addis_baseline_audit.py).

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/run_native_addis_baseline_audit.py \
  --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR" --jobs 4
```

Set `AETHMODULAR_ETAD_NATIVE_DIR` to a local folder containing the four source
files below. Their private provider URLs and raw contents are not committed.
The checked runtime used Python 3.12.14, NumPy 2.4.2 and pandas 2.3.3; the
repository recommends Python 3.13. `aeth doctor` passed all required modules.

| Source | Bytes | SHA-256 |
|---|---:|---|
| ETAD_FTIR_spectra.csv | 16,200,588 | `0f530df2cb48a1eca6c83717a7f051d02d74e848fb160178da9079687f253ed2` |
| ETAD_metadata.csv | 30,639 | `697696e98a4e10919374a34d72cb6fd41dc03b45632533e9ef9c46014f5025e8` |
| etad_spectra_lotmap.csv | 8,030 | `eae2d170fca33c504ae2d477843a603047fafa8ca1b99ccc1963e5382c382e3d` |
| spectra_baselined_AIRSPEC.csv | 13,443,028 | `441efd52d72b1c769ae44049611ac08a26f0060b68101d313b2e755d41e129cc` |

Outputs are generated under
`research/ftir_hips_chem/output/tables/native_baseline_audit_20261102/`.
`summary.json` records hashes of the source inputs, source measurement dataset,
packed descriptor and relevant code. The output ledger retains all physical
filters and records eligibility separately.

## Cohorts reconcile exactly

| Cohort | Physical filters | Scans |
|---|---:|---:|
| PM2.5 | 253 | 274 |
| Field blanks, identified by `ExternalFilterType=FB` | 37 | 39 |
| PM10 | 6 | 6 |
| Total | 296 | 319 |

The native grid has 2,722 points from 3998.423 to 500.124 cm⁻¹, spaced by
1.285667 cm⁻¹. The saved spline result contains 2,002 finite points in its
strict 1425–4000 cm⁻¹ region. Missing values outside that region are not zeros.

The packed gallery contains 247 PM2.5 and six PM10 Addis filters. Its six fewer
PM2.5 filters are **ETAD-0048, ETAD-0122, ETAD-0123, ETAD-0124, ETAD-0125 and
ETAD-0126**. All have finite native spectra but lack sampling dates and sampled
volumes in the supplied metadata. The exporter excludes undated rows explicitly;
it also excludes field blanks by type. There is no unexplained native-versus-
packed discrepancy. See `native_filters_absent_from_packed.csv` and
[`export_research_lot_spectra.py`](../gallery/data/export_research_lot_spectra.py).

The existing unified dataset supplies 189 eligible native PM2.5 filters with
both frozen unified FTIR EC and direct HIPS Fabs. HIPS is divided by the configured
MAC of 10 m²/g. All paired native/product collection dates agree. The six PM10
filters and field blanks remain in baseline diagnostics but are not counted as
PM2.5 EC–HIPS pairs. The undated samples remain in the ledger and spectral
summaries without being treated as dated optical pairs.

### Why the current portable screen has 246 pairs

The 189-pair native table uses the frozen unified dataset, whose paired Addis
coverage ends on 21 September 2024. The current chemistry export used by the
portable screen includes later products. The cohorts reconcile as follows:

| Membership | Physical filters | Explanation |
|---|---:|---|
| In both paired analyses | 188 | Native spectra and both product snapshots are present |
| Current portable pairs only | 58 | Later dates, 9 October 2024–1 March 2026; 44 added direct-HIPS pairs and 14 public-HIPS extensions |
| Frozen native pairs only | 1 | ETAD-0246 has EC 9.279024 µg/m³ in the frozen dataset but no EC in the current export |
| Frozen native paired total | 189 | 188 + 1 |
| Current portable paired total | 246 | 188 + 58; 232 direct-HIPS pairs plus 14 public-HIPS extensions |

Thus the difference is product availability and snapshot coverage, not missing
native spectra. The script writes `chemistry_cohort_reconciliation.csv` and
separate native-feature association tables for the current 246 combined-HIPS
and 232 direct-HIPS-only cohorts. It does not fill the current ETAD-0246 gap
with its older value or mix the public extension into the direct-HIPS label.

## Numerical baseline reproduction

The existing Python spline implementation was run on **all 319 scans**, at
segment parameters `(df1, df2)=(6,4)`, and compared with the saved R output.

| Check | Result |
|---|---:|
| Corrected cells compared | 638,638 |
| All-cell RMSE | 1.076 × 10⁻⁷ absorbance |
| Largest absolute cell difference | 6.014 × 10⁻⁷ absorbance |
| Largest per-scan maximum error / 95th percentile absolute corrected signal | 0.0002745, or 0.02745% |

This is evidence that the port reproduces the recorded calculation. It is
neither independent validation of that baseline model nor independent EC
validation. Results are in `r_python_validation.csv`.

## Confirmed acquisitions requiring review

These issues already appear in [`open-items.md`](open-items.md); their existence
is reproduced here, not newly discovered. The spectra and their source products
remain intact. Flags are diagnostic annotations in this audit, not new canonical
exclusions.

### ETAD-0037 is nearly flat before any baseline correction

| Filter / scan | Collection | Collected mass, µg | Raw full-grid peak-to-peak absorbance | Raw C–H local contrast |
|---|---|---:|---:|---:|
| ETAD-0036 / 3211 | 6 Jan 2023 | 107.7 | 0.383924 | 0.003755 |
| **ETAD-0037 / 406** | **9 Jan 2023** | **99.6** | **0.000149** | **0.000004** |
| ETAD-0040 / 3214 | 15 Jan 2023 | 106.2 | 0.434373 | 0.001842 |

All three are lot 248. Their HIPS-derived BC concentrations are approximately
3.93, 3.84 and 3.85 µg/m³, respectively. ETAD-0037's raw spectrum is entirely
negative, from −0.000193 to −0.000044 absorbance. Its corrected RMS is about
0.000006. A different baseline cannot recover absorption features absent in
the acquired spectrum. The existing sample/background-beam investigation in
`open-items.md` should guide laboratory review of sample/reference identity and
acquisition conditions. This audit alone cannot determine which upstream
measurement or identity is wrong.

### ETAD-0139 has a discordant repeat; averaging hides it

| Scan | Raw full-grid peak-to-peak absorbance | Corrected RMS | Corrected absorbance at 1617.369 cm⁻¹ |
|---|---:|---:|---:|
| 1626 | 0.001869 | 0.000092 | 0.000058 |
| 1668 | 0.445007 | 0.008556 | 0.005377 |
| Current mean of both scans | 0.222590 | 0.004247 | 0.002717 |

The same physical filter has a recorded collected mass of 162.4 µg. The two
raw spectra correlate at **r = −0.1821** over the full grid. A labelled
1668-only sensitivity calculation approximately doubles the corrected signal
relative to the two-scan average; the C–H local contrast rises from 0.001865 to
0.003804. See `etad0139_flagged_scan_sensitivity.csv`.

No EC value was recalculated in this test. In particular, the frozen unified
FTIR EC product, 3.9766 µg/m³, must not be confused with the separate historical
CAL-0 per-scan predictions quoted in `open-items.md`.

### ETAD-0158 remains a documented damage review item

The existing issue records a large hole in this filter. Its scan, 1995, is
preserved and flagged in `prior_documented_problem_scans.csv`. A spectrum with
recognizable features does not establish that a physically damaged filter is a
valid mass or optical comparator.

## Repeatability is generally good in shape, with qualifications

There are 22 physical filters with repeated scans: 20 PM2.5 filters and two
field blanks. The triplicate ETAD-0070 contributes three pairwise comparisons,
so there are 24 pairs; these are not 24 independent filters.

Aside from ETAD-0139, all pairwise full-grid raw correlations are at least
0.995584, reproducing the previous review. Median absolute RMS difference in
the modeled region is 0.003306 for raw spectra and 0.000291 after correction
over the 23 other pairs. Relative RMS differences use different signal scales:
their medians are 1.94% for raw, 3.52% for corrected, and 9.65% for derivatives.
The derivative calculation uses a 15-point cubic Savitzky–Golay filter,
converted from channel spacing to wavenumber units.

High correlation measures shape agreement and can coexist with amplitude
differences. Baseline subtraction reduces much of the absolute repeat
difference, but neither this nor high correlation establishes external accuracy.

## Field blanks reveal a signed residual that clipping would hide

After averaging repeated scans by physical filter, all **37 field blanks** have
negative corrected absorbance at 1617.369 cm⁻¹. The median is −0.0003715,
compared with +0.006153 for the 253 PM2.5 filters. The median blank corrected
RMS is 0.000551; the median fraction of negative points across the analyzed
region is 68.43%. For PM2.5 it is 14.44%.

| Blank lot | Physical filters | Median corrected 1617 absorbance | Median corrected RMS |
|---|---:|---:|---:|
| 248 | 5 | −0.000252 | 0.000661 |
| 251 | 23 | −0.000372 | 0.000545 |
| Unknown | 9 | −0.000423 | 0.000551 |

The pattern is not confined to one known lot. It is a reason to examine
background references, blank matching, and baseline residual shape. Field
blanks may carry handling material, and noise can be negative; this result
does not certify zero analyte or prove that the EC result is biased. Preserve
sign and scale when evaluating these traces. `signed_median_spectra.csv`
contains median and quartile curves for raw and corrected PM2.5, PM10 and
blank spectra.

The structure is already present before spline correction: all 37 physical
blanks have negative raw 1617 absorbance and negative raw local 1600–1635
contrasts. Therefore the negative corrected residual cannot, by itself, be
attributed to something introduced by the spline.

## Baseline sensitivity and spectral location

Four prespecified spline combinations were run on all scans: `(6,4)`, `(8,4)`,
`(6,3)` and `(6,6)`. No combination was selected by its HIPS agreement.

Changing **df1 from 6 to 8 leaves the native 1617 point exactly unchanged**:
that parameter controls the 4000–1820 cm⁻¹ segment. Testing the 1617 region
requires changing segment 2 or its boundary assumptions. Relative to `(6,4)`,
the median within-filter 1617 changes for PM2.5 are −0.000178 at `(6,3)` and
−0.000385 at `(6,6)`. Individual changes are heterogeneous, reaching +0.004776
at `(6,6)`. Parameter sensitivity is not an uncertainty interval or proof that
one result is more accurate.

The strongest measured-gap association concerns the broad component removed at
1617, not a unique chemical peak. For the 189 paired filters, the target below
is explicitly **HIPS/MAC minus frozen unified FTIR EC**. Rank adjustment includes HIPS
concentration, collected mass, sampled volume, month harmonics, elapsed time
and lot.

| Spectral measure | Spearman r² before adjustment | Partial-rank r | Partial-rank r² |
|---|---:|---:|---:|
| Raw 1617 absorbance | 0.700 | −0.855 | 0.731 |
| Removed component at 1617 | 0.679 | −0.857 | 0.734 |
| Corrected 1617 absorbance | 0.577 | −0.281 | 0.079 |
| Raw narrow local 1600–1635 contrast | 0.062 | +0.332 | 0.110 |
| Corrected narrow local contrast | 0.066 | +0.348 | 0.121 |
| Local mean second derivative | 0.112 | −0.510 | 0.260 |

These measures answer different questions. Their different directions after
adjustment demonstrate why a single raw-channel association cannot establish
an interfering compound. The shared FTIR term in the target also creates a
direct mathematical dependence. Results locate informative components for
investigation, not causal artifacts or independently measured EC errors.

The table above and `native_feature_associations.csv` use the **frozen
189-pair** cohort and **HIPS minus FTIR** sign. The companion
`current_chemistry_native_feature_associations.csv` deliberately uses **FTIR
minus HIPS**, matching the portable screen, and labels the 246 combined-HIPS
and 232 direct-HIPS cohorts separately. It adds optical-source indicators to
the adjustment for the combined cohort. Opposite signs across these files are
an explicit convention, not a contradictory observation.

| Native measure, current products | Combined HIPS n=246, partial-rank r² | Direct HIPS n=232, partial-rank r² |
|---|---:|---:|
| Raw 1617 absorbance | 0.593 | 0.622 |
| Removed component at 1617 | 0.599 | 0.624 |
| Corrected 1617 absorbance | 0.089 | 0.071 |
| Local mean second derivative | 0.198 | 0.209 |
| Segment 1 parameter sensitivity | 0.026 | 0.031 |
| Segment 2 parameter sensitivity | 0.00011 | 0.00098 |

The removed component remains associated with the current measured gap, but
filters whose spectra are most sensitive to the tested spline stiffnesses are
not strongly associated with that gap after the recorded adjustments. This
weakens a simple explanation in which choosing a different stiffness alone
solves the Addis discrepancy. It does not rule out other baseline/reference
errors or establish that the removed component is entirely physical aerosol.

### A within-Addis comparison illustrates the limitation of one channel

| Metric | ETAD-0202, 27 Apr 2024 | ETAD-0221, 18 Jun 2024 |
|---|---:|---:|
| Lot | 251 | 251 |
| Collected mass, µg | 260.4 | 377.9 |
| Frozen unified FTIR EC, µg/m³ | 6.854 | 11.735 |
| HIPS/MAC, µg/m³ | 6.828 | 6.891 |
| Raw 1617 absorbance | 0.058758 | 0.123795 |
| Corrected 1617 absorbance | 0.013291 | 0.013294 |
| Removed component at 1617 | 0.045466 | 0.110501 |
| Corrected narrow 1600–1635 contrast | 0.005415 | 0.002745 |
| Corrected C–H local contrast | 0.007105 | 0.011292 |

These are useful optical-matched, same-lot comparison candidates, **not a
mass-matched experiment**. Their nearly identical corrected point values
coexist with different local shape, organic-band signal, collected loading and
frozen unified EC estimates. Neither point height nor a uniform site offset can
capture all of these differences.

## What this evidence supports doing next

1. Review the native sample/background identities for ETAD-0037 and ETAD-0139;
   document a scan-level decision and regenerate affected averages consistently.
2. Resolve the documented ETAD-0158 damage flag through the shared exclusion
   process. Keep original observations recoverable.
3. Retain signed blank traces and compare matched-background/lot alternatives
   against repeatability and held-out blank or known-addition tests. Do not
   optimize the baseline against HIPS.
4. Examine the broad removed component alongside narrow contrasts and
   derivatives; a spectral novelty score alone cannot separate scattering,
   substrate effects and real broad aerosol absorption.
5. For quantitative changes, use identically processed calibration and test
   spectra with independent reference evaluation. The four native files do not
   include acquisition instrument IDs, scan times, background IDs, or an
   independent EC reference for these Addis PTFE filters.

The separate multisite screen should determine whether comparable spectral
regimes also occur at Bishoftu or other sites. The Addis numerical reproduction
and native blank diagnostics do not establish that those sites share the same
baseline behavior.
