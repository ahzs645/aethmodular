# PMF data, snapshot and engine contracts

## Native record input

`prepare` consumes separate CSVs, with string IDs retained:

* Filters: `MediaId,SiteId,ExternalFilterId,ExternalFilterType,ExternalLotId,SamplingStartDate,SampleVolume_m3`.
* Analyses: `Id,MediaId,AnalysisQcCode,SampleScanId,BackgroundScanId,InstrumentId,AnalysisDate` (other fields retained when available).
* Scans: `Id,DataPointFormat,NumberOfDataPoints,FrequencyOfFirstPoint,FrequencyOfLastPoint,YScalingFactor,XUnits,Type,Data`.

The native adapter validates the export actually supplied. Binary payloads are
base64-encoded little-endian float32 intensities with explicit scale and linear
wavenumber coordinates. Unsupported formats/units are held or rejected; they are
not guessed. A background grid may be interpolated only within its actual coverage.
Nonpositive intensity invalidates conversion rather than producing NaN absorbance.

The retained QC code defaults to the recovered code `1`. Other codes are not
reinterpreted as usable observations. Multiple retained analyses produce an
ambiguity hold. A different laboratory convention must be explicitly configured
and documented. Do not infer quality from the magnitude of a fitted contribution.

## Grid and uncertainty

An explicit target grid is recommended for stable longitudinal model versions.
A grid change is a new preparation protocol, and the frozen-profile projector
rejects incompatible columns. Without a configured grid, the first selected
native grid is limited to common coverage and its exact coordinates are recorded.
Different sites are not concatenated or harmonized implicitly.

The default model windows are 4000–2500 and 2300–1520 cm⁻¹. AIRSpec baseline settings
are explicit. The new ETAD raw overlap run excludes the old upper endpoint outside
native coverage, giving 1,771 modeled channels, not the old 1,772. It does not
recreate the unrecovered historical 1,656-channel mask. The endpoint is removed
by a coverage rule, not to improve fit statistics.

Default X = selected baseline-corrected absorbance / measured air volume.
Default U = sample SD (ddof=1) of matching independent physical field blanks /
measured air volume. Blank-subtracted sensitivity uses mean subtraction and
SD × sqrt(1+1/n_blanks) / volume. This is an exploratory noise proxy, not a complete
analytical error budget. Matching lot/instrument/batch criteria and the actual
blank IDs/counts are recorded. No uncertainty is manufactured from a single blank.
An invalid group holds its samples without aborting unrelated eligible groups.

## Snapshot identity and reproducibility

`dataset.npz` has no pickled objects. It contains X, U, exact ordered sample IDs,
features and string metadata. Its JSON sidecar records the schema, metadata,
protocol, original input hashes, content identity and binary checksum. Identity
includes numerical data, IDs, protocol, metadata **and source hashes**.
Compatibility includes the feature vector and protocol, allowing new observations
from new source files only when their preparation conventions actually match.

A run request additionally records settings, code hashes and numerical-library
versions. Per-start model files have their own checksums and recorded objectives,
which are re-evaluated on load. A changed value, source, feature order, code file,
seed list or setting invalidates strict resume. Original source files for the
runner are archived with the result. Existing output directories are not overwritten.

## Engine/objective distinctions

| Engine/mode | X | G / H constraints | Selection and qualification |
|---|---|---|---|
| Independent quadratic | Signed finite | G≥0, H≥0 | Weighted Q; objective and gradient convergence |
| Independent Huber | Signed finite | G≥0, H≥0 | Continuous Huber loss; independent from excluded-residual Q |
| Pinned ESAT LS | Nonnegative | G≥0, H≥0 | Actual original Python updates; native-controller and gradient checks |
| Pinned ESAT WS | Signed | G may be signed, H≥0 | Original failure retained when it fails; no substitution |
| Explicit local ESAT WS repair | Signed | G may be signed, H≥0 | Repaired updates under original controller, conspicuously non-stock |
| Native EPA ordinary / robust | Signed | Verified nonnegative G/H contract | Native Q_main and convergence log, independently verified Q_true |

All modes report meaningful actual status. `passed` refers to the stated operation:
export/source-validation success is not model execution. Failed and capped starts
remain separate. The per-run manifest says whether a native engine actually ran.
No adapter renames a Python result as EPA output.

## Output interpretation

Profiles and contributions retain explicit scaling. Factor matching is descriptive,
not proof of source identity. A strong same-FTIR total-OM regression does not prove
factor-specific mass allocation. Spectral amplitudes are not automatically µg/m³.
Blank/baseline issues, source-tracer evidence and mass calibration remain review
requirements even when every numerical start converges.
