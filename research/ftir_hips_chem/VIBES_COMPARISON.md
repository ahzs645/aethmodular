# VIBES versus AIRSpec on Addis FTIR spectra

Open `ftir_vibes_vs_airspec.ipynb` and run all cells with the repository environment:

```sh
uv sync --extra vibes
uv run aeth doctor
uv run --extra vibes pytest tests/test_vibes_baseline.py research/ftir_ec_phase3/scripts/test_airspec_baseline.py
```

The completed notebook with embedded figures is saved locally under
`notebooks/archive/executed/ftir_vibes_vs_airspec_executed.ipynb`. The active notebook
has no generated outputs. The notebook builder is
`workflows/build_vibes_comparison_notebook.py`.

The comparison uses identical replicate-averaged ETAD PM2.5 spectra and the same
1425–4000 cm⁻¹ analysis window. Canonical sample exclusion flags are applied and
all metadata rows appear in the inclusion ledger. Explicitly labeled field blanks
are split by physical filter within each known/unknown lot: 28 train, nine test.
PCA uses the pooled training blanks only, with upstream leave-one-out rank selection.
VIBES uses PB loss and fixed tau=0.1; AIRSpec uses DF1=6, DF2=4.

The adapter retains upstream mathematical functions and MAP solver. It additionally
checks array/grid alignment, prevents training/evaluation ID overlap, captures
warnings and optimizer status, and logs a single continuation for failed optimizations
(50,000 additional iterations, maxls=50). Remaining failures produce NaN outputs,
never an implicit fallback method. Use `retry_failed=False` to enforce strict
upstream optimizer settings.

Outputs in `output/tables/vibes_vs_airspec/` include the input/split ledger, PCA
rank curve, paired sample and band metrics, held-out blank residuals, known-addition
recovery, per-fit numerical diagnostics, runtime, corrected arrays and provenance.
Seven figure families are written to `output/plots/vibes_vs_airspec/`.

The VIBES source supplied at `~/Downloads/pyvibes-main` is copied
unchanged into `vendor/pyvibes/`, with its MIT license, citation and SHA-256 manifest.
It is not installed from PyPI (where the package name could be ambiguous), and no
runtime path depends on Downloads. The existing repository's NumPy/SciPy stack is
used with optional CVXPY; `uv.lock` records resolved versions. Numerical tests
compare the adapter directly with the supplied upstream VibeSpec implementation.

Interpretation: AIRSpec agreement is not ground truth. Held-out field blanks can
contain handling contamination. Known-addition recovery tests the difference
`correct(blank + peaks) - correct(blank)`, so it measures added-signal preservation
without assuming the original field blank is chemically empty. The three injection
amplitudes on each blank are repeated measures, not 27 independent filters.
Neither diagnostic validates EC predictions. Existing PLS calibrations are not
changed; a downstream EC comparison requires separately refitting each preprocessing
strategy on the same calibration pool and locked held-out TOR split.

## Larger calibration comparison

The larger Colab comparison completed on 2026-09-21: 12,808 processing cases,
zero failed fits. Its paired EC comparison and subsequent 12-case investigation
are summarized in the [current scientific synthesis](../../docs/current-research-summary.md).
The full-pool and restricted test populations must remain separate.

The portable, resumable workflow stages the full eligible IMPROVE library
and separately refits both calibration strategies. See
`colab/VIBES_AIRSpec_Large_Run_Colab.ipynb` and `colab/README.md` for the 212 MB bundle,
shared site split, training-only component selection, and checkpoint/restart steps.
