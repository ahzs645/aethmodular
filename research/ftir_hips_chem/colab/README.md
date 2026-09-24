# Large VIBES / AIRSpec comparison in Google Colab

Two files are needed:

- `VIBES_AIRSpec_Large_Run_Colab.ipynb` (this directory).
- `../output/tables/vibes_colab_bundle/vibes_large_run_bundle.zip` (~212 MB, generated locally).

Upload the ZIP to **My Drive / Aethmodular VIBES/**. Upload/open the notebook in
Colab, choose a CPU runtime, and run the cells in order. The default runs a small
installation check followed by the **full** experiment. The only interactive
cloud step is Google's Drive authorization. The bundle has not been uploaded by
these builders.

The analysis uses an isolated Python virtual environment with pinned dependencies,
so it does not replace Colab's NumPy or TensorFlow packages. Data/code are extracted
on the runtime's local disk; outputs and completed batch checkpoints are saved
under **My Drive / Aethmodular VIBES / results/**.

Profiles:

| Profile | Calibration filters | Addis targets | Purpose |
|---|---:|---:|---|
| `smoke` | 64 | 12 | Test installation, pipeline and report; not scientific conclusions |
| `pilot` | 762 historical-cohort members + 400 others | 253 | Intermediate run |
| `full` | 12,393 | 253 | Large comparison on all audited eligible rows |

Each profile also includes held-out blanks and known-peak additions. The source
library contains 13,634 scans / 13,632 physical filters. The full inclusion ledger
records reasons for ineligibility, including unverified filter purpose and absent
positive TOR labels. The historical 800 list has 762 eligible members here; it is
never silently called an intact 800-filter fit.

## What is compared

- AIRSpec DF1=6/DF2=4 versus blank-trained VIBES PB, tau=0.1.
- Raw replicates averaged identically before each baseline; same 2,002 channels.
- Independent blank bank: training-site IMPROVE blanks (25% reserved first;
  training capped at 50 per lot) plus 28 independent Addis training blanks.
  This default yields 87 training blanks and blank-only LOO rank 28 (cap 30).
- One frozen full-pool, site-disjoint 80/20 split shared by methods and cohorts.
  This differs from older cohort-specific splits and is labeled accordingly.
- Separate PLS refits with site-grouped training-only CV and the existing
  first-major-minimum selection rule; identical paired-valid train/test rows.
- Held-out predictive R², RMSE, MAE, bias, paired site-bootstrap ΔRMSE intervals,
  Addis transfer, band features, blank residuals, injected-peak recovery and timing.
- Eight figure families, portable PLS coefficients, per-fit diagnostics, inclusion
  ledgers, package versions and content hashes.

The historical OC/EC-defined cohort is a fixed restricted-domain study; its labels
were used historically in cohort construction. Addis HIPS is an optical comparator,
not independent thermal EC truth. The notebook does not select tau, a blank bank,
or a model based on favorable held-out results.

## Resume and output files

After Colab resets, rerun the same notebook with the same bundle and settings.
Completed `.npz` batches are reused after signature and row-order validation.
Checkpoint identity covers input/code hashes, package versions, and scientific
settings. Changing worker count is allowed; changing batch size or scientific
settings starts another run. Only the unfinished batch is lost on interruption.
Background PCA and final PLS/report generation are rerun.

Each run folder contains the corrected arrays, checkpoints, metadata, score tables,
predictions, selected component curves, coefficients and plots. The notebook also
creates `comparison_report.zip`, which contains tables, figures and models but
omits large corrected arrays and checkpoints.

## Rebuild locally

```sh
uv run --extra vibes python research/ftir_hips_chem/workflows/build_vibes_colab_bundle.py
uv run --extra vibes python research/ftir_hips_chem/workflows/build_vibes_colab_notebook.py
```

For a code-only update after staging the raw arrays, add `--code-only` to the first
command. Always rebuild the notebook afterward: it pins the archive checksum.

Test the exact notebook locally by setting `AETH_VIBES_LOCAL_BUNDLE` to the ZIP,
`AETH_VIBES_LOCAL_WORK` to a disposable working directory, and
`AETH_VIBES_PROFILE=smoke`, then execute with nbclient. It will extract the archive,
create the isolated environment, and run all cells without Google authentication.

```sh
uv run --extra vibes pytest tests/test_vibes_large_run.py tests/test_vibes_baseline.py \
  research/ftir_ec_phase3/scripts/test_airspec_baseline.py
```

Colab's runtime lifetime and resources are variable. Checkpointing and local reads
follow Google's [Colab guidance](https://research.google.com/colaboratory/faq.html).
No full Colab cloud run is claimed by local validation.

## CLI execution

The installed `google-colab-cli` can execute this notebook without an interactive
Drive mount. Use a named CPU session and the existing ADC login:

```sh
colab --auth=adc sessions
colab --auth=adc new -s aeth-vibes-full
```

Upload the bundle to `/content/vibes_large_run_bundle.zip`, this notebook to
`/content/VIBES_AIRSpec_Large_Run_Colab.ipynb`, and
`../workflows/run_vibes_colab_notebook.py` to `/content/run_vibes_colab_notebook.py`.
Large uploads may need splitting into 8 MiB parts, reconstructing on the VM, and
verifying the notebook's pinned SHA256 before execution.

Start the runner in a detached subprocess through `colab exec`. It uses nbclient
to execute every cell of the same notebook, with `PROFILE=full` and the smoke
check enabled. It writes an executed notebook, log and `execution_status.json`
under `/content/aeth_vibes/`. The environment override skips Drive authorization;
**the VM's `persistent_results` folder is ephemeral until downloaded**.

Keep the local checkpoint monitor running:

```sh
uv run python research/ftir_hips_chem/workflows/monitor_vibes_colab.py \
  --session aeth-vibes-full \
  --output research/ftir_hips_chem/output/tables/vibes_colab_cloud
```

This downloads completed batches while the notebook runs. At completion it also
downloads corrected arrays, tables, figures, models and the executed notebook,
then stops the Colab runtime. Transfers use checksum-verified 8 MiB chunks. If the
notebook fails, it downloads diagnostics and retains the runtime for recovery.
The monitor can be restarted against the same session and output directory; its
local mirror ledger avoids downloading unchanged checkpoints again. The Mac must
remain awake and connected for this monitor to work.

For CLI runs, `CLOUD_RUN.json` in the output directory records the session name
and exact runtime `endpoint`. The monitor uses the CLI's installed Python and
`refresh_vibes_colab_session.py` to refresh proxy credentials every 20 minutes.
Those credentials expire after one hour; the installed CLI otherwise mistakes
their expiry for a terminated runtime. Recovery only reconnects the recorded
endpoint and never allocates a replacement VM. `monitor_status.json` includes
`checked_utc` and `monitor_connection` so stale observations remain distinguishable
from a live connection.

Read `output/tables/vibes_colab_cloud/monitor_status.json` for the last mirrored
status. `colab --auth=adc url -s aeth-vibes-full` opens the existing runtime;
`colab --auth=adc stop -s aeth-vibes-full` releases it manually if needed.

## Validation completed

The exact notebook was executed locally with the smoke profile in a newly created,
isolated virtual environment: 106 evaluation spectra, zero failed fits, all eight
figures embedded. A second invocation reused all 106 checkpointed corrections and
reproduced the calibration tables. Sixteen focused tests passed, including upstream
numerical parity, blank/site isolation, stale-checkpoint rejection, and paired
bootstrap checks. Negative predictions are retained and remain visible in plots.
The Colab CLI smoke run also passed on 2026-09-21 UTC: 106 cases, zero failed
fits, 10 logged optimizer retries, and eight figures. The full cloud run completed
at 09:45:48 UTC that day: 12,808 cases, zero failed fits, 451 fits retried, and
eight figures embedded in an error-free executed notebook. All results downloaded
successfully and the runtime was stopped. Corrected arrays are finite with shape
12,808 × 2,002, and the downloaded report ZIP passed its integrity check.

Full-pool held-out RMSE was 3.224 µg/filter for AIRSpec versus 3.296 for VIBES
(predictive R² 0.679 versus 0.664; 2,327 held-out filters). In the 762-filter
historical cohort, held-out RMSE was 2.548 versus 2.420 (137 held-out filters).
The paired site-bootstrap 95% intervals for VIBES-minus-AIRSpec RMSE include zero
in both cohorts: full pool −0.077 to 0.262; historical cohort −0.930 to 0.457
µg/filter. These results do not establish a clear performance advantage for VIBES.

Full executed notebook:
`../notebooks/archive/executed/VIBES_AIRSpec_Colab_full_executed.ipynb`.
Full results and report ZIP:
`../output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09/`.
Completion record: `../output/tables/vibes_colab_cloud/COMPLETION.json`.

Executed preview: `../notebooks/archive/executed/VIBES_AIRSpec_Colab_smoke_executed.ipynb`.
Local validation record: `../output/tables/vibes_colab_validation/VALIDATION.json`.
