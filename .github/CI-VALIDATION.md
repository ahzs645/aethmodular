# Validation scope and local commands

Focused PMF checks and the native Windows probe gain separate stale-run cancellation and shared dependency invalidation. The native engine checksum and independent verification remain unchanged. Full scientific fits and experiment grids were not an every-PR workflow in this main snapshot; keep them deliberate with recorded source/data/configuration rather than adding a new automatic sweep.

## Local entry points

```sh
python -m pip install numpy scipy pytest
python -m pytest tests/test_pmf_runner.py tests/test_pmf_intake.py -q
# Native EPA PMF5: run the existing Windows workflow with its checksum-pinned official installer.
# Research grids need their own recorded data, seed and configuration.
```

Feature branches without an open PR use local commands. A local pass does not replace validation of the submitted revision. Main/release publication keeps the workflow's existing authorization and artifact conditions.

CI optimization batch: November 7, 2026. Workflow syntax and focused orchestration/selection checks are separate from complete application, native-platform and research validation.
