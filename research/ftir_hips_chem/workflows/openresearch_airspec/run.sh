#!/usr/bin/env bash
set -euo pipefail
# Fixed execution environment, inherited unchanged by future experiment children.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
uv run --python 3.13.9 --locked --no-default-groups --group notebooks python -u research/ftir_hips_chem/workflows/openresearch_airspec/run.py
