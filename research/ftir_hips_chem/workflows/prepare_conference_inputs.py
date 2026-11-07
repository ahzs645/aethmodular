"""Prepare traceable source-calibration or tracked-filter experiment manifests.

This performs local input preparation, not model fitting or data acquisition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))

from aethmodular_cli.env import load_repo_env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--tracked", action="store_true", help="Tracked Addis chemistry, direct HIPS and daily weather")
    inputs.add_argument("--source-dir", type=Path, help="Reviewed historical FTIR archive root containing apps/ and notebook/")
    parser.add_argument("--output", type=Path, required=True, help="A new or empty input directory")
    args = parser.parse_args(argv)
    load_repo_env()
    try:
        if args.tracked:
            from conference_experiments.tracked_inputs import prepare_tracked_inputs
            manifest = prepare_tracked_inputs(ROOT, args.output)
        else:
            from conference_experiments.preparation import prepare_source_inputs
            manifest = prepare_source_inputs(args.source_dir, args.output)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(2, f"Input error: {exc}\n")
    print(f"Prepared manifest: {manifest}")
    print("Review the manifest before running run_conference_experiments.py --config with a new output directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
