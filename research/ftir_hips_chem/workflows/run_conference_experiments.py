"""Run the conference experiment manifest, or a clearly synthetic demonstration.

    uv run python research/ftir_hips_chem/workflows/run_conference_experiments.py \
        --config /path/to/manifest.json --output research/ftir_hips_chem/output/tables/conference/run1

No data is downloaded, published or sent to collaborators by this workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))

from aethmodular_cli.env import load_repo_env
from conference_experiments.io import new_output_directory
from conference_experiments.runner import run_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Explicit inputs, units, evidence and experiment settings")
    source.add_argument("--demo", action="store_true", help="Generate artificial inputs and exercise all experiment families")
    parser.add_argument("--output", type=Path, required=True, help="A new or empty run directory")
    args = parser.parse_args(argv)
    load_repo_env()
    try:
        if args.demo:
            from conference_experiments.demo import create_demo_inputs
            new_output_directory(args.output)
            manifest = create_demo_inputs(args.output / "synthetic_inputs")
        else:
            manifest = args.config
        receipt = run_manifest(manifest, args.output, output_prepared=args.demo)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(2, f"Input error: {exc}\n")
    print(f"Report: {args.output / 'report.md'}")
    print(f"Run status: {receipt['status']}; scope: {receipt['data_scope']}")
    # Incomplete is distinct from an implementation failure and from completion.
    return {"completed": 0, "failed": 1, "incomplete": 3}[receipt["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
