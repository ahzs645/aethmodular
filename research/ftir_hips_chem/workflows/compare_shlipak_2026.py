"""CLI for the named Shlipak et al. comparison; see docs/aethalometer-paper-comparison.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from paper_comparison import compare_files, load_profile  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Print the study profile without loading observations.")
    parser.add_argument("--hourly", type=Path, help="QC-audited hourly CSV; timestamps must carry explicit offsets.")
    parser.add_argument("--metadata", type=Path, help="Deployment identity, units and upstream QC provenance JSON.")
    parser.add_argument("--output", type=Path, help="New JSON report destination; existing reports are not overwritten.")
    args = parser.parse_args(argv)
    if args.describe:
        if any((args.hourly, args.metadata, args.output)):
            parser.error("--describe cannot be combined with comparison inputs")
        print(json.dumps(load_profile(), indent=2, ensure_ascii=False))
        return 0
    if not all((args.hourly, args.metadata, args.output)):
        parser.error("provide --hourly, --metadata and --output, or use --describe")
    try:
        report = compare_files(args.hourly, args.metadata)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(serialized)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"comparison failed: {exc}\n")
    print(f"Wrote {args.output}; descriptive comparison, exact paper reproduction remains incomplete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
