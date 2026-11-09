"""Compatibility entrypoint for version-independent PMF intake."""
import sys
from src.pmf.intake import evaluate


def main(argv=None):
    from src.pmf.cli import main as pmf_main
    return pmf_main(['intake', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    raise SystemExit(main())
