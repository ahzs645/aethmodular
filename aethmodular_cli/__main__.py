"""Allow ``python -m aethmodular_cli`` to behave like the ``aeth`` command."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
