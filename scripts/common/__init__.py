"""Shared helpers for the standalone scripts under ``scripts/``.

``scripts/`` is deliberately not an installed package: ``aethmodular_cli.cli``
launches each script as ``[sys.executable, str(path), ...]``, which puts the
*script's own* directory on ``sys.path`` rather than the repository root. Each
script therefore prepends ``scripts/`` to ``sys.path`` before importing from
``common`` -- see the header of any file under ``scripts/pipelines/``.
"""
