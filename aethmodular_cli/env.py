"""Load machine-specific settings from the repo-root ``.env`` file.

Every external location this repo reads (the Google Drive mount, IMPROVE and
AERONET archives, the OpenResearch store, the Zoer server) is resolved from an
``AETHMODULAR_*`` / ``ZOER_*`` environment variable, with discovery as the
fallback. Setting those in a shell profile is easy to forget and does not reach
Jupyter kernels, so this module reads them from ``<repo>/.env`` instead.

``.env`` is gitignored; ``.env.example`` lists every variable with a comment.
Copy it to ``.env`` and fill in only the lines you need.

Rules, kept deliberately small so there is no dependency on python-dotenv:

* ``KEY=VALUE`` per line; blank lines and ``#`` comments are ignored;
  an optional leading ``export`` is accepted so the file can also be sourced.
* Matching single or double quotes around a value are stripped.
* A variable already set in the real environment wins, so a one-off
  ``AETHMODULAR_DRIVE_ROOT=... uv run ...`` still overrides the file.
* ``~`` is *not* expanded here; the resolvers call ``Path.expanduser()``.

The resolvers call :func:`load_repo_env` themselves, so scripts and notebooks
need no setup. It is safe to call repeatedly.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"

_loaded: set[Path] = set()


def _parse(text: str) -> dict[str, str]:
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_repo_env(path: Path | None = None) -> dict[str, str]:
    """Copy ``.env`` values into ``os.environ`` without overriding set ones.

    Returns the values that were applied (empty when the file is absent).
    """
    path = Path(path) if path is not None else ENV_FILE
    if path in _loaded or not path.is_file():
        return {}
    _loaded.add(path)
    applied = {}
    for key, value in _parse(path.read_text(encoding="utf-8")).items():
        if key not in os.environ and value:
            os.environ[key] = value
            applied[key] = value
    return applied


def display_path(path: str | os.PathLike) -> str:
    """Return ``path`` relative to the repo, or ``~``-shortened, for records.

    Receipts and reports written into the repo should not carry a home
    directory. Paths inside the checkout become repo-relative; other paths
    under the home directory become ``~/...``.
    """
    p = Path(path).expanduser()
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return "~/" + p.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(p)
