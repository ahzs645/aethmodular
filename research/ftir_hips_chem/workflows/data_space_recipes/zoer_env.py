"""Zoer server address and Drive archive location for the data-space recipes.

Both are machine-specific, so they come from the gitignored repo-root ``.env``
(see ``.env.example``) rather than being spelled out in each recipe:

* ``ZOER_URL`` -- the Zoer server root, e.g. ``https://zoer.example.org``.
* The Davis Drive archive resolves through ``data_paths.maia_data_root()``,
  i.e. ``AETHMODULAR_MAIA_DATA_ROOT`` / ``AETHMODULAR_DRIVE_ROOT`` or discovery.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
for _path in (_REPO, _REPO / "research/ftir_hips_chem/scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from aethmodular_cli.env import load_repo_env  # noqa: E402

load_repo_env()


def zoer_url() -> str:
    """Return the Zoer server root without a trailing slash."""
    url = os.environ.get("ZOER_URL", "").strip()
    if not url:
        raise SystemExit("ZOER_URL is not set. Add it to the repo-root .env (see .env.example).")
    return url.rstrip("/")


def zoer_api() -> str:
    """Return the Zoer REST API root."""
    return zoer_url() + "/api"


def davis_data_root() -> Path:
    """Return the Drive ``Davis Data`` directory holding the raw archives."""
    from data_paths import maia_data_root

    return maia_data_root()
