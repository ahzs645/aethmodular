"""Repository and data-root paths shared by every script under ``scripts/``.

Previously ``REPO_ROOT = Path(__file__).resolve().parents[2]`` and the
``AETHMODULAR_DATA_ROOT`` lookup were copy-pasted into each pipeline and
diagnostic.
"""

from __future__ import annotations

import os
from pathlib import Path

# scripts/common/paths.py -> scripts/common -> scripts -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_ROOT = REPO_ROOT / "research" / "ftir_hips_chem"


def data_root() -> Path:
    """Resolve the data root: ``AETHMODULAR_DATA_ROOT`` if set, else the in-repo default.

    Normalization follows research/ftir_hips_chem/scripts/config.py
    (``.expanduser().resolve()``), which is also what
    ``src.config.project_paths.get_data_root`` does.

    NOTE: this FIXES a latent bug. The scripts that inlined this used a bare
    ``Path(os.environ.get("AETHMODULAR_DATA_ROOT", <default>))`` with no
    normalization, so ``AETHMODULAR_DATA_ROOT=~/data`` produced a path with a
    literal "~" component that never existed and every downstream file check
    silently failed. ``.expanduser()`` makes the documented override work.
    The default branch is unaffected: ``REPO_ROOT`` is already resolved, so
    ``.expanduser().resolve()`` is a no-op on it.
    """
    return Path(
        os.environ.get("AETHMODULAR_DATA_ROOT", DEFAULT_DATA_ROOT)
    ).expanduser().resolve()
