"""Resolve external dataset locations without hardcoding a machine or account.

Every notebook that reads raw data used to spell out a full home-directory
path naming both a user and a Google account
(``~<name>/Library/CloudStorage/GoogleDrive-<account>/My Drive/...``).
That is unrunnable for anyone else and breaks on this machine too if the Drive
account changes. The helpers here resolve the same directories at runtime.

Resolution order is the same for every entry point, and is the one already used
by :func:`pls_transfer.drive_root`:

1. an environment variable, so a different layout can be pointed at directly;
2. a configured local directory, when it exists and is non-empty;
3. discovery of the Google Drive mount signed in on this machine.

Nothing here touches the network or creates directories. Use
:func:`describe` to see what actually resolved before debugging a load error.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from config import DATA_ROOT, WEATHER_DATA_DIR
    from pls_transfer import (
        FTIR_DIR_CANDIDATES,
        MAIA_DATA_CANDIDATES,
        drive_root,
        first_existing,
    )
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.*
    from .config import DATA_ROOT, WEATHER_DATA_DIR
    from .pls_transfer import (
        FTIR_DIR_CANDIDATES,
        MAIA_DATA_CANDIDATES,
        drive_root,
        first_existing,
    )


# ``MAIA_DATA_CANDIDATES`` and ``FTIR_DIR_CANDIDATES`` are the "My Drive"-relative
# layouts, defined once in ``pls_transfer`` and imported here so both modules
# resolve the same directories. Each is a newest-first tuple probed at call time,
# because the tree has been reorganised more than once.

# Subdirectories of the MAIA data root, by the names they actually have on the
# mount. "Aethelometry" is spelled that way on Drive; do not silently correct it.
AETHALOMETRY_SUBDIR = "Aethelometry Data"
WEATHER_SUBDIR = "Weather Data"
METEOSTAT_SUBDIR = "Meteostat"
AERONET_SUBDIR = "AERONET"
IMPROVE_SUBDIR = "Improve"
ETAD_SUBDIR = "DAVIS/ETAD FTIR"

LOCAL_DB_SUBDIR = "local_db"


def ftir_spectra_dir() -> Path:
    """Return the directory holding FTIR spectra exports and calibration tables."""
    env = os.environ.get("AETHMODULAR_FTIR_SPECTRA_DIR")
    if env:
        return Path(env).expanduser()
    root = drive_root()
    return first_existing([root / rel for rel in FTIR_DIR_CANDIDATES])


def maia_data_root() -> Path:
    """Return the directory on Drive holding the raw datasets.

    Named for the NASA MAIA tree it originally lived in; it has since moved, so
    every known layout is probed rather than one being assumed.
    """
    env = os.environ.get("AETHMODULAR_MAIA_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    root = drive_root()
    return first_existing([root / rel for rel in MAIA_DATA_CANDIDATES])


def aethalometry_dir() -> Path:
    """Return the directory holding raw aethalometer exports."""
    env = os.environ.get("AETHMODULAR_AETHALOMETRY_DIR")
    if env:
        return Path(env).expanduser()
    return maia_data_root() / AETHALOMETRY_SUBDIR


def etad_dir() -> Path:
    """Return the directory holding the Addis (ETAD) FTIR spectra and metadata.

    The same directory ``pls_transfer.FTIRTransferPaths.etad_dir`` resolves;
    exposed here so a caller that only wants a path need not build the whole
    dataclass.
    """
    env = os.environ.get("AETHMODULAR_ETAD_DIR")
    if env:
        return Path(env).expanduser()
    return maia_data_root() / ETAD_SUBDIR


def weather_dir() -> Path:
    """Return the weather-data directory, preferring the in-repo copy.

    Unlike the other datasets this one is small enough that a copy lives in the
    repo at ``config.WEATHER_DATA_DIR``. That copy wins when present so a
    checkout is self-sufficient; the Drive original is the fallback.

    The two locations do **not** hold the same files -- the repo copy has the
    Meteostat master/daily-average exports, the Drive copy has the
    ``addis_ababa_weather_data*`` series. Use :func:`weather_file` when you want
    a specific file; this function alone will happily return a directory that
    does not contain it.
    """
    env = os.environ.get("AETHMODULAR_WEATHER_DIR")
    if env:
        return Path(env).expanduser()

    local = Path(WEATHER_DATA_DIR)
    if local.is_dir() and any(local.iterdir()):
        return local

    return maia_data_root() / WEATHER_SUBDIR


def weather_file(*names, subdir=METEOSTAT_SUBDIR) -> Path:
    """Return the first of ``names`` found in either weather location.

    Searches the in-repo copy and the Drive copy, because they hold different
    files. Accepts several candidate names so a caller can tolerate a rename.

    Raises
    ------
    FileNotFoundError
        If none of ``names`` exists in any searched location. The message lists
        every location tried, since "which copy am I reading" is the usual
        question when this fails.
    """
    if not names:
        raise ValueError("weather_file() requires at least one filename")

    roots = []
    env = os.environ.get("AETHMODULAR_WEATHER_DIR")
    if env:
        roots.append(Path(env).expanduser())
    roots.append(Path(WEATHER_DATA_DIR))
    roots.append(maia_data_root() / WEATHER_SUBDIR)

    searched = []
    for root in roots:
        base = root / subdir if subdir else root
        for name in names:
            candidate = base / name
            searched.append(candidate)
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "No weather file found. Tried:\n  "
        + "\n  ".join(str(s) for s in searched)
        + "\nSet AETHMODULAR_WEATHER_DIR to point at the directory holding it."
    )


def ftir_local_db() -> Path:
    """Return the FTIR ``local_db`` directory holding calibration tables.

    Searches both known layouts. May still be absent on a machine without the
    mount, so check :meth:`Path.is_dir` and degrade gracefully rather than
    assuming it resolved to something real.
    """
    env = os.environ.get("AETHMODULAR_FTIR_LOCAL_DB")
    if env:
        return Path(env).expanduser()
    root = drive_root()
    return first_existing(
        [root / rel / LOCAL_DB_SUBDIR for rel in FTIR_DIR_CANDIDATES]
    )


def describe() -> dict[str, tuple[Path, bool]]:
    """Return ``{name: (resolved_path, exists)}`` for every entry point.

    Intended for a notebook to print once at setup, so a later load failure is
    obviously a missing-mount problem rather than a logic bug.
    """
    entries = {
        "repo_data_root": Path(DATA_ROOT),
        "drive_root": drive_root(),
        "maia_data_root": maia_data_root(),
        "aethalometry_dir": aethalometry_dir(),
        "etad_dir": etad_dir(),
        "weather_dir": weather_dir(),
        "ftir_spectra_dir": ftir_spectra_dir(),
        "ftir_local_db": ftir_local_db(),
    }
    return {name: (path, path.is_dir()) for name, path in entries.items()}
