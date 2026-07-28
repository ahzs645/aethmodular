"""Notebook bootstrap helpers for stable paths and plotting defaults."""

from pathlib import Path
from types import SimpleNamespace

try:
    from config import DATA_ROOT
    from plotting import apply_default_style
    from prep import find_repo_root, output_dirs
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.*
    from .config import DATA_ROOT
    from .plotting import apply_default_style
    from .prep import find_repo_root, output_dirs


def bootstrap(output_slug, subdirs=("plots", "tables"), style=True) -> SimpleNamespace:
    """Prepare absolute research paths and notebook output directories."""
    repo_root = find_repo_root()
    data_root = Path(DATA_ROOT).expanduser().resolve()
    # Resolve from this file rather than rebuilding the path from repo_root, so
    # the package keeps working if it is ever relocated.
    scripts_dir = Path(__file__).resolve().parent
    output_root = (data_root / "output").resolve()
    directories = output_dirs(output_slug, subdirs=subdirs, data_root=data_root)

    if style:
        apply_default_style()

    values = {
        "repo_root": repo_root.resolve(),
        "data_root": data_root,
        "scripts_dir": scripts_dir,
        "output_root": output_root,
        "plots_dir": directories.get("plots", (output_root / "plots" / output_slug).resolve()),
        "tables_dir": directories.get(
            "tables", (output_root / "tables" / output_slug).resolve()
        ),
    }
    values.update({f"{name}_dir": path for name, path in directories.items()})
    return SimpleNamespace(**values)
