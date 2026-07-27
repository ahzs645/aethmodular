"""Import smoke tests for canonical package entry points.

These absorb the only unique coverage that `scripts/diagnostics/test_system.py`
provided before it was retired on 2026-07-26: that every canonical subpackage
imports, and that the main analyzer classes actually construct. The rest of that
374-line script asserted a 2025-era filesystem layout (a `setup.py` this repo
deliberately does not have, plus gitignored `data/` and `outputs/` directories),
so it exited 1 permanently and was never a usable signal.

Unlike the old script these are real assertions, and they run under
`aeth check`.
"""

import importlib

import pytest


# Live library surface. Adding a module here is the cheap way to keep a
# subpackage from silently rotting.
SRC_MODULES = [
    "src",
    "src.core.base",
    "src.core.exceptions",
    "src.config.plotting",
    "src.config.project_paths",
    "src.config.multi_site_seasons",
    "src.data.loaders",
    "src.data.loaders.aethalometer",
    "src.data.processors.validation",
    "src.data.qc",
    "src.analysis.bc.black_carbon_analyzer",
    "src.analysis.bc.source_apportionment",
]

# Retained-but-unused code. It is expected to keep working; see attic/README.md.
ATTIC_MODULES = [
    "attic.analysis.advanced.statistical_analysis",
    "attic.analysis.aethalometer.smoothening",
    "attic.core.monitoring",
    "attic.core.parallel_processing",
    "attic.utils",
    "attic.utils.file_io",
    "attic.utils.memory_optimization",
]


@pytest.mark.parametrize("module_name", SRC_MODULES)
def test_src_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("module_name", ATTIC_MODULES)
def test_attic_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_src_data_exposes_its_public_api():
    """Guards the bare `except ImportError` in src/data/__init__.py.

    An undeclared `tqdm` once made this silently evaluate to an empty list,
    hiding the fact that the whole subpackage was unimportable.
    """
    import src.data

    assert src.data.__all__, "src.data.__all__ is empty - an import is being swallowed"


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("src.analysis.bc.black_carbon_analyzer", "BlackCarbonAnalyzer"),
        ("src.analysis.bc.source_apportionment", "SourceApportionmentAnalyzer"),
    ],
)
def test_analyzer_classes_construct(module_name, class_name):
    cls = getattr(importlib.import_module(module_name), class_name)
    assert cls() is not None
