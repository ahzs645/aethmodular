"""Keep hand-maintained dependency fallbacks aligned with pyproject.toml."""

from pathlib import Path
import re

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    pytest.skip("tomllib requires Python 3.11+", allow_module_level=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _distribution_name(spec):
    """Return a normalized distribution name without parsing full PEP 508 syntax."""
    match = NAME_RE.match(spec.strip())
    if match is None:
        return None
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _pyproject_dependencies():
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    core = {_distribution_name(spec) for spec in config["project"]["dependencies"]}
    notebooks = {
        _distribution_name(spec) for spec in config["dependency-groups"]["notebooks"]
    }
    return core, notebooks


def _requirements_dependencies():
    packages = set()
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            packages.add(_distribution_name(line))
    return packages


def _environment_dependencies():
    packages = set()
    for line in (REPO_ROOT / "environment.yml").read_text().splitlines():
        match = re.match(r"^\s*-\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match:
            packages.add(_distribution_name(match.group(1)))
    return packages


def _assert_present(expected, actual, filename):
    for package in sorted(expected):
        assert package in actual, f"{package} from pyproject.toml is missing from {filename}"


def test_core_dependencies_are_in_requirements():
    core, _ = _pyproject_dependencies()
    _assert_present(core, _requirements_dependencies(), "requirements.txt")


def test_core_and_notebook_dependencies_are_in_conda_environment():
    core, notebooks = _pyproject_dependencies()
    _assert_present(core | notebooks, _environment_dependencies(), "environment.yml")


def test_plotly_notebook_dependency_is_in_requirements_fallback():
    _, notebooks = _pyproject_dependencies()
    plotly = {"plotly"} & notebooks
    _assert_present(plotly, _requirements_dependencies(), "requirements.txt")
