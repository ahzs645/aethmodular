"""Unified command interface for repository diagnostics and workflows.

The command layer deliberately uses only the Python standard library. This
keeps ``aeth doctor`` useful even when the scientific environment is missing
or incomplete. Scientific implementation remains in its existing ``src/`` or
``research/ftir_hips_chem/scripts/`` home.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

DIAGNOSTICS = {
    "etad": "scripts/diagnostics/check_etad_data.py",
    "matching": "scripts/diagnostics/check_matching_statistics.py",
    "compare-pkl": "scripts/diagnostics/compare_pkl_files.py",
    "etad-stats": "scripts/diagnostics/get_etad_stats.py",
    "flow": "scripts/diagnostics/inspect_flow_columns.py",
    "system": "scripts/diagnostics/test_system.py",
}

SPARTAN_COMMANDS = {
    "pull": "scripts/pipelines/spartan_pull_and_summarize.py",
    "coverage": "scripts/pipelines/spartan_coverage_analysis.py",
    "coverage-plots": "scripts/pipelines/spartan_coverage_plots.py",
    "bridge": "scripts/pipelines/spartan_hips_bridge.py",
    "connections": "scripts/pipelines/spartan_master_connections.py",
    "extras": "scripts/pipelines/spartan_extras.py",
}

BUILD_GROUPS = {
    "ftir-calibration": "research/ftir_ec_calibration_2026_06_25",
    "spartan-ec": "research/spartan_ec_2026_06_16",
    "july07": "research/July07",
    "addis-deming": "research/addis_fabs_ec_deming",
    "catch-up": "research/catch_up",
}

MACHINE_PATH_MARKERS = ("/Users/", "C:\\Users\\", "/home/")
LEGACY_PATH_MARKERS = ("FTIR_HIPS_Chem", "aethmodular-clean")


@dataclass(frozen=True)
class CheckResult:
    label: str
    ok: bool
    detail: str
    required: bool = True


def _repo_path(relative: str | Path) -> Path:
    return REPO_ROOT / relative


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _run(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> int:
    print("+", " ".join(str(part) for part in command))
    try:
        return subprocess.run(list(command), cwd=cwd, check=False).returncode
    except OSError as exc:
        print(f"ERROR: unable to run {command[0]}: {exc}", file=sys.stderr)
        return 1


def _run_script(relative: str, script_args: Sequence[str]) -> int:
    path = _repo_path(relative)
    if not path.is_file():
        print(f"ERROR: command target is missing: {_display_path(path)}", file=sys.stderr)
        return 2
    return _run([sys.executable, str(path), *script_args])


def _active_notebooks(include_archive: bool = False) -> list[Path]:
    notebooks: set[Path] = set()
    for base in (_repo_path("notebooks"), _repo_path("research")):
        if not base.exists():
            continue
        for path in base.rglob("*.ipynb"):
            if not include_archive and "archive" in path.parts:
                continue
            notebooks.add(path)
    root_gallery = _repo_path("plotting_gaps_scenarios.ipynb")
    if root_gallery.exists():
        notebooks.add(root_gallery)
    return sorted(notebooks)


def _resolve_notebook_paths(paths: Sequence[str], include_archive: bool) -> list[Path]:
    if not paths:
        return _active_notebooks(include_archive=include_archive)
    resolved = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved.append(path.resolve())
    return resolved


def _changed_notebooks() -> list[Path]:
    """Return tracked modifications and untracked notebooks in the worktree."""
    commands = [
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD", "--", "*.ipynb"],
        ["git", "ls-files", "--others", "--exclude-standard", "--", "*.ipynb"],
    ]
    paths: set[Path] = set()
    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return []
        if result.returncode:
            return []
        for value in result.stdout.splitlines():
            if value:
                paths.add((REPO_ROOT / value).resolve())
    return sorted(paths)


def _cell_sources(notebook: dict) -> Iterable[str]:
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            yield "".join(str(part) for part in source)
        else:
            yield str(source)


def _notebook_issues(path: Path) -> list[str]:
    if not path.is_file():
        return ["file is missing"]
    try:
        with path.open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"invalid notebook JSON: {exc}"]

    source = "\n".join(_cell_sources(notebook))
    issues = []
    for marker in MACHINE_PATH_MARKERS:
        if marker in source:
            issues.append(f"machine-specific path ({marker})")
    for marker in LEGACY_PATH_MARKERS:
        if marker in source:
            issues.append(f"legacy path ({marker})")
    return issues


def command_doctor(args: argparse.Namespace) -> int:
    data_root = Path(
        os.environ.get("AETHMODULAR_DATA_ROOT", _repo_path("research/ftir_hips_chem"))
    ).expanduser()
    required_modules = ("numpy", "pandas", "matplotlib", "scipy")
    optional_modules = ("pytest", "ruff", "nbformat", "nbclient")

    results = [
        CheckResult("repository", (_repo_path("pyproject.toml")).is_file(), str(REPO_ROOT)),
        CheckResult(
            "Python",
            sys.version_info >= (3, 9),
            f"{sys.version.split()[0]} at {sys.executable} (3.13 recommended)",
        ),
        CheckResult("data root", data_root.is_dir(), str(data_root)),
        CheckResult(
            "filter dataset",
            (data_root / "Filter Data/unified_filter_dataset.pkl").is_file(),
            str(data_root / "Filter Data/unified_filter_dataset.pkl"),
        ),
        CheckResult(
            "processed sites",
            (data_root / "processed_sites").is_dir(),
            str(data_root / "processed_sites"),
        ),
    ]
    results.extend(
        CheckResult(module, importlib.util.find_spec(module) is not None, "required Python module")
        for module in required_modules
    )
    results.extend(
        CheckResult(
            module,
            importlib.util.find_spec(module) is not None,
            "development/notebook module",
            required=False,
        )
        for module in optional_modules
    )

    if args.json:
        print(json.dumps([result.__dict__ for result in results], indent=2))
    else:
        print("Aethmodular environment")
        for result in results:
            if result.ok:
                status = "PASS"
            elif result.required:
                status = "FAIL"
            else:
                status = "WARN"
            print(f"{status:4}  {result.label:18} {result.detail}")

    return 1 if any(not result.ok and result.required for result in results) else 0


def command_notebook_list(args: argparse.Namespace) -> int:
    paths = _active_notebooks(include_archive=args.all)
    for path in paths:
        print(_display_path(path))
    print(f"\n{len(paths)} notebook(s)")
    return 0


def command_notebook_check(args: argparse.Namespace) -> int:
    if args.changed and (args.notebooks or args.all):
        print("ERROR: --changed cannot be combined with notebook paths or --all.", file=sys.stderr)
        return 2
    if args.changed:
        paths = _changed_notebooks()
    else:
        paths = _resolve_notebook_paths(args.notebooks, include_archive=args.all)
    failures = 0
    for path in paths:
        issues = _notebook_issues(path)
        if issues:
            failures += 1
            print(f"FAIL  {_display_path(path)}")
            for issue in issues:
                print(f"      - {issue}")
        elif args.verbose:
            print(f"PASS  {_display_path(path)}")
    print(f"\nChecked {len(paths)} notebook(s); {failures} failed portability checks.")
    return 1 if failures else 0


def command_notebook_run(args: argparse.Namespace) -> int:
    script_args: list[str] = ["--timeout", str(args.timeout)]
    if args.all_root:
        script_args.append("--all-root")
    script_args.extend(args.notebooks)
    return _run_script("scripts/diagnostics/run_notebook_smoke.py", script_args)


def command_diagnose(args: argparse.Namespace) -> int:
    return _run_script(DIAGNOSTICS[args.diagnostic], args.script_args)


def command_data_resample(args: argparse.Namespace) -> int:
    script_args = []
    if args.list_sites:
        script_args.append("--list-sites")
    for site in args.site:
        script_args.extend(("--site", site))
    script_args.extend(args.script_args)
    return _run_script("scripts/pipelines/create_9am_resampled_datasets.py", script_args)


def command_spartan(args: argparse.Namespace) -> int:
    return _run_script(SPARTAN_COMMANDS[args.workflow], args.script_args)


def _build_targets(group: str) -> dict[str, Path]:
    directory = _repo_path(BUILD_GROUPS[group])
    targets = {}
    for path in sorted(directory.glob("_build*.py")):
        name = path.stem.removeprefix("_build_").removeprefix("_build")
        name = name.strip("_") or "default"
        targets[name] = path
    return targets


def command_build_list(args: argparse.Namespace) -> int:
    groups = [args.group] if args.group else sorted(BUILD_GROUPS)
    for group in groups:
        print(f"{group}:")
        targets = _build_targets(group)
        for target in targets:
            print(f"  {target}")
        if not targets:
            print("  (no registered builders found)")
    return 0


def command_build_run(args: argparse.Namespace) -> int:
    targets = _build_targets(args.group)
    if args.target == "all":
        selected = list(targets.items())
    elif args.target in targets:
        selected = [(args.target, targets[args.target])]
    else:
        available = ", ".join(targets) or "none"
        print(
            f"ERROR: unknown target {args.target!r} for {args.group}; available: {available}",
            file=sys.stderr,
        )
        return 2

    for name, path in selected:
        print(f"BUILD {args.group}:{name}")
        code = _run([sys.executable, str(path), *args.script_args])
        if code:
            return code
    return 0


def command_check(args: argparse.Namespace) -> int:
    commands: list[tuple[str, list[str]]] = []
    if not args.no_tests:
        commands.append(("tests", [sys.executable, "-m", "pytest", "-q"]))
    if not args.no_lint:
        commands.append(("lint", [sys.executable, "-m", "ruff", "check", "src", "tests", "aethmodular_cli"]))

    failures = []
    for label, command in commands:
        module = command[2]
        if importlib.util.find_spec(module) is None:
            print(f"FAIL  {label}: Python module {module!r} is not installed")
            failures.append(label)
            continue
        if _run(command):
            failures.append(label)

    if args.notebooks:
        notebook_args = argparse.Namespace(notebooks=[], all=False, changed=True, verbose=False)
        if command_notebook_check(notebook_args):
            failures.append("notebooks")

    if failures:
        print("\nChecks failed:", ", ".join(failures))
        return 1
    print("\nAll selected checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aeth",
        description="Aethmodular repository commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check the local environment and data root.")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable results.")
    doctor.set_defaults(func=command_doctor)

    check = subparsers.add_parser("check", help="Run repository quality gates.")
    check.add_argument("--no-tests", action="store_true", help="Skip pytest.")
    check.add_argument("--no-lint", action="store_true", help="Skip Ruff.")
    check.add_argument(
        "--notebooks", action="store_true", help="Also check every active notebook for portable paths."
    )
    check.set_defaults(func=command_check)

    notebook = subparsers.add_parser("notebook", help="List, check, or execute notebooks.")
    notebook_sub = notebook.add_subparsers(dest="notebook_command", required=True)

    notebook_list = notebook_sub.add_parser("list", help="List active notebooks.")
    notebook_list.add_argument("--all", action="store_true", help="Include archived notebooks.")
    notebook_list.set_defaults(func=command_notebook_list)

    notebook_check = notebook_sub.add_parser("check", help="Check notebook source portability.")
    notebook_check.add_argument("notebooks", nargs="*", help="Paths relative to the repository root.")
    notebook_check.add_argument("--all", action="store_true", help="Include archives when no paths are given.")
    notebook_check.add_argument(
        "--changed",
        action="store_true",
        help="Check only modified or untracked notebooks in the worktree.",
    )
    notebook_check.add_argument("--verbose", action="store_true", help="Print passing notebooks too.")
    notebook_check.set_defaults(func=command_notebook_check)

    notebook_run = notebook_sub.add_parser("run", help="Execute the portable smoke suite or given notebooks.")
    notebook_run.add_argument("notebooks", nargs="*", help="Paths relative to the repository root.")
    notebook_run.add_argument("--all-root", action="store_true", help="Run all notebooks under notebooks/.")
    notebook_run.add_argument("--timeout", type=int, default=300, help="Per-cell timeout in seconds.")
    notebook_run.set_defaults(func=command_notebook_run)

    diagnose = subparsers.add_parser("diagnose", help="Run a repository diagnostic.")
    diagnose.add_argument("diagnostic", choices=sorted(DIAGNOSTICS))
    diagnose.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the diagnostic.")
    diagnose.set_defaults(func=command_diagnose)

    data = subparsers.add_parser("data", help="Run data preparation workflows.")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    resample = data_sub.add_parser("resample", help="Build 9am-to-9am site datasets.")
    resample.add_argument(
        "--site",
        action="append",
        default=[],
        help="Site code or name; repeat for multiple sites (default: all).",
    )
    resample.add_argument("--list-sites", action="store_true", help="List configured sites and exit.")
    resample.add_argument("script_args", nargs=argparse.REMAINDER)
    resample.set_defaults(func=command_data_resample)

    spartan = subparsers.add_parser("spartan", help="Run SPARTAN data workflows.")
    spartan.add_argument("workflow", choices=sorted(SPARTAN_COMMANDS))
    spartan.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the workflow.")
    spartan.set_defaults(func=command_spartan)

    build = subparsers.add_parser("build", help="Discover and run research artifact builders.")
    build_sub = build.add_subparsers(dest="build_command", required=True)
    build_list = build_sub.add_parser("list", help="List registered build targets.")
    build_list.add_argument("group", nargs="?", choices=sorted(BUILD_GROUPS))
    build_list.set_defaults(func=command_build_list)
    build_run = build_sub.add_parser("run", help="Run one target or every target in a group.")
    build_run.add_argument("group", choices=sorted(BUILD_GROUPS))
    build_run.add_argument("target", help="Target shown by 'aeth build list', or 'all'.")
    build_run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the builder.")
    build_run.set_defaults(func=command_build_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
