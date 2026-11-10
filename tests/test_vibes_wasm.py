"""The WebAssembly VIBES solver (research/ftir_hips_chem/wasm_demo) against the Python one.

The comparison is skipped unless cargo with the wasm32-unknown-unknown target and node are installed.
The check on the gallery's built copy always runs.
"""

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ftir_hips_chem"


def _toolchain():
    if not (shutil.which("cargo") and shutil.which("node") and shutil.which("rustc")):
        return False
    sysroot = subprocess.run(["rustc", "--print", "sysroot"], capture_output=True, text=True).stdout.strip()
    return bool(sysroot) and (Path(sysroot) / "lib" / "rustlib" / "wasm32-unknown-unknown").is_dir()


needs_toolchain = pytest.mark.skipif(not _toolchain(), reason="needs cargo with wasm32-unknown-unknown, and node")


def _demo():
    spec = importlib.util.spec_from_file_location("wasm_demo", ROOT / "workflows" / "wasm_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_toolchain
def test_wasm_solver_matches_python(tmp_path):
    demo = _demo()
    demo.build(None)
    rng = np.random.default_rng(3)
    p, c, rows = 240, 6, 8
    t = np.linspace(0.0, 1.0, p)
    # A smooth background family with shrinking scales, like the scaled principal components of a blank bank.
    W = np.stack([0.03 * 0.4**k * np.cos(np.pi * k * t) for k in range(c)], axis=1)
    mean = 0.2 + 0.05 * t
    bands = np.exp(-(((t[None, :] - rng.uniform(0.2, 0.8, (rows, 1))) / 0.03) ** 2))
    amplitude = np.r_[0.0, 0.0, rng.uniform(0.002, 0.02, rows - 2)][:, None]  # two blank-like rows
    Y = mean + rng.normal(size=(rows, c)) @ W.T + amplitude * bands + 2e-5 * rng.normal(size=(rows, p))
    out = demo.write_folder(tmp_path, W, mean, Y)
    r = json.loads(subprocess.run(["node", "check.mjs", "."], cwd=out, check=True, capture_output=True, text=True).stdout)
    assert r["converged"] == rows and r["certified"] == rows
    assert r["sigma_rel_max"] < 1e-6
    assert r["baseline_abs_max"] < 1e-8


def test_gallery_module_is_built_from_the_current_source():
    """The gallery ships a built copy; its build record must name the source as it stands (no toolchain needed)."""
    demo = _demo()
    record = json.loads((demo.GALLERY / "vibes_wasm.build.json").read_text())
    assert record["source_sha256"] == demo.source_hash(), "rebuild with: python research/ftir_hips_chem/workflows/wasm_demo.py build --gallery"
    body = (demo.GALLERY / "vibes_wasm.wasm").read_bytes()
    assert hashlib.sha256(body).hexdigest() == record["wasm_sha256"]
