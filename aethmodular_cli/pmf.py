"""Versioned PMF matrix intake, independent Python fit, and EPA/ESAT handoff.

This is NOT EPA ME-2. External engines are never marked successful by export.
Run with: python -m aethmodular_cli.pmf --help
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"{path}: expected IDs and at least one variable")
    header = rows[0]
    if not header[0].strip() or any(not x.strip() for x in header[1:]):
        raise ValueError(f"{path}: missing header")
    if len(set(header)) != len(header):
        raise ValueError(f"{path}: duplicate column headers")
    ids = []
    values = []
    for row in rows[1:]:
        if len(row) != len(header) or not row[0].strip():
            raise ValueError(f"{path}: invalid row length or missing sample ID")
        ids.append(row[0])
        try:
            values.append([float(x) for x in row[1:]])
        except ValueError as exc:
            raise ValueError(f"{path}: nonnumeric cell for {row[0]}") from exc
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample ID")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: nonfinite values")
    return ids, header[1:], array


def load_pair(concentrations, uncertainties):
    ids, features, x = _read(concentrations)
    uid, uf, u = _read(uncertainties)
    if ids != uid or features != uf or x.shape != u.shape:
        raise ValueError("X and U sample/feature identities or shapes differ")
    if not np.all(u > 0):
        raise ValueError("U must be strictly positive")
    return ids, features, x, u


def _write_matrix(path, ids, features, values):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", *features])
        for name, row in zip(ids, values):
            writer.writerow([name, *[format(float(v), ".17g") for v in row]])


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _receipt(args, status, extra=None):
    data = {
        "schema": "aethmodular-pmf-v1",
        "status": status,
        "engine": args.engine,
        "dataset_id": args.dataset,
        "concentrations_sha256": _digest(args.x),
        "uncertainties_sha256": _digest(args.u),
        "source_concentrations": str(args.x),
        "source_uncertainties": str(args.u),
    }
    data.update(extra or {})
    return data


def _fit(x, u, rank, seed, iterations, tolerance):
    """Alternating weighted NNLS, preserving signed X and nonnegative G,H."""
    if rank < 1 or rank > min(x.shape):
        raise ValueError("factor count exceeds input dimensions")
    rng = np.random.default_rng(seed)
    g = rng.random((x.shape[0], rank)) + 0.1
    h = rng.random((rank, x.shape[1])) + 0.1
    weights = 1.0 / u
    history = []
    for iteration in range(iterations):
        for j in range(x.shape[1]):
            h[:, j] = nnls(g * weights[:, j, None], x[:, j] * weights[:, j])[0]
        for i in range(x.shape[0]):
            g[i] = nnls(h.T * weights[i, :, None], x[i] * weights[i])[0]
        q = float(np.sum(((x - g @ h) / u) ** 2))
        history.append(q)
        if len(history) > 2 and abs(history[-2] - q) <= tolerance * max(1.0, history[-2]):
            break
    return g, h, history


def _output_dir(args):
    dest = Path(args.output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("output directory is not empty; choose a new immutable run directory")
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def command_validate(args):
    ids, features, x, u = load_pair(args.x, args.u)
    out = _output_dir(args)
    _write_json(out / "manifest.json", _receipt(args, "passed", {
        "operation": "validate", "samples": len(ids), "features": len(features),
        "negative_observations": int(np.sum(x < 0)),
    }))


def command_fit(args):
    ids, features, x, u = load_pair(args.x, args.u)
    if args.engine != "python-independent":
        raise ValueError("fit is only implemented for python-independent; ESAT/EPA require verified native runs")
    out = _output_dir(args)
    runs = []
    best = None
    for seed in args.seeds:
        g, h, history = _fit(x, u, args.factors, seed, args.iterations, args.tolerance)
        result = {"seed": seed, "q_true": history[-1], "iterations": len(history),
                  "status": "capped" if len(history) == args.iterations else "passed"}
        runs.append(result)
        if best is None or history[-1] < best[0]:
            best = (history[-1], seed, g.copy(), h.copy())
    q, seed, g, h = best
    _write_matrix(out / "contributions.csv", ids, [f"F{i+1}" for i in range(args.factors)], g)
    _write_matrix(out / "profiles.csv", [f"F{i+1}" for i in range(args.factors)], features, h)
    _write_matrix(out / "residuals.csv", ids, features, (x - g @ h) / u)
    _write_json(out / "manifest.json", _receipt(args, "passed" if any(r["status"] == "passed" for r in runs) else "capped", {
        "operation": "fit", "samples": len(ids), "features": len(features),
        "factors": args.factors, "selected_seed": seed, "q_true": q, "runs": runs,
        "model_type": "independent alternating weighted NNLS; not EPA PMF5/ME-2",
        "contributions_unit": "spectral factor amplitude, not calibrated mass",
    }))


def command_project(args):
    ids, features, x, u = load_pair(args.x, args.u)
    factor_ids, model_features, h = _read(args.profiles)
    if features != model_features:
        raise ValueError("frozen profile feature identities/order do not match")
    if np.any(h < 0):
        raise ValueError("profiles must be nonnegative")
    out = _output_dir(args)
    g = np.stack([nnls(h.T / u[i, :, None], x[i] / u[i])[0] for i in range(len(ids))])
    residuals = (x - g @ h) / u
    _write_matrix(out / "contributions.csv", ids, factor_ids, g)
    _write_matrix(out / "residuals.csv", ids, features, residuals)
    _write_json(out / "manifest.json", _receipt(args, "passed", {
        "operation": "project", "frozen_profiles_sha256": _digest(args.profiles),
        "samples": len(ids), "features": len(features),
        "review_samples": [ids[i] for i in range(len(ids)) if np.sqrt(np.mean(residuals[i] ** 2)) > args.threshold],
        "residual_rms_review_threshold": args.threshold,
        "contributions_unit": "spectral factor amplitude, not calibrated mass",
    }))


def command_export(args):
    ids, features, x, u = load_pair(args.x, args.u)
    out = _output_dir(args)
    _write_matrix(out / "concentrations.csv", ids, features, x)
    _write_matrix(out / "uncertainties.csv", ids, features, u)
    load_pair(out / "concentrations.csv", out / "uncertainties.csv")
    _write_json(out / "manifest.json", _receipt(args, "not_run", {
        "operation": "epa-handoff", "target_engine": "EPA PMF5/ME-2",
        "samples": len(ids), "features": len(features),
        "warning": "CSV round trip verified; no native EPA PMF5 run performed",
    }))


def command_esat(args):
    ids, features, x, u = load_pair(args.x, args.u)
    out = _output_dir(args)
    _write_json(out / "manifest.json", _receipt(args, "blocked", {
        "operation": "esat-handoff", "samples": len(ids), "features": len(features),
        "reason": "ESAT backend/version not validated on signed FTIR inputs; no fit executed",
    }))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name, func in (("validate", command_validate), ("fit", command_fit),
                       ("project", command_project), ("epa-export", command_export),
                       ("esat-check", command_esat)):
        p = sub.add_parser(name)
        p.add_argument("--x", required=True, type=Path)
        p.add_argument("--u", required=True, type=Path)
        p.add_argument("--dataset", required=True)
        p.add_argument("--output", required=True, type=Path)
        p.set_defaults(func=func, engine={"epa-export": "epa-pmf5", "esat-check": "esat"}.get(name, "python-independent"))
        if name == "fit":
            p.add_argument("--factors", type=int, default=5)
            p.add_argument("--seeds", type=int, nargs="+", default=[1, 10, 100])
            p.add_argument("--iterations", type=int, default=1000)
            p.add_argument("--tolerance", type=float, default=1e-7)
        if name == "project":
            p.add_argument("--profiles", required=True, type=Path)
            p.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"PMF ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
