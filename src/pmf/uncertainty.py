"""Independent PMF resampling and normalized-profile sensitivity diagnostics.

These routines are not EPA ME-2 DISP, Fpeak or native BS-DISP implementations.
They preserve signed observations and state the explored uncertainty explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from .solver import fit, diagnostics, objective_from_residual, robust_weights_from_residual, validate_arrays


def temporal_segments(dates, strata=None, max_gap_days=14):
    """Contiguous index segments separated by observed strata changes or large gaps."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    if dates.ndim != 1 or len(dates) == 0 or np.isnat(dates).any():
        raise ValueError("Dates must be a nonempty, finite, ordered vector")
    gaps = np.diff(dates).astype(int)
    if np.any(gaps < 0) or max_gap_days <= 0:
        raise ValueError("Dates must be chronological; max_gap_days must be positive")
    strata = np.zeros(len(dates), dtype=int) if strata is None else np.asarray(strata)
    if strata.shape != dates.shape:
        raise ValueError("Strata and dates must have the same length")
    boundaries = np.flatnonzero((gaps > max_gap_days) | (strata[1:] != strata[:-1])) + 1
    return [s.astype(int) for s in np.split(np.arange(len(dates)), boundaries)]


def moving_block_indices(n, rng, block_length=6, segments=None):
    """Stratified overlapping noncircular blocks, preserving each segment's row count.

    Source blocks never cross a gap/stratum boundary. The resampled series is not
    chronological; the individual block membership is returned for auditing.
    """
    if block_length < 1 or int(block_length) != block_length or n < 1:
        raise ValueError("Positive integer block length and sample count are required")
    segments = [np.arange(n)] if segments is None else [np.asarray(s, dtype=int) for s in segments]
    if not segments or not np.array_equal(np.concatenate(segments), np.arange(n)):
        raise ValueError("Segments must partition ordered sample indices exactly once")
    selected, blocks = [], []
    for number, segment in enumerate(segments):
        size = min(block_length, len(segment))
        remaining = len(segment)
        while remaining:
            start = int(rng.integers(0, len(segment) - size + 1))
            block = segment[start:start + min(size, remaining)]
            selected.extend(block.tolist())
            blocks.append({"segment": number, "source_indices": block.tolist()})
            remaining -= len(block)
    return np.asarray(selected, dtype=int), blocks


def match_profiles(reference, candidate, feature_weights=None, threshold=0.90):
    """One-to-one weighted cosine assignment; retain low-similarity rejection flags."""
    a, b = np.asarray(reference, float), np.asarray(candidate, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Profiles must have a shared feature dimension")
    if not 0 <= threshold <= 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid profile values or threshold")
    weights = np.ones(a.shape[1]) if feature_weights is None else np.asarray(feature_weights, float)
    if weights.shape != (a.shape[1],) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Feature weights must be positive and finite")
    # A common rescaling avoids overflow without changing the cosine.
    weights = weights / np.max(weights)
    aw, bw = a * weights, b * weights
    an, bn = np.linalg.norm(aw, axis=1), np.linalg.norm(bw, axis=1)
    similarity = np.zeros((len(a), len(b)))
    valid = (an[:, None] > 0) & (bn[None, :] > 0)
    np.divide(aw @ bw.T, an[:, None] * bn[None, :], out=similarity, where=valid)
    similarity = np.clip(similarity, -1, 1)
    rows, cols = linear_sum_assignment(-similarity)
    mapping = np.full(len(a), -1, dtype=int)
    cosine = np.full(len(a), np.nan)
    accepted = np.zeros(len(a), dtype=bool)
    mapping[rows], cosine[rows] = cols, similarity[rows, cols]
    accepted[rows] = valid[rows, cols] & (cosine[rows] >= threshold)
    return {"candidate_index": mapping, "cosine": cosine, "accepted": accepted, "similarity_matrix": similarity}


def weighted_simplex_projection(values, curvature, total=1.0):
    """Exact diagonal-quadratic minimizer h>=0, sum(h)=total.

    Minimizes sum(curvature * (h-values)**2). Zero-curvature coordinates are
    handled as free coordinates, not floored. This is an optimization constraint,
    never a modification of the measured X or its uncertainty U.
    """
    v, d = np.asarray(values, float), np.asarray(curvature, float)
    if v.ndim != 1 or v.shape != d.shape or not len(v) or not np.isfinite(v).all() or not np.isfinite(d).all():
        raise ValueError("Projection requires equally sized finite vectors")
    if np.any(d < 0) or not np.isfinite(total) or total < 0:
        raise ValueError("Curvature and target sum must be nonnegative")
    if total == 0:
        return np.zeros_like(v)
    positive = d > 0
    if not positive.all():
        answer = np.zeros_like(v)
        answer[positive] = np.maximum(v[positive], 0)
        if answer.sum() <= total:
            free = ~positive
            proportions = np.maximum(v[free], 0)
            if proportions.sum() == 0:
                proportions = np.ones(int(free.sum()))
            answer[free] = (total - answer.sum()) * proportions / proportions.sum()
            return answer
        answer[positive] = weighted_simplex_projection(v[positive], d[positive], total)
        return answer
    scale = d.max()
    d = d / scale
    if np.any(d == 0) or not np.isfinite(1 / d).all():
        raise ValueError("Curvature range exceeds stable floating-point representation")
    breakpoint = d * v
    order = np.argsort(-breakpoint, kind="stable")
    lam = (np.cumsum(v[order]) - total) / np.cumsum(1 / d[order])
    active = np.flatnonzero(breakpoint[order] > lam)
    if not len(active):
        raise RuntimeError("No active projection coordinate")
    answer = np.maximum(0, v - lam[active[-1]] / d)
    discrepancy = total - answer.sum()
    if abs(discrepancy) > 1e-9 * max(1, total):
        raise RuntimeError("Weighted simplex projection lost its sum constraint")
    answer[np.argmax(answer)] += discrepancy
    if np.any(answer < -1e-14):
        raise RuntimeError("Projection produced a negative coordinate")
    return np.maximum(answer, 0)


def _normalized_pair(G, H):
    g, h = np.array(G, float, copy=True), np.array(H, float, copy=True)
    if g.ndim != 2 or h.ndim != 2 or g.shape[1] != len(h):
        raise ValueError("Incompatible factor dimensions")
    if not np.isfinite(g).all() or not np.isfinite(h).all() or np.any(g < 0) or np.any(h < 0):
        raise ValueError("Factor values must be finite and nonnegative")
    sums = h.sum(axis=1)
    if np.any(sums <= 0):
        raise ValueError("Collapsed profile cannot define a normalized-profile sensitivity")
    h /= sums[:, None]
    g *= sums[None, :]
    return g, h


def _g_update(a, weights, h, g, sweeps):
    p = len(h)
    gram = ((h[:, None, :] * h[None, :, :]).reshape(p * p, -1) @ weights.T).reshape(p, p, -1)
    rhs = (weights * a) @ h.T
    for _ in range(sweeps):
        for k in range(p):
            den = gram[k, k]
            num = rhs[:, k] - np.sum(gram[k].T * g, axis=1)
            step = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            g[:, k] = np.maximum(0, g[:, k] + step)
    return g


def _constrain_row(values, curvature, row, factor, band_mask, target):
    if factor is None or row != factor:
        return weighted_simplex_projection(values, curvature, 1.0)
    result = np.empty_like(values)
    result[band_mask] = weighted_simplex_projection(values[band_mask], curvature[band_mask], target)
    result[~band_mask] = weighted_simplex_projection(values[~band_mask], curvature[~band_mask], 1 - target)
    return result


def _h_simplex_update(a, weights, g, h, factor, band_mask, target, sweeps):
    residual = a - g @ h
    for _ in range(sweeps):
        for k in range(len(h)):
            den = np.sum(weights * g[:, k, None] ** 2, axis=0)
            rhs = np.sum(weights * g[:, k, None] * residual, axis=0)
            v = h[k] + np.divide(rhs, den, out=np.zeros_like(rhs), where=den > 0)
            updated = _constrain_row(v, den, k, factor, band_mask, target)
            difference = updated - h[k]
            residual -= g[:, k, None] * difference[None, :]
            h[k] = updated
    return h


def constrained_refit(X, U, G, H, *, factor=None, band_mask=None, target=None,
                      loss="quadratic", delta=4.0, max_iter=500, tol=1e-9,
                      inner_g_sweeps=12, inner_h_sweeps=2, check_every=5):
    """Locally refit with exact unit-sum H rows and optional fixed band fraction.

    A band may contain one channel. Fixing its normalized fraction avoids the
    trivial G/H scaling escape of fixing an unnormalized H value. Every other
    factor and profile value can adjust. This is not native EPA DISP.
    """
    start = time.perf_counter()
    x, u = validate_arrays(X, U)
    g, h = _normalized_pair(G, H)
    if g.shape[0] != len(x) or h.shape[1] != x.shape[1]:
        raise ValueError("Factors do not match X/U dimensions")
    if max_iter < 1 or check_every < 1 or inner_g_sweeps < 1 or inner_h_sweeps < 1 or tol <= 0:
        raise ValueError("Invalid iteration controls")
    if factor is not None:
        band_mask = np.asarray(band_mask, dtype=bool)
        if not 0 <= factor < len(h) or band_mask.shape != (x.shape[1],) or not band_mask.any() or band_mask.all():
            raise ValueError("Target requires a valid factor and proper nonempty band")
        if target is None or not np.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Normalized band target must be between zero and one")
        h[factor] = _constrain_row(h[factor], np.ones(x.shape[1]), factor, factor, band_mask, target)
    elif band_mask is not None or target is not None:
        raise ValueError("Band and target require a factor")
    # Common positive scale leaves normalized H and every residual invariant.
    scale = float(np.median(u))
    a, b, g = x / scale, u / scale, g / scale
    base_weights = 1 / b ** 2
    if not np.isfinite(base_weights).all():
        raise ValueError("Scaled inverse-variance weights overflow")
    objective = objective_from_residual((a - g @ h) / b, loss, delta)
    history = [(0, objective)]
    stable, plateau, increases = 0, False, 0
    for iteration in range(1, max_iter + 1):
        residual = (a - g @ h) / b
        weights = base_weights * robust_weights_from_residual(residual, loss, delta)
        g = _g_update(a, weights, h, g, inner_g_sweeps)
        h = _h_simplex_update(a, weights, g, h, factor, band_mask, target, inner_h_sweeps)
        if iteration % check_every == 0 or iteration == max_iter:
            current = objective_from_residual((a - g @ h) / b, loss, delta)
            if current > objective + 1e-8 * max(1, objective):
                increases += 1
                raise RuntimeError("Constrained block descent increased its objective")
            relative = (objective - current) / max(1, abs(objective))
            stable = stable + 1 if -1e-12 <= relative < tol else 0
            objective = current
            history.append((iteration, current))
            if stable >= 3:
                plateau = True
                break
    g = g * scale
    info = diagnostics(x, u, g, h, loss, delta)
    info.update({"iterations": iteration, "objective_plateau": plateau,
                 "objective_plateau_is_global_optimum": False,
                 "objective_increases_above_roundoff": increases,
                 "profile_sum_max_error": float(abs(h.sum(axis=1) - 1).max()),
                 "target_band_error": None if factor is None else float(abs(h[factor, band_mask].sum() - target)),
                 "elapsed_seconds": time.perf_counter() - start,
                 "method": "alternating_NNLS_G_and_exact_diagonal_quadratic_H_simplex_updates",
                 "native_EPA_DISP": False})
    if info["profile_sum_max_error"] > 1e-10 or (factor is not None and info["target_band_error"] > 1e-10):
        raise RuntimeError("Normalized-profile constraints were not retained")
    return {"G": g, "H": h, "history": np.asarray(history), "metrics": info,
            "config": {"factor": factor, "target": target, "loss": loss, "delta": delta,
                       "max_iter": max_iter, "tol": tol, "inner_g_sweeps": inner_g_sweeps,
                       "inner_h_sweeps": inner_h_sweeps, "check_every": check_every}}


def exact_pair_rotation_envelope(G, H, *, fractions=(0.25, 0.5, 0.75, 1.0)):
    """Explore feasible pair shears G'=G T,H'=T^-1 H, preserving GH exactly.

    This enumerates a limited family of rotations. Its envelope demonstrates
    feasible variation but is neither a global range nor a confidence interval.
    Collapsed boundary profiles are recorded and excluded from normalized ranges.
    """
    g, h = _normalized_pair(G, H)
    product = g @ h
    records, hs, gs = [], [h], [g]
    k = len(h)
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            active_g, active_h = g[:, a] > 0, h[b] > 0
            if not active_g.any() or not active_h.any():
                continue
            lower = -float(np.min(g[active_g, b] / g[active_g, a]))
            upper = float(np.min(h[a, active_h] / h[b, active_h]))
            for endpoint, sign in ((lower, "lower"), (upper, "upper")):
                for fraction in fractions:
                    if not 0 < fraction <= 1:
                        raise ValueError("Rotation fractions must be in (0,1]")
                    amount = endpoint * fraction
                    if amount == 0:
                        continue
                    gr, hr = g.copy(), h.copy()
                    gr[:, b] += amount * g[:, a]
                    hr[a] -= amount * h[b]
                    coefficient_scale = max(float(g.max()), float(h.max()), 1.0)
                    if min(gr.min(), hr.min()) < -1e-12 * coefficient_scale:
                        raise RuntimeError("Analytic feasible shear produced a negative factor")
                    gr = np.maximum(gr, 0)
                    hr = np.maximum(hr, 0)
                    err = float(np.max(abs(gr @ hr - product)))
                    relative_err = err / max(float(abs(product).max()), np.finfo(float).tiny)
                    if relative_err > 1e-10:
                        raise RuntimeError("Shear failed product invariance")
                    collapsed = bool((hr.sum(axis=1) <= 1e-15).any())
                    records.append({"factor_a": a, "factor_b": b, "endpoint": sign,
                                    "fraction_of_feasible_bound": fraction, "shear": amount,
                                    "product_max_absolute_error": err, "product_max_relative_error": relative_err,
                                    "collapsed_profile": collapsed})
                    if not collapsed:
                        gr, hr = _normalized_pair(gr, hr)
                        gs.append(gr)
                        hs.append(hr)
    return {"records": records, "profile_min": np.min(hs, axis=0), "profile_max": np.max(hs, axis=0),
            "contribution_min": np.min(gs, axis=0), "contribution_max": np.max(gs, axis=0),
            "profiles": np.asarray(hs), "contributions": np.asarray(gs),
            "method": "exact_feasible_pair_shears_with_unit_sum_profiles",
            "scope": "limited feasible rotation family; demonstrated ranges, not global or statistical bounds",
            "native_EPA_Fpeak_or_DISP": False}


def profile_displacement_search(X, U, G, H, *, factor, band_mask, thresholds=(4, 8, 15, 25),
                                loss="quadratic", delta=4, max_iter=500, tol=1e-9,
                                max_steps=12, relative_target_tol=0.001, feature_weights=None):
    """Search local delta-loss brackets for a normalized band's profile fraction.

    This is a targeted independent diagnostic. Interval endpoints carry cap,
    objective-decrease and factor-swap flags; they are not EPA DISP intervals.
    """
    x, u = validate_arrays(X, U)
    g, h = _normalized_pair(G, H)
    mask = np.asarray(band_mask, bool)
    if mask.shape != (x.shape[1],) or not mask.any() or mask.all() or not 0 <= factor < len(h):
        raise ValueError("Invalid factor or band")
    thresholds = sorted(set(float(v) for v in thresholds))
    if not thresholds or any(v <= 0 or not np.isfinite(v) for v in thresholds):
        raise ValueError("Positive finite thresholds required")
    if max_steps < 1 or not 0 < relative_target_tol < 1:
        raise ValueError("Invalid search controls")
    original_fraction = float(h[factor, mask].sum())
    base_objective = objective_from_residual((x - g @ h) / u, loss, delta)
    feature_weights = 1 / np.median(u, axis=0) if feature_weights is None else feature_weights
    trace, intervals = [], []
    cache = {original_fraction: {"G": g, "H": h, "metrics": {"objective": base_objective, "objective_plateau": True}}}
    loss_drop_tolerance = max(1e-6, 1e-8 * base_objective)

    def evaluate(target, direction):
        target = float(np.clip(target, 0, 1))
        if target in cache:
            return cache[target]
        closest = min(cache, key=lambda s: abs(s - target))
        warm = cache[closest]
        result = constrained_refit(x, u, warm["G"], warm["H"], factor=factor, band_mask=mask,
                                   target=target, loss=loss, delta=delta, max_iter=max_iter, tol=tol)
        mapping = match_profiles(h, result["H"], feature_weights)
        change = float(result["metrics"]["objective"] - base_objective)
        result["profile_diagnostic"] = {"target": target, "direction": direction, "delta_loss": change,
                                         "objective_decrease": change < -loss_drop_tolerance,
                                         "factor_assignment_changed": bool(np.any(mapping["candidate_index"] != np.arange(len(h)))),
                                         "min_weighted_profile_cosine": float(np.nanmin(mapping["cosine"])),
                                         "objective_plateau": bool(result["metrics"]["objective_plateau"]),
                                         "target_band_error": result["metrics"]["target_band_error"],
                                         "iterations": result["metrics"]["iterations"]}
        cache[target] = result
        trace.append(result["profile_diagnostic"])
        return result

    def change_at(target):
        return float(cache[target]["metrics"]["objective"] - base_objective)

    for sign, direction, bound in ((-1, "decrease", 0.0), (1, "increase", 1.0)):
        room = abs(bound - original_fraction)
        if room < 1e-14:
            for threshold in thresholds:
                intervals.append({"direction": direction, "threshold_delta_loss": threshold, "inside_target": original_fraction,
                                  "outside_target": None, "inside_delta_loss": 0.0, "outside_delta_loss": None,
                                  "status": "parameter_boundary_at_reference", "valid_as_native_EPA_interval": False})
            continue
        # Geometric outward continuation gives feasible local starts without
        # fixing all other profile values or silently changing the reference.
        step = min(room, max(1e-5, room / 256))
        outer = original_fraction
        for _ in range(max_steps):
            outer = float(np.clip(original_fraction + sign * step, 0, 1))
            evaluate(outer, direction)
            if change_at(outer) >= thresholds[-1] or outer == bound:
                break
            step = min(room, step * 2)
        for threshold in thresholds:
            points = sorted((s for s in cache if sign * (s - original_fraction) >= -1e-14), key=lambda s: abs(s - original_fraction))
            inside = original_fraction
            outside = None
            for candidate in points[1:]:
                if change_at(candidate) >= threshold:
                    outside = candidate
                    break
                inside = candidate
            if outside is not None:
                for _ in range(max_steps):
                    if abs(outside - inside) <= relative_target_tol * max(room, 1e-6):
                        break
                    middle = (inside + outside) / 2
                    evaluate(middle, direction)
                    if change_at(middle) < threshold:
                        inside = middle
                    else:
                        outside = middle
                status = "local_threshold_bracket"
            else:
                status = "parameter_boundary_reached" if inside == bound else "outward_search_limit"
            evaluated = [cache[s].get("profile_diagnostic") for s in (inside, outside) if s is not None and s != original_fraction]
            flags = {"endpoint_iteration_cap": any(not p["objective_plateau"] for p in evaluated),
                     "endpoint_objective_decrease": any(p["objective_decrease"] for p in evaluated),
                     "endpoint_factor_assignment_changed": any(p["factor_assignment_changed"] for p in evaluated)}
            intervals.append({"direction": direction, "threshold_delta_loss": threshold,
                              "inside_target": inside, "outside_target": outside,
                              "inside_delta_loss": change_at(inside), "outside_delta_loss": None if outside is None else change_at(outside),
                              "status": status, **flags, "valid_as_native_EPA_interval": False})
    return {"reference_fraction": original_fraction, "reference_objective": base_objective,
            "factor": factor, "band_indices": np.flatnonzero(mask).tolist(), "loss": loss, "delta": delta,
            "trace": trace, "intervals": intervals, "models": cache,
            "any_objective_decrease": any(p["objective_decrease"] for p in trace),
            "any_factor_assignment_changed": any(p["factor_assignment_changed"] for p in trace),
            "any_iteration_cap": any(not p["objective_plateau"] for p in trace),
            "scope": "local constrained normalized-band profile sensitivity; not native DISP, confidence coverage, or a global interval"}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_bootstrap(input_path, reference_path, out, *, repeats=100, starts=2, block_length=6,
                  max_gap_days=14, seed=64718, loss="huber", delta=4, max_iter=2000,
                  tol=1e-7, similarity_threshold=0.90, project_original=False):
    """Checkpointed segmented-block bootstrap; summaries filter caps and poor maps."""
    from . import solver
    from .solver import project_contributions
    input_path, reference_path, out = Path(input_path), Path(reference_path), Path(out)
    data = np.load(input_path, allow_pickle=False)
    reference = np.load(reference_path, allow_pickle=False)
    x, u = validate_arrays(data["X"], data["U"])
    g, h = _normalized_pair(reference["G"], reference["H"])
    if g.shape[0] != len(x) or h.shape[1] != x.shape[1]:
        raise ValueError("Reference factors have incompatible dimensions")
    for key in ("sample_ids", "wavenumbers"):
        if key not in reference or key not in data:
            raise ValueError("Both reference and input require sample_ids and wavenumbers")
        if not np.array_equal(reference[key], data[key]):
            raise ValueError("Reference identity/grid differs from bootstrap data")
    if repeats < 1 or starts < 1:
        raise ValueError("At least one bootstrap and optimizer start required")
    segments = temporal_segments(data["dates"], data["lot_ids"] if "lot_ids" in data else None, max_gap_days)
    protocol = {"input_sha256": _hash(input_path), "reference_sha256": _hash(reference_path),
                "solver_sha256": _hash(solver.__file__), "uncertainty_module_sha256": _hash(__file__),
                "repeats": repeats, "starts": starts, "block_length": block_length, "max_gap_days": max_gap_days,
                "bootstrap_seed": seed, "loss": loss, "delta": delta, "max_iter": max_iter, "tol": tol,
                "fit_seed_rule": "100000 + starts * zero_based_resample + zero_based_start",
                "similarity_threshold": similarity_threshold, "project_original": project_original,
                "segments": [s.tolist() for s in segments], "segment_counts_preserved": True,
                "resampling": "overlapping noncircular blocks independently within contiguous lot/time segments",
                "mapping": "one-to-one cosine of profiles weighted by 1/median(reference U); rejected below threshold",
                "summary_eligibility": "selected model must meet solver convergence and mapped factor cosine threshold",
                "limitations": ["Blank selection and baseline uncertainty fixed", "Not EPA native BS", "Segment frequencies conditioned on their observed counts", "Profile percentile ranges are descriptive, not calibrated confidence coverage", "Projected original-sample G is conditional on full original observations and is not held-out validation"]}
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Bootstrap cache settings/source differ; use a new output directory")
    elif any(out.iterdir()):
        raise ValueError("Bootstrap output directory has unrecorded existing content")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    rng = np.random.default_rng(seed)
    profiles, accepted, convergence, rows, starts_rows, indices = [], [], [], [], [], []
    projected_values = []
    weights = 1 / np.median(u, axis=0)
    for number in range(repeats):
        index, blocks = moving_block_indices(len(x), rng, block_length, segments)
        model_path = out / f"resample_{number:04d}.npz"
        info_path = model_path.with_suffix(".json")
        if model_path.exists() and info_path.exists():
            cached = np.load(model_path, allow_pickle=False)
            info = json.loads(info_path.read_text())
            if not np.array_equal(cached["indices"], index):
                raise ValueError("Cached bootstrap indices differ")
            bh, bg = cached["H_aligned"], cached["G_resampled_aligned"]
            cos, good = cached["cosine"], cached["accepted"].astype(bool)
            history = cached["history"]
            projected = cached["G_projected_original"] if "G_projected_original" in cached else None
            check = objective_from_residual((x[index] - bg @ bh) / u[index], loss, delta)
            if not np.isclose(check, info["selected_metrics"]["objective"], rtol=1e-9, atol=1e-8):
                raise ValueError("Cached bootstrap objective does not match its data")
        elif model_path.exists() or info_path.exists():
            raise ValueError("Incomplete bootstrap checkpoint; inspect before resuming")
        else:
            solutions = []
            with threadpool_limits(limits=1):
                for start in range(starts):
                    fit_seed = 100000 + starts * number + start
                    solutions.append(fit(x[index], u[index], len(h), fit_seed, loss=loss, delta=delta, max_iter=max_iter, tol=tol))
                best = min(solutions, key=lambda s: s["metrics"]["objective"])
                matching = match_profiles(h, best["H"], weights, similarity_threshold)
                order, cos, good = matching["candidate_index"], matching["cosine"], matching["accepted"]
                if np.any(order < 0):
                    raise RuntimeError("Bootstrap factor counts differ")
                bg, bh, history = best["G"][:, order], best["H"][order], best["history"]
                projected = project_contributions(x, u, bh, loss=loss, delta=delta) if project_original else None
            info = {"resample": number, "blocks": blocks, "selected_metrics": best["metrics"],
                    "selected_config": best["config"], "selected_provenance": best["provenance"],
                    "starts": [{"resample": number, "start": start, **s["metrics"]} for start, s in enumerate(solutions)],
                    "profile_alignment_candidate_indices": order.tolist()}
            arrays = {"G_resampled_aligned": bg, "H_aligned": bh, "indices": index, "cosine": cos, "accepted": good,
                      "history": history, "wavenumbers": data["wavenumbers"], "resampled_sample_ids": data["sample_ids"][index]}
            if projected is not None:
                arrays["G_projected_original"] = projected
            np.savez_compressed(model_path, **arrays)
            info_path.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
        converged = bool(info["selected_metrics"].get("converged", info["selected_metrics"].get("objective_plateau", False)))
        profiles.append(bh)
        accepted.append(good)
        convergence.append(converged)
        indices.append(index)
        if projected is not None:
            projected_values.append(projected)
        starts_rows.extend(info["starts"])
        for k in range(len(h)):
            rows.append({"resample": number, "factor": k + 1, "weighted_profile_cosine": float(cos[k]),
                         "mapped": bool(good[k]), "selected_fit_converged": converged,
                         "eligible_for_profile_percentiles": bool(good[k] and converged),
                         "objective": info["selected_metrics"]["objective"]})
        print(json.dumps({"bootstrap": number + 1, "of": repeats, "converged": converged,
                          "mapped_factors": int(good.sum()), "minimum_cosine": float(cos.min())}), flush=True)
    profiles, accepted, convergence = np.asarray(profiles), np.asarray(accepted), np.asarray(convergence)
    eligible = accepted & convergence[:, None]
    percentiles = np.full((5, len(h), x.shape[1]), np.nan)
    summary = []
    for k in range(len(h)):
        valid = eligible[:, k]
        if valid.any():
            percentiles[:, k] = np.percentile(profiles[valid, k], [5, 25, 50, 75, 95], axis=0)
        values = [r["weighted_profile_cosine"] for r in rows if r["factor"] == k + 1]
        summary.append({"factor": k + 1, "repeats": repeats, "mapped_count": int(accepted[:, k].sum()),
                        "converged_selected_count": int(convergence.sum()), "percentile_eligible_count": int(valid.sum()),
                        "median_weighted_profile_cosine": float(np.median(values)), "minimum_weighted_profile_cosine": float(np.min(values))})
    ensemble = {"H_aligned": profiles, "accepted": accepted, "converged": convergence, "eligible": eligible,
                "profile_percentiles": percentiles, "percentile_levels": [5, 25, 50, 75, 95],
                "indices": indices, "wavenumbers": data["wavenumbers"]}
    if projected_values:
        ensemble["G_projected_original"] = np.asarray(projected_values)
    np.savez_compressed(out / "ensemble.npz", **ensemble)
    for name, values in (("matches.csv", rows), ("summary.csv", summary), ("all_start_metrics.csv", starts_rows)):
        keys = list(dict.fromkeys(k for row in values for k in row))
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(values)
    return {"protocol": protocol, "summary": summary, "matches": rows, "ensemble_path": str(out / "ensemble.npz")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bs = sub.add_parser("bootstrap")
    bs.add_argument("--input", type=Path, required=True)
    bs.add_argument("--reference", type=Path, required=True)
    bs.add_argument("--out", type=Path, required=True)
    bs.add_argument("--repeats", type=int, default=100)
    bs.add_argument("--starts", type=int, default=2)
    bs.add_argument("--block-length", type=int, default=6)
    bs.add_argument("--max-gap-days", type=int, default=14)
    bs.add_argument("--max-iter", type=int, default=2000)
    bs.add_argument("--tol", type=float, default=1e-7)
    bs.add_argument("--seed", type=int, default=64718)
    bs.add_argument("--loss", choices=("quadratic", "huber"), default="huber")
    bs.add_argument("--delta", type=float, default=4)
    bs.add_argument("--similarity-threshold", type=float, default=0.90)
    bs.add_argument("--project-original", action="store_true")
    args = parser.parse_args()
    run_bootstrap(args.input, args.reference, args.out, repeats=args.repeats, starts=args.starts,
                  block_length=args.block_length, max_gap_days=args.max_gap_days, seed=args.seed,
                  loss=args.loss, delta=args.delta, max_iter=args.max_iter, tol=args.tol,
                  similarity_threshold=args.similarity_threshold, project_original=args.project_original)


if __name__ == "__main__":
    main()
