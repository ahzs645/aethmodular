"""Batched VIBES (PB loss, fixed tau): ELBO by vectorised L-BFGS, MAP by batched IPM.

Opt-in experiment next to ``vibes_baseline.vibes_baseline_matrix``; it does not
change that adapter. The objective is the vendored upstream one
(``vibes.absorbance_estimators.vibes.vibes_pb_elbo`` / ``vibes_pb_jac_elbo`` and
``map.map_pb``); only the way many spectra are executed differs:

* ELBO: every spectrum keeps its own L-BFGS state (history, step, stopping
  decision). Rows are evaluated together as matrix products, so B spectra cost
  four (B x c) @ (c x p) products per iteration instead of B Python round trips.
  A rolling pool refills slots as rows finish, so slow fits never idle the rest.
* MAP: the pinball QP is solved through its box-constrained dual with a
  Mehrotra predictor-corrector interior-point method. Every Newton step is a
  c x c Cholesky per spectrum (Woodbury on the rank-c Hessian), batched. Each
  row is accepted only with a primal-dual gap certificate on the *original*
  pinball objective, independent of the solver internals.

Backends: ``numpy`` (reference, CPU) or ``cupy`` (CUDA). No silent fallback.
Only ``loss='PB'`` with a fixed ``tau`` is implemented; tau estimation raises.
L-BFGS stopping follows SciPy's L-BFGS-B rules (factr/pgtol/maxiter/maxfun)
and the line search is SciPy's More-Thuente DCSRCH, vectorised; the bound on d
is handled by projection rather than L-BFGS-B's Cauchy point, so iterates are
not bit-identical to the reference. The reference is itself only defined to
optimiser tolerance (1e-15 changes in the gradient move SciPy's own end point),
so compare against that noise floor.

``fit_elbo_newton`` is a different optimiser for the same objective: damped
Newton with the exact Hessian (built by GEMM like the MAP step). It targets the
ELBO optimum, not SciPy's stopping point, so compare it with polished fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- backend
class _Backend:
    def __init__(self, name):
        self.name = name
        if name == "numpy":
            from scipy.special import ndtr

            self.xp, self.ndtr = np, ndtr
            self.host = np.asarray
            self.sync = lambda: None
        elif name == "cupy":
            try:
                import cupy as cp
                from cupyx.scipy.special import ndtr
            except ImportError as exc:  # pragma: no cover - GPU only
                raise RuntimeError("backend='cupy' requires CuPy") from exc
            if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover
                raise RuntimeError("backend='cupy' found no CUDA device")
            self.xp, self.ndtr = cp, ndtr
            self.host = cp.asnumpy
            self.sync = cp.cuda.Device().synchronize
            # theta = v - wn, delta = sqrt(dw): one pass gives the three ELBO terms.
            self.elbo_terms = cp.ElementwiseKernel(
                "float64 v, float64 wn, float64 dw, float64 tau",
                "float64 g, float64 a, float64 b",
                """
                double theta = v - wn;
                double delta = sqrt(dw);
                double z = theta / delta;
                double Phi = normcdf(z);
                double phi = exp(-0.5 * z * z) * 0.3989422804014327;
                g = theta * Phi + delta * phi + (tau - 1.0) * theta;
                a = (1.0 - tau) - Phi;
                b = phi / delta;
                """,
                "vibes_pb_elbo_terms")
        else:
            raise ValueError("backend must be 'numpy' or 'cupy'")


def _check_inputs(spectra, mean, components):
    y = np.asarray(spectra, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    W = np.asarray(components, dtype=np.float64)
    if y.ndim != 2 or not len(y):
        raise ValueError("spectra must be a nonempty rows-by-channels matrix")
    if mu.shape != (y.shape[1],) or W.ndim != 2 or W.shape[0] != y.shape[1]:
        raise ValueError("mean/components must match the spectrum channels")
    for a in (y, mu, W):
        if not np.isfinite(a).all():
            raise ValueError("inputs must be finite")
    return y, mu, W


# --------------------------------------------------------------------------- line search
def _dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax, xp=np):
    """Vectorised transcription of SciPy's ``_dcsrch.dcstep`` (More-Thuente).

    ``xp`` is numpy or jax.numpy (vibes_jax compiles the same arithmetic)."""
    with np.errstate(all="ignore"):
        sgnd = xp.sign(dp) * xp.sign(dx)
        c1 = fp > fx
        c2 = ~c1 & (sgnd < 0)
        c3 = ~c1 & ~c2 & (xp.abs(dp) < xp.abs(dx))
        c4 = ~c1 & ~c2 & ~c3
        # case 1
        th = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = xp.maximum(xp.maximum(xp.abs(th), xp.abs(dx)), xp.abs(dp))
        gm = s * xp.sqrt((th / s) ** 2 - (dx / s) * (dp / s))
        gm1 = xp.where(stp < stx, -gm, gm)
        r = ((gm1 - dx) + th) / (((gm1 - dx) + gm1) + dp)
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        f1 = xp.where(xp.abs(stpc - stx) <= xp.abs(stpq - stx), stpc, stpc + (stpq - stpc) / 2.0)
        # case 2
        gm2 = xp.where(stp > stx, -gm, gm)
        r = ((gm2 - dp) + th) / (((gm2 - dp) + gm2) + dx)
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        f2 = xp.where(xp.abs(stpc - stp) > xp.abs(stpq - stp), stpc, stpq)
        # case 3
        g3 = s * xp.sqrt(xp.maximum(0.0, (th / s) ** 2 - (dx / s) * (dp / s)))
        g3 = xp.where(stp > stx, -g3, g3)
        r = ((g3 - dp) + th) / ((g3 + (dx - dp)) + g3)
        stpc = xp.where((r < 0) & (g3 != 0), stp + r * (stx - stp), xp.where(stp > stx, stpmax, stpmin))
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        fb = xp.where(xp.abs(stpc - stp) < xp.abs(stpq - stp), stpc, stpq)
        fb = xp.where(stp > stx, xp.minimum(stp + 0.66 * (sty - stp), fb),
                      xp.maximum(stp + 0.66 * (sty - stp), fb))
        fnb = xp.clip(xp.where(xp.abs(stpc - stp) > xp.abs(stpq - stp), stpc, stpq), stpmin, stpmax)
        f3 = xp.where(brackt, fb, fnb)
        # case 4
        th4 = 3.0 * (fp - fy) / (sty - stp) + dy + dp
        s4 = xp.maximum(xp.maximum(xp.abs(th4), xp.abs(dy)), xp.abs(dp))
        g4 = s4 * xp.sqrt((th4 / s4) ** 2 - (dy / s4) * (dp / s4))
        g4 = xp.where(stp > sty, -g4, g4)
        r = ((g4 - dp) + th4) / (((g4 - dp) + g4) + dy)
        f4 = xp.where(brackt, stp + r * (sty - stp), xp.where(stp > stx, stpmax, stpmin))
        stpf = xp.select([c1, c2, c3, c4], [f1, f2, f3, f4])
        brackt = brackt | c1 | c2
        # interval update
        new_sty = xp.where(c1, stp, xp.where(sgnd < 0, stx, sty))
        new_fy = xp.where(c1, fp, xp.where(sgnd < 0, fx, fy))
        new_dy = xp.where(c1, dp, xp.where(sgnd < 0, dx, dy))
        new_stx = xp.where(c1, stx, stp)
        new_fx = xp.where(c1, fx, fp)
        new_dx = xp.where(c1, dx, dp)
    return new_stx, new_fx, new_dx, new_sty, new_fy, new_dy, stpf, brackt


class _MoreThuente:
    """Vectorised SciPy DCSRCH state machine; one independent search per row.

    Status codes: 0 = evaluate at stp (FG), 1 = CONVERGENCE, 2 = WARNING (L-BFGS-B
    accepts both), -1 = ERROR.
    """

    def __init__(self, stp, f, g, stpmax, ftol=1e-3, gtol=0.9, xtol=0.1, stpmin=0.0):
        n = len(stp)
        self.ftol, self.gtol, self.xtol = ftol, gtol, xtol
        self.stpmin = np.full(n, stpmin)
        self.stpmax = np.asarray(stpmax, dtype=float).copy()
        err = (stp < self.stpmin) | (stp > self.stpmax) | ~(g < 0) | (self.stpmax < self.stpmin)
        self.status = np.where(err, -1, 0)
        self.brackt = np.zeros(n, bool)
        self.stage = np.ones(n, np.int64)
        self.finit, self.ginit = f.copy(), g.copy()
        self.gtest = ftol * g
        self.width = self.stpmax - self.stpmin
        self.width1 = self.width / 0.5
        self.stx = np.zeros(n); self.fx = f.copy(); self.gx = g.copy()
        self.sty = np.zeros(n); self.fy = f.copy(); self.gy = g.copy()
        self.stmin = np.zeros(n)
        self.stmax = stp + 4.0 * stp
        self.stp = np.asarray(stp, dtype=float).copy()

    def iterate(self, idx, f, g):
        """Update rows ``idx`` with f, g evaluated at ``self.stp[idx]``."""
        i = idx
        stp = self.stp[i]
        ftest = self.finit[i] + stp * self.gtest[i]
        st2 = (self.stage[i] == 1) & (f <= ftest) & (g >= 0)
        stage = np.where(st2, 2, self.stage[i])
        br, stmin, stmax = self.brackt[i], self.stmin[i], self.stmax[i]
        warn = (br & ((stp <= stmin) | (stp >= stmax))) | (br & (stmax - stmin <= self.xtol * stmax))
        warn |= (stp == self.stpmax[i]) & (f <= ftest) & (g <= self.gtest[i])
        warn |= (stp == self.stpmin[i]) & ((f > ftest) | (g >= self.gtest[i]))
        conv = (f <= ftest) & (np.abs(g) <= self.gtol * -self.ginit[i])
        status = np.where(conv, 1, np.where(warn, 2, 0))
        go = status == 0
        gtest = self.gtest[i]
        mod = (stage == 1) & (f <= self.fx[i]) & (f > ftest)
        stx, fx, gx, sty, fy, gy = (self.stx[i], self.fx[i], self.gx[i],
                                     self.sty[i], self.fy[i], self.gy[i])
        # modified function in stage 1
        fm, fxm, fym = f - stp * gtest, fx - stx * gtest, fy - sty * gtest
        gm, gxm, gym = g - gtest, gx - gtest, gy - gtest
        a = _dcstep(np.where(mod, stx, stx), np.where(mod, fxm, fx), np.where(mod, gxm, gx),
                    sty, np.where(mod, fym, fy), np.where(mod, gym, gy), stp,
                    np.where(mod, fm, f), np.where(mod, gm, g), br, stmin, stmax)
        nstx, nfx, ngx, nsty, nfy, ngy, nstp, nbr = a
        nfx = np.where(mod, nfx + nstx * gtest, nfx)
        nfy = np.where(mod, nfy + nsty * gtest, nfy)
        ngx = np.where(mod, ngx + gtest, ngx)
        ngy = np.where(mod, ngy + gtest, ngy)
        width, width1 = self.width[i], self.width1[i]
        bis = nbr & (np.abs(nsty - nstx) >= 0.66 * width1)
        nstp = np.where(bis, nstx + 0.5 * (nsty - nstx), nstp)
        nwidth1 = np.where(nbr, width, width1)
        nwidth = np.where(nbr, np.abs(nsty - nstx), width)
        nstmin = np.where(nbr, np.minimum(nstx, nsty), nstp + 1.1 * (nstp - nstx))
        nstmax = np.where(nbr, np.maximum(nstx, nsty), nstp + 4.0 * (nstp - nstx))
        nstp = np.clip(nstp, self.stpmin[i], self.stpmax[i])
        stuck = (nbr & ((nstp <= nstmin) | (nstp >= nstmax))) | (nbr & (nstmax - nstmin <= self.xtol * nstmax))
        nstp = np.where(stuck, nstx, nstp)
        # commit rows that continue searching
        def put(arr, val):
            arr[i[go]] = val[go]
        for arr, val in ((self.stage, stage), (self.brackt, nbr), (self.stx, nstx), (self.fx, nfx),
                         (self.gx, ngx), (self.sty, nsty), (self.fy, nfy), (self.gy, ngy),
                         (self.width, nwidth), (self.width1, nwidth1), (self.stmin, nstmin),
                         (self.stmax, nstmax), (self.stp, nstp)):
            put(arr, val)
        status = np.where(go & ~np.isfinite(nstp), -1, status)
        self.status[i] = status
        return status


# --------------------------------------------------------------------------- ELBO
def _elbo_and_grad(be, V, W, W2, tau, nu, d):
    """Negative PB ELBO and gradient for rows (V = y - mu), fixed tau.

    Identical algebra to vibes_pb_elbo / vibes_pb_jac_elbo; the d-gradient of
    tau is not returned because tau is fixed.
    """
    xp = be.xp
    p = V.shape[1]
    c = W.shape[1]
    if hasattr(be, "elbo_terms"):
        g, a_term, b_term = be.elbo_terms(V, nu @ W.T, (d * d) @ W2.T, tau)
    else:
        theta = V - nu @ W.T
        delta = xp.sqrt((d * d) @ W2.T)
        z = theta / delta
        Phi = be.ndtr(z)
        phi = xp.exp(-0.5 * z * z) / SQRT_2PI
        g = theta * Phi + delta * phi + (tau - 1.0) * theta
        a_term = (1.0 - tau) - Phi
        b_term = phi / delta
    s = g.mean(axis=1)
    elbo = (p * math.log(tau) + p * math.log(1.0 - tau) - p * xp.log(s)
            - 0.5 * xp.sum(nu * nu, axis=1) - 0.5 * xp.sum(d * d, axis=1)
            + xp.sum(xp.log(d), axis=1) + (c / 2.0 - p))
    dnu = -(a_term @ W) / s[:, None] - nu
    dd = 1.0 / d - d - (b_term @ W2) * (d / s[:, None])
    return -elbo, -xp.concatenate([dnu, dd], axis=1), s


@dataclass
class ElboResult:
    sigma: np.ndarray
    elbo: np.ndarray
    nu: np.ndarray
    d: np.ndarray
    success: np.ndarray
    iterations: np.ndarray
    evaluations: np.ndarray
    message: list
    projected_grad: np.ndarray
    seconds: float
    backend: str
    stats: dict = field(default_factory=dict)


_MSG_FTOL = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
_MSG_PGTOL = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
_MSG_MAXIT = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
_MSG_MAXFUN = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
_MSG_LS = "ABNORMAL: LINE SEARCH FAILED"
_MSG_NAN = "ABNORMAL: NON-FINITE OBJECTIVE"


def fit_elbo_batch(
    spectra,
    mean,
    components,
    *,
    tau=0.1,
    backend="numpy",
    pool=256,
    maxiter=10000,
    maxfun=15000,
    m=10,
    factr=1e7,
    pgtol=1e-5,
    sd_min=1e-10,
    maxls=20,
    retry_failed=True,
    polish_restarts=0,
    progress=None,
):
    """Maximise the PB ELBO for every spectrum independently (fixed tau).

    Mirrors SciPy's L-BFGS-B as called by the adapter: same objective, bounds
    (d >= sd_min), init (nu=0, d=1), memory m=10, More-Thuente line search
    (SciPy's own DCSRCH, ftol=1e-3, gtol=0.9, xtol=0.1), first step
    min(1/||g||, 1), skip rule for BFGS pairs, memory reset on a failed line
    search, and the stopping tests factr/pgtol/maxiter/maxfun. Then the adapter's
    one retry from the last iterate (5x maxiter, 20x maxiter evaluations,
    maxls=50, fresh memory). Every row has its own state; only the objective and
    gradient evaluations are batched.

    ``polish_restarts`` (default 0 = the reference procedure) restarts a
    converged row from its end point with fresh memory up to that many times,
    stopping once a restart no longer lowers -ELBO by more than the factr
    threshold. This is NOT the reference: SciPy's relative-reduction rule can
    stop the reference after a handful of iterations far from the optimum, and
    polishing measures how much that moves sigma.
    """

    if tau is None or not 0 < float(tau) < 1:
        raise ValueError("only a fixed tau in (0, 1) is implemented")
    tau = float(tau)
    started = time.perf_counter()
    be = _Backend(backend)
    xp = be.xp
    y, mu, W_h = _check_inputs(spectra, mean, components)
    rows, p = y.shape
    c = W_h.shape[1]
    n = 2 * c
    W = xp.asarray(W_h)
    W2 = W * W
    mu_d = xp.asarray(mu)
    eps = float(np.finfo(np.float64).eps)
    ftol = factr * eps
    lower_d = sd_min

    out_sigma = np.full(rows, np.nan)
    out_elbo = np.full(rows, np.nan)
    out_nu = np.full((rows, c), np.nan)
    out_d = np.full((rows, c), np.nan)
    out_ok = np.zeros(rows, bool)
    out_it = np.zeros(rows, np.int64)
    out_fev = np.zeros(rows, np.int64)
    out_pg = np.full(rows, np.nan)
    out_retry = np.zeros(rows, np.int64)
    out_polish = np.zeros(rows, np.int64)
    out_elbo_unpolished = np.full(rows, np.nan)
    out_first_msg = [""] * rows
    out_msg = [""] * rows

    B = int(min(pool, rows))
    slot_row = np.full(B, -1, np.int64)
    attempt = np.zeros(B, np.int64)
    it_prior = np.zeros(B, np.int64)
    fev_prior = np.zeros(B, np.int64)
    first_msg = [""] * B
    polish = np.zeros(B, np.int64)
    f_polish_start = np.zeros(B)
    V = xp.zeros((B, p))
    x = xp.zeros((B, n))
    f = np.zeros(B)  # host copy of objective (scalars per row)
    g = xp.zeros((B, n))
    S = xp.zeros((B, m, n))
    Yh = xp.zeros((B, m, n))
    rho = xp.zeros((B, m))
    gamma = xp.ones(B)
    k = np.zeros(B, np.int64)
    fev = np.zeros(B, np.int64)
    lim_it = np.full(B, maxiter, np.int64)
    lim_fev = np.full(B, maxfun, np.int64)
    lim_ls = np.full(B, maxls, np.int64)
    queue = list(range(rows))[::-1]
    stats = {"pool": B, "evaluation_batches": 0, "row_evaluations": 0,
             "linesearch_restarts": 0}

    def evaluate(slots, xs):
        stats["evaluation_batches"] += 1
        stats["row_evaluations"] += len(slots)
        return _elbo_and_grad(be, V[xp.asarray(slots)], W, W2, tau, xs[:, :c], xs[:, c:])

    def clear_memory(slots):
        si = xp.asarray(slots)
        S[si] = 0.0
        Yh[si] = 0.0
        rho[si] = 0.0
        gamma[si] = 1.0

    def start_attempt(slots, x0):
        slots = np.asarray(slots)
        x[xp.asarray(slots)] = x0
        clear_memory(slots)
        k[slots] = 0
        fev[slots] = 1
        fv, gv, _ = evaluate(slots, x0)
        f[slots] = be.host(fv)
        g[xp.asarray(slots)] = gv

    def load(slots):
        slots = np.asarray(slots, dtype=np.int64)
        if not len(slots):
            return
        rws = [queue.pop() for _ in slots]
        slot_row[slots] = rws
        attempt[slots] = 0
        polish[slots] = 0
        it_prior[slots] = 0
        fev_prior[slots] = 0
        lim_it[slots], lim_fev[slots], lim_ls[slots] = maxiter, maxfun, maxls
        V[xp.asarray(slots)] = xp.asarray(y[rws]) - mu_d
        x0 = xp.concatenate([xp.zeros((len(slots), c)), xp.ones((len(slots), c))], axis=1)
        start_attempt(slots, x0)

    def projected_gradient_norm(xs, gs):
        pg = gs.copy()
        dpart = xs[:, c:] - gs[:, c:]
        pg[:, c:] = xp.where(dpart < lower_d, xs[:, c:] - lower_d, gs[:, c:])
        return be.host(xp.max(xp.abs(pg), axis=1))

    def finish(slot, msg, good):
        if not good and polish[slot] > 0 and msg != _MSG_NAN and np.isfinite(f[slot]):
            # L-BFGS is monotone: a polish restart that runs out of iterations
            # still ends at least as high as the converged point it started from.
            good = True
        if not good and retry_failed and attempt[slot] == 0 and bool(xp.all(xp.isfinite(x[slot]))):
            attempt[slot] = 1
            first_msg[slot] = msg
            it_prior[slot] = k[slot]
            fev_prior[slot] = fev[slot]
            lim_it[slot], lim_fev[slot], lim_ls[slot] = 5 * maxiter, 20 * maxiter, 50
            start_attempt([slot], x[slot:slot + 1].copy())
            return
        r = slot_row[slot]
        if good and polish_restarts:
            if polish[slot] == 0:
                out_elbo_unpolished[r] = -float(f[slot])
            gained = (f_polish_start[slot] - f[slot]) > ftol * max(abs(f[slot]), 1.0)
            if polish[slot] < polish_restarts and (polish[slot] == 0 or gained):
                polish[slot] += 1
                f_polish_start[slot] = f[slot]
                it_prior[slot] += k[slot]
                fev_prior[slot] += fev[slot]
                lim_it[slot], lim_fev[slot], lim_ls[slot] = maxiter, maxfun, maxls
                start_attempt([slot], x[slot:slot + 1].copy())
                return
        xs = x[slot:slot + 1]
        _, gv, s = evaluate([slot], xs)
        out_sigma[r] = float(be.host(s)[0])
        out_elbo[r] = -float(f[slot])
        out_nu[r] = be.host(xs[0, :c])
        out_d[r] = be.host(xs[0, c:])
        out_ok[r] = bool(good) and np.isfinite(out_sigma[r]) and out_sigma[r] > 0
        out_it[r] = it_prior[slot] + k[slot]
        out_fev[r] = fev_prior[slot] + fev[slot]
        out_pg[r] = float(projected_gradient_norm(xs, gv)[0])
        out_retry[r] = attempt[slot]
        out_polish[r] = polish[slot]
        out_first_msg[r] = first_msg[slot] if attempt[slot] else msg
        out_msg[r] = msg
        slot_row[slot] = -1
        stats["finished"] = stats.get("finished", 0) + 1
        if progress is not None:
            progress(stats["finished"], rows)

    def directions(active):
        a = xp.asarray(active)
        ga = g[a]
        q = -ga
        full = len(active) == B
        Sa, Ya, ra = (S, Yh, rho) if full else (S[a], Yh[a], rho[a])
        alphas = []
        for j in range(m):  # newest -> oldest; empty pairs have rho = 0
            al = ra[:, j] * xp.sum(Sa[:, j] * q, axis=1)
            q = q - al[:, None] * Ya[:, j]
            alphas.append(al)
        r_ = (gamma if full else gamma[a])[:, None] * q
        for j in reversed(range(m)):
            bt = ra[:, j] * xp.sum(Ya[:, j] * r_, axis=1)
            r_ = r_ + Sa[:, j] * (alphas[j] - bt)[:, None]
        return r_

    load(np.arange(B))
    # Rows whose initial point already satisfies pgtol.
    while True:
        active = np.flatnonzero(slot_row >= 0)
        if not len(active):
            break
        a = xp.asarray(active)
        xa, ga = x[a], g[a]
        dirs = directions(active)
        # Variables already on their bound that the direction pushes further out
        # are held fixed this iteration (the projected step of L-BFGS-B).
        blocked = (xa[:, c:] <= lower_d) & (dirs[:, c:] < 0)
        dirs[:, c:] = xp.where(blocked, 0.0, dirs[:, c:])
        gd = be.host(xp.sum(ga * dirs, axis=1))
        # L-BFGS-B: a non-descent direction resets memory (steepest descent next).
        nondesc = np.flatnonzero(~(gd < 0))
        if len(nondesc):
            ni = xp.asarray(nondesc)
            pgd = -ga[ni]
            pgd[:, c:] = xp.where(blocked[ni], 0.0, pgd[:, c:])
            dirs[ni] = pgd
            clear_memory(active[nondesc])
            gd = be.host(xp.sum(ga * dirs, axis=1))
        dnorm = be.host(xp.sqrt(xp.sum(dirs * dirs, axis=1)))
        # Max feasible step along d for the lower-bounded d-block.
        dd_ = dirs[:, c:]
        room = xa[:, c:] - lower_d
        ratio = xp.where(dd_ < 0, room / xp.where(dd_ < 0, -dd_, 1.0), xp.inf)
        stpmx = np.minimum(be.host(xp.min(ratio, axis=1)), 1e10)
        first = k[active] == 0
        stpmx = np.where(first, 1.0, stpmx)
        stp = np.where(first, np.minimum(1.0 / np.maximum(dnorm, 1e-300), stpmx), np.minimum(1.0, stpmx))
        # Line search: one More-Thuente state per row, evaluations batched.
        nA = len(active)
        stpmx_safe = np.maximum(stpmx, 0.0)
        stp = np.minimum(stp, stpmx_safe)
        mt = _MoreThuente(stp, f[active].copy(), gd, stpmx_safe)
        status = mt.status.copy()
        status[~(stpmx_safe > 0)] = -1
        ifun = np.zeros(nA, np.int64)
        f_try = f[active].copy()
        x_try = xa.copy()
        g_try = ga.copy()
        while (status == 0).any():
            sj = np.flatnonzero(status == 0)
            si = xp.asarray(sj)
            stp_d = xp.asarray(mt.stp[sj])
            cand = xa[si] + stp_d[:, None] * dirs[si]
            cand[:, c:] = xp.maximum(cand[:, c:], lower_d)
            fv, gv, _ = evaluate(active[sj], cand)
            fv_h = be.host(fv)
            gdv_h = be.host(xp.sum(gv * dirs[si], axis=1))
            x_try[si] = cand
            g_try[si] = gv
            f_try[sj] = fv_h
            ifun[sj] += 1
            fev[active[sj]] += 1
            finite = np.isfinite(fv_h) & np.isfinite(gdv_h)
            st = np.full(len(sj), -1)
            if finite.any():
                st[finite] = mt.iterate(sj[finite], fv_h[finite], gdv_h[finite])
            # Searches that need another evaluation but exhausted maxls fail.
            st = np.where((st == 0) & (ifun[sj] >= lim_ls[active[sj]]), -1, st)
            status[sj] = np.where(st == 2, 1, st)
        # Failed searches: restore x; reset memory, or stop if memory was empty.
        failed = np.flatnonzero(status == -1)
        if len(failed):
            empty_memory = np.ones(nA, bool)
            empty_memory[failed] = be.host(xp.all(rho[xp.asarray(active[failed])] == 0, axis=1))
        for j in failed:
            sl = active[j]
            if empty_memory[j]:
                finish(sl, _MSG_LS, False)
            else:
                stats["linesearch_restarts"] += 1
                clear_memory([sl])
                if fev[sl] > lim_fev[sl]:
                    finish(sl, _MSG_MAXFUN, False)
        acc = np.flatnonzero(status == 1)
        if len(acc):
            ai = xp.asarray(acc)
            sl = active[acc]
            sli = xp.asarray(sl)
            s_vec = x_try[ai] - xa[ai]
            y_vec = g_try[ai] - ga[ai]
            dr = xp.sum(s_vec * y_vec, axis=1)
            ddum = -xp.sum(s_vec * ga[ai], axis=1)
            yy = xp.sum(y_vec * y_vec, axis=1)
            keep_h = be.host(dr > eps * ddum)
            ku = np.flatnonzero(keep_h)
            if len(ku):
                kidx = xp.asarray(ku)
                rows_k = xp.asarray(sl[ku])
                # shift the selected rows' histories one place older, insert newest at 0
                S[rows_k, 1:] = S[rows_k, :-1]
                Yh[rows_k, 1:] = Yh[rows_k, :-1]
                rho[rows_k, 1:] = rho[rows_k, :-1]
                S[rows_k, 0] = s_vec[kidx]
                Yh[rows_k, 0] = y_vec[kidx]
                rho[rows_k, 0] = 1.0 / dr[kidx]
                gamma[rows_k] = dr[kidx] / yy[kidx]
            f_old = f[sl].copy()
            x[sli] = x_try[ai]
            g[sli] = g_try[ai]
            f[sl] = f_try[acc]
            k[sl] += 1
            pg = projected_gradient_norm(x[sli], g[sli])
            fn = f[sl]
            rel = (f_old - fn) <= ftol * np.maximum(np.maximum(np.abs(f_old), np.abs(fn)), 1.0)
            # SciPy checks the iteration/evaluation limits before the
            # convergence tests run inside setulb.
            c_it = k[sl] >= lim_it[sl]
            c_fev = ~c_it & (fev[sl] > lim_fev[sl])
            c_pg = ~c_it & ~c_fev & (pg <= pgtol)
            c_rel = ~c_it & ~c_fev & ~c_pg & rel
            c_nan = ~np.isfinite(fn)
            for j in np.flatnonzero(c_it | c_fev | c_pg | c_rel | c_nan):
                s_ = sl[j]
                if c_nan[j]:
                    finish(s_, _MSG_NAN, False)
                elif c_it[j]:
                    finish(s_, _MSG_MAXIT, False)
                elif c_fev[j]:
                    finish(s_, _MSG_MAXFUN, False)
                elif c_pg[j]:
                    finish(s_, _MSG_PGTOL, True)
                else:
                    finish(s_, _MSG_FTOL, True)
        free = np.flatnonzero(slot_row < 0)
        if len(free) and queue:
            load(free[: len(queue)])
    be.sync()
    res = ElboResult(out_sigma, out_elbo, out_nu, out_d, out_ok, out_it, out_fev, out_msg,
                     out_pg, time.perf_counter() - started, backend, stats)
    res.stats["retry_count"] = out_retry
    res.stats["polish_restarts"] = out_polish
    res.stats["elbo_before_polish"] = out_elbo_unpolished
    res.stats["initial_message"] = out_first_msg
    return res


# --------------------------------------------------------------------------- ELBO, Newton
def _elbo_newton_terms(be, V, W, W2, K, tau, nu, d, hess_dtype):
    """-ELBO, its gradient, and the pieces of its Hessian for rows of V = y - mu.

    With theta_j = v_j - W_j nu, D_j = sum_k W_jk^2 d_k^2 and the per-channel
    expected pinball g_j(theta, D), S = sum_j g_j and s = S / p,

        -ELBO = p log s + |nu|^2/2 + |d|^2/2 - sum log d + const
        Hess  = Hs / s + diag(1, 1 + 1/d^2) - u u^T,   u = grad S / (s sqrt p),

    where Hs = sum_j Hess g_j is positive semidefinite (g_j is convex and
    increasing in delta = sqrt(D), which is convex in d). Hs is built from three
    GEMMs against fixed Khatri-Rao products of W and W^2 (``K``), the same trick
    as the MAP step. Returns f, grad, A = Hs/s + diag (>= I), u, s.
    """
    xp = be.xp
    B, p = V.shape
    c = W.shape[1]
    theta = V - nu @ W.T
    D = (d * d) @ W2.T
    delta = xp.sqrt(D)
    z = theta / delta
    Phi = be.ndtr(z)
    phi = xp.exp(-0.5 * z * z) / SQRT_2PI
    gj = theta * Phi + delta * phi + (tau - 1.0) * theta
    a_term = (1.0 - tau) - Phi          # -dg/dtheta
    b_term = phi / delta                # d2g/dtheta2 = 2 dg/dD
    s = gj.mean(axis=1)
    elbo = (p * math.log(tau) + p * math.log(1.0 - tau) - p * xp.log(s)
            - 0.5 * xp.sum(nu * nu, axis=1) - 0.5 * xp.sum(d * d, axis=1)
            + xp.sum(xp.log(d), axis=1) + (c / 2.0 - p))
    bW2 = b_term @ W2
    G = xp.concatenate([a_term @ W, bW2 * d], axis=1)          # grad S
    grad = G / s[:, None] + xp.concatenate([nu, d - 1.0 / d], axis=1)
    # Hessian weights per channel: theta-theta, theta-D (x2 for dD/dd), D-D (x4).
    w1 = b_term
    w2 = z * b_term / delta
    w3 = b_term * (z * z - 1.0) / D
    hd = hess_dtype
    iu0, iu1 = K["iu"]
    gemm = K.get("gemm") or (lambda name, w: w.astype(hd) @ K[name])
    Hnn_u = gemm("WW", w1).astype(xp.float64)
    Hnd = gemm("WW2", w2).astype(xp.float64).reshape(B, c, c) * d[:, None, :]
    Hdd_u = gemm("W2W2", w3).astype(xp.float64)
    n = 2 * c
    A = xp.zeros((B, n, n))
    A[:, iu0, iu1] = Hnn_u
    A[:, iu1, iu0] = Hnn_u
    dd_ = Hdd_u * (d[:, iu0] * d[:, iu1])
    A[:, c + iu0, c + iu1] = dd_
    A[:, c + iu1, c + iu0] = dd_
    A[:, :c, c:] = Hnd
    A[:, c:, :c] = xp.swapaxes(Hnd, 1, 2)
    A /= s[:, None, None]
    diag = xp.concatenate([xp.ones_like(nu), 1.0 + bW2 / s[:, None] + 1.0 / (d * d)], axis=1)
    ar = xp.arange(n)
    A[:, ar, ar] += diag
    u = G / (s[:, None] * math.sqrt(p))
    return -elbo, grad, A, u, s


def _khatri_rao(be, W, hess_dtype):
    xp = be.xp
    p, c = W.shape
    iu0, iu1 = np.triu_indices(c)
    W2 = W * W
    i0, i1 = xp.asarray(iu0), xp.asarray(iu1)
    return {"iu": (i0, i1),
            "WW": (W[:, i0] * W[:, i1]).astype(hess_dtype),
            "WW2": (W[:, :, None] * W2[:, None, :]).reshape(p, c * c).astype(hess_dtype),
            "W2W2": (W2[:, i0] * W2[:, i1]).astype(hess_dtype)}


def fit_elbo_newton(
    spectra,
    mean,
    components,
    *,
    tau=0.1,
    backend="numpy",
    chunk=1024,
    maxiter=300,
    gtol=1e-5,
    dec_rtol=1e-13,
    kappa_max=0.99,
    max_backtracks=50,
    hess_dtype="float64",
    x0=None,
    gemm=None,
    progress=None,
):
    """Maximise the PB ELBO per spectrum by damped Newton with the exact Hessian.

    Same objective, parameters and start (nu = 0, d = 1) as the adapter, but a
    second-order optimiser: each iteration solves A p = -grad with A = Hs/s +
    diag (positive definite), then applies the rank-one correction -u u^T by
    Sherman-Morrison when the true Hessian is positive definite (u^T A^-1 u <
    ``kappa_max``); otherwise the A step alone, which is still a descent
    direction. An Armijo backtracking search (fraction to the d > 0 boundary
    0.95) globalises it. The log-barrier in the ELBO keeps d away from 0, so the
    d >= 1e-10 bound of the reference never binds.

    A row converges when the Newton decrement g.p is <= dec_rtol * max(1, |f|)
    (no representable progress left) or max|grad| <= gtol (SciPy's pgtol).
    ``hess_dtype='float32'`` builds the Hessian GEMMs in single precision (an
    inexact Newton step; the gradient and objective stay float64).
    ``gemm(name, weights)`` replaces those products with an external one (name is
    WW, WW2 or W2W2; see vibes_webgpu.py, where they run on WebGPU in float32).
    This is NOT the reference procedure: it aims at the optimum of the same
    objective, so compare it with polished L-BFGS fits, not with SciPy's stops.
    """
    if tau is None or not 0 < float(tau) < 1:
        raise ValueError("only a fixed tau in (0, 1) is implemented")
    tau = float(tau)
    started = time.perf_counter()
    be = _Backend(backend)
    xp = be.xp
    hd = {"float64": xp.float64, "float32": xp.float32}[hess_dtype]
    y, mu, W_h = _check_inputs(spectra, mean, components)
    rows, p = y.shape
    c = W_h.shape[1]
    n = 2 * c
    W = xp.asarray(W_h)
    W2 = W * W
    K = _khatri_rao(be, W, hd)
    if gemm is not None:
        K["gemm"] = gemm
    mu_d = xp.asarray(mu)
    out = {"sigma": np.full(rows, np.nan), "elbo": np.full(rows, np.nan),
           "pg": np.full(rows, np.nan), "dec": np.full(rows, np.nan)}
    out_x = np.full((rows, n), np.nan)
    out_ok = np.zeros(rows, bool)
    out_it = np.zeros(rows, np.int64)
    out_fev = np.zeros(rows, np.int64)
    out_exact = np.zeros(rows, np.int64)
    out_final_exact = np.zeros(rows, bool)
    out_msg = [""] * rows
    stats = {"chunk": chunk, "hess_dtype": hess_dtype, "newton_iterations": 0, "row_hessians": 0,
             "row_evaluations": 0, "batches": 0}

    for lo in range(0, rows, chunk):
        sl = np.arange(lo, min(rows, lo + chunk))
        b = len(sl)
        V = xp.asarray(y[sl]) - mu_d
        if x0 is None:
            x = xp.concatenate([xp.zeros((b, c)), xp.ones((b, c))], axis=1)
        else:
            x = xp.asarray(np.asarray(x0, dtype=np.float64)[sl])
        done = np.zeros(b, bool)
        msg = [""] * b
        it_n = np.zeros(b, np.int64)
        fev = np.zeros(b, np.int64)
        exact_n = np.zeros(b, np.int64)
        f_fin = np.full(b, np.nan)
        s_fin = np.full(b, np.nan)
        pg_fin = np.full(b, np.nan)
        dec_fin = np.full(b, np.nan)
        final_exact = np.zeros(b, bool)
        for it in range(maxiter + 1):
            act = np.flatnonzero(~done)
            if not len(act):
                break
            ai = xp.asarray(act)
            xa = x[ai]
            f, g, A, u, s = _elbo_newton_terms(be, V[ai], W, W2, K, tau, xa[:, :c], xa[:, c:], hd)
            stats["row_hessians"] += len(act)
            stats["batches"] += 1
            fev[act] += 1
            sol = xp.linalg.solve(A, xp.stack([-g, u], axis=2))
            q, w = sol[:, :, 0], sol[:, :, 1]
            kappa = xp.sum(u * w, axis=1)
            uq = xp.sum(u * q, axis=1)
            exact = kappa < kappa_max
            step = q + xp.where(exact, uq / xp.where(exact, 1.0 - kappa, 1.0), 0.0)[:, None] * w
            gd = xp.sum(g * step, axis=1)
            # Exact step that is not a descent direction (numerically indefinite): use the A step.
            bad = ~(gd < 0)
            step = xp.where(bad[:, None], q, step)
            gd = xp.where(bad, xp.sum(g * q, axis=1), gd)
            exact = exact & ~bad
            f_h, gd_h = be.host(f), be.host(gd)
            pg_h = be.host(xp.max(xp.abs(g), axis=1))
            s_h = be.host(s)
            dec = -gd_h
            exact_h = be.host(exact)
            # The decrement test needs the exact (positive definite) Hessian step:
            # A >= Hess, so a small decrement on A alone does not bound the remaining gain.
            conv = ((dec <= dec_rtol * np.maximum(1.0, np.abs(f_h))) & exact_h) | (pg_h <= gtol)
            nonfin = ~np.isfinite(f_h) | ~np.isfinite(gd_h)
            stop = conv | nonfin | (it == maxiter)
            for j in np.flatnonzero(stop):
                r = act[j]
                done[r] = True
                f_fin[r], s_fin[r], pg_fin[r], dec_fin[r] = f_h[j], s_h[j], pg_h[j], dec[j]
                final_exact[r] = exact_h[j]
                msg[r] = ("CONVERGENCE: NEWTON DECREMENT" if conv[j] and pg_h[j] > gtol
                          else "CONVERGENCE: MAX |GRAD| <= GTOL" if conv[j]
                          else _MSG_NAN if nonfin[j] else _MSG_MAXIT)
            go = np.flatnonzero(~stop)
            if not len(go):
                continue
            gi = xp.asarray(go)
            rows_go = act[go]
            exact_n[rows_go] += exact_h[go].astype(np.int64)
            it_n[rows_go] += 1
            xg, stg, fg_, gdg = xa[gi], step[gi], f_h[go], gd_h[go]
            dpart, dstep = xg[:, c:], stg[:, c:]
            ratio = xp.where(dstep < 0, dpart / xp.where(dstep < 0, -dstep, 1.0), xp.inf)
            t = np.minimum(1.0, 0.95 * be.host(xp.min(ratio, axis=1)))
            pending = np.ones(len(go), bool)
            x_new = xg.copy()
            for _ in range(max_backtracks):
                pj = np.flatnonzero(pending)
                if not len(pj):
                    break
                pji = xp.asarray(pj)
                cand = xg[pji] + xp.asarray(t[pj])[:, None] * stg[pji]
                fv, _, _ = _elbo_and_grad(be, V[xp.asarray(rows_go[pj])], W, W2, tau, cand[:, :c], cand[:, c:])
                stats["row_evaluations"] += len(pj)
                fev[rows_go[pj]] += 1
                fv = be.host(fv)
                okj = np.isfinite(fv) & (fv <= fg_[pj] + 1e-4 * t[pj] * gdg[pj])
                if okj.any():
                    acc = pj[okj]
                    x_new[xp.asarray(acc)] = cand[xp.asarray(np.flatnonzero(okj))]
                    pending[acc] = False
                t[pj[~okj]] *= 0.5
            x[xp.asarray(rows_go)] = x_new
            # Rows whose search found no decrease cannot progress: stop them as converged
            # only if already near-stationary, else flag.
            for j in np.flatnonzero(pending):
                r = rows_go[j]
                done[r] = True
                f_fin[r], s_fin[r] = fg_[j], s_h[go[j]]
                pg_fin[r], dec_fin[r] = pg_h[go[j]], -gdg[j]
                final_exact[r] = exact_h[go[j]]
                msg[r] = "ABNORMAL: NEWTON LINE SEARCH FAILED"
            stats["newton_iterations"] = max(stats["newton_iterations"], it + 1)
            if progress is not None:
                progress(lo + int(done.sum()), rows)
        xh = be.host(x)
        out["sigma"][sl], out["elbo"][sl] = s_fin, -f_fin
        out["pg"][sl], out["dec"][sl] = pg_fin, dec_fin
        out_x[sl] = xh
        out_it[sl], out_fev[sl], out_exact[sl] = it_n, fev, exact_n
        out_final_exact[sl] = final_exact
        for j, r in enumerate(sl):
            out_msg[r] = msg[j]
            # A failed search at a point whose exact decrement is already ~1e-9
            # relative is a converged point limited by roundoff in f.
            near = (msg[j].startswith("ABNORMAL: NEWTON") and final_exact[j]
                    and dec_fin[j] <= 1e-9 * max(1.0, abs(f_fin[j])))
            out_ok[r] = (msg[j].startswith("CONVERGENCE") or near) and np.isfinite(s_fin[j]) and s_fin[j] > 0
    be.sync()
    res = ElboResult(out["sigma"], out["elbo"], out_x[:, :c], out_x[:, c:], out_ok, out_it, out_fev,
                     out_msg, out["pg"], time.perf_counter() - started, backend, stats)
    res.stats["newton_decrement"] = out["dec"]
    res.stats["exact_steps"] = out_exact
    res.stats["final_step_exact"] = out_final_exact
    return res


# --------------------------------------------------------------------------- MAP
@dataclass
class MapResult:
    baseline: np.ndarray
    latent: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    primal_objective: np.ndarray
    dual_objective: np.ndarray
    duality_gap: np.ndarray
    gap_tolerance: np.ndarray
    seconds: float
    backend: str


def solve_map_batch(
    spectra,
    mean,
    components,
    *,
    sigma,
    tau=0.1,
    backend="numpy",
    chunk=512,
    maxiter=100,
    gap_rtol=1e-8,
    gap_atol=1e-9,
    hess_dtype="float64",
    gemm=None,
    refine_steps=25,
):
    """Pinball MAP for each row:  min_x sum pb_tau(v - W x) + sigma/2 |x|^2,  v = y - mu.

    Primal-dual interior point (Mehrotra predictor-corrector) on the KKT system
        sigma x - W^T a = 0,   W x + p - n = v,   tau-1 <= a <= tau,
        p (tau - a) = 0,  n (a - tau + 1) = 0,  p, n >= 0,
    i.e. the dual box QP with the primal latent x carried as its own variable,
    so small sigma does not amplify dual roundoff into x. Each Newton step is
    one c x c Cholesky per row: (sigma I + W^T diag(1/D) W) dx = rhs.

    A row is accepted only if the primal-dual gap of the *original* pinball
    objective, P(x) - D(clip(a)), is within gap_atol + gap_rtol*max(1,|P|,|D|)
    (CLARABEL's default gap tolerances are 1e-8 abs/rel). P(x) >= optimum >=
    D(a) for any x and any box-feasible a, so this certificate does not trust
    the solver. Failed rows return NaN. ``sigma`` must be > 0 (no EBS).
    ``hess_dtype='float32'`` or ``gemm('WW', weights)`` (upper triangle) builds
    only the system matrix inexactly; it then preconditions conjugate gradients on
    the exact float64 operator (at most ``refine_steps`` per solve), and residuals
    and the certificate stay float64.
    """
    started = time.perf_counter()
    be = _Backend(backend)
    xp = be.xp
    y, mu, W_h = _check_inputs(spectra, mean, components)
    rows, p = y.shape
    c = W_h.shape[1]
    sig_all = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (rows,)).copy()
    tau_all = np.broadcast_to(np.asarray(tau, dtype=np.float64), (rows,)).copy()
    if np.any(~np.isfinite(sig_all) | (sig_all <= 0)):
        raise ValueError("sigma must be finite and > 0 (sigma=0 EBS unsupported)")
    if np.any((tau_all <= 0) | (tau_all >= 1)):
        raise ValueError("tau must lie in (0, 1)")
    W = xp.asarray(W_h)
    WT = xp.ascontiguousarray(W.T)
    # W (x) W, so every row's W^T diag(1/D) W is one GEMM: (b x p) @ (p x c^2).
    hd = {"float64": xp.float64, "float32": xp.float32}[hess_dtype]
    inexact = hess_dtype == "float32" or gemm is not None
    WW = (W[:, :, None] * W[:, None, :]).reshape(p, c * c).astype(hd)
    iu0, iu1 = (xp.asarray(i) for i in np.triu_indices(c))
    mu_d = xp.asarray(mu)
    eye = xp.eye(c)
    out = {k: np.full(rows, np.nan) for k in ("P", "D", "gap", "tol")}
    base = np.full((rows, p), np.nan)
    lat = np.full((rows, c), np.nan)
    conv = np.zeros(rows, bool)
    its = np.zeros(rows, np.int64)

    for lo_r in range(0, rows, chunk):
        sl = slice(lo_r, min(rows, lo_r + chunk))
        b = sl.stop - sl.start
        v = xp.asarray(y[sl]) - mu_d
        sg = xp.asarray(sig_all[sl])[:, None]
        tu = xp.asarray(tau_all[sl])[:, None]
        lo, hi = tu - 1.0, tu

        def certificate(x_, a_):
            r_ = v - x_ @ WT
            P = xp.sum(xp.maximum(tu * r_, (tu - 1.0) * r_), axis=1) + 0.5 * sg[:, 0] * xp.sum(x_ * x_, axis=1)
            ac = xp.clip(a_, lo, hi)
            wa = ac @ W
            Dv = xp.sum(ac * v, axis=1) - xp.sum(wa * wa, axis=1) / (2.0 * sg[:, 0])
            gap = P - Dv
            tol = gap_atol + gap_rtol * xp.maximum(1.0, xp.maximum(xp.abs(P), xp.abs(Dv)))
            return P, Dv, gap, tol

        x = xp.zeros((b, c))
        a = xp.broadcast_to(tu - 0.5, (b, p)).copy()  # box centre
        scale = xp.maximum(xp.mean(xp.abs(v), axis=1, keepdims=True), 1e-12)
        z1 = xp.maximum(-v, 0.0) + scale  # ~ n (negative part of the residual)
        z2 = xp.maximum(v, 0.0) + scale   # ~ p (positive part)
        done = xp.zeros(b, bool)
        it_c = xp.zeros(b, dtype=xp.int64)
        x_best, a_best = x.copy(), a.copy()

        def max_step(s1, s2, z1, z2, da, dz1, dz2):
            al = xp.ones(b)
            for val, dv in ((s1, da), (s2, -da), (z1, dz1), (z2, dz2)):
                neg = dv < 0
                ratio = xp.where(neg, val / xp.where(neg, -dv, 1.0), xp.inf)
                al = xp.minimum(al, xp.min(ratio, axis=1))
            return al

        for it in range(1, maxiter + 1):
            s1 = a - lo
            s2 = hi - a
            if inexact:  # slacks can round to exactly 0 while PCG finishes the last digits
                s1 = xp.maximum(s1, 1e-15)
                s2 = xp.maximum(s2, 1e-15)
            r_x = sg * x - a @ W                 # stationarity in x
            r_a = x @ WT - v - z1 + z2           # W x + p - n - v  (z1 = n, z2 = p)
            mu_c = (xp.sum(s1 * z1, axis=1) + xp.sum(s2 * z2, axis=1)) / (2 * p)
            Dg = z1 / s1 + z2 / s2
            Dinv = 1.0 / Dg
            if inexact:
                frozen = ~xp.all(xp.isfinite(Dinv), axis=1)  # stays uncertified -> NaN result
                Dinv = xp.where(frozen[:, None], 1.0, Dinv)
            if gemm is None:
                M = (Dinv.astype(hd) @ WW).astype(xp.float64).reshape(b, c, c)
            else:
                M = xp.zeros((b, c, c))
                up = gemm("WW", Dinv).astype(xp.float64)
                M[:, iu0, iu1] = up
                M[:, iu1, iu0] = up
            M = M + sg[:, :, None] * eye
            if inexact:
                # Single-precision rounding can leave M indefinite: scale to unit
                # diagonal, add jitter until it factorises. PCG removes the error.
                Si = 1.0 / xp.sqrt(xp.diagonal(M, axis1=1, axis2=2))
                Ms = M * Si[:, :, None] * Si[:, None, :]
                jitter = 1e-5
                while True:
                    try:
                        L = xp.linalg.cholesky(Ms + jitter * eye)
                        # one dense inverse is far cheaper than repeated solves inside PCG
                        Li = xp.linalg.inv(L)
                        Minv = xp.swapaxes(Li, 1, 2) @ Li
                        break
                    except np.linalg.LinAlgError:
                        jitter *= 10
                        if jitter > 1.0:
                            raise
            else:
                Si = 1.0
                L = xp.linalg.cholesky(M)
            LT = xp.swapaxes(L, 1, 2)

            def prec(r):
                if inexact:
                    return Si * (Minv @ (Si * r)[:, :, None])[:, :, 0]
                return xp.linalg.solve(LT, xp.linalg.solve(L, r[:, :, None]))[:, :, 0]

            def pcg(rhs, x_):
                op = lambda q: sg * q + (Dinv * (q @ WT)) @ W
                r_ = rhs - op(x_)
                z_ = prec(r_)
                d_ = z_
                rz = xp.sum(r_ * z_, axis=1)
                norm = xp.sqrt(xp.sum(rhs * rhs, axis=1))
                for _ in range(refine_steps):
                    if bool(be.host(xp.all(xp.sqrt(xp.sum(r_ * r_, axis=1)) <= 1e-12 * norm))):
                        break
                    Ad = op(d_)
                    al_ = rz / xp.where(xp.sum(d_ * Ad, axis=1) == 0, 1.0, xp.sum(d_ * Ad, axis=1))
                    x_ = x_ + al_[:, None] * d_
                    r_ = r_ - al_[:, None] * Ad
                    z_ = prec(r_)
                    rz2 = xp.sum(r_ * z_, axis=1)
                    d_ = z_ + (rz2 / xp.where(rz == 0, 1.0, rz))[:, None] * d_
                    rz = rz2
                return x_

            def direction(t1, t2):
                h = -r_a + t1 / s1 - t2 / s2
                rhs = (Dinv * h) @ W - r_x
                dx = prec(rhs)
                if inexact:  # mixed precision: float32 M preconditions an exact float64 operator
                    dx = pcg(rhs, dx)
                da = Dinv * (h - dx @ WT)
                dz1 = (t1 - z1 * da) / s1
                dz2 = (t2 + z2 * da) / s2
                return dx, da, dz1, dz2

            dx, da, dz1, dz2 = direction(-s1 * z1, -s2 * z2)
            al = max_step(s1, s2, z1, z2, da, dz1, dz2)[:, None]
            mu_aff = (xp.sum((s1 + al * da) * (z1 + al * dz1), axis=1)
                      + xp.sum((s2 - al * da) * (z2 + al * dz2), axis=1)) / (2 * p)
            sm = (((mu_aff / mu_c) ** 3) * mu_c)[:, None]
            dx, da, dz1, dz2 = direction(-s1 * z1 + sm - da * dz1, -s2 * z2 + sm + da * dz2)
            al = xp.minimum(1.0, 0.99 * max_step(s1, s2, z1, z2, da, dz1, dz2))
            al = xp.where(done, 0.0, al)
            if inexact:
                bad = frozen | ~xp.isfinite(al)
                bad = bad | ~xp.all(xp.isfinite(dx), axis=1) | ~xp.all(xp.isfinite(da), axis=1)
                al = xp.where(bad, 0.0, al)
                dx, da = xp.where(bad[:, None], 0.0, dx), xp.where(bad[:, None], 0.0, da)
                dz1, dz2 = xp.where(bad[:, None], 0.0, dz1), xp.where(bad[:, None], 0.0, dz2)
            al = al[:, None]
            x = x + al * dx
            a = a + al * da
            z1 = z1 + al * dz1
            z2 = z2 + al * dz2
            P, Dv, gap, tol = certificate(x, a)
            good = xp.isfinite(gap) & (gap <= tol) & (gap >= -tol)
            newly = (~done) & good
            x_best = xp.where(newly[:, None], x, x_best)
            a_best = xp.where(newly[:, None], a, a_best)
            it_c = xp.where(newly, it, it_c)
            done = done | newly
            if bool(be.host(xp.all(done))):
                break
        x_fin = xp.where(done[:, None], x_best, x)
        a_fin = xp.where(done[:, None], a_best, a)
        P, Dv, gap, tol = certificate(x_fin, a_fin)
        ok = xp.isfinite(gap) & (gap <= tol) & (gap >= -tol)
        bl = mu_d[None, :] + x_fin @ WT
        okh = be.host(ok)
        base[sl] = np.where(okh[:, None], be.host(bl), np.nan)
        lat[sl] = np.where(okh[:, None], be.host(x_fin), np.nan)
        conv[sl] = okh
        its[sl] = np.where(be.host(it_c) > 0, be.host(it_c), maxiter)
        out["P"][sl], out["D"][sl] = be.host(P), be.host(Dv)
        out["gap"][sl], out["tol"][sl] = be.host(gap), be.host(tol)
    be.sync()
    return MapResult(base, lat, conv, its, out["P"], out["D"], out["gap"], out["tol"],
                     time.perf_counter() - started, backend)


def vibes_batch(spectra, mean, components, *, tau=0.1, backend="numpy", pool=256,
                map_chunk=512, **elbo_kwargs):
    """ELBO then MAP; returns (baseline, ElboResult, MapResult). Failed rows are NaN."""
    e = fit_elbo_batch(spectra, mean, components, tau=tau, backend=backend, pool=pool, **elbo_kwargs)
    good = e.success
    baseline = np.full(np.shape(spectra), np.nan)
    mres = None
    if good.any():
        mres = solve_map_batch(np.asarray(spectra)[good], mean, components, sigma=e.sigma[good],
                               tau=tau, backend=backend, chunk=map_chunk)
        baseline[good] = mres.baseline
    return baseline, e, mres


# --------------------------------------------------------------------------- CPU reference, fused
def fused_pb_objective(y, mean, components, tau):
    """SciPy ``jac=True`` objective: identical algebra to vibes_pb_optim_prep's
    fun/jac (negative ELBO) for fixed tau, computed once per point with W**2
    cached. Returns (fun, init, bounds, sigma_hat)."""
    from scipy.special import ndtr

    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    W = np.asarray(components, dtype=np.float64)
    W2 = W * W
    p, c = W.shape
    const = p * math.log(tau) + p * math.log(1.0 - tau) + (c / 2.0 - p)
    v = y - mu

    def parts(pars):
        nu, d = pars[1:1 + c], pars[1 + c:]
        theta = v - W @ nu
        delta = np.sqrt(W2 @ (d * d))
        z = theta / delta
        Phi = ndtr(z)
        phi = np.exp(-0.5 * z * z) / SQRT_2PI
        s = np.mean(theta * Phi + delta * phi + (tau - 1.0) * theta)
        return nu, d, theta, delta, Phi, phi, s

    def fun(pars):
        nu, d, theta, delta, Phi, phi, s = parts(pars)
        elbo = const - p * math.log(s) - 0.5 * nu @ nu - 0.5 * d @ d + np.sum(np.log(d))
        dtau = p / tau - p / (1 - tau) - np.sum(theta) / s
        dnu = -(((1.0 - tau) - Phi) @ W) / s - nu
        dd = 1.0 / d - d - ((phi / delta) @ W2) * (d / s)
        return -elbo, -np.concatenate(([dtau], dnu, dd))

    def sigma_hat(pars):
        return float(parts(pars)[-1])

    init = np.concatenate(([tau], np.zeros(c), np.ones(c)))
    bounds = ((tau, tau),) + ((-np.inf, np.inf),) * c + ((1e-10, np.inf),) * c
    return fun, init, bounds, sigma_hat


def fit_elbo_scipy_fused(y, mean, components, *, tau=0.1, maxiter=10000, retry_failed=True,
                         polish_restarts=0):
    """The adapter's ELBO procedure (L-BFGS-B, same options, same one retry)
    with the fused objective. Returns a dict like the adapter's diagnostics.
    ``polish_restarts`` > 0 adds restarts from the converged point (fresh
    memory) while they still lower -ELBO beyond the factr threshold; this is a
    deliberate departure from the reference, see fit_elbo_batch."""
    from scipy.optimize import minimize

    fun, init, bounds, sigma_hat = fused_pb_objective(y, mean, components, tau)
    opt = minimize(fun, init, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": maxiter})
    nit0, retry, first = int(opt.nit), 0, str(opt.message)
    if retry_failed and not opt.success and np.isfinite(opt.x).all():
        opt = minimize(fun, opt.x, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": 5 * maxiter, "maxls": 50, "maxfun": 20 * maxiter})
        retry = 1
    elbo0, polished = float(-opt.fun), 0
    ftol = 1e7 * np.finfo(float).eps
    ok_any = bool(opt.success)
    while ok_any and polished < polish_restarts:
        # A restart that hits maxiter still counts if it raised the ELBO: the
        # point of polishing is to escape SciPy's premature relative-reduction stop.
        nxt = minimize(fun, opt.x, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": maxiter})
        polished += 1
        gained = np.isfinite(nxt.fun) and (opt.fun - nxt.fun) > ftol * max(abs(nxt.fun), 1.0)
        if np.isfinite(nxt.fun) and nxt.fun <= opt.fun:
            opt = nxt
        if not gained:
            break
    if polish_restarts and ok_any:
        opt.success = True
    sig = sigma_hat(opt.x)
    if polish_restarts:
        return {"success": bool(opt.success) and np.isfinite(sig) and sig > 0, "sigma": sig,
                "elbo": float(-opt.fun), "elbo_before_polish": elbo0, "polish_restarts": polished,
                "iterations": int(opt.nit),
                "retry_count": retry, "initial_message": first, "message": str(opt.message)}
    return {"success": bool(opt.success) and np.isfinite(sig) and sig > 0, "sigma": sig,
            "elbo": float(-opt.fun), "iterations": int(opt.nit) + (nit0 if retry else 0),
            "retry_count": retry, "initial_message": first, "message": str(opt.message)}
