"""Batched VIBES ELBO (PB loss, fixed tau) as a compiled JAX loop. Experiment.

Two engines:

* ``engine="lbfgsb"``: a port of SciPy's L-BFGS-B (generalised Cauchy point with breakpoints, direct
  primal subspace step, Morales-Nocedal projection, memory refreshes, the fixed tau parameter's
  gradient in theta). Same mathematics as SciPy's ``__lbfgsb.c``; the matrices are formed explicitly
  per row rather than factorised incrementally, so it matches SciPy to roundoff, then separates the
  way any non-identical arithmetic does.
* ``engine="lbfgs"`` (default): projected L-BFGS with SciPy's line search and stopping rules but no
  Cauchy point or subspace step. It leaves SciPy's path at the first iteration and stops early on
  different spectra.

``lbfgs`` is the same optimiser as ``vibes_batch.fit_elbo_batch`` (per-row L-BFGS with m = 10,
SciPy's More-Thuente line search, projected handling of d >= 1e-10, SciPy's
stopping rules, the adapter's one retry, optional polish restarts), but the
iteration runs on the device: ``chunk_iters`` L-BFGS iterations are one
``jax.jit``-compiled ``lax.while_loop`` with the line search as a nested
``while_loop``, so there is no Python and no device->host sync inside a chunk.
Between chunks the host reads the stop codes, applies the retry / polish /
finish decisions and refills free slots from the queue (the rolling pool).

A compiled batch runs in lock-step: a finished row idles until the chunk ends,
and every row in the batch is computed on every iteration. To keep the long
tail (a few spectra need ~60,000 iterations) cheap, the pool shrinks to a
smaller compiled size once the queue is empty and few rows are left.

Importing this module turns on ``jax_enable_x64`` (the fit needs float64).
"""

from __future__ import annotations

import math
import time
from functools import partial

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.scipy.special import ndtr  # noqa: E402

import vibes_batch as vb  # noqa: E402

SQRT_2PI = math.sqrt(2.0 * math.pi)
eps64 = float(np.finfo(np.float64).eps)
RUN, PGTOL, FTOL, MAXIT, MAXFUN, LSFAIL, NAN, EMPTY = range(8)
_MSG = {PGTOL: vb._MSG_PGTOL, FTOL: vb._MSG_FTOL, MAXIT: vb._MSG_MAXIT, MAXFUN: vb._MSG_MAXFUN,
        LSFAIL: vb._MSG_LS, NAN: vb._MSG_NAN}
_STATE_KEYS = ("V", "x", "f", "g", "s", "S", "Y", "rho", "gamma", "k", "fev", "status",
               "lim_it", "lim_fev", "lim_ls")


def _fg(V, x, W, W2, tau):
    """-ELBO, -gradient, s and the -gradient in tau; algebra of vibes_batch._elbo_and_grad.

    tau is fixed, but L-BFGS-B keeps its gradient in y, so it enters y'y and hence theta."""
    p, c = W.shape
    nu, d = x[:, :c], x[:, c:]
    theta = V - nu @ W.T
    delta = jnp.sqrt((d * d) @ W2.T)
    z = theta / delta
    Phi = ndtr(z)
    phi = jnp.exp(-0.5 * z * z) / SQRT_2PI
    g = theta * Phi + delta * phi + (tau - 1.0) * theta
    a_term = (1.0 - tau) - Phi
    b_term = phi / delta
    s = g.mean(axis=1)
    elbo = (p * math.log(tau) + p * math.log(1.0 - tau) - p * jnp.log(s)
            - 0.5 * jnp.sum(nu * nu, axis=1) - 0.5 * jnp.sum(d * d, axis=1)
            + jnp.sum(jnp.log(d), axis=1) + (c / 2.0 - p))
    dnu = -(a_term @ W) / s[:, None] - nu
    dd = 1.0 / d - d - (b_term @ W2) * (d / s[:, None])
    gt = -(p / tau - p / (1.0 - tau) - jnp.sum(theta, axis=1) / s)
    return -elbo, -jnp.concatenate([dnu, dd], axis=1), s, gt


# --------------------------------------------------------------------------- More-Thuente
def _mt_init(stp, f, g, stpmax, ftol=1e-3):
    stpmin = jnp.zeros_like(stp)
    err = (stp < stpmin) | (stp > stpmax) | ~(g < 0) | (stpmax < stpmin)
    mt = {"brackt": jnp.zeros(stp.shape, bool), "stage": jnp.ones(stp.shape, jnp.int32),
          "finit": f, "ginit": g, "gtest": ftol * g, "width": stpmax - stpmin,
          "width1": (stpmax - stpmin) / 0.5, "stx": jnp.zeros_like(stp), "fx": f, "gx": g,
          "sty": jnp.zeros_like(stp), "fy": f, "gy": g, "stmin": jnp.zeros_like(stp),
          "stmax": stp + 4.0 * stp, "stp": stp, "stpmax": stpmax, "stpmin": stpmin}
    return mt, jnp.where(err, -1, 0)


def _mt_iterate(mt, f, g, gtol=0.9, xtol=0.1):
    """vibes_batch._MoreThuente.iterate for all rows; rows that stop keep their state."""
    stp = mt["stp"]
    ftest = mt["finit"] + stp * mt["gtest"]
    stage = jnp.where((mt["stage"] == 1) & (f <= ftest) & (g >= 0), 2, mt["stage"])
    br, stmin, stmax = mt["brackt"], mt["stmin"], mt["stmax"]
    warn = (br & ((stp <= stmin) | (stp >= stmax))) | (br & (stmax - stmin <= xtol * stmax))
    warn |= (stp == mt["stpmax"]) & (f <= ftest) & (g <= mt["gtest"])
    warn |= (stp == mt["stpmin"]) & ((f > ftest) | (g >= mt["gtest"]))
    conv = (f <= ftest) & (jnp.abs(g) <= gtol * -mt["ginit"])
    status = jnp.where(conv, 1, jnp.where(warn, 2, 0))
    go = status == 0
    gtest = mt["gtest"]
    mod = (stage == 1) & (f <= mt["fx"]) & (f > ftest)
    stx, fx, gx, sty, fy, gy = (mt[k] for k in ("stx", "fx", "gx", "sty", "fy", "gy"))
    fm, fxm, fym = f - stp * gtest, fx - stx * gtest, fy - sty * gtest
    gm, gxm, gym = g - gtest, gx - gtest, gy - gtest
    nstx, nfx, ngx, nsty, nfy, ngy, nstp, nbr = vb._dcstep(
        stx, jnp.where(mod, fxm, fx), jnp.where(mod, gxm, gx), sty, jnp.where(mod, fym, fy),
        jnp.where(mod, gym, gy), stp, jnp.where(mod, fm, f), jnp.where(mod, gm, g), br, stmin, stmax,
        xp=jnp)
    nfx = jnp.where(mod, nfx + nstx * gtest, nfx)
    nfy = jnp.where(mod, nfy + nsty * gtest, nfy)
    ngx = jnp.where(mod, ngx + gtest, ngx)
    ngy = jnp.where(mod, ngy + gtest, ngy)
    width, width1 = mt["width"], mt["width1"]
    nstp = jnp.where(nbr & (jnp.abs(nsty - nstx) >= 0.66 * width1), nstx + 0.5 * (nsty - nstx), nstp)
    nwidth1 = jnp.where(nbr, width, width1)
    nwidth = jnp.where(nbr, jnp.abs(nsty - nstx), width)
    nstmin = jnp.where(nbr, jnp.minimum(nstx, nsty), nstp + 1.1 * (nstp - nstx))
    nstmax = jnp.where(nbr, jnp.maximum(nstx, nsty), nstp + 4.0 * (nstp - nstx))
    nstp = jnp.clip(nstp, mt["stpmin"], mt["stpmax"])
    stuck = (nbr & ((nstp <= nstmin) | (nstp >= nstmax))) | (nbr & (nstmax - nstmin <= xtol * nstmax))
    nstp = jnp.where(stuck, nstx, nstp)
    new = {"stage": stage, "brackt": nbr, "stx": nstx, "fx": nfx, "gx": ngx, "sty": nsty, "fy": nfy,
           "gy": ngy, "width": nwidth, "width1": nwidth1, "stmin": nstmin, "stmax": nstmax, "stp": nstp}
    out = dict(mt)
    for k_, v in new.items():
        out[k_] = jnp.where(go, v, mt[k_])
    status = jnp.where(go & ~jnp.isfinite(nstp), -1, status)
    return out, status


# --------------------------------------------------------------------------- one iteration
def _two_loop(g, S, Y, rho, gamma, m):
    q = -g
    alphas = []
    for j in range(m):  # newest -> oldest; empty pairs have rho = 0
        al = rho[:, j] * jnp.sum(S[:, j] * q, axis=1)
        q = q - al[:, None] * Y[:, j]
        alphas.append(al)
    r = gamma[:, None] * q
    for j in reversed(range(m)):
        bt = rho[:, j] * jnp.sum(Y[:, j] * r, axis=1)
        r = r + S[:, j] * (alphas[j] - bt)[:, None]
    return r


def _pg_norm(x, g, c, lower):
    dpart = x[:, c:] - g[:, c:]
    pgd = jnp.where(dpart < lower, x[:, c:] - lower, g[:, c:])
    return jnp.maximum(jnp.max(jnp.abs(g[:, :c]), axis=1), jnp.max(jnp.abs(pgd), axis=1))


def _iteration(st, W, W2, tau, m, ftol, pgtol, lower):
    c = W.shape[1]
    eps = float(np.finfo(np.float64).eps)
    run = st["status"] == RUN
    x, g, S, Y, rho, gamma = st["x"], st["g"], st["S"], st["Y"], st["rho"], st["gamma"]
    dirs = _two_loop(g, S, Y, rho, gamma, m)
    xd = x[:, c:]
    blocked = (xd <= lower) & (dirs[:, c:] < 0)
    dirs = dirs.at[:, c:].set(jnp.where(blocked, 0.0, dirs[:, c:]))
    gd = jnp.sum(g * dirs, axis=1)
    nondesc = ~(gd < 0)
    pgd = -g
    pgd = pgd.at[:, c:].set(jnp.where(blocked, 0.0, pgd[:, c:]))
    dirs = jnp.where(nondesc[:, None], pgd, dirs)
    clr = nondesc & run
    S = jnp.where(clr[:, None, None], 0.0, S)
    Y = jnp.where(clr[:, None, None], 0.0, Y)
    rho = jnp.where(clr[:, None], 0.0, rho)
    gamma = jnp.where(clr, 1.0, gamma)
    gd = jnp.sum(g * dirs, axis=1)
    dnorm = jnp.sqrt(jnp.sum(dirs * dirs, axis=1))
    dd_ = dirs[:, c:]
    ratio = jnp.where(dd_ < 0, (xd - lower) / jnp.where(dd_ < 0, -dd_, 1.0), jnp.inf)
    stpmx = jnp.minimum(jnp.min(ratio, axis=1), 1e10)
    first = st["k"] == 0
    stpmx = jnp.where(first, 1.0, stpmx)
    stp = jnp.where(first, jnp.minimum(1.0 / jnp.maximum(dnorm, 1e-300), stpmx), jnp.minimum(1.0, stpmx))
    stpmx_safe = jnp.maximum(stpmx, 0.0)
    stp = jnp.minimum(stp, stpmx_safe)
    mt, lss = _mt_init(stp, st["f"], gd, stpmx_safe)
    lss = jnp.where(~(stpmx_safe > 0), -1, lss)
    lss = jnp.where(run, lss, 3)  # 3: not searching this iteration
    ifun = jnp.zeros_like(st["k"])

    def ls_cond(cr):
        return jnp.any(cr[1] == 0)

    def ls_body(cr):
        mt, lss, ifun, xt, ft, gt, sv = cr
        sel = lss == 0
        cand = x + mt["stp"][:, None] * dirs
        cand = cand.at[:, c:].set(jnp.maximum(cand[:, c:], lower))
        fv, gv, s_new, _ = _fg(st["V"], cand, W, W2, tau)
        gdv = jnp.sum(gv * dirs, axis=1)
        xt = jnp.where(sel[:, None], cand, xt)
        gt = jnp.where(sel[:, None], gv, gt)
        ft = jnp.where(sel, fv, ft)
        sv = jnp.where(sel, s_new, sv)
        ifun = ifun + sel.astype(ifun.dtype)
        finite = jnp.isfinite(fv) & jnp.isfinite(gdv)
        mt2, stn = _mt_iterate(mt, fv, gdv)
        upd = sel & finite
        mt = {k_: jnp.where(upd, mt2[k_], mt[k_]) for k_ in mt}
        stn = jnp.where(finite, stn, -1)
        stn = jnp.where((stn == 0) & (ifun >= st["lim_ls"]), -1, stn)
        stn = jnp.where(stn == 2, 1, stn)
        lss = jnp.where(sel, stn, lss)
        return mt, lss, ifun, xt, ft, gt, sv

    mt, lss, ifun, xt, ft, gt, sv = lax.while_loop(
        ls_cond, ls_body, (mt, lss, ifun, x, st["f"], g, st["s"]))
    fev = st["fev"] + ifun
    status = st["status"]
    failed = run & (lss == -1)
    empty = jnp.all(rho == 0, axis=1)
    status = jnp.where(failed & empty, LSFAIL, status)
    clr2 = failed & ~empty
    S = jnp.where(clr2[:, None, None], 0.0, S)
    Y = jnp.where(clr2[:, None, None], 0.0, Y)
    rho = jnp.where(clr2[:, None], 0.0, rho)
    gamma = jnp.where(clr2, 1.0, gamma)
    status = jnp.where(clr2 & (fev > st["lim_fev"]), MAXFUN, status)
    acc = run & (lss == 1)
    s_vec = xt - x
    y_vec = gt - g
    dr = jnp.sum(s_vec * y_vec, axis=1)
    ddum = -jnp.sum(s_vec * g, axis=1)
    yy = jnp.sum(y_vec * y_vec, axis=1)
    keep = acc & (dr > eps * ddum)
    kk = keep[:, None, None]
    S = jnp.where(kk, jnp.concatenate([s_vec[:, None], S[:, :-1]], axis=1), S)
    Y = jnp.where(kk, jnp.concatenate([y_vec[:, None], Y[:, :-1]], axis=1), Y)
    rho = jnp.where(keep[:, None], jnp.concatenate([(1.0 / jnp.where(keep, dr, 1.0))[:, None], rho[:, :-1]], axis=1), rho)
    gamma = jnp.where(keep, dr / jnp.where(keep, yy, 1.0), gamma)
    f_old = st["f"]
    xn = jnp.where(acc[:, None], xt, x)
    gn = jnp.where(acc[:, None], gt, g)
    fn = jnp.where(acc, ft, f_old)
    sn = jnp.where(acc, sv, st["s"])
    k = st["k"] + acc.astype(st["k"].dtype)
    pg = _pg_norm(xn, gn, c, lower)
    rel = (f_old - fn) <= ftol * jnp.maximum(jnp.maximum(jnp.abs(f_old), jnp.abs(fn)), 1.0)
    c_it = k >= st["lim_it"]
    c_fev = ~c_it & (fev > st["lim_fev"])
    c_pg = ~c_it & ~c_fev & (pg <= pgtol)
    c_rel = ~c_it & ~c_fev & ~c_pg & rel
    c_nan = ~jnp.isfinite(fn)
    code = jnp.select([c_nan, c_it, c_fev, c_pg, c_rel], [NAN, MAXIT, MAXFUN, PGTOL, FTOL], RUN)
    status = jnp.where(acc, code, status)
    out = dict(st)
    out.update(x=xn, f=fn, g=gn, s=sn, S=S, Y=Y, rho=rho, gamma=gamma, k=k, fev=fev, status=status)
    return out


# --------------------------------------------------------------------------- L-BFGS-B
def _middle(S, Y, theta, col, m):
    """Middle matrix M of the compact form B = theta I - W M W', W = [Y, theta S], by the same block
    factorisation as ``bmv`` (D, and the Cholesky factor of T = theta S'S + L D^-1 L'), which keeps
    the rounding of a badly scaled matrix down. Rows hold the col newest pairs, oldest first; unused
    rows get unit blocks, so products with vectors that are zero there stay zero.
    Returns M, the T test of ``formt``, D (unit where unused) and L."""
    SY = S @ jnp.swapaxes(Y, 1, 2)
    SS = S @ jnp.swapaxes(S, 1, 2)
    used = jnp.arange(m)[None, :] < col[:, None]
    pair = used[:, :, None] & used[:, None, :]
    D = jnp.where(used, jnp.diagonal(SY, axis1=1, axis2=2), 1.0)
    L = jnp.where(pair, jnp.tril(SY, -1), 0.0)
    T = theta[:, None, None] * jnp.where(pair, SS, 0.0) + (L / D[:, None, :]) @ jnp.swapaxes(L, 1, 2)
    T = jnp.where(pair, T, jnp.eye(m)[None])
    Tc = jnp.linalg.cholesky(T)
    t_ok = jnp.all(jnp.isfinite(Tc), axis=(1, 2))
    Tinv = jax.scipy.linalg.cho_solve((Tc, True), jnp.broadcast_to(jnp.eye(m), T.shape))
    DiLt = jnp.swapaxes(L, 1, 2) / D[:, :, None]
    M12 = DiLt @ Tinv
    M11 = jax.vmap(jnp.diag)(-1.0 / D) + M12 @ jnp.swapaxes(DiLt, 1, 2)
    M = jnp.concatenate([jnp.concatenate([M11, M12], axis=2),
                         jnp.concatenate([jnp.swapaxes(M12, 1, 2), Tinv], axis=2)], axis=1)
    return M, t_ok, D, L


def _cauchy(x, g, theta, col, S, Y, M, c, lower):
    """Generalised Cauchy point (``cauchy`` in __lbfgsb.c): piecewise search along projected -g.

    Only the d block has (lower) bounds; nu is free. Returns xcp, c = W'(xcp - x), and the d
    variables held at the bound (not free for the subspace step)."""
    B, n = x.shape
    rows = jnp.arange(B)
    xd, neg = x[:, c:], -g[:, c:]
    at_l = (xd - lower) <= 0.0
    held = at_l & (neg <= 0.0)
    flat = ~at_l & (neg == 0.0)
    dird = jnp.where(held | flat, 0.0, neg)
    dirs = jnp.concatenate([-g[:, :c], dird], axis=1)
    f1 = -jnp.sum(dirs * dirs, axis=1)
    p_ = jnp.concatenate([jnp.sum(Y * dirs[:, None], axis=2),
                          theta[:, None] * jnp.sum(S * dirs[:, None], axis=2)], axis=1)
    bp = dird < 0.0
    t_ = jnp.where(bp, (xd - lower) / jnp.where(bp, -neg, 1.0), jnp.inf)
    order = jnp.argsort(t_, axis=1)
    ts = jnp.take_along_axis(t_, order, axis=1)
    nb = jnp.sum(bp, axis=1)
    bnded = jnp.all(g[:, :c] == 0.0, axis=1) & ~jnp.any(dird > 0.0, axis=1)
    f2 = -theta * f1
    f2_org = f2
    f2 = f2 - jnp.sum(p_ * jnp.squeeze(M @ p_[:, :, None], 2), axis=1)
    moved = jnp.any(dirs != 0.0, axis=1)
    dtm = -f1 / jnp.where(moved, f2, 1.0)
    state = (jnp.int32(0), moved & (nb > 0), f1, f2, dtm, jnp.zeros(B), jnp.zeros(B), dirs, x,
             p_, jnp.zeros_like(p_), held)

    def cond(st):
        return jnp.any(st[1]) & (st[0] < c)

    def body(st):
        j, live, f1, f2, dtm, tsum, tj, dirs, z, p_, cv, held = st
        ibp = order[:, j]
        tnew = ts[:, j]
        dt = tnew - tj
        go = live & ~(dtm < dt)
        sel = rows, c + ibp
        dibp = dirs[sel]
        zibp = lower - x[sel]
        cv2 = cv + dt[:, None] * p_
        wbp = jnp.concatenate([Y[rows, :, c + ibp], theta[:, None] * S[rows, :, c + ibp]], axis=1)
        v = jnp.squeeze(M @ wbp[:, :, None], 2)
        wmc, wmp, wmw = jnp.sum(cv2 * v, axis=1), jnp.sum(p_ * v, axis=1), jnp.sum(wbp * v, axis=1)
        d2 = dibp * dibp
        f1n = f1 + dt * f2 + d2 - theta * dibp * zibp + dibp * wmc
        f2n = jnp.maximum(eps64 * f2_org, f2 - theta * d2 + 2.0 * dibp * wmp - d2 * wmw)
        more = nb > j + 1
        dtmn = jnp.where(more | ~bnded, -f1n / f2n, 0.0)
        return (j + 1, go & more,
                jnp.where(go, f1n, f1), jnp.where(go, f2n, f2), jnp.where(go, dtmn, dtm),
                jnp.where(go, tsum + dt, tsum), jnp.where(go, tnew, tj),
                dirs.at[sel].set(jnp.where(go, 0.0, dibp)),
                z.at[sel].set(jnp.where(go, lower, z[sel])),
                jnp.where(go[:, None], p_ - dibp[:, None] * wbp, p_),
                jnp.where(go[:, None], cv2, cv),
                held.at[rows, ibp].set(held[rows, ibp] | go))

    _, _, _, _, dtm, tsum, _, dirs, z, p_, cv, held = lax.while_loop(cond, body, state)
    dtm = jnp.where(moved, jnp.maximum(dtm, 0.0), 0.0)
    return z + (tsum + dtm)[:, None] * dirs, cv + dtm[:, None] * p_, held


def _subspace(x, g, z, cv, theta, col, S, Y, D, L, M, held, c, m, lower):
    """Direct primal subspace step (``cmprlb`` + ``formk`` + ``subsm``), with the projection.

    K = [[-A, B'], [B, C]] with A = D + Y'ZZ'Y/theta, B = L_a - R_z, C = theta S'AA'S is solved by its
    Schur complement, as ``formk``'s LEL' does; A and the complement are positive definite."""
    free = jnp.concatenate([jnp.ones((x.shape[0], c), bool), ~held], axis=1)
    fm = free[:, None].astype(x.dtype)
    Yt, St = jnp.swapaxes(Y, 1, 2), jnp.swapaxes(S, 1, 2)
    th = theta[:, None, None]
    W = jnp.concatenate([Y, th * S], axis=1)   # (B, 2m, n)
    Mc = jnp.squeeze(M @ cv[:, :, None], 2)
    r = (-theta[:, None] * (z - x) - g + jnp.sum(W * Mc[:, :, None], axis=1)) * free
    YF, SF = Y * fm, S * fm
    w1 = jnp.sum(YF * r[:, None], axis=2)
    w2 = theta[:, None] * jnp.sum(SF * r[:, None], axis=2)
    used = jnp.arange(m)[None, :] < col[:, None]
    pair = used[:, :, None] & used[:, None, :]
    A = jax.vmap(jnp.diag)(D) + jnp.where(pair, YF @ Yt, 0.0) / th
    Bm = jnp.where(pair, L - SF @ Yt, 0.0)
    Cm = th * jnp.where(pair, (S * (1.0 - fm)) @ St, 0.0)
    Ac = jnp.linalg.cholesky(A)
    AiBt = jax.scipy.linalg.cho_solve((Ac, True), jnp.swapaxes(Bm, 1, 2))
    Sch = Cm + Bm @ AiBt
    Sch = jnp.where(pair, Sch, jnp.eye(m)[None])
    Sc = jnp.linalg.cholesky(Sch)
    Aiw1 = jax.scipy.linalg.cho_solve((Ac, True), w1[:, :, None])
    u2 = jax.scipy.linalg.cho_solve((Sc, True), w2[:, :, None] + Bm @ Aiw1)
    u1 = AiBt @ u2 - Aiw1
    step = (r + (jnp.sum(YF * u1, axis=1) + theta[:, None] * jnp.sum(SF * u2, axis=1)) / theta[:, None]) / theta[:, None]
    bad = ~jnp.all(jnp.isfinite(step), axis=1)
    step = jnp.where(bad[:, None], 0.0, step)
    trial = z + step
    trial_d = jnp.maximum(trial[:, c:], lower)
    hit = free[:, c:] & (trial_d == lower)
    trial = jnp.concatenate([trial[:, :c], jnp.where(free[:, c:], trial_d, z[:, c:])], axis=1)
    iword = jnp.any(hit, axis=1)
    ddp = jnp.sum((trial - x) * g, axis=1)
    # backtrack from the Cauchy point along the step when the projected point is not a descent point
    back = iword & (ddp > 0.0)
    dk, zd = step[:, c:], z[:, c:]
    neg = free[:, c:] & (dk < 0.0)
    room = lower - zd
    ratio = jnp.where(neg, jnp.where(room >= 0.0, 0.0, room / jnp.where(neg, dk, 1.0)), jnp.inf)
    ibd = jnp.argmin(ratio, axis=1)
    alpha = jnp.minimum(jnp.min(ratio, axis=1), 1.0)
    rows = jnp.arange(x.shape[0])
    blocked = alpha < 1.0
    zb = z + alpha[:, None] * step
    zb = zb.at[rows, c + ibd].set(jnp.where(blocked, lower, zb[rows, c + ibd]))
    out = jnp.where(back[:, None], zb, trial)
    return jnp.where((col > 0)[:, None] & ~bad[:, None], out, z), bad & (col > 0)


def _iteration_lbfgsb(st, W, W2, tau, m, ftol, pgtol, lower):
    c = W.shape[1]
    run = st["status"] == RUN
    x, g, S, Y, theta, col = st["x"], st["g"], st["S"], st["Y"], st["theta"], st["col"]
    M, _, D, L = _middle(S, Y, theta, col, m)
    z, cv, held = _cauchy(x, g, theta, col, S, Y, M, c, lower)
    z, sub_bad = _subspace(x, g, z, cv, theta, col, S, Y, D, L, M, held, c, m, lower)
    d = z - x
    dtd = jnp.sum(d * d, axis=1)
    dnorm = jnp.sqrt(dtd)
    gd = jnp.sum(g * d, axis=1)
    first = st["k"] == 0
    dd_ = d[:, c:]
    a2 = lower - x[:, c:]
    neg = dd_ < 0.0
    ratio = jnp.where(neg, jnp.where(a2 >= 0.0, 0.0, a2 / jnp.where(neg, dd_, 1.0)), jnp.inf)
    stpmx = jnp.where(first, 1.0, jnp.minimum(jnp.min(ratio, axis=1), 1e10))
    stp = jnp.where(first, jnp.minimum(1.0 / jnp.where(dnorm > 0, dnorm, 1.0), stpmx), 1.0)
    mt, lss = _mt_init(stp, st["f"], gd, stpmx)
    lss = jnp.where(sub_bad | ~jnp.isfinite(gd) | ~(gd < 0.0) | ~(stpmx > 0.0), -1, lss)
    lss = jnp.where(run, lss, 3)
    ifun = jnp.zeros_like(st["k"])

    def ls_cond(cr):
        return jnp.any(cr[1] == 0)

    def ls_body(cr):
        mt, lss, ifun, xt, ft, gt_, sv, gdt, gtau = cr
        sel = lss == 0
        cs = mt["stp"]
        cand = jnp.where((cs == 1.0)[:, None], z, x + cs[:, None] * d)
        cand = cand.at[:, c:].set(jnp.maximum(cand[:, c:], lower))
        fv, gv, s_new, tg = _fg(st["V"], cand, W, W2, tau)
        gdv = jnp.sum(gv * d, axis=1)
        xt = jnp.where(sel[:, None], cand, xt)
        gt_ = jnp.where(sel[:, None], gv, gt_)
        ft = jnp.where(sel, fv, ft)
        sv = jnp.where(sel, s_new, sv)
        gdt = jnp.where(sel, gdv, gdt)
        gtau = jnp.where(sel, tg, gtau)
        ifun = ifun + sel.astype(ifun.dtype)
        finite = jnp.isfinite(fv) & jnp.isfinite(gdv)
        mt2, stn = _mt_iterate(mt, fv, gdv)
        upd = sel & finite
        mt = {k_: jnp.where(upd, mt2[k_], mt[k_]) for k_ in mt}
        stn = jnp.where(finite, stn, -1)
        stn = jnp.where((stn == 0) & (ifun >= st["lim_ls"]), -1, stn)
        stn = jnp.where(stn == 2, 1, stn)
        lss = jnp.where(sel, stn, lss)
        return mt, lss, ifun, xt, ft, gt_, sv, gdt, gtau

    mt, lss, ifun, xt, ft, gt_, sv, gdt, gtau = lax.while_loop(
        ls_cond, ls_body, (mt, lss, ifun, x, st["f"], g, st["s"], gd, st["gt"]))
    fev = st["fev"] + ifun
    status = st["status"]
    failed = run & (lss == -1)
    empty = col == 0
    status = jnp.where(failed & empty, LSFAIL, status)
    refresh = failed & ~empty
    status = jnp.where(refresh & (fev > st["lim_fev"]), MAXFUN, status)
    acc = run & (lss == 1)
    fin = mt["stp"]
    dr = jnp.where(fin == 1.0, gdt - gd, (gdt - gd) * fin)
    ddum = jnp.where(fin == 1.0, -gd, -gd * fin)
    y_vec = gt_ - g
    rr = jnp.sum(y_vec * y_vec, axis=1) + (gtau - st["gt"]) ** 2
    s_vec = fin[:, None] * d
    keep = acc & (dr > eps64 * ddum)
    full = col == m
    k1 = keep & ~full
    k2 = keep & full
    rows = jnp.arange(x.shape[0])
    slot = jnp.minimum(col, m - 1)
    S1, Y1 = S.at[rows, slot].set(s_vec), Y.at[rows, slot].set(y_vec)
    S2 = jnp.concatenate([S[:, 1:], s_vec[:, None]], axis=1)
    Y2 = jnp.concatenate([Y[:, 1:], y_vec[:, None]], axis=1)
    Sn = jnp.where(k1[:, None, None], S1, jnp.where(k2[:, None, None], S2, S))
    Yn = jnp.where(k1[:, None, None], Y1, jnp.where(k2[:, None, None], Y2, Y))
    coln = jnp.where(keep, jnp.minimum(col + 1, m), col)
    thetan = jnp.where(keep, rr / jnp.where(keep, dr, 1.0), theta)
    t_ok = _middle(Sn, Yn, thetan, coln, m)[1]
    clear = refresh | (keep & ~t_ok)
    Sn = jnp.where(clear[:, None, None], 0.0, Sn)
    Yn = jnp.where(clear[:, None, None], 0.0, Yn)
    coln = jnp.where(clear, 0, coln)
    thetan = jnp.where(clear, 1.0, thetan)
    f_old = st["f"]
    xn = jnp.where(acc[:, None], xt, x)
    gn = jnp.where(acc[:, None], gt_, g)
    fn = jnp.where(acc, ft, f_old)
    sn = jnp.where(acc, sv, st["s"])
    gtn = jnp.where(acc, gtau, st["gt"])
    k = st["k"] + acc.astype(st["k"].dtype)
    pg = _pg_norm(xn, gn, c, lower)
    rel = (f_old - fn) <= ftol * jnp.maximum(jnp.maximum(jnp.abs(f_old), jnp.abs(fn)), 1.0)
    c_it = k >= st["lim_it"]
    c_fev = ~c_it & (fev > st["lim_fev"])
    c_pg = ~c_it & ~c_fev & (pg <= pgtol)
    c_rel = ~c_it & ~c_fev & ~c_pg & rel
    c_nan = ~jnp.isfinite(fn)
    code = jnp.select([c_nan, c_it, c_fev, c_pg, c_rel], [NAN, MAXIT, MAXFUN, PGTOL, FTOL], RUN)
    status = jnp.where(acc, code, status)
    out = dict(st)
    out.update(x=xn, f=fn, g=gn, s=sn, gt=gtn, S=Sn, Y=Yn, theta=thetan, col=coln, k=k, fev=fev, status=status)
    return out


@partial(jax.jit, static_argnames=("K", "m", "ftol", "pgtol", "lower", "tau", "engine"), donate_argnums=0)
def _chunk(st, W, W2, *, K, m, ftol, pgtol, lower, tau, engine="lbfgs"):
    step = _iteration_lbfgsb if engine == "lbfgsb" else _iteration

    def cond(cr):
        return (cr[0] < K) & jnp.any(cr[1]["status"] == RUN)

    def body(cr):
        return cr[0] + 1, step(cr[1], W, W2, tau, m, ftol, pgtol, lower)

    it, st = lax.while_loop(cond, body, (jnp.int32(0), st))
    return st, it


@partial(jax.jit, static_argnames=("tau",), donate_argnums=0)
def _reset(st, idx, Vrows, newV, x0, lim_it, lim_fev, lim_ls, W, W2, *, tau):
    """(Re)start the slots ``idx`` (padded with repeats) at x0: fresh memory, k = 0."""
    V_rows = jnp.where(newV[:, None], Vrows, st["V"][idx])
    f, g, s, gt = _fg(V_rows, x0, W, W2, tau)
    out = dict(st)
    out["V"] = st["V"].at[idx].set(V_rows)
    out["x"] = st["x"].at[idx].set(x0)
    out["f"] = st["f"].at[idx].set(f)
    out["g"] = st["g"].at[idx].set(g)
    out["s"] = st["s"].at[idx].set(s)
    out["gt"] = st["gt"].at[idx].set(gt)
    out["theta"] = st["theta"].at[idx].set(1.0)
    out["col"] = st["col"].at[idx].set(0)
    out["S"] = st["S"].at[idx].set(0.0)
    out["Y"] = st["Y"].at[idx].set(0.0)
    out["rho"] = st["rho"].at[idx].set(0.0)
    out["gamma"] = st["gamma"].at[idx].set(1.0)
    out["k"] = st["k"].at[idx].set(0)
    out["fev"] = st["fev"].at[idx].set(1)
    out["status"] = st["status"].at[idx].set(RUN)
    out["lim_it"] = st["lim_it"].at[idx].set(lim_it)
    out["lim_fev"] = st["lim_fev"].at[idx].set(lim_fev)
    out["lim_ls"] = st["lim_ls"].at[idx].set(lim_ls)
    return out


def _pad(n):
    return 1 << max(0, int(n - 1).bit_length())


def _empty_state(B, p, n, m):
    z = lambda *s: jnp.zeros(s)
    zi = lambda: jnp.zeros(B, jnp.int64)
    return {"V": z(B, p), "x": jnp.ones((B, n)), "f": z(B), "g": z(B, n), "s": jnp.ones(B),
            "S": z(B, m, n), "Y": z(B, m, n), "rho": z(B, m), "gamma": jnp.ones(B), "k": zi(),
            "fev": zi(), "status": jnp.full(B, EMPTY, jnp.int32), "lim_it": zi(), "lim_fev": zi(),
            "lim_ls": zi(), "gt": z(B), "theta": jnp.ones(B), "col": zi()}


def fit_elbo_jax(
    spectra,
    mean,
    components,
    *,
    tau=0.1,
    pool=2048,
    chunk_iters=100,
    min_pool=8,
    maxiter=10000,
    maxfun=15000,
    m=10,
    factr=1e7,
    pgtol=1e-5,
    sd_min=1e-10,
    maxls=20,
    retry_failed=True,
    polish_restarts=0,
    x0=None,
    engine="lbfgs",
    progress=None,
):
    """Same contract and result as vibes_batch.fit_elbo_batch, compiled with JAX.

    ``chunk_iters`` is the number of L-BFGS iterations per compiled call;
    ``min_pool`` the smallest compiled batch for the tail. Returns ElboResult
    with extra stats: compiled chunk calls, host seconds, pool sizes used.
    ``x0`` (rows x 2c, [nu, d]) replaces the adapter's start nu = 0, d = 1, e.g. to warm start
    from a previous fit; retries and polish restarts still start from the current point.
    """
    if tau is None or not 0 < float(tau) < 1:
        raise ValueError("only a fixed tau in (0, 1) is implemented")
    tau = float(tau)
    started = time.perf_counter()
    y, mu, W_h = vb._check_inputs(spectra, mean, components)
    rows, p = y.shape
    c = W_h.shape[1]
    n = 2 * c
    W = jnp.asarray(W_h)
    W2 = W * W
    V_all = y - mu
    init = np.r_[np.zeros(c), np.ones(c)]
    x0_rows = None if x0 is None else np.asarray(x0, dtype=np.float64).reshape(rows, 2 * c)
    eps = float(np.finfo(np.float64).eps)
    ftol = factr * eps
    kw = dict(m=m, ftol=ftol, pgtol=pgtol, lower=sd_min, tau=tau, engine=engine)

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
    out_unpol = np.full(rows, np.nan)
    out_first = [""] * rows
    out_msg = [""] * rows

    B = int(min(pool, _pad(rows)))
    st = _empty_state(B, p, n, m)
    slot_row = np.full(B, -1, np.int64)
    attempt = np.zeros(B, np.int64)
    polish = np.zeros(B, np.int64)
    it_prior = np.zeros(B, np.int64)
    fev_prior = np.zeros(B, np.int64)
    f_pol = np.zeros(B)
    first_msg = [""] * B
    queue = list(range(rows))[::-1]
    stats = {"pool": B, "chunk_iters": chunk_iters, "chunk_calls": 0, "device_iterations": 0,
             "pool_sizes": [B], "host_seconds": 0.0, "device_wait_seconds": 0.0, "finished": 0}

    def restart(entries):
        """entries: list of (slot, x0 or None for a fresh row, (lim_it, lim_fev, lim_ls))."""
        nonlocal st
        if not entries:
            return
        k_ = len(entries)
        P = _pad(k_)
        idx = np.empty(P, np.int64)
        Vr = np.zeros((P, p))
        newV = np.zeros(P, bool)
        x0 = np.empty((P, n))
        li = np.empty((3, P), np.int64)
        for j, (sl, xs, lims) in enumerate(entries):
            idx[j] = sl
            if xs is None:
                Vr[j] = V_all[slot_row[sl]]
                newV[j] = True
                x0[j] = init if x0_rows is None else x0_rows[slot_row[sl]]
            else:
                x0[j] = xs
            li[:, j] = lims
        idx[k_:], Vr[k_:], newV[k_:], x0[k_:], li[:, k_:] = idx[0], Vr[0], newV[0], x0[0], li[:, :1]
        st = _reset(st, jnp.asarray(idx), jnp.asarray(Vr), jnp.asarray(newV), jnp.asarray(x0),
                    jnp.asarray(li[0]), jnp.asarray(li[1]), jnp.asarray(li[2]), W, W2, tau=tau)

    default_lims = (maxiter, maxfun, maxls)

    def load(slots):
        ent = []
        for sl in slots:
            slot_row[sl] = queue.pop()
            attempt[sl] = polish[sl] = it_prior[sl] = fev_prior[sl] = 0
            first_msg[sl] = ""
            ent.append((sl, None, default_lims))
        return ent

    restart(load(list(range(min(B, rows)))))
    while (slot_row >= 0).any():
        t0 = time.perf_counter()
        st, it = _chunk(st, W, W2, K=chunk_iters, **kw)
        status = np.asarray(st["status"])
        t1 = time.perf_counter()
        stats["device_wait_seconds"] += t1 - t0
        stats["chunk_calls"] += 1
        stats["device_iterations"] += int(it)
        stopped = np.flatnonzero((slot_row >= 0) & (status != RUN) & (status != EMPTY))
        entries = []
        freed = []
        if len(stopped):
            hx, hf, hg, hs = (np.asarray(st[k_]) for k_ in ("x", "f", "g", "s"))
            hk, hfev = np.asarray(st["k"]), np.asarray(st["fev"])
            for sl in stopped:
                code = int(status[sl])
                msg, good = _MSG[code], code in (PGTOL, FTOL)
                if not good and polish[sl] > 0 and code != NAN and np.isfinite(hf[sl]):
                    good = True  # polish restarts are monotone, see vibes_batch.finish
                if not good and retry_failed and attempt[sl] == 0 and np.isfinite(hx[sl]).all():
                    attempt[sl] = 1
                    first_msg[sl] = msg
                    it_prior[sl], fev_prior[sl] = hk[sl], hfev[sl]
                    entries.append((sl, hx[sl], (5 * maxiter, 20 * maxiter, 50)))
                    continue
                r = slot_row[sl]
                if good and polish_restarts:
                    if polish[sl] == 0:
                        out_unpol[r] = -hf[sl]
                    gained = (f_pol[sl] - hf[sl]) > ftol * max(abs(hf[sl]), 1.0)
                    if polish[sl] < polish_restarts and (polish[sl] == 0 or gained):
                        polish[sl] += 1
                        f_pol[sl] = hf[sl]
                        it_prior[sl] += hk[sl]
                        fev_prior[sl] += hfev[sl]
                        entries.append((sl, hx[sl], default_lims))
                        continue
                xs, gs = hx[sl], hg[sl]
                dpart = xs[c:] - gs[c:]
                pgd = np.where(dpart < sd_min, xs[c:] - sd_min, gs[c:])
                out_sigma[r], out_elbo[r] = hs[sl], -hf[sl]
                out_nu[r], out_d[r] = xs[:c], xs[c:]
                out_ok[r] = bool(good) and np.isfinite(hs[sl]) and hs[sl] > 0
                out_it[r] = it_prior[sl] + hk[sl]
                out_fev[r] = fev_prior[sl] + hfev[sl]
                out_pg[r] = max(np.abs(gs[:c]).max(), np.abs(pgd).max())
                out_retry[r], out_polish[r] = attempt[sl], polish[sl]
                out_first[r] = first_msg[sl] if attempt[sl] else msg
                out_msg[r] = msg
                slot_row[sl] = -1
                freed.append(sl)
                stats["finished"] += 1
            if progress is not None:
                progress(stats["finished"], rows)
        if freed and queue:
            entries += load(freed[: len(queue)])
        if freed:
            idle = np.asarray([sl for sl in freed if slot_row[sl] < 0], np.int64)
            if len(idle):
                st["status"] = st["status"].at[jnp.asarray(idle)].set(EMPTY)
        restart(entries)
        # Shrink the compiled batch for the tail once the queue is empty.
        occ = np.flatnonzero(slot_row >= 0)
        if not queue and len(occ) and B > min_pool and len(occ) <= B // 4:
            nB = max(min_pool, _pad(len(occ)))
            take = np.r_[occ, np.full(nB - len(occ), occ[0])]
            ti = jnp.asarray(take)
            st = {k_: v[ti] for k_, v in st.items()}
            st["status"] = st["status"].at[len(occ):].set(EMPTY)
            remap = lambda a: np.r_[a[occ], np.zeros(nB - len(occ), a.dtype)]
            slot_row = np.r_[slot_row[occ], np.full(nB - len(occ), -1, np.int64)]
            attempt, polish, it_prior, fev_prior = map(remap, (attempt, polish, it_prior, fev_prior))
            f_pol = remap(f_pol)
            first_msg = [first_msg[o] for o in occ] + [""] * (nB - len(occ))
            B = nB
            stats["pool_sizes"].append(B)
        stats["host_seconds"] += time.perf_counter() - t1
    res = vb.ElboResult(out_sigma, out_elbo, out_nu, out_d, out_ok, out_it, out_fev, out_msg,
                        out_pg, time.perf_counter() - started, "jax:" + jax.default_backend(), stats)
    res.stats["retry_count"] = out_retry
    res.stats["polish_restarts"] = out_polish
    res.stats["elbo_before_polish"] = out_unpol
    res.stats["initial_message"] = out_first
    return res


def precompile(p, c, *, tau=0.1, pool=2048, chunk_iters=100, min_pool=8, m=10, factr=1e7,
               pgtol=1e-5, sd_min=1e-10, engine="lbfgs"):
    """Compile every batch shape fit_elbo_jax can use (pool down to min_pool,
    and the padded restart sizes). Returns seconds spent compiling."""
    t = time.perf_counter()
    n = 2 * c
    W = jnp.zeros((p, c)) + 0.01
    W2 = W * W
    kw = dict(m=m, ftol=factr * float(np.finfo(np.float64).eps), pgtol=pgtol, lower=sd_min, tau=float(tau),
              engine=engine)
    B = pool
    sizes = []
    while True:
        sizes.append(B)
        if B <= min_pool:
            break
        B = max(min_pool, B // 2)
    for B in sizes:
        st = _empty_state(B, p, n, m)
        st, _ = _chunk(st, W, W2, K=chunk_iters, **kw)
        P = 1
        while P <= B:
            idx = jnp.zeros(P, jnp.int64)
            st = _reset(st, idx, jnp.zeros((P, p)), jnp.zeros(P, bool), jnp.ones((P, n)),
                        jnp.zeros(P, jnp.int64), jnp.zeros(P, jnp.int64), jnp.zeros(P, jnp.int64),
                        W, W2, tau=float(tau))
            P *= 2
        st["status"].block_until_ready()
    return time.perf_counter() - t
