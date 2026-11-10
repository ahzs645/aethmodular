//! VIBES baseline for one spectrum (pinball loss, fixed tau), for WebAssembly.
//!
//! A port of `fit_elbo_newton` and `solve_map_batch` in `scripts/vibes_batch.py`:
//! the noise scale by damped Newton on the ELBO with the exact Hessian, then the
//! baseline by a primal-dual interior point on the pinball MAP, accepted only when
//! its duality gap certifies it. Everything is float64. No dependencies.

const SQRT_2PI: f64 = 2.5066282746310002;

/// Standard normal CDF (West 2005, Hart 5666), double precision.
fn ndtr(x: f64) -> f64 {
    let a = x.abs();
    let c;
    if a > 37.0 {
        c = 0.0;
    } else {
        let e = (-a * a / 2.0).exp();
        if a < 7.07106781186547 {
            let mut b = 3.52624965998911e-02 * a + 0.700383064443688;
            b = b * a + 6.37396220353165;
            b = b * a + 33.912866078383;
            b = b * a + 112.079291497871;
            b = b * a + 221.213596169931;
            b = b * a + 220.206867912376;
            let num = e * b;
            let mut d = 8.83883476483184e-02 * a + 1.75566716318264;
            d = d * a + 16.064177579207;
            d = d * a + 86.7807322029461;
            d = d * a + 296.564248779674;
            d = d * a + 637.333633378831;
            d = d * a + 793.826512519948;
            d = d * a + 440.413735824752;
            c = num / d;
        } else {
            let mut b = a + 0.65;
            b = a + 4.0 / b;
            b = a + 3.0 / b;
            b = a + 2.0 / b;
            b = a + 1.0 / b;
            c = e / b / 2.506628274631;
        }
    }
    if x > 0.0 { 1.0 - c } else { c }
}

// The kernels below are kept out of line on purpose. A browser first runs WebAssembly in a fast-to-compile
// tier and swaps in optimised code for a function only between calls to it. Kernels that are called once
// per iteration are optimised after the first iteration; inlined into one long call they never would be.

/// In-place lower Cholesky of a row-major n x n matrix. False if not positive definite.
#[inline(never)]
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let (head, tail) = a.split_at_mut((j + 1) * n);
        let rj = &mut head[j * n..];
        let mut d = rj[j];
        for k in 0..j {
            d -= rj[k] * rj[k];
        }
        if !(d > 0.0) {
            return false;
        }
        let d = d.sqrt();
        rj[j] = d;
        for ri in tail.chunks_exact_mut(n) {
            let mut s = ri[j];
            for k in 0..j {
                s -= ri[k] * rj[k];
            }
            ri[j] = s / d;
        }
    }
    true
}

#[inline(never)]
fn chol_solve(l: &[f64], n: usize, b: &[f64], x: &mut [f64]) {
    for i in 0..n {
        let row = &l[i * n..i * n + n];
        let mut s = b[i];
        for k in 0..i {
            s -= row[k] * x[k];
        }
        x[i] = s / row[i];
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
}

struct Elbo<'a> {
    w: &'a [f64],
    w2: &'a [f64],
    v: &'a [f64],
    p: usize,
    c: usize,
    tau: f64,
    theta: Vec<f64>,
    delta: Vec<f64>,
    cdf: Vec<f64>,
    pdf: Vec<f64>,
    dsq: Vec<f64>,
}

impl Elbo<'_> {
    /// -ELBO and the mean expected pinball at x = [nu, d]; fills the per-channel buffers.
    #[inline(never)]
    fn eval(&mut self, x: &[f64]) -> (f64, f64) {
        let (p, c, tau) = (self.p, self.c, self.tau);
        let (nu, d) = x.split_at(c);
        for k in 0..c {
            self.dsq[k] = d[k] * d[k];
        }
        let mut total = 0.0;
        for j in 0..p {
            let wr = &self.w[j * c..j * c + c];
            let w2r = &self.w2[j * c..j * c + c];
            let mut a = 0.0;
            let mut b = 0.0;
            for k in 0..c {
                a += nu[k] * wr[k];
                b += self.dsq[k] * w2r[k];
            }
            let t = self.v[j] - a;
            let dl = b.sqrt();
            let z = t / dl;
            let cdf = ndtr(z);
            let pdf = (-0.5 * z * z).exp() / SQRT_2PI;
            self.theta[j] = t;
            self.delta[j] = dl;
            self.cdf[j] = cdf;
            self.pdf[j] = pdf;
            total += t * cdf + dl * pdf + (tau - 1.0) * t;
        }
        let s = total / p as f64;
        let (mut sn, mut sd, mut ld) = (0.0, 0.0, 0.0);
        for k in 0..c {
            sn += nu[k] * nu[k];
            sd += d[k] * d[k];
            ld += d[k].ln();
        }
        let pf = p as f64;
        let elbo = pf * tau.ln() + pf * (1.0 - tau).ln() - pf * s.ln() - 0.5 * sn - 0.5 * sd + ld
            + (c as f64 / 2.0 - pf);
        (-elbo, s)
    }
    /// Gradient sums and the three weighted Gram blocks of the Hessian at the last `eval` point,
    /// one channel at a time: `hnn` and `hdd` upper triangles, `hnd` full.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn accumulate(&self, big_g: &mut [f64], bw2: &mut [f64], hnn: &mut [f64], hnd: &mut [f64], hdd: &mut [f64], ta: &mut [f64], tb: &mut [f64]) {
        let (p, c, tau, w, w2) = (self.p, self.c, self.tau, self.w, self.w2);
        big_g.iter_mut().for_each(|t| *t = 0.0);
        bw2.iter_mut().for_each(|t| *t = 0.0);
        hnn.iter_mut().for_each(|t| *t = 0.0);
        hnd.iter_mut().for_each(|t| *t = 0.0);
        hdd.iter_mut().for_each(|t| *t = 0.0);
        for j in 0..p {
            let wr = &w[j * c..j * c + c];
            let w2r = &w2[j * c..j * c + c];
            let dl = self.delta[j];
            let b = self.pdf[j] / dl;
            let z = self.theta[j] / dl;
            let a_term = (1.0 - tau) - self.cdf[j];
            let (w1, w2w, w3) = (b, z * b / dl, b * (z * z - 1.0) / (dl * dl));
            for k in 0..c {
                big_g[k] += a_term * wr[k];
                bw2[k] += b * w2r[k];
                ta[k] = w1 * wr[k];
                tb[k] = w3 * w2r[k];
            }
            for i in 0..c {
                let (ai, bi, ci) = (ta[i], tb[i], w2w * wr[i]);
                let rn = &mut hnn[i * c + i..i * c + c];
                for (h, &wk) in rn.iter_mut().zip(&wr[i..]) {
                    *h += ai * wk;
                }
                let rd = &mut hdd[i * c + i..i * c + c];
                for (h, &wk) in rd.iter_mut().zip(&w2r[i..]) {
                    *h += bi * wk;
                }
                let rx = &mut hnd[i * c..i * c + c];
                for (h, &wk) in rx.iter_mut().zip(w2r) {
                    *h += ci * wk;
                }
            }
        }
    }
}

pub struct ElboFit {
    pub sigma: f64,
    pub elbo: f64,
    pub iterations: u32,
    pub hessians: u32,
    /// 0 converged, 1 iteration limit, 2 line search failed, 3 not finite, 4 factorisation failed
    pub status: u32,
}

/// Noise scale by damped Newton on the ELBO, from the reference start (nu = 0, d = 1).
pub fn fit_elbo(w: &[f64], w2: &[f64], v: &[f64], p: usize, c: usize, tau: f64, maxiter: u32) -> ElboFit {
    let n = 2 * c;
    let (gtol, dec_rtol, kappa_max) = (1e-5, 1e-13, 0.99);
    let mut e = Elbo {
        w, w2, v, p, c, tau,
        theta: vec![0.0; p], delta: vec![0.0; p], cdf: vec![0.0; p], pdf: vec![0.0; p], dsq: vec![0.0; c],
    };
    let mut x = vec![0.0; n];
    for k in 0..c {
        x[c + k] = 1.0;
    }
    let mut hnn = vec![0.0; c * c];
    let mut hnd = vec![0.0; c * c];
    let mut hdd = vec![0.0; c * c];
    let mut a = vec![0.0; n * n];
    let mut big_g = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut bw2 = vec![0.0; c];
    let (mut mg, mut u, mut q, mut wv, mut step, mut cand) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut ta, mut tb) = (vec![0.0; c], vec![0.0; c]);
    let mut fit = ElboFit { sigma: f64::NAN, elbo: f64::NAN, iterations: 0, hessians: 0, status: 1 };

    for it in 0..=maxiter {
        let (f, s) = e.eval(&x);
        fit.sigma = s;
        fit.elbo = -f;
        fit.hessians += 1;
        e.accumulate(&mut big_g, &mut bw2, &mut hnn, &mut hnd, &mut hdd, &mut ta, &mut tb);
        let d = &x[c..];
        for k in 0..c {
            big_g[c + k] = bw2[k] * d[k];
            g[k] = big_g[k] / s + x[k];
            g[c + k] = big_g[c + k] / s + d[k] - 1.0 / d[k];
        }
        for i in 0..c {
            for k in i..c {
                let h1 = hnn[i * c + k] / s;
                a[i * n + k] = h1;
                a[k * n + i] = h1;
                let h3 = hdd[i * c + k] * d[i] * d[k] / s;
                a[(c + i) * n + c + k] = h3;
                a[(c + k) * n + c + i] = h3;
            }
            for k in 0..c {
                let h2 = hnd[i * c + k] * d[k] / s;
                a[i * n + c + k] = h2;
                a[(c + k) * n + i] = h2;
            }
        }
        for i in 0..c {
            a[i * n + i] += 1.0;
            a[(c + i) * n + c + i] += 1.0 + bw2[i] / s + 1.0 / (d[i] * d[i]);
        }
        if !f.is_finite() {
            fit.status = 3;
            return fit;
        }
        if !cholesky(&mut a, n) {
            fit.status = 4;
            return fit;
        }
        let sq = (p as f64).sqrt() * s;
        for i in 0..n {
            mg[i] = -g[i];
            u[i] = big_g[i] / sq;
        }
        chol_solve(&a, n, &mg, &mut q);
        chol_solve(&a, n, &u, &mut wv);
        let (mut kappa, mut uq) = (0.0, 0.0);
        for i in 0..n {
            kappa += u[i] * wv[i];
            uq += u[i] * q[i];
        }
        // Sherman-Morrison for the rank-one term when the true Hessian is positive definite.
        let mut exact = kappa < kappa_max;
        let fac = if exact { uq / (1.0 - kappa) } else { 0.0 };
        let mut gd = 0.0;
        for i in 0..n {
            step[i] = q[i] + fac * wv[i];
            gd += g[i] * step[i];
        }
        if !(gd < 0.0) {
            exact = false;
            gd = 0.0;
            for i in 0..n {
                step[i] = q[i];
                gd += g[i] * step[i];
            }
        }
        let pg = g.iter().fold(0.0f64, |m, t| m.max(t.abs()));
        let converged = (exact && -gd <= dec_rtol * f.abs().max(1.0)) || pg <= gtol;
        if converged {
            fit.status = 0;
            return fit;
        }
        if !gd.is_finite() {
            fit.status = 3;
            return fit;
        }
        if it == maxiter {
            fit.status = 1;
            return fit;
        }
        let mut t = 1.0f64;
        for k in 0..c {
            let dk = step[c + k];
            if dk < 0.0 {
                t = t.min(0.95 * x[c + k] / -dk);
            }
        }
        let mut accepted = false;
        for _ in 0..50 {
            for i in 0..n {
                cand[i] = x[i] + t * step[i];
            }
            let (fv, _) = e.eval(&cand);
            if fv.is_finite() && fv <= f + 1e-4 * t * gd {
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            fit.status = if exact && -gd <= 1e-9 * f.abs().max(1.0) { 0 } else { 2 };
            return fit;
        }
        x.copy_from_slice(&cand);
        fit.iterations += 1;
    }
    fit
}

pub struct MapFit {
    pub iterations: u32,
    pub certified: bool,
    pub gap: f64,
    pub tolerance: f64,
}

#[inline(never)]
fn w_times(w: &[f64], c: usize, x: &[f64], out: &mut [f64]) {
    for (o, wr) in out.iter_mut().zip(w.chunks_exact(c)) {
        let mut s = 0.0;
        for k in 0..c {
            s += wr[k] * x[k];
        }
        *o = s;
    }
}

#[inline(never)]
fn wt_times(w: &[f64], c: usize, y: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|t| *t = 0.0);
    for (&yj, wr) in y.iter().zip(w.chunks_exact(c)) {
        for k in 0..c {
            out[k] += yj * wr[k];
        }
    }
}

/// W^T diag(weights) W + ridge I, full symmetric, row-major.
#[inline(never)]
fn weighted_gram(w: &[f64], c: usize, weights: &[f64], ridge: f64, m: &mut [f64], row: &mut [f64]) {
    m.iter_mut().for_each(|t| *t = 0.0);
    for (j, wr) in w.chunks_exact(c).enumerate() {
        let dj = weights[j];
        for k in 0..c {
            row[k] = dj * wr[k];
        }
        for i in 0..c {
            let ri = row[i];
            for (mk, &wk) in m[i * c + i..i * c + c].iter_mut().zip(&wr[i..]) {
                *mk += ri * wk;
            }
        }
    }
    for i in 0..c {
        for k in i + 1..c {
            m[k * c + i] = m[i * c + k];
        }
        m[i * c + i] += ridge;
    }
}

/// Pinball MAP baseline: min_x sum pinball_tau(v - W x) + sigma/2 |x|^2. Writes W x to `fitted`.
pub fn solve_map(w: &[f64], v: &[f64], p: usize, c: usize, tau: f64, sigma: f64, maxiter: u32, fitted: &mut [f64]) -> MapFit {
    let (lo, hi) = (tau - 1.0, tau);
    let mut x = vec![0.0; c];
    let mut a = vec![tau - 0.5; p];
    let scale = (v.iter().map(|t| t.abs()).sum::<f64>() / p as f64).max(1e-12);
    let mut z1: Vec<f64> = v.iter().map(|&t| (-t).max(0.0) + scale).collect();
    let mut z2: Vec<f64> = v.iter().map(|&t| t.max(0.0) + scale).collect();
    let (mut s1, mut s2, mut dinv, mut r_a) = (vec![0.0; p], vec![0.0; p], vec![0.0; p], vec![0.0; p]);
    let (mut t1, mut t2, mut h, mut tmp_p) = (vec![0.0; p], vec![0.0; p], vec![0.0; p], vec![0.0; p]);
    let (mut da, mut dz1, mut dz2) = (vec![0.0; p], vec![0.0; p], vec![0.0; p]);
    let (mut r_x, mut rhs, mut dx, mut tmp_c, mut row) = (vec![0.0; c], vec![0.0; c], vec![0.0; c], vec![0.0; c], vec![0.0; c]);
    let mut m = vec![0.0; c * c];
    let mut fit = MapFit { iterations: maxiter, certified: false, gap: f64::NAN, tolerance: f64::NAN };

    for it in 1..=maxiter {
        for j in 0..p {
            s1[j] = (a[j] - lo).max(1e-15);
            s2[j] = (hi - a[j]).max(1e-15);
            dinv[j] = 1.0 / (z1[j] / s1[j] + z2[j] / s2[j]);
        }
        wt_times(w, c, &a, &mut tmp_c);
        for k in 0..c {
            r_x[k] = sigma * x[k] - tmp_c[k];
        }
        w_times(w, c, &x, &mut tmp_p);
        let mut mu_c = 0.0;
        for j in 0..p {
            r_a[j] = tmp_p[j] - v[j] - z1[j] + z2[j];
            mu_c += s1[j] * z1[j] + s2[j] * z2[j];
        }
        mu_c /= 2.0 * p as f64;
        weighted_gram(w, c, &dinv, sigma, &mut m, &mut row);
        if !cholesky(&mut m, c) {
            return fit;
        }
        let mut direction = |t1: &[f64], t2: &[f64], dx: &mut [f64], da: &mut [f64], dz1: &mut [f64], dz2: &mut [f64]| {
            for j in 0..p {
                h[j] = -r_a[j] + t1[j] / s1[j] - t2[j] / s2[j];
                tmp_p[j] = dinv[j] * h[j];
            }
            wt_times(w, c, &tmp_p, &mut rhs);
            for k in 0..c {
                rhs[k] -= r_x[k];
            }
            chol_solve(&m, c, &rhs, dx);
            w_times(w, c, dx, &mut tmp_p);
            for j in 0..p {
                da[j] = dinv[j] * (h[j] - tmp_p[j]);
                dz1[j] = (t1[j] - z1[j] * da[j]) / s1[j];
                dz2[j] = (t2[j] + z2[j] * da[j]) / s2[j];
            }
        };
        let max_step = |da: &[f64], dz1: &[f64], dz2: &[f64]| {
            let mut al = 1.0f64;
            for j in 0..p {
                if da[j] < 0.0 { al = al.min(s1[j] / -da[j]); }
                if da[j] > 0.0 { al = al.min(s2[j] / da[j]); }
                if dz1[j] < 0.0 { al = al.min(z1[j] / -dz1[j]); }
                if dz2[j] < 0.0 { al = al.min(z2[j] / -dz2[j]); }
            }
            al
        };
        for j in 0..p {
            t1[j] = -s1[j] * z1[j];
            t2[j] = -s2[j] * z2[j];
        }
        direction(&t1, &t2, &mut dx, &mut da, &mut dz1, &mut dz2);
        let al = max_step(&da, &dz1, &dz2);
        let mut mu_aff = 0.0;
        for j in 0..p {
            mu_aff += (s1[j] + al * da[j]) * (z1[j] + al * dz1[j]) + (s2[j] - al * da[j]) * (z2[j] + al * dz2[j]);
        }
        mu_aff /= 2.0 * p as f64;
        let sm = (mu_aff / mu_c).powi(3) * mu_c;
        for j in 0..p {
            t1[j] = -s1[j] * z1[j] + sm - da[j] * dz1[j];
            t2[j] = -s2[j] * z2[j] + sm + da[j] * dz2[j];
        }
        direction(&t1, &t2, &mut dx, &mut da, &mut dz1, &mut dz2);
        let al = (0.99 * max_step(&da, &dz1, &dz2)).min(1.0);
        if !al.is_finite() {
            return fit;
        }
        for k in 0..c {
            x[k] += al * dx[k];
        }
        for j in 0..p {
            a[j] += al * da[j];
            z1[j] += al * dz1[j];
            z2[j] += al * dz2[j];
        }
        // Certificate on the original pinball problem: primal at x, dual at the clipped a.
        w_times(w, c, &x, fitted);
        let (mut primal, mut dual) = (0.0, 0.0);
        for j in 0..p {
            let r = v[j] - fitted[j];
            primal += (tau * r).max((tau - 1.0) * r);
            let ac = a[j].clamp(lo, hi);
            tmp_p[j] = ac;
            dual += ac * v[j];
        }
        wt_times(w, c, &tmp_p, &mut tmp_c);
        primal += 0.5 * sigma * x.iter().map(|t| t * t).sum::<f64>();
        dual -= tmp_c.iter().map(|t| t * t).sum::<f64>() / (2.0 * sigma);
        let gap = primal - dual;
        let tol = 1e-9 + 1e-8 * primal.abs().max(dual.abs()).max(1.0);
        fit.gap = gap;
        fit.tolerance = tol;
        if gap.is_finite() && gap.abs() <= tol {
            fit.iterations = it;
            fit.certified = true;
            return fit;
        }
    }
    fit
}

// ---------------------------------------------------------------- WebAssembly interface

#[no_mangle]
pub extern "C" fn vibes_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0f64; len];
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// # Safety
/// `ptr` and `len` must come from `vibes_alloc`.
#[no_mangle]
pub unsafe extern "C" fn vibes_free(ptr: *mut f64, len: usize) {
    drop(Vec::from_raw_parts(ptr, len, len));
}

/// Fit one spectrum. `w` is p x c row-major, `w2` its elementwise square, `mean` and `y` have
/// length p. Writes the baseline to `baseline` (NaN if not certified) and eight numbers to `out`:
/// sigma, ELBO, Newton iterations, Newton status, MAP iterations, certified, gap, Hessians built.
///
/// # Safety
/// Every pointer must reference a live buffer of the stated length.
#[no_mangle]
pub unsafe extern "C" fn vibes_fit(
    w: *const f64, w2: *const f64, mean: *const f64, y: *const f64, p: usize, c: usize, tau: f64,
    baseline: *mut f64, out: *mut f64,
) {
    let w = core::slice::from_raw_parts(w, p * c);
    let w2 = core::slice::from_raw_parts(w2, p * c);
    let mean = core::slice::from_raw_parts(mean, p);
    let y = core::slice::from_raw_parts(y, p);
    let baseline = core::slice::from_raw_parts_mut(baseline, p);
    let out = core::slice::from_raw_parts_mut(out, 8);
    let v: Vec<f64> = y.iter().zip(mean).map(|(a, b)| a - b).collect();
    let e = fit_elbo(w, w2, &v, p, c, tau, 300);
    out[0] = e.sigma;
    out[1] = e.elbo;
    out[2] = e.iterations as f64;
    out[3] = e.status as f64;
    out[7] = e.hessians as f64;
    let ok = e.status == 0 && e.sigma.is_finite() && e.sigma > 0.0;
    baseline_into(w, mean, &v, p, c, tau, if ok { e.sigma } else { f64::NAN }, baseline, out);
}

/// The baseline at a noise level given by the caller, with no fit of the noise level: the way to
/// repeat a saved run from its recorded sigma. `out` as in `vibes_fit`, with sigma echoed, the ELBO
/// NaN and the Newton fields zero.
///
/// # Safety
/// Every pointer must reference a live buffer of the stated length.
#[no_mangle]
pub unsafe extern "C" fn vibes_baseline_at(
    w: *const f64, mean: *const f64, y: *const f64, p: usize, c: usize, tau: f64, sigma: f64,
    baseline: *mut f64, out: *mut f64,
) {
    let w = core::slice::from_raw_parts(w, p * c);
    let mean = core::slice::from_raw_parts(mean, p);
    let y = core::slice::from_raw_parts(y, p);
    let baseline = core::slice::from_raw_parts_mut(baseline, p);
    let out = core::slice::from_raw_parts_mut(out, 8);
    let v: Vec<f64> = y.iter().zip(mean).map(|(a, b)| a - b).collect();
    out[0] = sigma;
    out[1] = f64::NAN;
    out[2] = 0.0;
    out[3] = 0.0;
    out[7] = 0.0;
    baseline_into(w, mean, &v, p, c, tau, sigma, baseline, out);
}

/// Solve the baseline at `sigma` (skipped unless it is finite and positive) and fill out[4..7].
#[allow(clippy::too_many_arguments)]
fn baseline_into(w: &[f64], mean: &[f64], v: &[f64], p: usize, c: usize, tau: f64, sigma: f64, baseline: &mut [f64], out: &mut [f64]) {
    let m = if sigma.is_finite() && sigma > 0.0 {
        solve_map(w, v, p, c, tau, sigma, 100, baseline)
    } else {
        MapFit { iterations: 0, certified: false, gap: f64::NAN, tolerance: f64::NAN }
    };
    out[4] = m.iterations as f64;
    out[5] = if m.certified { 1.0 } else { 0.0 };
    out[6] = m.gap;
    for j in 0..p {
        baseline[j] = if m.certified { mean[j] + baseline[j] } else { f64::NAN };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth two-component background under one positive band: the fit must converge, certify,
    /// and leave most of the band above the baseline.
    #[test]
    fn fits_a_band_on_a_smooth_background() {
        let (p, c, tau) = (400usize, 2usize, 0.1);
        let mut w = vec![0.0; p * c];
        let mut y = vec![0.0; p];
        for j in 0..p {
            let t = j as f64 / p as f64;
            w[j * c] = 0.02;
            w[j * c + 1] = 0.02 * (t - 0.5);
            let band = 0.01 * (-((t - 0.3) / 0.02).powi(2)).exp();
            let ripple = 2e-4 * (37.0 * t).sin();
            y[j] = 0.03 + 0.012 * (t - 0.5) + band + ripple;
        }
        let w2: Vec<f64> = w.iter().map(|v| v * v).collect();
        let e = fit_elbo(&w, &w2, &y, p, c, tau, 300);
        assert_eq!(e.status, 0);
        assert!(e.sigma > 0.0 && e.sigma.is_finite());
        let mut fitted = vec![0.0; p];
        let m = solve_map(&w, &y, p, c, tau, e.sigma, 100, &mut fitted);
        assert!(m.certified);
        let peak = (0.3 * p as f64) as usize;
        assert!(y[peak] - fitted[peak] > 0.008);
    }
}
