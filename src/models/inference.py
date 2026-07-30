"""
src/models/inference.py

Shared helper for computing asymptotic standard errors, z-stats, and
p-values for parameters fit via scipy.optimize.minimize(method="L-BFGS-B").

WHY THIS EXISTS / WHY NOT result.hess_inv:
L-BFGS-B is a limited-memory quasi-Newton method. The `hess_inv` it
returns (a `LbfgsInvHessProduct`) is a low-rank running approximation
built from the last handful of gradient/step pairs seen *during* the
search - it is NOT the actual inverse Hessian of the objective at the
optimum. Using it for standard errors will silently over- or under-state
parameter uncertainty, sometimes substantially, and there's no reliable
way to tell from the result alone how wrong it is.

The only trustworthy route with L-BFGS-B is to throw away hess_inv after
optimization finishes and compute a fresh numerical Hessian of the
negative log-likelihood AT the converged point, then invert that. That's
what this module does. It costs extra objective evaluations (see
numerical_hessian's docstring for the count) - for a cheap NLL (EPS) this
is negligible; for an NLL with an expensive inner solve per call (NRL's
Bellman value iteration) it can matter, which is why NRLModel.fit exposes
compute_se as an explicit opt-in rather than defaulting it on.
"""

import numpy as np
from scipy import stats


def numerical_hessian(func, x, eps=1e-4):
    """Central-difference Hessian of a scalar function `func` at point `x`.

    Cost: 4 evaluations of `func` per unique (i, j) pair with i <= j, i.e.
    2 * n * (n + 1) total calls for an n-parameter model (60 calls for
    EPS's 5 params, 84 for NRL's 6). Each `func` call here is a full
    objective evaluation over the whole dataset, same as one step scipy
    took during optimization - so this Hessian pass costs roughly as much
    as ~2n additional gradient-free optimizer iterations, evaluated once,
    after fit() has already converged.

    Step size scales with |x[i]| (relative step) rather than using a fixed
    eps everywhere, since a fixed absolute step is too coarse for
    large-magnitude parameters and too noisy for near-zero ones.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            step_i = eps * max(abs(x[i]), 1.0)
            step_j = eps * max(abs(x[j]), 1.0)

            x_pp, x_pm, x_mp, x_mm = x.copy(), x.copy(), x.copy(), x.copy()
            x_pp[i] += step_i; x_pp[j] += step_j
            x_pm[i] += step_i; x_pm[j] -= step_j
            x_mp[i] -= step_i; x_mp[j] += step_j
            x_mm[i] -= step_i; x_mm[j] -= step_j

            hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * step_i * step_j)
            hess[j, i] = hess[i, j]
    return hess


def standard_errors_from_nll(nll_func, x_hat, param_names, eps=1e-4):
    """Given a converged NLL function and its minimizer x_hat, returns a
    dict keyed by each name in param_names with 'estimate', 'se', 'z', 'pval'.

    z / pval use a normal reference distribution (asymptotic MLE theory),
    matching the z-stat convention in the target table rather than a
    t-distribution - appropriate here since these are large-sample MLE
    estimates, not small-sample OLS coefficients.

    Falls back to NaN SEs (with a printed warning) for any parameter whose
    Hessian-implied variance comes out negative or where the Hessian isn't
    invertible at all, rather than raising or printing a nonsense value -
    this happens when the likelihood is flat/ridge-shaped at the optimum
    (e.g. a parameter that barely affects the objective, or one pinned at
    its box-constraint bound) and signals that parameter's estimate should
    be treated with real caution even though it still has a point value.
    """
    x_hat = np.asarray(x_hat, dtype=float)
    hessian = numerical_hessian(nll_func, x_hat, eps=eps)

    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        print("WARNING: Hessian of the NLL is singular at the fitted point - falling back "
              "to a pseudo-inverse. Standard errors below may be unreliable.")
        cov = np.linalg.pinv(hessian)

    diag = np.diag(cov)
    bad = diag < 0
    if np.any(bad):
        bad_names = [n for n, b in zip(param_names, bad) if b]
        print(f"WARNING: negative implied variance for {bad_names} - the Hessian isn't "
              f"positive definite there (optimizer may not have fully converged, or the "
              f"parameter is sitting at/near a box-constraint bound). Reporting SE/z/pval "
              f"as NaN for these rather than sqrt of a negative number.")

    se = np.where(diag >= 0, np.sqrt(np.clip(diag, 0, None)), np.nan)
    z = np.where(se > 0, x_hat / se, np.nan)
    pval = np.where(np.isfinite(z), 2 * (1 - stats.norm.cdf(np.abs(z))), np.nan)

    return {
        name: {"estimate": est, "se": s, "z": zz, "pval": p}
        for name, est, s, zz, p in zip(param_names, x_hat, se, z, pval)
    }
