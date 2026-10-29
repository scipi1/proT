"""
Post-hoc sigma temperature scaling for the censored log-normal head.

The trained model is frozen. A single scalar T (per variable, optionally
per conditioning fraction) rescales the log-space scale, sigma -> T * sigma,
and is fitted on the validation split by minimizing the censored NLL:

    observed : log(T s) + 0.5 (z / T)^2 (+ constants), z = (log(y+shift) - mu)/s
    survival : -log Phi((mu - log(thr + shift)) / (T s))   (honest tails only)

mu is unchanged, so the median, MAE/R2 and the "median crosses the
threshold" decision are unchanged; NLL, CRPS, coverage and P(fail) change.

Reading T: T > 1 -> the model is overconfident; T ~ 1 -> calibrated (a rising
val NLL then comes from mu errors); T < 1 -> underconfident.

All functions are numpy-only so they work on the notebook caches and can
later run as the last step of training.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr

SIGMA_EPS = 1e-3  # matches the losses in proT/training/losses.py


def softplus(x):
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _flat(mu, sigma, val, observed, survival, shift, threshold):
    """Flattened observed / survival terms needed by the NLL in T."""
    log_y = np.log(np.clip(np.nan_to_num(val) + shift, 1e-8, None))
    z_obs = ((log_y - mu) / sigma)[observed]
    s_obs = sigma[observed]
    z_thr = ((mu - np.log(threshold + shift)) / sigma)[survival]
    return z_obs, s_obs, z_thr


def censored_nll_T(T, z_obs, s_obs, z_thr):
    """Mean censored NLL (without the T-independent Jacobian) at temperature T."""
    n = z_obs.size + z_thr.size
    if n == 0:
        return np.nan
    obs = np.sum(np.log(T * s_obs) + 0.5 * (z_obs / T) ** 2) \
        + 0.5 * np.log(2 * np.pi) * z_obs.size
    surv = -np.sum(log_ndtr(z_thr / T))
    return (obs + surv) / n


def fit_temperature(mu, sigma, val, observed, survival, shift=1e-3,
                    threshold=1.0, bounds=(-3.0, 3.0)):
    """Fit T (scalar) minimizing the censored NLL over the given positions.

    Args:
        mu, sigma: log-space mean and positive std, same shape.
        val: targets (NaN where censored/padding).
        observed, survival: bool masks of the positions entering the NLL
            (use classify_termination: observed / survival).
        bounds: search interval for log T.

    Returns:
        dict(T, nll_before, nll_after, closed_form_obs_T, n_obs, n_surv).
    """
    z_obs, s_obs, z_thr = _flat(mu, sigma, val, observed, survival, shift, threshold)
    if z_obs.size + z_thr.size == 0:
        return {"T": 1.0, "nll_before": np.nan, "nll_after": np.nan,
                "closed_form_obs_T": np.nan, "n_obs": 0, "n_surv": 0}
    res = minimize_scalar(lambda lt: censored_nll_T(np.exp(lt), z_obs, s_obs, z_thr),
                          bounds=bounds, method="bounded")
    T = float(np.exp(res.x))
    return {"T": T,
            "nll_before": float(censored_nll_T(1.0, z_obs, s_obs, z_thr)),
            "nll_after": float(censored_nll_T(T, z_obs, s_obs, z_thr)),
            # observed-only optimum (sanity check): T^2 = mean(z^2)
            "closed_form_obs_T": float(np.sqrt(np.mean(z_obs ** 2)))
            if z_obs.size else np.nan,
            "n_obs": int(z_obs.size), "n_surv": int(z_thr.size)}


def fit_temperature_per_var(mu, sigma, val, var, observed, survival,
                            shift=1e-3, threshold=1.0):
    """One T per target variable id. Returns {var_id: fit_temperature dict}."""
    out = {}
    for v in np.unique(var[observed | survival]):
        m = var == v
        out[int(v)] = fit_temperature(mu, sigma, val, observed & m,
                                      survival & m, shift, threshold)
    return out


def apply_temperature(sigma, var, temps):
    """sigma scaled per variable: temps = {var_id: T} or {var_id: fit dict}."""
    out = sigma.copy()
    for v, t in temps.items():
        T = t["T"] if isinstance(t, dict) else float(t)
        out = np.where(var == v, sigma * T, out)
    return out


def lognormal_crps_np(mu, sigma, val, shift=1e-3):
    """Element-wise closed-form log-normal CRPS (NaN where val is NaN)."""
    y_s = np.clip(val + shift, 1e-12, None)
    w = (np.log(y_s) - mu) / sigma
    return y_s * (2 * ndtr(w) - 1) - 2 * np.exp(mu + 0.5 * sigma ** 2) * (
        ndtr(w - sigma) + ndtr(sigma / np.sqrt(2)) - 1)


def calibration_summary(mu, sigma, val, observed, survival, shift=1e-3,
                        threshold=1.0):
    """Censored NLL (incl. Jacobian), CRPS, 1/2-sigma coverage over observed."""
    z_obs, s_obs, z_thr = _flat(mu, sigma, val, observed, survival, shift, threshold)
    log_y = np.log(np.clip(np.nan_to_num(val) + shift, 1e-8, None))[observed]
    n = max(z_obs.size + z_thr.size, 1)
    nll = (censored_nll_T(1.0, z_obs, s_obs, z_thr) * n + log_y.sum()) / n
    crps = lognormal_crps_np(mu[observed], sigma[observed], val[observed], shift)
    return {"nll": float(nll), "crps": float(np.mean(crps)) if crps.size else np.nan,
            "cov_1s": float(np.mean(np.abs(z_obs) <= 1.0)) if z_obs.size else np.nan,
            "cov_2s": float(np.mean(np.abs(z_obs) <= 1.96)) if z_obs.size else np.nan}
