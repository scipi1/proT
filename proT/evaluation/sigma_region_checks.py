"""Diagnose WHERE the lognormal-censored model is miscalibrated.

Runs the four checks for one experiment / fold on the cached raw rollouts:
  1. coverage of the +/-1 sigma and +/-1.96 sigma bands, per region
  2. mean z^2 per region (1 = calibrated)
  3. per-region optimal sigma temperature T (fitted on the validation split,
     evaluated on the test set), incl. NLL and CRPS before/after
  4. share of the censored NLL contributed by each region (val and test)

Regions: cycle bins and target-value bins. Everything works on the npz caches
written by nb_eval_seq_v1 (pred_k<fold>_<split>_f000_raw.npz), so no retraining.

Usage:
    python -m proT.evaluation.sigma_region_checks <experiment_dir> [--fold 3]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from proT.training.losses import CensoredLogNormalNLL
from proT.evaluation.calibration import softplus, lognormal_crps_np

POS, VAR, VAL = 1, 2, 3
CYC_BINS = [(0, 15), (15, 50), (50, 150), (150, math.inf)]
Y_BINS = [(-1, 0.05), (0.05, 0.3), (0.3, 0.8), (0.8, math.inf)]
def load_split(exp, fold, split):
    c = np.load(Path(exp) / "cache" / f"pred_k{fold}_{split}_f000_raw.npz")
    o, t = c["outputs"], c["targets"]
    mu, s = o[:, :, 0].astype(float), softplus(o[:, :, 1].astype(float)) + 1e-3
    tt = torch.tensor(t, dtype=torch.float32)
    m = CensoredLogNormalNLL.classify_termination(
        tt[:, :, VAL], tt[:, :, VAR], tt[:, :, POS],
        expected_len=args.expected_len, threshold=args.threshold,
        fail_margin=args.fail_margin)
    return dict(mu=mu, s=s, val=t[:, :, VAL], pos=t[:, :, POS],
                obs=m["observed"].numpy(),
                surv=(m["active"] & ~m["observed"]).numpy())


def regions(p):
    y = np.nan_to_num(p["val"])
    out = {}
    for lo, hi in CYC_BINS:
        out[f"cyc({lo:g},{hi:g}]"] = p["obs"] & (p["pos"] > lo) & (p["pos"] <= hi)
    for lo, hi in Y_BINS:
        out[f"y({lo:g},{hi:g}]"] = p["obs"] & (y > lo) & (y <= hi)
    out["all"] = p["obs"]
    return out


def z_and_logy(p):
    log_y = np.log(np.clip(np.nan_to_num(p["val"]) + args.shift, 1e-8, None))
    return (log_y - p["mu"]) / p["s"], log_y


def nll_regions(p, reg):
    """Censored NLL pieces per region: observed NLL (with Jacobian) + surv."""
    z, log_y = z_and_logy(p)
    nll = 0.5 * np.log(2 * np.pi) + np.log(p["s"]) + 0.5 * z ** 2 + log_y
    z_thr = (p["mu"] - math.log(args.threshold + args.shift)) / p["s"]
    surv_nll = -log_ndtr(z_thr)[p["surv"]]
    tot = np.nansum(nll[p["obs"]]) + np.sum(surv_nll)
    out = {name: np.nansum(nll[m]) / max(tot, 1e-12) for name, m in reg.items()}
    out["survival_tail"] = np.sum(surv_nll) / max(tot, 1e-12)
    n = max(int(p["obs"].sum()), 1) + max(int(p["surv"].sum()), 1)
    return out, tot / n


def check12(p, reg):
    z, _ = z_and_logy(p)
    rows = []
    for name, m in reg.items():
        zm = z[m]
        rows.append(dict(region=name, n=int(m.sum()),
                         cov_1s=np.mean(np.abs(zm) <= 1.0) if m.sum() else np.nan,
                         cov_2s=np.mean(np.abs(zm) <= 1.96) if m.sum() else np.nan,
                         mean_z2=np.mean(zm ** 2) if m.sum() else np.nan))
    return rows


def opt_T_obs(z):
    """Observed-only optimum: T^2 = mean(z^2); 1-D search as a guard."""
    if len(z) == 0:
        return 1.0
    f = lambda lt: np.sum(lt + 0.5 * (z * np.exp(-lt)) ** 2)
    return float(np.exp(minimize_scalar(f, bounds=(-3, 3), method="bounded").x))


def check3(va, te, reg_va, reg_te):
    z_va, _ = z_and_logy(va)
    z_te, _ = z_and_logy(te)
    crps_te = lognormal_crps_np(te["mu"], te["s"], np.nan_to_num(te["val"]),
                                args.shift)
    rows = []
    for name in reg_va:
        m_v, m_t = reg_va[name], reg_te[name]
        T = opt_T_obs(z_va[m_v])
        zm = z_te[m_t]
        nll_raw = np.mean(np.log(te["s"][m_t]) + 0.5 * zm ** 2) \
            + 0.5 * np.log(2 * np.pi) if m_t.sum() else np.nan
        nll_T = np.mean(np.log(T * te["s"][m_t]) + 0.5 * (zm / T) ** 2) \
            + 0.5 * np.log(2 * np.pi) if m_t.sum() else np.nan
        T2 = np.clip(T, 1e-2, 1e2)
        crps_T = lognormal_crps_np(te["mu"], te["s"] * T2,
                                   np.nan_to_num(te["val"]), args.shift)
        rows.append(dict(region=name, n_test=int(m_t.sum()), T_val=T,
                         nll_raw=nll_raw, nll_T=nll_T,
                         crps_raw=crps_te[m_t].mean() if m_t.sum() else np.nan,
                         crps_T=crps_T[m_t].mean() if m_t.sum() else np.nan))
    return rows


def fmt_tab(rows, cols):
    w = [max(len(c), max(len(f"{r[c]:.4g}") if isinstance(r[c], float)
                         else len(str(r[c])) for r in rows)) for c in cols]
    out = ["  ".join(c.ljust(wi) for c, wi in zip(cols, w))]
    for r in rows:
        out.append("  ".join((f"{r[c]:.4g}" if isinstance(r[c], float)
                              else str(r[c])).ljust(wi)
                             for c, wi in zip(cols, w)))
    return "\n".join(out)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--shift", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--expected-len", dest="expected_len", type=int, default=200)
    ap.add_argument("--fail-margin", dest="fail_margin", type=float, default=0.8)
    args = ap.parse_args()

    va = load_split(args.experiment_dir, args.fold, "val")
    te = load_split(args.experiment_dir, args.fold, "test")
    reg_va, reg_te = regions(va), regions(te)

    print(f"experiment: {Path(args.experiment_dir).name}  fold: {args.fold}\n")
    print("CHECK 1+2: coverage and mean z^2 (raw sigma)")
    for tag, p, reg in [("validation", va, reg_va), ("test", te, reg_te)]:
        print(f"  [{tag}]")
        print(fmt_tab(check12(p, reg), ["region", "n", "cov_1s", "cov_2s", "mean_z2"]))
        print()

    print("CHECK 3: per-region optimal T (fit: validation, eval: test)")
    print(fmt_tab(check3(va, te, reg_va, reg_te),
                  ["region", "n_test", "T_val", "nll_raw", "nll_T", "crps_raw", "crps_T"]))
    print()

    print("CHECK 4: share of the censored NLL per region")
    for tag, p, reg in [("validation", va, reg_va), ("test", te, reg_te)]:
        share, mean_nll = nll_regions(p, reg)
        print(f"  [{tag}] mean NLL = {mean_nll:.4g}")
        for k, v in share.items():
            print(f"    {k:16s} {100 * v:5.1f}%")
    sys.exit(0)