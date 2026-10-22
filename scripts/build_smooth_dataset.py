"""Build dyconex_260701_seq2seq_smooth: idealized (smooth, monotone, zero-anchored)
IST targets for the delta-mode RNN arm.

Pipeline (validated in notebooks/nb_ist_smoothing.ipynb):
1. Split the target sequences into per-variable curves (var 1 = delta_A_norm,
   var 2 = delta_B_norm).
2. Anchor: subtract the global min; points before the min are set to 0 (NOT
   dropped, so length and cycle alignment are preserved). Drop guards: dip
   deeper than DIP_DEPTH_MAX or fewer than MIN_LEN_AFTER_TRIM points after
   the min.
3. Fit: Savitzky-Golay (local cubic) + isotonic monotonization, zero-centered.
4. Failure classification, PER VARIABLE (the sibling has no influence; the
   two variables fail independently), from the ORIGINAL observed length:
   failed = short and last value >= FAIL_MARGIN; interrupted = short and
   ending lower; full = full-length.
5. Failed curves: cubic Bezier over the last BEZIER_K points, C1-continuous
   at the junction, ending exactly at 1.0 at the last observed point.
   NaN tails are preserved everywhere - nothing past the last measurement.

The dataset stays SEQUENTIAL (absolute values): the increment decomposition
is done in-training by EncoderForecaster._increment_targets when
rnn_output_mode == "delta".

Usage: python scripts/build_smooth_dataset.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

SRC = Path(__file__).resolve().parent.parent / "data" / "input" / "dyconex_260701_seq2seq"
DST = SRC.parent / "dyconex_260701_seq2seq_smooth"

FULL_LEN = 200          # points per variable in a complete test
FAIL_MARGIN = 0.8       # short curve with last raw value >= this -> failed
                        # (matches training.loss_fail_margin * threshold 1.0)
DIP_DEPTH_MAX = 0.1     # deeper dip -> faulty curve
MIN_LEN_AFTER_TRIM = 20
SG_POLYORDER = 3
SG_MAX_WINDOW = 21
BEZIER_K = 10
BEZIER_END_FRAC = 0.85
BEZIER_END_RISE = 0.1

GRP_CH, POS_CH, VAR_CH, VAL_CH = 0, 1, 2, 3
VAR_IDS = (1.0, 2.0)


def anchor(val):
    """Re-zero at the global min. Returns (val_tail, imin, drop_reason).

    val_tail = val[imin:] - val[imin]; the imin points before the min are NOT
    dropped by the caller but set to 0, so the curve keeps its original length
    and cycle alignment (a full-length test must stay full-length).
    """
    if len(val) == 0:
        return val, 0, "empty"
    imin = int(np.argmin(val))
    if val[imin] < -DIP_DEPTH_MAX:
        return val, imin, f"deep dip ({val[imin]:.3f})"
    val2 = val[imin:] - val[imin]
    if len(val2) < MIN_LEN_AFTER_TRIM:
        return val2, imin, f"only {len(val2)} pts after min"
    return val2, imin, None


def fit_sg_isotonic(val):
    """Monotone smooth, zero-centered fit of one (anchored) curve."""
    n = len(val)
    w = min(SG_MAX_WINDOW, max(SG_POLYORDER + 2, n // 8))
    w += 1 - w % 2
    w = min(w, n if n % 2 else n - 1)
    sm = savgol_filter(val, w, SG_POLYORDER) if w > SG_POLYORDER and n >= 5 else val.copy()
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    sm = iso.fit_transform(np.arange(n), sm)
    return np.clip(sm - sm[0], 0.0, None)


def bezier_approach(sm, k=BEZIER_K):
    """Replace the last k points with a cubic Bezier ending exactly at 1.0,
    C1-continuous at the junction."""
    n = len(sm)
    k = int(min(k, n))
    y0 = sm[-k]
    w = min(20, n - k)
    m = np.median(np.diff(sm[-k - w:-k])) if w >= 5 else 0.0
    P0 = np.array([0.0, y0])
    P1 = np.array([1.0 / 3.0, y0 + m * k / 3.0])
    P2 = np.array([BEZIER_END_FRAC, y0 + BEZIER_END_RISE * (1.0 - y0)])
    P3 = np.array([1.0, 1.0])
    t = np.linspace(0.0, 1.0, k)
    B = ((1 - t) ** 3)[:, None] * P0 + (3 * (1 - t) ** 2 * t)[:, None] * P1 \
        + (3 * (1 - t) * t ** 2)[:, None] * P2 + (t ** 3)[:, None] * P3
    out = sm.copy()
    out[-k:] = np.clip(B[:, 1], 0.0, 1.0)
    out = np.maximum.accumulate(out)
    return np.clip(out, 0.0, 1.0)


def classify(n, last):
    """failed / interrupted / full / empty, per variable (independent of the
    sibling), from the ORIGINAL observed length and last raw value."""
    if n == 0:
        return "empty"
    if n >= FULL_LEN:
        return "full"
    if last >= FAIL_MARGIN:
        return "failed"
    return "interrupted"


def process_split(npz_path):
    """Return (x_keep, y_keep, y_raw_keep, report)."""
    d = np.load(npz_path)
    x, y = d["x"], d["y"]
    n_samples = y.shape[0]

    # pass 1: extract, anchor, classify -> per-var idealized curves
    idealized = {}   # (i, v) -> (n, smooth_values)
    drops = {}
    stats = {"reanchored": 0,
             "class": {s: {float(v): 0 for v in VAR_IDS}
                       for s in ("failed", "interrupted", "full", "empty")}}
    for i in range(n_samples):
        entries = {}
        for v in VAR_IDS:
            rows = np.where(y[i, :, VAR_CH] == v)[0]
            val = y[i, rows, VAL_CH]
            ok = ~np.isnan(val)
            val_raw = val[ok]
            vp, imin, reason = anchor(val_raw)
            if reason:
                drops.setdefault(i, {})[v] = reason
                continue
            entries[v] = (rows[ok], val_raw, vp, imin)
            if imin > 0:
                stats["reanchored"] += 1
        if i in drops:
            continue
        for v in VAR_IDS:
            if v not in entries:
                continue
            rows, val_raw, vp, imin = entries[v]
            # per-variable classification: the sibling has no influence
            status = classify(len(val_raw),
                              val_raw[-1] if len(val_raw) else np.nan)
            stats["class"][status][v] += 1
            sm = fit_sg_isotonic(vp) if len(vp) else np.array([])
            # pre-min points -> 0: original length and alignment preserved
            sm = np.concatenate([np.zeros(imin), sm])
            if status == "failed" and len(sm):
                sm = bezier_approach(sm)
            idealized[(i, v)] = (rows, sm)

    # pass 2: rebuild y with idealized values, re-indexed positions
    keep = [i for i in range(n_samples) if i not in drops]
    y_new = y.copy()
    y_raw = y.copy()
    for i in keep:
        for v in VAR_IDS:
            rows_all = np.where(y[i, :, VAR_CH] == v)[0]
            # re-index positions 1..len(rows_all) for this variable's rows
            y_new[i, rows_all, POS_CH] = np.arange(1, len(rows_all) + 1)
            y_new[i, rows_all, VAL_CH] = np.nan  # default: unobserved
            if (i, v) not in idealized:
                continue
            rows, sm = idealized[(i, v)]
            # same rows as the raw observations (length/alignment preserved)
            y_new[i, rows, VAL_CH] = sm

    report = {
        "split": npz_path.name,
        "n_samples_in": int(n_samples),
        "n_samples_kept": len(keep),
        "dropped": {int(i): {str(v): r for v, r in dct.items()}
                    for i, dct in drops.items()},
        "reanchored_curves": stats["reanchored"],
        "classification": {s: {str(v): c for v, c in dct.items()}
                           for s, dct in stats["class"].items()},
    }
    return x[keep], y_new[keep], y_raw[keep], report


def main():
    DST.mkdir(parents=True, exist_ok=True)
    reports = {}
    for split in ("train", "test"):
        x, y_new, y_raw, rep = process_split(SRC / f"{split}_data.npz")
        np.savez(DST / f"{split}_data.npz", x=x, y=y_new)
        np.savez(DST / f"{split}_raw_values.npz", y_raw=y_raw)
        reports[split] = rep
        print(f"[{split}] kept {rep['n_samples_kept']}/{rep['n_samples_in']} "
              f"-> x{x.shape} y{y_new.shape}")

    # copy metadata and write the report
    shutil.copytree(SRC / "meta_seq", DST / "meta_seq", dirs_exist_ok=True)
    report = {
        "source_dataset": SRC.name,
        "pipeline": "anchor (pre-min -> 0, length kept) -> SG(poly3)+isotonic"
                    " -> Bezier-to-1.0 for failed",
        "classification": "per-variable (independent of the sibling)",
        "config": {
            "FULL_LEN": FULL_LEN, "FAIL_MARGIN": FAIL_MARGIN,
            "DIP_DEPTH_MAX": DIP_DEPTH_MAX,
            "MIN_LEN_AFTER_TRIM": MIN_LEN_AFTER_TRIM,
            "SG_POLYORDER": SG_POLYORDER, "SG_MAX_WINDOW": SG_MAX_WINDOW,
            "BEZIER_K": BEZIER_K, "BEZIER_END_FRAC": BEZIER_END_FRAC,
            "BEZIER_END_RISE": BEZIER_END_RISE,
        },
        "splits": reports,
    }
    with open(DST / "meta_seq" / "smoothing_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("written:", DST)


if __name__ == "__main__":
    main()

