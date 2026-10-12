FAIL_TOL = 0.98        # last raw value >= this -> failed (at threshold)
BEZIER_K = 10          # last k observed points replaced by the Bezier segment
BEZIER_END_FRAC = 0.85 # x-position of P2 in the segment (steepness handle)
BEZIER_END_RISE = 0.1  # fraction of the gap gained by P2 (small = steep finish)


def classify(entry, partner_entry):
    """failed / interrupted / full / empty, from ORIGINAL observed lengths."""
    n = len(entry[1])
    last = entry[1][-1] if n else np.nan
    n_p = len(partner_entry[1])
    last_p = partner_entry[1][-1] if n_p else np.nan
    if n == 0:
        return "empty"
    if n >= FULL_LEN:
        return "full"
    if last >= FAIL_TOL:
        return "failed"
    if n_p < FULL_LEN and last_p < FAIL_TOL:
        return "failed"  # ambiguous: both short, both low -> both failed
    return "interrupted"


def bezier_approach(sm, k=BEZIER_K, end_frac=BEZIER_END_FRAC,
                    end_rise=BEZIER_END_RISE):
    """Replace the last k points of the smooth fit by a cubic Bezier ending
    exactly at 1.0, C1-continuous at the junction."""
    n = len(sm)
    k = int(min(k, n))
    y0 = sm[-k]
    # local slope of the fit at the junction (per point)
    w = min(20, n - k)
    m = np.median(np.diff(sm[-k - w:-k])) if w >= 5 else 0.0
    P0 = np.array([0.0, y0])
    P1 = np.array([1.0 / 3.0, y0 + m * k / 3.0])
    P2 = np.array([end_frac, y0 + end_rise * (1.0 - y0)])
    P3 = np.array([1.0, 1.0])
    t = np.linspace(0.0, 1.0, k)
    B = ((1 - t) ** 3)[:, None] * P0 + (3 * (1 - t) ** 2 * t)[:, None] * P1 \
        + (3 * (1 - t) * t ** 2)[:, None] * P2 + (t ** 3)[:, None] * P3
    seg = np.clip(B[:, 1], 0.0, 1.0)
    out = sm.copy()
    out[-k:] = seg
    out = np.maximum.accumulate(out)  # enforce monotonicity (safety)
    return np.clip(out, 0.0, 1.0)


def idealize(entry, status, full_len=FULL_LEN):
    """Return (smooth_obs, full_curve) for one processed curve.

    full_curve: (full_len,) with NaN tail - positions after the last
    observation stay NaN, exactly as in the raw data.
    """
    pos, val_raw, val, _ = entry
    n = len(val)
    sm = fit_sg_isotonic(val) if n else np.array([])
    if status == "failed" and n:
        sm = bezier_approach(sm)
    full = np.full(full_len, np.nan)
    full[:n] = sm
    return sm, full


# classification inventory
counts = {s: {v: 0 for v in VAR_IDS} for s in ("failed", "interrupted", "full", "empty")}
status_map = {}
for i, c in enumerate(train_proc):
    if i in train_drop:
        continue
    for v in VAR_IDS:
        s = classify(c[v], c[2.0 if v == 1.0 else 1.0])
        counts[s][v] += 1
        status_map[(i, v)] = s
for s in counts:
    print(f"{s:12s}: var1={counts[s][1.0]:4d}  var2={counts[s][2.0]:4d}")
