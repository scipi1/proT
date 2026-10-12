def sg_window(n, polyorder=SG_POLYORDER, cap=SG_MAX_WINDOW):
    w = min(cap, max(polyorder + 2, n // 8))
    w += 1 - w % 2  # odd
    return max(w, polyorder + 2 + (polyorder % 2 == 0))


def fit_sg_isotonic(val):
    """Monotone smooth, zero-centered fit of one (already re-anchored) curve."""
    n = len(val)
    w = min(sg_window(n), n if n % 2 else n - 1)
    sm = savgol_filter(val, w, SG_POLYORDER) if w > SG_POLYORDER and n >= 5 else val.copy()
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    sm = iso.fit_transform(np.arange(n), sm)
    sm = sm - sm[0]
    return np.clip(sm, 0.0, None)


def fit_cubic(val):
    n = len(val)
    if n < 5:
        return np.full(n, np.nan)
    c = np.polyfit(np.arange(n), val, 3)
    return np.polyval(c, np.arange(n))
