def load_curves(npz_path):
    """Return list of dicts: per sample, per variable -> (pos, val) observed."""
    d = np.load(npz_path)
    y = d["y"]
    curves = []
    for i in range(y.shape[0]):
        sample = {}
        for v in VAR_IDS:
            m = y[i, :, VAR_CH] == v
            pos = y[i, m, POS_CH]
            val = y[i, m, VAL_CH]
            ok = ~np.isnan(val)
            order = np.argsort(pos[ok])
            sample[v] = (pos[ok][order], val[ok][order])
        curves.append(sample)
    return curves

train_curves = load_curves(DATA_DIR / "train_data.npz")
test_curves = load_curves(DATA_DIR / "test_data.npz")
print(f"train samples: {len(train_curves)}, test samples: {len(test_curves)}")
for v in VAR_IDS:
    lens = np.array([len(c[v][1]) for c in train_curves])
    print(f"var {v:.0f}: len min/med/max = {lens.min()}/{int(np.median(lens))}/{lens.max()}, "
          f"short (<{FULL_LEN}): {(lens < FULL_LEN).sum()}")
