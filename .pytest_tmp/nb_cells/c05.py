DIP_DEPTH_MAX = 0.1       # normalized units; deeper dip -> faulty curve
MIN_LEN_AFTER_TRIM = 20   # too few points left after trimming -> drop


def reanchor(val):
    """Salvage a curve dipping below 0.

    Returns (val_proc, drop_reason):
    - val_proc: curve trimmed before the dip and shifted so the dip is the
      new zero (== input unchanged if the curve never goes negative);
    - drop_reason: None if kept, else a string explaining the drop.
    """
    if len(val) == 0:
        return val, "empty"
    imin = int(np.argmin(val))
    if val[imin] >= 0:
        return val, None
    if val[imin] < -DIP_DEPTH_MAX:
        return val, f"deep dip ({val[imin]:.3f})"
    val2 = val[imin:] - val[imin]
    if len(val2) < MIN_LEN_AFTER_TRIM:
        return val2, f"only {len(val2)} pts after trim"
    return val2, None


# apply to every curve; a SAMPLE is dropped if any of its variables is dropped
def process_split(curves):
    proc, drop = [], {}
    for i, c in enumerate(curves):
        sample, reasons = {}, {}
        for v in VAR_IDS:
            pos, val = c[v]
            vp, reason = reanchor(val)
            sample[v] = (pos, val, vp, reason)  # keep raw for review plots
            if reason:
                reasons[v] = reason
        proc.append(sample)
        if reasons:
            drop[i] = reasons
    return proc, drop

train_proc, train_drop = process_split(train_curves)
test_proc, test_drop = process_split(test_curves)
print(f"train: kept {len(train_curves) - len(train_drop)}/{len(train_curves)} "
      f"({100*(1 - len(train_drop)/len(train_curves)):.1f}%), dropped {len(train_drop)}")
print(f"test : kept {len(test_curves) - len(test_drop)}/{len(test_curves)} "
      f"({100*(1 - len(test_drop)/len(test_curves)):.1f}%), dropped {len(test_drop)}")
for i, r in list(train_drop.items())[:10]:
    print(f"  drop sample {i}: " + ", ".join(f"var{v:.0f}: {s}" for v, s in r.items()))

# how many curves were re-anchored?
n_reanch = sum(1 for c in train_proc for v in VAR_IDS
               if len(c[v][2]) != len(c[v][1]))
print(f"train curves re-anchored (trimmed+shifted): {n_reanch}")
