def collect_increments(proc, drop, smoothed):
    incs = {v: [] for v in VAR_IDS}
    for i, c in enumerate(proc):
        if i in drop:
            continue
        for v in VAR_IDS:
            val = c[v][2]  # re-anchored raw
            if len(val) < 5:
                continue
            x = fit_sg_isotonic(val) if smoothed else val - val[0]
            incs[v].append(np.diff(x))
    return {v: np.concatenate(lst) for v, lst in incs.items()}

raw_inc = collect_increments(train_proc, train_drop, smoothed=False)
sm_inc = collect_increments(train_proc, train_drop, smoothed=True)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
for ax, v in zip(axes, VAR_IDS):
    lo, hi = np.percentile(raw_inc[v], [0.5, 99.5])
    bins = np.linspace(lo, hi, 80)
    ax.hist(raw_inc[v], bins=bins, density=True, histtype="stepfilled",
            alpha=0.5, color=CB["raw"], label="raw deltas")
    ax.hist(sm_inc[v], bins=bins, density=True, histtype="step", lw=1.8,
            color=CB["sg"], label="smooth deltas")
    ax.set_yscale("log")
    ax.set_title(f"var {v:.0f}")
    ax.set_xlabel("increment [norm.]")
    ax.set_xlim(lo, hi)
axes[0].set_ylabel("density [-]")
axes[0].legend(frameon=False)
fig.suptitle("First differences: raw vs. idealized", fontweight="bold")
fig.tight_layout()
plt.show()

for v in VAR_IDS:
    print(f"var {v:.0f}: raw delta std={raw_inc[v].std():.5f}, "
          f"smooth delta std={sm_inc[v].std():.5f}, "
          f"smooth negative fraction={np.mean(sm_inc[v] < 0):.4f}")
