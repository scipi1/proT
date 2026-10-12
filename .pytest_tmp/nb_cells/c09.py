# comparison grid: raw vs SG-isotonic vs global cubic (on kept samples)
keep_idx = np.array([i for i in range(len(train_proc)) if i not in train_drop])
rng = np.random.default_rng(0)
show = rng.choice(keep_idx, size=6, replace=False)

fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.8))
for k, i in enumerate(show):
    ax = axes[k // 2, k % 2]
    for v in VAR_IDS:
        pos, val, vp, reason = train_proc[i][v]
        if len(vp) < 5:
            continue
        lbl = (k == 0 and v == 1.0)
        ax.plot(np.arange(1, len(vp) + 1), vp, ".", ms=2.5, color=CB["raw"],
                alpha=0.6, label="re-anchored raw" if lbl else None)
        ax.plot(np.arange(1, len(vp) + 1), fit_sg_isotonic(vp), "-", lw=1.8,
                color=CB["sg"], label="SG+isotonic" if lbl else None)
        ax.plot(np.arange(1, len(vp) + 1), fit_cubic(vp), "--", lw=1.5,
                color=CB["cubic"], label="global cubic" if lbl else None)
    ax.set_title(f"sample {i}")
    ax.set_xlim(0, FULL_LEN)
axes[0, 0].legend(frameon=False)
fig.supxlabel("position [-]", fontweight="bold")
fig.supylabel("value [norm.]", fontweight="bold")
fig.tight_layout()
plt.show()
