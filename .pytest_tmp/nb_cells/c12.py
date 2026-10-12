# review panel: failed curves - zoom on the Bezier approach to 1.0
failed_list = [(i, v) for (i, v), s in status_map.items() if s == "failed"]
failed_list.sort(key=lambda t: train_proc[t[0]][t[1]][2][-1])  # lowest enders first
n_show = min(6, len(failed_list))
fig, axes = plt.subplots((n_show + 1) // 2, 2, figsize=(7.2, 2.8 * ((n_show + 1) // 2)))
axes = np.atleast_2d(axes)
ZOOM = 30
for k in range(n_show):
    i, v = failed_list[k]
    entry = train_proc[i][v]
    pos, val_raw, val, _ = entry
    sm, full = idealize(entry, "failed")
    n = len(val)
    ax = axes[k // 2, k % 2]
    lo = max(0, n - ZOOM)
    ax.plot(np.arange(lo + 1, n + 1), val[lo:], ".", ms=4, color=CB["raw"], label="raw")
    ax.plot(np.arange(lo + 1, n + 1), fit_sg_isotonic(val)[lo:], "-", lw=1.5,
            color=CB["sg"], label="SG+isotonic")
    ax.plot(np.arange(lo + 1, n + 1), sm[lo:], "-", lw=2.0,
            color=CB["fail"], label="Bezier to 1.0")
    ax.axhline(FAIL_VALUE, color=CB["fail"], lw=0.8, ls=":")
    ax.axvspan(n - BEZIER_K + 1, n, color=CB["fail"], alpha=0.08)
    low = val[-1] < 0.8
    ax.set_title(f"sample {i} var {v:.0f} (last raw={val[-1]:.2f})"
                 + ("  LOW-ENDER" if low else ""),
                 color=CB["fail"] if low else "black")
    ax.set_xlim(lo + 1, n)
    if k == 0:
        ax.legend(frameon=False, loc="lower right")
for k in range(n_show, axes.size):
    axes.flat[k].axis("off")
fig.supxlabel("position [-]", fontweight="bold")
fig.supylabel("value [norm.]", fontweight="bold")
fig.tight_layout()
plt.show()
print(f"failed curves: {len(failed_list)}; shaded band = Bezier window (last {BEZIER_K} pts)")
