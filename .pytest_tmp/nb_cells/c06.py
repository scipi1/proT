# review: re-anchored curves (top) and dropped curves (bottom)
reanch = [(i, v) for i, c in enumerate(train_proc) for v in VAR_IDS
          if i not in train_drop and len(c[v][2]) != len(c[v][1])]
rng = np.random.default_rng(0)
sel_r = rng.choice(len(reanch), size=min(4, len(reanch)), replace=False)
drops = [(i, v) for i, r in train_drop.items() for v in r]
sel_d = range(min(2, len(drops)))
panels = [("reanch", reanch[k]) for k in sel_r] + [("drop", drops[k]) for k in sel_d]

fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.8))
for k, (kind, (i, v)) in enumerate(panels):
    pos, val, vp, reason = train_proc[i][v]
    ax = axes[k // 2, k % 2]
    ax.plot(pos, val, ".", ms=3, color=CB["raw"], label="raw")
    ax.axhline(0, color=CB["fail"], lw=0.8)
    if kind == "reanch":
        n_trim = len(val) - len(vp)
        ax.axvline(pos[n_trim], color=CB["sg"], lw=1, ls="--")
        ax.plot(np.arange(1, len(vp) + 1), vp, ".", ms=3, color=CB["sg"],
                label="re-anchored")
        ax.set_title(f"sample {i} var {v:.0f} - SALVAGED ({n_trim} pts trimmed)")
    else:
        ax.set_title(f"sample {i} var {v:.0f} - DROPPED ({reason})", color=CB["fail"])
    ax.set_xlim(0, FULL_LEN)
    if k == 0:
        ax.legend(frameon=False, loc="lower right")
for k in range(len(panels), 6):
    axes.flat[k].axis("off")
fig.supxlabel("position [-]", fontweight="bold")
fig.supylabel("value [norm.]", fontweight="bold")
fig.tight_layout()
plt.show()
