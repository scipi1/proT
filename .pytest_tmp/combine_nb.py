"""Combine .pytest_tmp/nb_cells/* into notebooks/nb_ist_smoothing.ipynb and validate."""
import json
import os
from pathlib import Path

CELL_DIR = Path(r"c:\Users\ScipioneFrancesco\Documents\Projects\proT\.pytest_tmp\nb_cells")
OUT = Path(r"c:\Users\ScipioneFrancesco\Documents\Projects\proT\notebooks\nb_ist_smoothing.ipynb")

cells = []
for f in sorted(CELL_DIR.iterdir()):
    src = f.read_text(encoding="utf-8")
    if f.suffix == ".md":
        cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
    else:
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {}, "outputs": [], "source": src})

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")

# validate: JSON round-trip + syntax-compile every code cell
loaded = json.loads(OUT.read_text(encoding="utf-8"))
for j, c in enumerate(loaded["cells"]):
    if c["cell_type"] == "code":
        compile(c["source"], f"cell{j}", "exec")
print("notebook written and validated:", OUT, f"({len(cells)} cells)")
