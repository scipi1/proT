import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------- config
DATA_DIR = Path("../data/input/dyconex_260701_seq2seq")
FULL_LEN = 200            # points per variable in a complete test
FAIL_VALUE = 1.0          # idealized failure level (normalized)
DROP_ANY_NEGATIVE = True  # drop curves with any raw value < 0
SG_POLYORDER = 3          # local cubic
SG_MAX_WINDOW = 21        # cap on the Savitzky-Golay window (odd)

# Okabe-Ito colorblind palette
CB = {
    "raw": "#999999", "sg": "#0072B2", "cubic": "#E69F00",
    "iso": "#009E73", "fail": "#D55E00", "drop": "#CC79A7",
}
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "axes.titleweight": "bold", "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "figure.dpi": 110,
})

VAL_CH, POS_CH, VAR_CH = 3, 1, 2
VAR_IDS = (1.0, 2.0)
