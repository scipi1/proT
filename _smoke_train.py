"""Short local smoke run of the fail-stop/monotone config (2 epochs, 2 folds)."""
import sys, shutil
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from proT.training.experiment_control import update_config
from proT.training.trainer import trainer

import os
SRC = Path(os.environ.get("SMOKE_SRC",
    "experiments/training/RNN/proTEncGRUpervar_LogNCensSWHM_dyconex_2000ep_warmup_90_to_0"))
OUT = Path("experiments/training/tests/smoke_failstop_tmp")
OUT.mkdir(parents=True, exist_ok=True)

config = OmegaConf.load(next(SRC.glob("config*.yaml")))
config.training.max_epochs = 2
config.training.k_fold = 2
config.training.num_workers = 0
config.training.save_ckpt_every_n_epochs = 100000  # no periodic ckpts
config.training.warmup_epoch = 1                   # warmup path, scaled to 2 epochs
config.data.max_data_size = None                   # full data; few epochs only
cfg = update_config(config)
print("fail_mode keys:", cfg.training.loss_expected_len, cfg.training.loss_fail_margin,
      cfg.training.loss_fail_anchor, "| model:", cfg.model.kwargs.rnn_monotone,
      cfg.model.kwargs.rnn_absorbing_fail)

df = trainer(config=cfg, data_dir="data/input/", save_dir=str(OUT), cluster=False,
             experiment_tag="smoke_failstop", plot_pred_check=False, debug=False)
print(df)
print("SMOKE TRAIN OK")
