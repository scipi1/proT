"""Short local smoke runs of the two smooth-delta arms (2 epochs, 2 folds)."""
import sys
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from proT.training.experiment_control import update_config
from proT.training.trainer import trainer

ARMS = [
    "experiments/training/RNN/proTEncGRUpervar_smooth_delta_warmup",
    "experiments/training/RNN/proTEncGRUpervar_smooth_delta_nocurr",
]

for src in ARMS:
    name = Path(src).name
    out = Path(f"experiments/training/tests/smoke_{name}")
    out.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(next(Path(src).glob("config*.yaml")))
    config.training.max_epochs = 2
    config.training.k_fold = 2
    config.training.num_workers = 0
    config.training.save_ckpt_every_n_epochs = 100000  # no periodic ckpts
    config.data.max_data_size = None
    cfg = update_config(config)
    print(f"[{name}] dataset={cfg.data.dataset}, "
          f"curriculum={cfg.training.get('target_show_mode')}, "
          f"delta_max={cfg.model.kwargs.rnn_delta_max}")

    df = trainer(config=cfg, data_dir="data/input/", save_dir=str(out),
                 cluster=False, experiment_tag=f"smoke_{name}",
                 plot_pred_check=False, debug=False)
    print(df)
    print(f"SMOKE TRAIN OK: {name}")
