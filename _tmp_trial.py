import optuna, tempfile
from pathlib import Path
from omegaconf import OmegaConf
from proT.euler_optuna.cli import sample_params_for_optuna, train_function_for_optuna, get_metrics_for_optuna
cfg = OmegaConf.load('experiments/benchmark/bench_proT_bench_dyconex/config_bench_proT_bench_dyconex.yaml')
OmegaConf.set_struct(cfg, False)
trial = optuna.trial.FixedTrial({'e_layers': 1, 'n_heads': 2, 'd_ff': 200, 'd_qk': 100, 'dropout': 0.1, 'lr': 5e-4})
for k, v in sample_params_for_optuna(trial, cfg).items():
    OmegaConf.update(cfg, k, v, merge=False)
cfg.training.max_epochs = 1; cfg.training.k_fold = 2
out = Path(tempfile.mkdtemp())
df = train_function_for_optuna(cfg, save_dir=out, data_dir=Path('data/input'), experiment_tag='smoke', cluster=True)
print(get_metrics_for_optuna(df))
