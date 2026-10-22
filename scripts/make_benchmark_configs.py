"""
Generate the encoder benchmark experiment folders.

For every dataset base config (config/benchmark/base_<ds>.yaml) and every
encoder (config/benchmark/encoders.yaml), writes
    experiments/benchmark/bench_<encoder>_<ds>/config_bench_<encoder>_<ds>.yaml
The embedding and head blocks are copied verbatim from the base, so they
are identical across encoders by construction.

Usage:
    python scripts/make_benchmark_configs.py
"""

from os.path import abspath, dirname, join
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = dirname(dirname(abspath(__file__)))
CFG_DIR = join(ROOT_DIR, "config", "benchmark")
OUT_DIR = join(ROOT_DIR, "experiments", "benchmark")
DATASETS = ["dyconex", "ishigami"]


def main():
    encoders = OmegaConf.load(join(CFG_DIR, "encoders.yaml"))
    for ds in DATASETS:
        base = OmegaConf.load(join(CFG_DIR, f"base_{ds}.yaml"))
        for name, enc in encoders.items():
            cfg = base.copy()
            cfg.model.kwargs.encoder = enc
            exp_id = f"bench_{name}_{ds}"
            out = Path(OUT_DIR) / exp_id
            out.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, out / f"config_{exp_id}.yaml")
            print(f"written {out}")


if __name__ == "__main__":
    main()
