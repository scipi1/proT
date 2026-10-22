"""Temporary in-situ test: a few training steps of SimulatorForecaster with softplus on GPU."""
import sys, os
sys.path.append(os.path.abspath("./"))
import numpy as np
import torch
from omegaconf import OmegaConf
from proT.training.experiment_control import update_config
from proT.training.forecasters.simulator_forecaster import SimulatorForecaster

config = OmegaConf.load("experiments/training/proTPhySim_dyconex_2000ep/config_proTPhySim_dyconex_2000ep.yaml")
config = update_config(config)

forecaster = SimulatorForecaster(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
forecaster.to(device)
forecaster.train()

npz = np.load("notebooks/output/case_study_sim.npz")
X = npz["inputs"][:16]
T = npz["targets"][:16]

opt = torch.optim.AdamW(forecaster.parameters(), lr=1e-4)
for step in range(3):
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    tb = torch.tensor(T, dtype=torch.float32, device=device)
    forecast, _, _, _, _ = forecaster.forward(xb, tb)
    # forecast: (B, L, 2) simulator curves; dummy loss against zeros just to exercise backward
    loss = ((forecast - 0.0) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"step {step}: loss={loss.item():.6f}, forecast shape={tuple(forecast.shape)}")

print("SIM_TRAIN_TEST_OK")
