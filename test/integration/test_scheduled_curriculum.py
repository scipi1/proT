"""
Integration test for the scheduled curriculum mode of OnlineTargetForecaster.

Runs a tiny training with target_show_mode="scheduled" and verifies that the
logged forcing threshold (show_trg_max_idx) follows the expected annealing
schedule and that training completes end-to-end.
"""

import os
import shutil
import tempfile

import pandas as pd
import pytest
from omegaconf import OmegaConf

from proT.training.trainer import trainer
from proT.training.experiment_control import update_config

from .utils import modify_config_for_fast_testing, DATA_DIR, ROOT_DIR

CONFIG_PATH = (
    ROOT_DIR / "experiments" / "training" / "tests" / "protocol"
    / "test_forecaster_proT_adaptive_random" / "config_proT_dyconex_v5_2.yaml"
)

MAX_EPOCHS = 4
EPOCH_SHOW_TRG = 1
START_IDX = 6


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs, cleanup after test."""
    temp_dir = tempfile.mkdtemp(prefix="sched_curriculum_test_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_scheduled_curriculum_run(temp_output_dir):
    """Scheduled mode runs and the logged threshold follows the schedule."""
    config = OmegaConf.load(CONFIG_PATH)
    config_dev = modify_config_for_fast_testing(
        config, max_epochs=MAX_EPOCHS, k_fold=2, max_data_size=10
    )

    # Override curriculum settings: deterministic scheduled mode
    config_dev["training"]["target_show_mode"] = "scheduled"
    config_dev["training"]["epoch_show_trg"] = EPOCH_SHOW_TRG
    config_dev["training"]["show_trg_max_idx_start"] = START_IDX
    config_dev["training"]["show_trg_max_idx_end"] = 0
    config_dev["training"]["show_trg_schedule"] = "linear"
    config_dev["training"]["show_trg_schedule_end_frac"] = 1.0
    config_dev["training"]["show_trg_random_draw"] = False

    config_updated = update_config(config_dev)

    result_df = trainer(
        config=config_updated,
        data_dir=str(DATA_DIR),
        save_dir=temp_output_dir,
        cluster=False,
        experiment_tag="test_scheduled_curriculum",
        plot_pred_check=False,
        debug=False,
        best=True,
    )

    assert len(result_df) == 2, "Both folds should complete"

    # Check the logged forcing threshold in fold 0.
    # end_ep = min(3, round(1.0 * 4)) = 3, so expected bounds per epoch:
    # ep1 (curriculum start): 6, ep2: 3, ep3: 0
    csv_path = os.path.join(
        temp_output_dir, "k_0", "logs", "csv", "version_0", "metrics.csv"
    )
    csv = pd.read_csv(csv_path)
    assert "show_trg_max_idx" in csv.columns, "threshold should be logged"

    logged = csv.dropna(subset=["show_trg_max_idx"])[["epoch", "show_trg_max_idx"]]
    logged = logged.groupby("epoch").last()["show_trg_max_idx"]

    assert list(logged.index) == [1, 2, 3], f"unexpected epochs: {list(logged.index)}"
    assert logged.loc[1] == START_IDX
    assert logged.loc[2] == pytest.approx(3, abs=1)
    assert logged.loc[3] == 0


def test_censored_gaussian_loss_run(temp_output_dir):
    """loss_fn="gaussian_censored" trains end-to-end with finite losses."""
    config = OmegaConf.load(CONFIG_PATH)
    config_dev = modify_config_for_fast_testing(
        config, max_epochs=2, k_fold=2, max_data_size=10
    )

    # Censored Gaussian NLL (Tobit): channels are (mu, raw_sigma)
    config_dev["training"]["loss_fn"] = "gaussian_censored"
    config_dev["training"]["loss_censor_threshold"] = 1.0
    config_dev["training"]["loss_bce_on_observed"] = False

    config_updated = update_config(config_dev)

    result_df = trainer(
        config=config_updated,
        data_dir=str(DATA_DIR),
        save_dir=temp_output_dir,
        cluster=False,
        experiment_tag="test_censored_gaussian",
        plot_pred_check=False,
        debug=False,
        best=True,
    )

    assert len(result_df) == 2, "Both folds should complete"
    for col in ("train_loss", "val_loss", "val_mae"):
        assert col in result_df.columns
        assert result_df[col].map(lambda v: v == v and abs(v) != float("inf")).all(), \
            f"{col} must be finite: {result_df[col].tolist()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
