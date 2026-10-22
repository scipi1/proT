"""
Integration smoke test for experiment A on the dyconex_260701_seq2seq dataset.

Runs the actual experiment config (scheduled curriculum + group mask +
learnable decoder positions + censored Gaussian loss + design/panel_position
embeddings) for 2 epochs x 2 folds on a data subset, verifying the full
wiring against the real data layout.
"""

import os
import shutil
import tempfile

import pytest
from omegaconf import OmegaConf

from proT.training.trainer import trainer
from proT.training.experiment_control import update_config

from .utils import modify_config_for_fast_testing, DATA_DIR, ROOT_DIR

CONFIG_PATH = (
    ROOT_DIR / "experiments" / "training" / "proTCVSched_dyconex_2000ep"
    / "config_proTCVSched_dyconex_2000ep.yaml"
)

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "dyconex_260701_seq2seq").exists(),
    reason="dyconex_260701_seq2seq dataset not available",
)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs, cleanup after test."""
    temp_dir = tempfile.mkdtemp(prefix="new_ds_wiring_test_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_experiment_a_wiring_on_new_dataset(temp_output_dir):
    """Experiment A config trains end-to-end on dyconex_260701_seq2seq."""
    config = OmegaConf.load(CONFIG_PATH)
    config_dev = modify_config_for_fast_testing(
        config, max_epochs=2, k_fold=2, max_data_size=10
    )

    # Shrink the model for the smoke test: wiring validation does not need the
    # full-size model (avoids memory exhaustion during checkpoint serialization)
    config_dev["experiment"]["d_model_set"] = 110
    config_dev["experiment"]["e_layers"] = 1
    config_dev["experiment"]["d_layers"] = 1
    config_dev["experiment"]["n_heads"] = 1
    config_dev["experiment"]["d_ff"] = 200
    config_dev["experiment"]["d_qk"] = 200

    config_updated = update_config(config_dev)

    result_df = trainer(
        config=config_updated,
        data_dir=str(DATA_DIR),
        save_dir=temp_output_dir,
        cluster=False,
        experiment_tag="test_new_ds_wiring",
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
