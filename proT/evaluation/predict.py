import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import dirname, abspath, join
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from typing import Tuple, Callable
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
from tqdm import tqdm
import yaml

# Local imports
from proT.training.dataloader import ProcessDataModule
from proT.training.experiment_control import update_config
from proT.evaluation.predictors import (
    BasePredictor,
    PredictionResult,
    TransformerPredictor,
    BaselinePredictor
)


def create_predictor(
    config: dict,
    checkpoint_path: Path,
    datadir_path: Path = None
) -> BasePredictor:
    """
    Factory function that returns the appropriate predictor based on config.
    
    Similar to get_model_object() in trainer.py, this function automatically
    selects the correct predictor class based on the model type specified
    in the configuration.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        datadir_path: Path to data directory (optional)
        
    Returns:
        BasePredictor: Instance of appropriate predictor class
        
    Raises:
        ValueError: If model type is not recognized
    """
    model_obj = config["model"]["model_object"]
    
    # Registry maps model types to predictor classes
    PREDICTOR_REGISTRY = {
        # Transformer-based models
        "proT": TransformerPredictor,
        "proT_sim": TransformerPredictor,
        "proT_adaptive": TransformerPredictor,
        
        # Baseline models
        "LSTM": BaselinePredictor,
        "GRU": BaselinePredictor,
        "TCN": BaselinePredictor,
        "MLP": BaselinePredictor,
        "S6": BaselinePredictor,
    }
    
    predictor_class = PREDICTOR_REGISTRY.get(model_obj)
    if predictor_class is None:
        available = list(PREDICTOR_REGISTRY.keys())
        raise ValueError(f"Unknown model type: {model_obj}. Available: {available}")
    
    print(config["model"]["kwargs"].get("causal_mask", "No Causal mask found!"))
    # Initialize predictor - it handles everything internally
    return predictor_class(config, checkpoint_path, datadir_path)


def create_input_blanking_fn(
    beta: float,
    val_idx: int,
    seed: int = 42
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create an input blanking function using Bernoulli sampling.
    
    This function generates a conditioning function that randomly blanks tokens
    in the input sequence using Bernoulli(beta) sampling. Each token position
    is independently sampled, and if sampled as 1, the value feature at that
    position is set to NaN.
    
    Args:
        beta: Probability of blanking each token (0.0 to 1.0)
        val_idx: Index of the value feature to blank
        seed: Random seed for reproducibility
        
    Returns:
        Conditioning function with signature: fn(X: torch.Tensor) -> torch.Tensor
        
    Example:
        >>> blanking_fn = create_input_blanking_fn(beta=0.2, val_idx=0, seed=42)
        >>> X_blanked = blanking_fn(X)  # 20% of tokens will have val_idx set to NaN
    """
    generator = torch.Generator().manual_seed(seed)
    
    def blanking_fn(X: torch.Tensor) -> torch.Tensor:
        """
        Blank tokens in input tensor using Bernoulli(beta) sampling.
        
        Args:
            X: Input tensor (B x L x D)
            
        Returns:
            Blanked tensor with same shape
        """
        B, L, D = X.shape
        X_blanked = X.clone()
        
        # Sample Bernoulli(beta) for each position in the batch
        # Shape: B x L
        blank_mask = torch.bernoulli(
            torch.full((B, L), beta),
            generator=generator
        ).bool()
        
        # Blank the value feature where mask is True
        X_blanked[blank_mask, val_idx] = float('nan')
        
        return X_blanked
    
    return blanking_fn


def predict_test_from_ckpt(
    config: dict, 
    datadir_path: Path, 
    checkpoint_path: Path,
    external_dataset: dict = None,
    dataset_label: str = "test",
    cluster: bool = False,
    input_conditioning_fn: Callable[[torch.Tensor], torch.Tensor] = None
) -> PredictionResult:
    """
    Run standard prediction on specified dataset using a trained model checkpoint.
    
    This function:
    1. Creates appropriate predictor based on model type in config
    2. Loads the model from checkpoint
    3. Creates data module
    4. Runs prediction
    5. Returns PredictionResult object
    
    Args:
        config: Configuration dictionary
        datadir_path: Path to data directory
        checkpoint_path: Path to model checkpoint
        external_dataset: Optional dict with 'dataset', 'filename_input', 'filename_target'
        dataset_label: One of ["train", "test", "all"]
        cluster: Whether running on cluster
        
    Returns:
        PredictionResult: Object containing:
            - inputs: input data array
            - outputs: model predictions
            - targets: target data
            - attention_weights: dict with 'encoder', 'decoder', 'cross' keys (None for baseline)
            - metadata: dict with model info, dataset info, etc.
    """
    assert dataset_label in ["train", "test", "all"], \
        f"{dataset_label} is not a proper label!"
    
    # Set seed
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
        
    # Update config
    config_updated = update_config(config)
    
    # Create predictor (auto-selects correct type from config)
    predictor = create_predictor(config_updated, checkpoint_path, datadir_path)
    
    # Setup data module
    dm = predictor.create_data_module(
        external_dataset=external_dataset,
        cluster=cluster
    )
    
    # Run prediction
    results = predictor.predict(
        dm=dm,
        dataset_label=dataset_label,
        input_conditioning_fn=input_conditioning_fn
    )
    
    return results


def predict_test_from_ckpt_adaptive(
    config: dict, 
    datadir_path: Path, 
    checkpoint_path: Path,
    show_trg_max_idx: int,
    external_dataset: dict = None,
    dataset_label: str = "test",
    cluster: bool = False
) -> PredictionResult:
    """
    Run adaptive prediction with target revelation for transformer models.
    
    This function is specifically for models that support curriculum learning
    (OnlineTargetForecaster, TransformerForecaster with show_trg configuration).
    
    Args:
        config: Configuration dictionary
        datadir_path: Path to data directory
        checkpoint_path: Path to model checkpoint
        show_trg_max_idx: Max index to reveal in target sequence
        external_dataset: Optional dict with 'dataset', 'filename_input', 'filename_target'
        dataset_label: One of ["train", "test", "all"]
        cluster: Whether running on cluster
        
    Returns:
        PredictionResult: Object containing predictions and attention weights
        
    Raises:
        ValueError: If model doesn't support adaptive prediction
    """
    assert dataset_label in ["train", "test", "all"], \
        f"{dataset_label} is not a proper label!"
    
    # Set seed
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
        
    # Update config
    config_updated = update_config(config)
    
    # Create predictor (must be TransformerPredictor)
    predictor = create_predictor(config_updated, checkpoint_path, datadir_path)
    
    # Verify it's a transformer predictor
    if not isinstance(predictor, TransformerPredictor):
        raise ValueError(
            f"Adaptive prediction with show_trg_max_idx is only supported for "
            f"transformer models. Got {type(predictor).__name__}"
        )
    
    # Setup data module
    dm = predictor.create_data_module(
        external_dataset=external_dataset,
        cluster=cluster
    )
    
    # Run adaptive prediction
    results = predictor.predict(
        dm=dm,
        dataset_label=dataset_label,
        show_trg_max_idx=show_trg_max_idx
    )
    
    return results


def mk_quick_pred_plot(
    model: pl.LightningModule, 
    dm: pl.LightningDataModule, 
    val_idx: int, 
    save_dir: Path
):
    """
    Create quick prediction plots for visualization.
    
    Note: This function works with the old API for backward compatibility.
    It creates a temporary predictor-like wrapper around the model.
    
    Args:
        model: PyTorch Lightning model
        dm: Data module
        val_idx: Index of value feature to plot
        save_dir: Directory to save plots
    """
    # Use the old predict function for this utility
    input_array, output_array, target_array, cross_att_array, _, _ = predict(
        model=model, 
        dm=dm, 
        dataset_label="test"
    )
    
    # in case we have only one sample
    if len(output_array.shape) == 1:
        output_array = np.expand_dims(output_array, axis=0)
        target_array = np.expand_dims(target_array, axis=0)
        cross_att_array = np.expand_dims(cross_att_array, axis=0)
        input_array = np.expand_dims(input_array, axis=0)
    
    y_out = output_array[0, :]
    y_trg = target_array[0, :, val_idx]
    cross_att = cross_att_array[0]
    input_miss_bool = np.isnan(input_array[0, :, val_idx].squeeze())
    input_miss = input_miss_bool[np.newaxis, :].astype(int)
    
    N, M = cross_att.shape
    assert len(y_out) == len(y_trg)
    x = np.arange(len(y_out))

    fig, ax = plt.subplots()
    ax.plot(x, y_out)
    ax.plot(x, y_trg)
    fig.savefig(join(save_dir, "quick_pred_plot.png"))
    
    # Create figure and grid
    fig2 = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1], hspace=0.5)

    # Main heatmap axis
    ax0 = fig2.add_subplot(gs[0])
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    im0 = ax0.imshow(cross_att, cmap='viridis', aspect='auto', origin='upper')
    fig2.colorbar(im0, cax=cax0, label='Value')
    ax0.set_xticks([])
    ax0.set_ylabel("Rows")
    ax0.set_title("Heatmap with Boolean Mask")

    # Boolean mask axis
    ax1 = fig2.add_subplot(gs[1], sharex=ax0)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    im1 = ax1.imshow(input_miss, cmap='Greys', aspect='auto', origin='upper', vmin=0, vmax=1)
    fig2.colorbar(im1, cax=cax1, ticks=[0, 1], label='Missing')
    cax1.set_yticklabels(['False', 'True'])

    ax1.set_yticks([])
    ax1.set_xlabel("Columns")
    num_labels = min(M, 10)
    step = M // num_labels if M > 10 else 1

    ax1.set_xticks(np.arange(0, M, step))
    ax1.set_xticklabels(np.arange(0, M, step))
    
    fig2.savefig(join(save_dir, "cross_att.png"), dpi=300, bbox_inches='tight')


# Legacy function for backward compatibility
def predict(
    model: pl.LightningModule,
    dm: pl.LightningDataModule,
    dataset_label: str,
    debug_flag: bool = False,
    show_trg_max_idx: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    LEGACY FUNCTION: Makes prediction using the old API.
    
    This function is kept for backward compatibility with existing code.
    New code should use predict_test_from_ckpt() which returns a PredictionResult.
    
    Args:
        model: transformer model (assumed to be TransformerForecaster)
        dm: data module
        dataset_label: choose between ["test", "train", "all"]
        debug_flag: Predicts one batch and stops. Defaults to False.
        show_trg_max_idx: Max index to show target

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            input_array, output_array, target_array, cross_att_array, 
            enc_self_att_array, dec_self_att_array
    """
    assert dataset_label in ["train", "test", "all"], \
        AssertionError("Invalid dataset label!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set model for prediction
    forecaster = model.to(device)
    forecaster.eval()
    
    input_list, output_list, target_list = [], [], []
    cross_att_list, enc_self_att_list, dec_self_att_list = [], [], []
    
    # prepare data module
    dm.prepare_data()
    dm.setup(stage=None)
    
    # select dataset
    if dataset_label == "train":
        dataset = dm.train_dataloader()
        print("Train dataset selected.")
    elif dataset_label == "test":
        dataset = dm.test_dataloader()
        print("Test dataset selected (default).")
    elif dataset_label == "all":
        dataset = dm.all_dataloader()
        print("All data selected (default).")
        
    # loop over prediction batches
    print("Predicting...")
    for batch in tqdm(dataset):
        if isinstance(batch, (list, tuple)):
            batch = [item.to(device) for item in batch]
        else:
            batch = batch.to(device)

        X, trg = batch
        
        with torch.no_grad():
            forecast_output, _, (enc_self_att, dec_self_att, dec_cross_att), _, _ = forecaster.forward(
                data_input=X,
                data_trg=trg,
                show_trg_max_idx=show_trg_max_idx,
            )
            
        # append batch predictions
        input_list.append(X)
        output_list.append(forecast_output)
        target_list.append(trg)
        cross_att_list.append(dec_cross_att[0])
        enc_self_att_list.append(enc_self_att[0])
        dec_self_att_list.append(dec_self_att[0])

        if debug_flag:
            print("Debug mode: stopping after one batch...")
            break
    
    # detach predictions
    input_tensor = torch.cat([t.cpu().detach() for t in input_list], dim=0)
    output_tensor = torch.cat([t.cpu().detach() for t in output_list], dim=0)
    target_tensor = torch.cat([t.cpu().detach() for t in target_list], dim=0)
    cross_att_tensor = torch.cat([t.cpu().detach() for t in cross_att_list], dim=0)
    enc_self_att_tensor = torch.cat([t.cpu().detach() for t in enc_self_att_list], dim=0)
    dec_self_att_tensor = torch.cat([t.cpu().detach() for t in dec_self_att_list], dim=0)
    
    # convert predictions to numpy
    input_array = input_tensor.numpy().squeeze()
    output_array = output_tensor.numpy().squeeze()
    target_array = target_tensor.numpy().squeeze()
    cross_att_array = cross_att_tensor.numpy().squeeze()
    enc_self_att_array = enc_self_att_tensor.numpy().squeeze()
    dec_self_att_array = dec_self_att_tensor.numpy().squeeze()
    
    return input_array, output_array, target_array, cross_att_array, enc_self_att_array, dec_self_att_array


if __name__ == "__main__":
    datadir_path = r"../data/input"
    config_path = r"../experiments/training/proT/proT_cat_dyconex_optimized/config_proT_dyconex_v5_1.yaml"
    checkpoint_path = r"../experiments/training/proT/proT_cat_dyconex_optimized/k_0/checkpoints/epoch0-initial.ckpt"

    external_dataset = {
        "dataset": "ds_dx_pred_panel_MSI_01_01_2022-07_07_2025",
        "filename_input": "X.npy",
        "filename_target": "Y.npy",
    }

    config = OmegaConf.load(config_path)
    
    # Use new API
    results = predict_test_from_ckpt(
        config, 
        datadir_path, 
        checkpoint_path, 
        external_dataset,
        dataset_label="all",
        cluster=False
    )
    
    # Access results
    print(f"Input shape: {results.inputs.shape}")
    print(f"Output shape: {results.outputs.shape}")
    print(f"Target shape: {results.targets.shape}")
    print(f"Has attention: {results.attention_weights is not None}")
    print(f"Metadata: {results.metadata}")
