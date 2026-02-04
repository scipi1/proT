# proT - Transformer for Manufacturing Modeling

## Overview

proT is a specialized transformer-based model for manufacturing modeling, particularly designed for process data. It implements a modified transformer architecture based on the Spacetimeformer (Grigsby et al., 2023) with custom embedding layers, attention mechanisms, and encoder-decoder structures tailored for sequential process data analysis and prediction.

The model is capable of handling missing values in time series data and provides interpretable attention mechanisms to understand the relationships between different parts of the process sequence.

## Architecture

### Core Components

#### Embedding System
- **ModularEmbedding**: Flexible embedding system that supports multiple embedding types:
  - **Time2Vec**: Temporal embeddings based on the Time2Vec paper (Kazemi et al., 2019)
  - **SinusoidalPosition**: Positional embeddings as used in the original Transformer
  - **nn_embedding**: Standard embedding lookup tables
  - **identity_emb**: Identity embeddings for direct value representation
  - **linear_emb**: Linear transformation embeddings

#### Attention Mechanism
- **ScaledDotAttention**: Implementation of the scaled dot-product attention
- **AttentionLayer**: Wrapper for attention with projection layers
- Support for causal masking and missing value handling

#### Encoder-Decoder Structure
- **Encoder**: Processes input sequences with self-attention
- **Decoder**: Processes target sequences with self-attention and cross-attention to encoder outputs
- Pre-norm transformer architecture with residual connections

### Model Flow
1. Input data is embedded using the modular embedding system
2. Encoder processes the embedded input
3. Decoder processes the target sequence while attending to encoder outputs
4. Final linear layers produce forecasting outputs

## Project Structure

```
proT/
├── config/                 # Configuration files
├── data/                   # Data directory (gitignored)
├── docs/                   # Documentation
├── experiments/            # Experiment results (gitignored)
├── notebooks/              # Jupyter notebooks for analysis
├── proT/                   # Main source code
│   ├── core/               # Core model architecture
│   │   ├── model.py        # Main ProT model
│   │   └── modules/        # Transformer components
│   │       ├── attention.py    # Attention mechanisms
│   │       ├── decoder.py      # Decoder implementation
│   │       ├── embedding.py    # Embedding system
│   │       ├── embedding_layers.py # Base embedding layers
│   │       ├── encoder.py      # Encoder implementation
│   │       ├── extra_layers.py # Additional utility layers
│   │       └── utils.py        # Utility functions
│   ├── training/           # Training infrastructure
│   │   ├── trainer.py      # Training orchestration
│   │   ├── dataloader.py   # Data loading utilities
│   │   ├── experiment_control.py # Experiment management
│   │   ├── forecasters/    # Lightning module wrappers
│   │   └── callbacks/      # Training callbacks
│   ├── evaluation/         # Prediction & evaluation
│   │   ├── predict.py      # Prediction utilities
│   │   └── predictors/     # Predictor classes
│   ├── euler_optuna/       # Optuna hyperparameter optimization
│   ├── euler_sweep/        # Parameter sweep framework
│   ├── baseline/           # Baseline models (RNN, TCN, MLP, S6)
│   ├── proj_specific/      # Project-specific code
│   │   ├── GSA/            # Global sensitivity analysis
│   │   ├── simulator/      # Trajectory simulator
│   │   └── subroutines/    # Evaluation subroutines
│   ├── utils/              # Shared utilities
│   ├── cli.py              # Command-line interface
│   └── labels.py           # Constants
├── scripts/                # Utility scripts for cluster execution
├── test/                   # Unit and integration tests
├── trained_models/         # Pre-trained models (download separately)
└── tutorials/              # Usage tutorials
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- PyTorch Lightning
- Other dependencies listed in requirements.txt

### Setting Up the Environment

1. Clone the repository:
```bash
git clone https://github.com/scipi1/proT.git
cd proT
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- macOS/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Command Line Interface

The project provides a command-line interface for training and prediction:

#### Training

```bash
python -m proT.cli train --exp_id <experiment_id> [--debug] [--cluster] [--resume_checkpoint <checkpoint_path>]
```

Options:
- `--exp_id`: Experiment folder containing the config file
- `--debug`: Enable debug mode (default: False)
- `--cluster`: Enable cluster mode for distributed training (default: False)
- `--resume_checkpoint`: Resume training from a checkpoint
- `--plot_pred_check`: Generate prediction plots after training (default: True)
- `--sweep_mode`: Sweep mode, either 'independent' or 'combination' (default: 'combination')

#### Prediction

```bash
python -m proT.cli predict --exp_id <experiment_id> --out_id <output_id> [--checkpoint <checkpoint_path>]
```

Options:
- `--exp_id`: Path to experiment from experiment/training
- `--out_id`: Path to output from experiment/evaluations
- `--checkpoint`: Checkpoint path from exp_id path
- `--cluster`: Enable cluster mode (default: False)
- `--debug`: Enable debug mode (default: False)

### Configuration

The model is configured using YAML files. A typical configuration includes:

```yaml
data:
  dataset: "your_dataset_name"
  filename_input: "X.npy"
  filename_target: "Y.npy"
  val_idx: 0  # Index of the value to predict

model:
  model_object: "proT"  # Options: proT, proT_sim, proT_adaptive
  
  # Embedding configuration
  ds_embed_enc: {...}
  ds_embed_dec: {...}
  comps_embed_enc: "spatiotemporal"
  comps_embed_dec: "spatiotemporal"
  
  # Attention configuration
  enc_attention_type: "ScaledDotProduct"
  dec_self_attention_type: "ScaledDotProduct"
  dec_cross_attention_type: "ScaledDotProduct"
  n_heads: 4
  causal_mask: true
  
  # Architecture configuration
  e_layers: 3
  d_layers: 3
  d_model_enc: 128
  d_model_dec: 128
  d_ff: 256
  d_qk: 32
  activation: "gelu"
  norm: "layer"
  out_dim: 1
  
  # Dropout configuration
  dropout_emb: 0.1
  dropout_attn_out: 0.1
  dropout_ff: 0.1

training:
  batch_size: 32
  max_epochs: 100
  loss_fn: "mse"
  lr: 0.0001
```

## Examples

### Basic Training Example

1. Prepare your data in NumPy format (X.npy for input, Y.npy for target)
2. Create a configuration file in the experiments/training/<exp_id> directory
3. Run the training command:

```bash
python -m proT.cli train --exp_id your_experiment_id
```

### Prediction Example

After training, you can generate predictions using:

```bash
python -m proT.cli predict --exp_id your_experiment_id --out_id your_output_id
```

### Using the Model in Code

```python
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from proT.core import ProT
from proT.training.forecasters import SimpleForecaster
from proT.training import ProcessDataModule

# Load configuration
config = OmegaConf.load("path/to/config.yaml")

# Create data module
data_module = ProcessDataModule(
    data_dir="path/to/data",
    input_file="X.npy",
    target_file="Y.npy",
    batch_size=32,
    num_workers=4,
    data_format="float32"
)

# Create model
model = SimpleForecaster(config)

# Train model (using PyTorch Lightning)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)

# Make predictions
trainer.predict(model, data_module)
```

### Using Pre-trained Models

Pre-trained models can be downloaded from [Polybox](https://polybox.ethz.ch/index.php/s/R9xCXwfFbZYs5tq). See `trained_models/README.md` for setup instructions.

```python
from proT.evaluation import predict_test_from_ckpt
from omegaconf import OmegaConf

# Load config and run prediction
config = OmegaConf.load("trained_models/proT_dyconex_sum_200/config.yaml")
results = predict_test_from_ckpt(
    config=config,
    datadir_path="data/input",
    checkpoint_path="trained_models/proT_dyconex_sum_200/k_0/checkpoints/best_checkpoint.ckpt",
    dataset_label="test"
)

# Access results
print(f"Predictions shape: {results.outputs.shape}")
print(f"Targets shape: {results.targets.shape}")
```

## Supported Model Types

| Model | Description | Forecaster |
|-------|-------------|------------|
| `proT` | Standard transformer | SimpleForecaster |
| `proT_sim` | Physics-informed (PINN) | SimulatorForecaster |
| `proT_adaptive` | Curriculum learning | OnlineTargetForecaster |
| `LSTM` | Long Short-Term Memory | BaselineForecaster |
| `GRU` | Gated Recurrent Unit | BaselineForecaster |
| `TCN` | Temporal Convolutional Network | BaselineForecaster |
| `MLP` | Multi-Layer Perceptron | BaselineForecaster |

## Documentation

Detailed documentation is available in the following locations:
- `proT/training/forecasters/FORECASTERS_GUIDE.md` - Guide to forecaster modules
- `proT/euler_optuna/README_PARALLEL.md` - Parallel hyperparameter optimization
- `proT/euler_sweep/README.md` - Parameter sweep framework
- `proT/evaluation/README.md` - Prediction system documentation
- `test/README.md` - Testing guide
- `tutorials/tutorial_external_prediction.ipynb` - External prediction tutorial

## Testing

Run the test suite:

```bash
# All tests
pytest test/ -v

# Unit tests only (fast)
pytest test/unit/ -v

# Integration tests only
pytest test/integration/ -v
```

## References

- Vaswani, A., et al. (2017). "Attention is all you need." Advances in neural information processing systems.
- Kazemi, S. M., et al. (2019). "Time2Vec: Learning a Vector Representation of Time." arXiv preprint arXiv:1907.05321.
- Grigsby, J., et al. (2023). "Spacetimeformer: High-dimensional time series forecasting with self-attention." arXiv preprint.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

[Add citation information for the paper here]
