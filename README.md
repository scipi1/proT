# ProChain Transformer

## Overview
ProChain Transformer is a specialized transformer-based model for time series forecasting, particularly designed for process data. It implements a modified transformer architecture based on the Spacetimeformer (Grigsby et al., 2023) with custom embedding layers, attention mechanisms, and encoder-decoder structures tailored for sequential process data analysis and prediction.

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
prochain_transformer/
├── config/                 # Configuration files
├── data/                   # Data directory
│   ├── input/              # Input data
│   └── output/             # Output data
├── docs/                   # Documentation
├── experiments/            # Experiment results
├── notebooks/              # Jupyter notebooks for analysis
├── prochain_transformer/   # Main source code
│   ├── modules/            # Core model components
│   │   ├── attention.py    # Attention mechanisms
│   │   ├── decoder.py      # Decoder implementation
│   │   ├── embedding.py    # Embedding system
│   │   ├── embedding_layers.py # Base embedding layers
│   │   ├── encoder.py      # Encoder implementation
│   │   ├── extra_layers.py # Additional utility layers
│   │   └── utils.py        # Utility functions
│   ├── subroutines/        # Task-specific subroutines
│   ├── callbacks.py        # Training callbacks
│   ├── cli.py              # Command-line interface
│   ├── dataloader.py       # Data loading utilities
│   ├── experiment_control.py # Experiment management
│   ├── forecaster.py       # Lightning module wrapper
│   ├── kfold_train.py      # K-fold cross-validation
│   ├── model.py            # Main model definition
│   ├── predict.py          # Prediction utilities
│   └── train.py            # Training utilities
├── scripts/                # Utility scripts
└── test/                   # Unit tests
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
git clone https://github.com/yourusername/prochain_transformer.git
cd prochain_transformer
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
```

## Usage

### Command Line Interface

The project provides a command-line interface for training and prediction:

#### Training

```bash
python -m prochain_transformer.cli train --exp_id <experiment_id> [--debug] [--cluster] [--resume_checkpoint <checkpoint_path>]
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
python -m prochain_transformer.cli predict --exp_id <experiment_id> --out_id <output_id> [--checkpoint <checkpoint_path>]
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
  filename_input: "X_np.npy"
  filename_target: "Y_np.npy"
  val_idx: 0  # Index of the value to predict

model:
  # Embedding configuration
  ds_embed_enc: {...}
  ds_embed_dec: {...}
  comps_embed_enc: "spatiotemporal"
  comps_embed_dec: "spatiotemporal"
  
  # Attention configuration
  enc_attention_type: "ScaledDotProduct"
  dec_self_attention_type: "ScaledDotProduct"
  dec_cross_attention_type: "ScaledDotProduct"
  enc_mask_type: "Uniform"
  dec_self_mask_type: "Uniform"
  dec_cross_mask_type: "Uniform"
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
  use_final_norm: true
  out_dim: 1
  
  # Dropout configuration
  dropout_emb: 0.1
  dropout_data: 0.0
  dropout_attn_out: 0.1
  dropout_ff: 0.1
  enc_dropout_qkv: 0.1
  enc_attention_dropout: 0.1
  dec_self_dropout_qkv: 0.1
  dec_self_attention_dropout: 0.1
  dec_cross_dropout_qkv: 0.1
  dec_cross_attention_dropout: 0.1

training:
  batch_size: 32
  max_epochs: 100
  loss_fn: "mse"
  base_lr: 0.0001
  emb_lr: 0.001
  emb_start_lr: 0.01
  optimization: 2
  switch_epoch: 50
  switch_step: 50
  warmup_steps: 1000
```

## Examples

### Basic Training Example

1. Prepare your data in NumPy format (X_np.npy for input, Y_np.npy for target)
2. Create a configuration file in the experiments/training/<exp_id> directory
3. Run the training command:

```bash
python -m prochain_transformer.cli train --exp_id your_experiment_id
```

### Prediction Example

After training, you can generate predictions using:

```bash
python -m prochain_transformer.cli predict --exp_id your_experiment_id --out_id your_output_id
```

### Using the Model in Code

```python
import torch
import pytorch_lightning as pl
from prochain_transformer.model import Spacetimeformer
from prochain_transformer.forecaster import TransformerForecaster
from prochain_transformer.dataloader import ProcessDataModule

# Load configuration
config = {...}  # Your model configuration

# Create data module
data_module = ProcessDataModule(
    data_dir="path/to/data",
    input_file="X_np.npy",
    target_file="Y_np.npy",
    batch_size=32,
    num_workers=4,
    data_format="float32"
)

# Create model
model = TransformerForecaster(config)

# Train model (using PyTorch Lightning)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)

# Make predictions
trainer.predict(model, data_module)
```

## References

- Vaswani, A., et al. (2017). "Attention is all you need." Advances in neural information processing systems.
- Kazemi, S. M., et al. (2019). "Time2Vec: Learning a Vector Representation of Time." arXiv preprint arXiv:1907.05321.
- Grigsby, J., et al. (2023). "Spacetimeformer: High-dimensional time series forecasting with self-attention." arXiv preprint.
