# Trained Models

This directory contains pre-trained proT models for use in predictions.

## Download

Download the trained models from [Polybox](https://polybox.ethz.ch/index.php/s/R9xCXwfFbZYs5tq).

## Setup

1. Download the model folder(s) from the link above
2. Extract and place them in this `trained_models/` directory
3. Ensure the folder structure matches:
   ```
   trained_models/
   ├── README.md
   └── <model_name>/
       ├── config.yaml
       ├── kfold_summary.json
       └── k_*/
           └── checkpoints/
               └── best_checkpoint.ckpt
   ```

## Available Models

| Model | Description |
|-------|-------------|
| `proT_dyconex_sum_200` | Dyconex data with sum fusion (200 epochs) |
