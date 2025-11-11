"""
Predictor for baseline models (RNN, TCN, MLP, S6).
"""

from typing import Any, Dict
import torch
from pathlib import Path

from .base_predictor import BasePredictor
from proT.baseline.baseline_pl_modules import RNNForecaster


class BaselinePredictor(BasePredictor):
    """
    Predictor for baseline forecasting models.
    
    Handles:
    - LSTM
    - GRU
    - TCN
    - MLP
    - S6
    
    These models don't produce attention weights, so attention_weights will be None.
    """
    
    def _load_model(self) -> RNNForecaster:
        """
        Load baseline model from checkpoint.
        
        Returns:
            Loaded baseline forecaster model
        """
        # All baseline models use RNNForecaster wrapper
        model = RNNForecaster.load_from_checkpoint(self.checkpoint_path)
        
        # Verify model loaded correctly
        if model is None:
            raise RuntimeError("Model failed to load from checkpoint.")
        
        if not any(param.requires_grad for param in model.parameters()):
            raise RuntimeError("Model parameters seem uninitialized. Check the checkpoint path.")
        
        return model
    
    def _forward(self, X: torch.Tensor, trg: torch.Tensor, **kwargs) -> Any:
        """
        Perform forward pass through baseline model.
        
        Args:
            X: Input tensor (B x L x F)
            trg: Target tensor (B x L x F)
            **kwargs: Additional arguments (ignored for baseline models)
            
        Returns:
            Prediction tensor
        """
        # Extract value index from config
        val_idx = self.config["data"]["val_idx"]
        
        # Prepare masked target (set value feature to zero)
        masked_target = trg.clone()
        masked_target[:, :, val_idx] = 0
        
        # Call model forward
        output = self.model.forward(x=X, y=masked_target)
        
        return output
    
    def _process_forward_output(self, output: Any) -> Dict[str, Any]:
        """
        Process baseline model output.
        
        Args:
            output: Prediction tensor from model.forward()
        
        Returns:
            Dictionary with:
                - 'forecast': Prediction tensor
                - 'attention_weights': None (baseline models don't have attention)
        """
        return {
            'forecast': output,
            'attention_weights': None
        }
