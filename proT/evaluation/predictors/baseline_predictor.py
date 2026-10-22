"""
Predictor for the encoder benchmark models (model_object "BENCH").
"""

from typing import Any, Dict
import torch

from .base_predictor import BasePredictor
from proT.baseline.baseline_pl_modules import BenchmarkForecaster


class BaselinePredictor(BasePredictor):
    """
    Predictor for the encoder benchmark (GRU, LSTM, TCN, MLP, proT_bench
    encoders). The forecast is the (B, n_out) scalar prediction; no
    attention weights are returned.
    """
    
    def _load_model(self) -> BenchmarkForecaster:
        """
        Load benchmark model from checkpoint.
        
        Returns:
            Loaded BenchmarkForecaster
        """
        model = BenchmarkForecaster.load_from_checkpoint(self.checkpoint_path)
        
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
        # Encoder benchmark: prediction depends on the input chain only
        return self.model.forward(X)                       # (B, n_out)
    
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
