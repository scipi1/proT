"""
Simulator-based Forecaster for Physics-Informed Neural Network (PINN) training.
Extends SimpleForecaster with trajectory simulation capabilities.
"""

from typing import Any
import torch

from proT.training.forecasters.simple_forecaster import SimpleForecaster
from proT.proj_specific.simulator.trajectory_simulator import ISTSimulator


class SimulatorForecaster(SimpleForecaster):
    """
    Physics-informed forecaster that uses a trajectory simulator.
    
    This forecaster integrates a physics-based simulator to:
    1. Generate decoder inputs based on physical models
    2. Post-process model outputs through trajectory simulation
    
    Useful for problems where you have known physics/dynamics that should
    constrain or guide the neural network predictions.
    
    Args:
        config: Configuration dictionary containing:
            - All parameters from SimpleForecaster
            - training.pinn: Boolean to enable PINN mode
            - training.simulator_model: Model name for ISTSimulator (default: "F")
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize simulator components
        simulator_model = config["training"].get("simulator_model", "D")
        simulator = ISTSimulator(model=simulator_model)
        
        self.decoder_input_module = simulator.get_decoder_input
        self.trajectory_simulator = simulator.forward
        
        print(f"✓ SimulatorForecaster initialized with model: {simulator_model}")
    
    def forward(self, data_input: torch.Tensor, data_trg: torch.Tensor) -> Any:
        """
        Forward pass with simulator integration.
        
        Args:
            data_input: Encoder input tensor (B, L_enc, D_enc)
            data_trg: Decoder target tensor (B, L_dec, D_dec)
            
        Returns:
            Tuple of (forecast_output, recon_output, attention_weights, masks, entropy)
        """
        # Generate decoder input using simulator
        dec_input = self.decoder_input_module(
            batch_size=data_trg.size(0),
            device=data_input.device
        )
        
        # Forward pass through model
        model_output, recon_output, attn_weights, enc_mask, entropy = self.model.forward(
            input_tensor=data_input,
            target_tensor=dec_input,
            trg_pos_mask=None
        )
        
        # Post-process through trajectory simulator
        forecast_output = self.trajectory_simulator(model_output)
        
        return forecast_output, recon_output, attn_weights, enc_mask, entropy
