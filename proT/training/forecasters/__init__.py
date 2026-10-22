"""
Lightning Module forecasters for different model architectures.
"""

from .transformer_forecaster import TransformerForecaster
from .simple_forecaster import SimpleForecaster
from .entropy_regularized_forecaster import EntropyRegularizedForecaster
from .simulator_forecaster import SimulatorForecaster
from .online_target_forecaster import OnlineTargetForecaster
from .opt_forecaster import OptimizationForecaster

__all__ = [
    'TransformerForecaster',
    'SimpleForecaster',
    'EntropyRegularizedForecaster',
    'SimulatorForecaster',
    'OnlineTargetForecaster',
    'OptimizationForecaster',
]
