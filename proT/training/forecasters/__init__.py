"""
Lightning Module forecasters for different model architectures.
"""

from .transformer_forecaster import TransformerForecaster
from .simple_forecaster import SimpleForecaster
from .simulator_forecaster import SimulatorForecaster
from .online_target_forecaster import OnlineTargetForecaster

__all__ = [
    'TransformerForecaster',
    'SimpleForecaster',
    'SimulatorForecaster',
    'OnlineTargetForecaster',
]
