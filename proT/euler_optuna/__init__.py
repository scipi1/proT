"""
Euler Optuna optimization module for proT.

This module provides a standardized interface for hyperparameter optimization
using Optuna, following the euler_optuna skeleton pattern.
"""

from .optuna_opt import OptunaStudy

__all__ = ['OptunaStudy']
