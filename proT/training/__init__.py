"""
Training infrastructure for ProT models.
"""

# Avoid circular imports by not importing at module level
# Users should import directly from submodules:
# from proT.training.trainer import trainer
# from proT.training.dataloader import ProcessDataModule
# etc.

__all__ = [
    'trainer',
    'get_model_object',
    'ProcessDataModule',
    'update_config',
    'combination_sweep',
]
