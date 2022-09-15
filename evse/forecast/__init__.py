"""
The :mod:`evse.forecast` module includes tools related to forecasting demand
"""

from ._training import ForecasterTrainingResult, TrainingResult

__all__ = [
    "TrainingResult",
    "ForecasterTrainingResult",
]
