"""
The :mod:`evse.optimisation` module includes tools related to optimisation of Ev infrastructures
"""

from ._exceptions import SolverInitiationError
from ._metrics import compute_cost_of_infrastructure, compute_customer_dissatisfaction
from ._modelisation import DataModel, create_data_model
from ._result import compute_all_years_results

__all__ = [
    "create_data_model",
    "DataModel",
    "compute_customer_dissatisfaction",
    "compute_cost_of_infrastructure",
    "SolverInitiationError",
    "compute_all_years_results",
]
