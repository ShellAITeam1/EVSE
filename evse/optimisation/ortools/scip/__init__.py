from typing import List, Optional

from ortools.linear_solver import pywraplp

from evse.logger import logger
from evse.optimisation import (
    DataModel,
    SolverInitiationError,
    compute_all_years_results,
    compute_cost_of_infrastructure,
    compute_customer_dissatisfaction,
)
from evse.optimisation._result import summarize_optimization_output
from evse.scoring import Result

from ._constraint import (
    add_demand_point_delivery_should_be_less_than_capacity_constraint,
    add_demand_point_partition_over_all_supply_point_constraint,
    add_demand_supply_positive_or_null_constraint,
    add_incremental_ev_infrastructure_constraint,
    add_number_of_charging_station_is_non_null_positive_integer,
    add_parking_slot_per_supply_point_index_constraint,
)
from ._variables import (
    create_demand_supply_variable,
    create_supply_point_index_capacities_variable,
)


def apply_scip_optimizer(
    data_model: DataModel,
    limit_in_time: Optional[int] = None,
    show_output: bool = False,
    gap_limit: float = 0.01,
) -> List[Result]:
    # Create the mip solver with the SCIP backend.
    logger.info("Create the mip solver with the SCIP backend")
    solver = pywraplp.Solver("my_model", pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    if limit_in_time:
        logger.info(f"Time limit set to {limit_in_time} seconds.")
        solver.set_time_limit(limit_in_time)

    if not solver:
        raise SolverInitiationError("Solver not created")

    # set a stopping gap limit for MIP
    solver_parameters = pywraplp.MPSolverParameters()
    solver_parameters.SetDoubleParam(solver_parameters.RELATIVE_MIP_GAP, gap_limit)

    # Variables
    logger.info("Initializing Variables")
    logger.info("Initializing Variables: demand_supply")
    demand_supply = create_demand_supply_variable(data_model, solver)

    logger.info("Initializing Variables: supply_point_index_capacities")
    supply_point_index_capacities = create_supply_point_index_capacities_variable(data_model, solver)

    # Constraints
    logger.info("Setting Constraints")
    add_demand_supply_positive_or_null_constraint()
    add_number_of_charging_station_is_non_null_positive_integer()
    add_parking_slot_per_supply_point_index_constraint(data_model, solver, supply_point_index_capacities)
    add_incremental_ev_infrastructure_constraint(data_model, solver, supply_point_index_capacities)
    add_demand_point_delivery_should_be_less_than_capacity_constraint(
        data_model, demand_supply, solver, supply_point_index_capacities
    )
    add_demand_point_partition_over_all_supply_point_constraint(data_model, demand_supply, solver)

    logger.info("Setting Objective Function")
    solver.Minimize(
        solver.Sum(
            [
                solver.Sum(
                    compute_customer_dissatisfaction(
                        data_model.distance_matrix,
                        demand_supply,
                        data_model.demand_forecast,
                        data_model.years_list,
                    )
                ),
                solver.Sum(
                    compute_cost_of_infrastructure(
                        supply_point_index_capacities,
                        data_model.years_list,
                        data_model.supply_point_indexes,
                    )
                ),
            ]
        )
    )

    solver.EnableOutput()
    logger.info("Start Solving")
    status = solver.Solve(solver_parameters)

    all_years_results = compute_all_years_results(status, demand_supply, supply_point_index_capacities, data_model)

    if show_output and len(all_years_results) > 0:
        summarize_optimization_output(all_years_results, data_model)

    return all_years_results
