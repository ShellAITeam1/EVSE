"""Main module."""
import logging
from pathlib import Path

import pandas as pd

from evse.scoring import YearlyResult, get_scoring_dataframe


def main(data_path: Path):

    yearly_results = [
        YearlyResult(
            year=2019,
            slow_charging_stations_on_supply_point_matrix=pd.read_csv(
                data_path / "slow_charging_stations_on_supply_point_matrix_2019.csv"
            ),
            fast_charging_stations_on_supply_point_matrix=pd.read_csv(
                data_path / "fast_charging_stations_on_supply_point_matrix_2019.csv"
            ),
            demand_supply_matrix=pd.read_csv(
                data_path / "demand_supply_matrix_2019.csv",
                header=[0, 1],
                index_col=[0],
            ),
        ),
        YearlyResult(
            year=2020,
            slow_charging_stations_on_supply_point_matrix=pd.read_csv(
                data_path / "slow_charging_stations_on_supply_point_matrix_2020.csv"
            ),
            fast_charging_stations_on_supply_point_matrix=pd.read_csv(
                data_path / "fast_charging_stations_on_supply_point_matrix_2020.csv",
            ),
            demand_supply_matrix=pd.read_csv(
                data_path / "demand_supply_matrix_2020.csv",
                header=[0, 1],
                index_col=[0],
            ),
        ),
    ]

    submission_dataframe = get_scoring_dataframe(yearly_results)

    return submission_dataframe


if __name__ == "__main__":
    logging.info("Start Main")
    data_path = Path(__file__).parent.parent / "tests/test_data"

    main(data_path)
