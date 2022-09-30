import unittest
from pathlib import Path

import pandas as pd

from evse.scoring import YearlyResult, get_scoring_dataframe

TESTES_MODULE = "evse.scoring._submission"


class TestSubmission(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.parent / "test_data/"

    def test_get_scoring_dataframe(self):
        # Given
        given_yearly_results = [
            YearlyResult(
                year=2019,
                slow_charging_stations_on_supply_point_matrix=pd.read_csv(
                    self.data_path
                    / "slow_charging_stations_on_supply_point_matrix_2019.csv"
                ),
                fast_charging_stations_on_supply_point_matrix=pd.read_csv(
                    self.data_path
                    / "fast_charging_stations_on_supply_point_matrix_2019.csv"
                ),
                demand_supply_matrix=pd.read_csv(
                    self.data_path / "demand_supply_matrix_2019.csv",
                    header=[0, 1],
                    index_col=[0],
                ),
            ),
            YearlyResult(
                year=2020,
                slow_charging_stations_on_supply_point_matrix=pd.read_csv(
                    self.data_path
                    / "slow_charging_stations_on_supply_point_matrix_2020.csv"
                ),
                fast_charging_stations_on_supply_point_matrix=pd.read_csv(
                    self.data_path
                    / "fast_charging_stations_on_supply_point_matrix_2020.csv",
                ),
                demand_supply_matrix=pd.read_csv(
                    self.data_path / "demand_supply_matrix_2020.csv",
                    header=[0, 1],
                    index_col=[0],
                ),
            ),
        ]

        # When
        submission_dataframe = get_scoring_dataframe(given_yearly_results)

        # Then
        expected_submission_dataframe = pd.read_csv(
            self.data_path / "sample_submission.csv"
        )
        pd.testing.assert_frame_equal(
            expected_submission_dataframe, submission_dataframe
        )
