import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evse.scoring import YearlyResult
from evse.scoring._metrics import (
    compute_customer_dissatisfaction_cost_over_years,
    score_loss,
)

TESTES_MODULE = "evse.scoring._metrics"


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.parent / "test_data/"

    def test_compute_customer_dissatisfaction_cost_over_years_on_zero(self):
        # Given
        given_distance_matrix = np.zeros((3, 3))
        given_demand_supply_values_list = [np.zeros((3, 3)), np.zeros((3, 3))]

        # When
        customer_dissatisfaction_cost_over_years = compute_customer_dissatisfaction_cost_over_years(
            given_distance_matrix, given_demand_supply_values_list
        )

        # Then
        expected_customer_dissatisfaction_cost_over_years = 0.0
        assert expected_customer_dissatisfaction_cost_over_years == customer_dissatisfaction_cost_over_years

    def test_compute_customer_dissatisfaction_cost_over_years_with_distance_non_null_and_demand_supply_null(self):
        # Given
        given_distance_matrix = np.ones((3, 3))
        given_demand_supply_values_list = [np.zeros((3, 3)), np.zeros((3, 3))]

        # When
        customer_dissatisfaction_cost_over_years = compute_customer_dissatisfaction_cost_over_years(
            given_distance_matrix, given_demand_supply_values_list
        )

        # Then
        expected_customer_dissatisfaction_cost_over_years = 0.0
        assert expected_customer_dissatisfaction_cost_over_years == customer_dissatisfaction_cost_over_years

    def test_compute_customer_dissatisfaction_cost_over_years(self):
        # Given
        given_distance_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        given_demand_supply_values_list = [
            np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 0.2], [0.9, 0, 0, 0.1]]),
            np.array([[0.2, 0.2, 0.2, 0.4], [0.7, 0.1, 0.2, 0.0], [0.7, 0, 0.2, 0.1]]),
        ]

        # When
        customer_dissatisfaction_cost_over_years = compute_customer_dissatisfaction_cost_over_years(
            given_distance_matrix, given_demand_supply_values_list
        )

        # Then
        expected_customer_dissatisfaction_cost_over_years = 36.4
        np.testing.assert_almost_equal(
            expected_customer_dissatisfaction_cost_over_years, customer_dissatisfaction_cost_over_years
        )

    def test_score_loss(self):
        # Given
        given_yearly_results = [
            YearlyResult(
                year=2019,
                slow_charging_stations_on_supply_point_matrix=pd.read_csv(
                    self.data_path / "slow_charging_stations_on_supply_point_matrix_2019.csv"
                ),
                fast_charging_stations_on_supply_point_matrix=pd.read_csv(
                    self.data_path / "fast_charging_stations_on_supply_point_matrix_2019.csv"
                ),
                demand_supply_matrix=pd.read_csv(
                    self.data_path / "demand_supply_matrix_2019.csv", header=[0, 1], index_col=[0]
                ),
            ),
            YearlyResult(
                year=2020,
                slow_charging_stations_on_supply_point_matrix=pd.read_csv(
                    self.data_path / "slow_charging_stations_on_supply_point_matrix_2020.csv"
                ),
                fast_charging_stations_on_supply_point_matrix=pd.read_csv(
                    self.data_path / "fast_charging_stations_on_supply_point_matrix_2020.csv",
                ),
                demand_supply_matrix=pd.read_csv(
                    self.data_path / "demand_supply_matrix_2020.csv", header=[0, 1], index_col=[0]
                ),
            ),
        ]

        with open(self.data_path / "distance_matrix_in_array.pickle", "rb") as handle:
            given_distance_matrix_in_array = pickle.load(handle)

        # When
        score = score_loss(given_yearly_results, given_distance_matrix_in_array)

        # Then
        expected_score = 10
        np.testing.assert_almost_equal(expected_score, score)
