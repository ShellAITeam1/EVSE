from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from evse.const import (
    DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
    DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
    RFR,
    SPL,
    YEAR_INDEX_COLUMN_NAME,
)


def forecaster_estimator(forecaster: Literal[RFR, SPL], **kwargs) -> BaseEstimator:
    """Returns the chosen forecaster"""
    if forecaster == RFR:
        return RandomForestRegressor(
            **{"max_depth": 30, "max_features": "sqrt", "n_estimators": 200, "random_state": 42}
        )
    if forecaster == SPL:
        return SplineEstimator(**kwargs)


class SplineEstimator(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Custom Spline Estimator as per template in
    https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py
    """

    def __init__(self, predictor_columns, weights=None):
        self.interpolater = UnivariateSpline
        if self.interpolater == UnivariateSpline:
            self.kwargs = {"w": weights, "ext": "extrapolate", "k": 2, "s": 100}
        if self.interpolater == interp1d:
            self.kwargs = {"fill_value": "extrapolate", "kind": "quadratic"}
        self.columns = predictor_columns

    def format_data(self, x: np.array, y: np.array = None) -> pd.DataFrame:
        """Formats time series as numpy arrays"""
        return pd.DataFrame(x, columns=self.columns).assign(y=y)

    def fit_poly(self, df: pd.DataFrame):
        """fit poly"""
        return self.interpolater(df[YEAR_INDEX_COLUMN_NAME], df["y"], **self.kwargs)

    def fit(self, x: np.array, y: np.array) -> BaseEstimator:
        """Fit splines in one go"""
        df = self.format_data(x, y)

        self.spline_estimator_ = df.groupby(
            [
                DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
                DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
            ]
        ).apply(self.fit_poly)
        return self

    def apply_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(results=df["estimator"].values[0](df[YEAR_INDEX_COLUMN_NAME]))

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    def transform(self, X, y=None):
        return self.format_data(X).drop(columns=["y"]).assign(spline_forecast=self.predict(X))

    def predict(self, X: Union[pd.DataFrame, np.array]) -> np.array:
        """Predicts based on the trained model in one go"""
        check_is_fitted(self, "spline_estimator_")
        X = self._validate_data(X=X, ensure_2d=True)
        df = self.format_data(X).join(
            pd.DataFrame(self.spline_estimator_, columns=["estimator"]),
            on=[
                DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
                DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
            ],
        )

        return (
            df.groupby(
                [
                    DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
                    DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
                ]
            )
            .apply(self.apply_predict)["results"]
            .values
        )
