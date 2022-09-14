from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from evse.forecast import ForecasterTrainingResult, TrainingResult


def train_model_with_cross_validation(demand_history_df: pd.DataFrame, forecast_horizon: List[str]) -> TrainingResult:
    # TODO: Use constant after rebasing
    full_stacked_df = feature_engineering(demand_history_df)

    stacked_demand_history_test_df, stacked_demand_history_train_df = custom_train_test_split(
        forecast_horizon, full_stacked_df
    )
    predictor_columns = ["x_coordinate", "y_coordinate", "year", "value_last_year"]
    target_column = ["value"]

    feature_space_steps = [
        (
            "column_selector",
            ColumnTransformer(
                [
                    ("selector", "passthrough", predictor_columns),
                ],
                remainder="drop",
            ),
        ),
        ("min_max_scaler", MinMaxScaler()),
    ]

    feature_space_pipe = Pipeline(feature_space_steps)

    X_train = feature_space_pipe.fit_transform(stacked_demand_history_train_df)
    groups = stacked_demand_history_train_df["demand_point_index"].values
    y_train = np.log1p(stacked_demand_history_train_df[target_column])

    steps = [("random_forest", RandomForestRegressor(n_estimators=100))]
    pipe = Pipeline(steps)

    scoring = ["neg_root_mean_squared_error"]
    cv = GroupKFold(n_splits=10)
    scores = cross_validate(
        pipe,
        X_train,
        y_train.values.reshape(-1),
        cv=cv,
        scoring=scoring,
        error_score="raise",
        groups=groups,
        return_estimator=True,
    )
    train_pred = pd.Series(scores.get("estimator")[0].predict(X_train), index=y_train.index)
    print(mean_absolute_error(train_pred, y_train))

    X_test = feature_space_pipe.transform(stacked_demand_history_test_df)
    y_test = np.log1p(stacked_demand_history_test_df[target_column])
    test_pred = pd.Series(scores.get("estimator")[0].predict(X_test), index=y_test.index)
    print(mean_absolute_error(test_pred, y_test))

    original_y_test = stacked_demand_history_test_df[target_column]
    exp_test_pred = pd.Series(np.exp(scores.get("estimator")[0].predict(X_test)), index=y_test.index)
    print(mean_absolute_error(exp_test_pred, original_y_test))
    pred = stacked_demand_history_test_df.copy()
    pred["pred"] = exp_test_pred
    pred = pred.set_index(["demand_point_index", "x_coordinate", "y_coordinate", "year"])

    forecaster_training_result = ForecasterTrainingResult(
        predictors_transformation_pipeline=feature_space_pipe, model_pipeline=pipe, target_transformation=np.log1p
    )

    return forecaster_training_result


def custom_train_test_split(forecast_horizon, full_stacked_df):
    stacked_demand_history_train_df = full_stacked_df[
        full_stacked_df["year"].isin(set(full_stacked_df["year"]).difference(forecast_horizon))
    ]
    stacked_demand_history_test_df = full_stacked_df[full_stacked_df["year"].isin(forecast_horizon)]
    return stacked_demand_history_test_df, stacked_demand_history_train_df


def feature_engineering(demand_history_df):
    stacked_demand_history_df = demand_history_df.set_index(
        ["demand_point_index", "x_coordinate", "y_coordinate"]
    ).stack(0)
    stacked_demand_history_df.index.names = ["demand_point_index", "x_coordinate", "y_coordinate", "year"]
    stacked_demand_history_df.name = "value"
    stacked_demand_history_df = stacked_demand_history_df.to_frame()
    last_year_index = list(
        map(
            lambda index_tuple: (index_tuple[0], index_tuple[1], index_tuple[2], str(int(index_tuple[3]) + 1)),
            stacked_demand_history_df.index.values,
        )
    )
    one_year_shifted = stacked_demand_history_df.copy()
    one_year_shifted.index = pd.MultiIndex.from_tuples(last_year_index, names=stacked_demand_history_df.index.names)
    one_year_shifted.columns = ["value_last_year"]
    full_stacked_df = pd.concat([stacked_demand_history_df, one_year_shifted], axis=1).dropna(how="any")
    full_stacked_df = full_stacked_df.reset_index()
    return full_stacked_df


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data"

    demand_history_df = pd.read_csv(data_path / "Demand_History.csv")

    forecast_horizon = ["2017", "2018"]

    forecaster_training_result = train_model_with_cross_validation(demand_history_df, forecast_horizon)
