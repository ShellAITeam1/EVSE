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

from evse.const import YEAR_INDEX_COLUMN_NAME


def train_forecast(
    demand_history_df: pd.DataFrame, forecast_horizon: List[int]
) -> pd.DataFrame:
    # TODO: Use constant after rebasing
    stacked_demand_history_df = demand_history_df.set_index(
        ["demand_point_index", "x_coordinate", "y_coordinate"]
    ).stack(0)
    stacked_demand_history_df.index.names = [
        "demand_point_index",
        "x_coordinate",
        "y_coordinate",
        "year",
    ]
    stacked_demand_history_df.name = "value"
    stacked_demand_history_df = stacked_demand_history_df.to_frame()
    last_year_index = list(
        map(
            lambda index_tuple: (
                index_tuple[0],
                index_tuple[1],
                index_tuple[2],
                str(int(index_tuple[3]) + 2),
            ),
            stacked_demand_history_df.index.values,
        )
    )
    one_year_shifted = stacked_demand_history_df.copy()
    one_year_shifted.index = pd.MultiIndex.from_tuples(
        last_year_index, names=stacked_demand_history_df.index.names
    )
    one_year_shifted.columns = ["value_last_year"]

    def get_neighbour(index, radius):
        x_lower_bound = index[1] - radius
        y_lower_bound = index[2] - radius

        filtered_neighbour_indexes = [
            (demand_point, x_lower_bound + x, y_lower_bound + y, index[3])
            for demand_point in range(
                index[0] - 2 * radius * 64, index[0] + 2 * radius * 64
            )
            for x in range(2 * radius + 1)
            for y in range(2 * radius + 1)
            if (index[0], x, y, index[3]) != index
        ]
        return filtered_neighbour_indexes

    def compute_neighbour(index, dataframe, radius):
        neighbour_indexes = get_neighbour(index.name, radius)
        neighbour_df = dataframe.loc[
            list(set(neighbour_indexes).intersection(dataframe.index)), :
        ]
        return [
            neighbour_df.mean(),
            neighbour_df.min(),
            neighbour_df.max(),
            neighbour_df.std(),
        ]

    neighbour_df = one_year_shifted.copy()
    computed_features = np.array(
        one_year_shifted.progress_apply(
            lambda x: compute_neighbour(x, one_year_shifted, 1), axis=1
        ).values.tolist()
    )
    neighbour_df["neighbour_mean"] = computed_features[:, 0]
    neighbour_df["neighbour_min"] = computed_features[:, 1]
    neighbour_df["neighbour_max"] = computed_features[:, 2]
    neighbour_df["neighbour_std"] = computed_features[:, 3]
    neighbour_df["neighbour_std"] = neighbour_df["neighbour_std"].fillna(0)

    full_stacked_df = pd.concat(
        [stacked_demand_history_df, neighbour_df], axis=1
    ).dropna(how="any")
    full_stacked_df = full_stacked_df.reset_index()
    full_stacked_df[YEAR_INDEX_COLUMN_NAME] = full_stacked_df[
        YEAR_INDEX_COLUMN_NAME
    ].astype(int)

    stacked_demand_history_train_df = full_stacked_df[
        full_stacked_df[YEAR_INDEX_COLUMN_NAME].isin(
            set(full_stacked_df[YEAR_INDEX_COLUMN_NAME]).difference(forecast_horizon)
        )
    ]
    stacked_demand_history_test_df = full_stacked_df[
        full_stacked_df[YEAR_INDEX_COLUMN_NAME].isin(forecast_horizon)
    ]

    predictor_columns = [
        "x_coordinate",
        "y_coordinate",
        YEAR_INDEX_COLUMN_NAME,
        "value_last_year",
        "neighbour_mean",
        "neighbour_min",
        "neighbour_max",
    ]

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
    y_train = np.log1p(stacked_demand_history_train_df["value"])

    steps = [("random_forest", RandomForestRegressor(n_estimators=50))]
    pipe = Pipeline(steps)

    scoring = ["neg_mean_absolute_error"]
    cv = GroupKFold(n_splits=10)
    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        error_score="raise",
        groups=groups,
        return_estimator=True,
    )
    train_pred = pd.Series(
        scores.get("estimator")[0].predict(X_train), index=y_train.index
    )
    print(mean_absolute_error(train_pred, y_train))

    X_test = feature_space_pipe.transform(stacked_demand_history_test_df)
    y_test = np.log1p(stacked_demand_history_test_df["value"])
    test_pred = pd.Series(
        scores.get("estimator")[0].predict(X_test), index=y_test.index
    )
    print(mean_absolute_error(test_pred, y_test))

    original_y_test = stacked_demand_history_test_df["value"]
    exp_test_pred = pd.Series(
        np.exp(scores.get("estimator")[0].predict(X_test)), index=y_test.index
    )
    print(mean_absolute_error(exp_test_pred, original_y_test))
    pred = stacked_demand_history_test_df.copy()
    pred["pred"] = exp_test_pred
    pred = pred.set_index(
        ["demand_point_index", "x_coordinate", "y_coordinate", "year"]
    )

    forecast_df = pd.DataFrame()
    return forecast_df


if __name__ == "__main__":
    from tqdm import tqdm

    tqdm.pandas()

    data_path = Path(__file__).parent.parent.parent / "data"

    demand_history_df = pd.read_csv(data_path / "Demand_History.csv")

    forecast_horizon = [2017, 2018]

    train_forecast(demand_history_df, forecast_horizon)
