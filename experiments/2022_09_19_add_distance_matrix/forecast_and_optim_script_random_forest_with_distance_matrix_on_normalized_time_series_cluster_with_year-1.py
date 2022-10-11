from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans

import evse
from evse.const import (
    DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
    DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
    DEMAND_POINT_INDEX_COLUMN_NAME,
    VALUE_COLUMN_NAME,
    YEAR_INDEX_COLUMN_NAME, SUPPLY_POINT_INDEX_COLUMN_NAME,
)
from evse.forecast import ForecasterTrainingResult, TrainingResult
from evse.optimisation import create_data_model
from evse.optimisation._modelisation import extract_distance_matrix
from evse.optimisation.ortools.scip import apply_scip_optimizer
from evse.scoring import get_scoring_dataframe


def train_model_with_cross_validation(
    demand_history_df: pd.DataFrame, distance_matrix_df: pd.DataFrame, forecast_horizon: List[int]
) -> TrainingResult:
    # TODO: Use constant after rebasing
    full_stacked_df = feature_engineering(demand_history_df, distance_matrix_df)

    stacked_demand_history_test_df, stacked_demand_history_train_df = custom_train_test_split(
        forecast_horizon, full_stacked_df
    )
    predictor_columns = [
        DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
        DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
        YEAR_INDEX_COLUMN_NAME,
        "value_1_years_ago",
        "value_2_years_ago",
        "min",
        "min-1",
        "min-2",
    ]
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
    groups = stacked_demand_history_train_df[DEMAND_POINT_INDEX_COLUMN_NAME].values
    y_train = stacked_demand_history_train_df[target_column].values.reshape(-1)

    grid = {
        "random_forest__n_estimators": [10, 20, 100, 200],
        "random_forest__max_features": ["sqrt", "log2"],
        "random_forest__max_depth": [2, 3, 7, 8, 20, 30],
        "random_forest__random_state": [42],
    }

    steps = [("random_forest", RandomForestRegressor())]
    pipe = Pipeline(steps)

    cv = GroupKFold(n_splits=10)
    cv.get_n_splits(X_train, y_train, groups=groups)

    rf_cv = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        cv=cv.get_n_splits(X_train, y_train, groups=groups),
        verbose=2,
    )
    rf_cv.fit(X_train, y_train)
    print(f"Best Parameters resulting from GridSearch:\n{rf_cv.best_params_}")

    best_param = rf_cv.best_params_

    # best_param = {
    #     'random_forest__max_depth': 20,
    #     'random_forest__max_features': 'sqrt',
    #     'random_forest__n_estimators': 200,
    #     'random_forest__random_state': 42
    # }

    best_pipe = pipe.set_params(**best_param).fit(X_train, y_train)

    train_pred = pd.Series(best_pipe.predict(X_train), index=stacked_demand_history_train_df.index)
    print("Score on Train:", mean_absolute_error(train_pred, y_train))

    X_test = feature_space_pipe.transform(stacked_demand_history_test_df)
    y_test = stacked_demand_history_test_df[target_column].values.reshape(-1)
    test_pred = pd.Series(best_pipe.predict(X_test), index=stacked_demand_history_test_df.index)
    print("Score on Test:", mean_absolute_error(test_pred, y_test))
    print("Score on Test RMSE:", mean_squared_error(test_pred, y_test, squared=False))
    print("Score on Test MAPE", mean_absolute_percentage_error(test_pred, y_test))
    pred = stacked_demand_history_test_df.copy()
    pred["pred"] = test_pred
    pred = pred.set_index(
        [
            DEMAND_POINT_INDEX_COLUMN_NAME,
            DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
            DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
            YEAR_INDEX_COLUMN_NAME,
        ]
    )

    # Fit on whole data
    full_X_train = feature_space_pipe.transform(full_stacked_df)
    full_y_train = full_stacked_df["value"].values.reshape(-1)
    best_pipe = best_pipe.fit(full_X_train, full_y_train)

    forecaster_training_result = ForecasterTrainingResult(
        predictors_transformation_pipeline=feature_space_pipe,
        model_pipeline=best_pipe,
        target_transformation=lambda x: x,
    )

    return forecaster_training_result, pred


def custom_train_test_split(
    forecast_horizon: List[int], full_stacked_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stacked_demand_history_train_df = full_stacked_df[
        full_stacked_df[YEAR_INDEX_COLUMN_NAME].isin(
            set(full_stacked_df[YEAR_INDEX_COLUMN_NAME]).difference(forecast_horizon)
        )
    ]
    stacked_demand_history_test_df = full_stacked_df[full_stacked_df[YEAR_INDEX_COLUMN_NAME].isin(forecast_horizon)]
    return stacked_demand_history_test_df, stacked_demand_history_train_df


def feature_engineering(
    demand_history_df: pd.DataFrame, distance_matrix_df: pd.DataFrame, mode: str = "training"
) -> pd.DataFrame:
    stacked_demand_history_df = demand_history_df.set_index(
        [
            DEMAND_POINT_INDEX_COLUMN_NAME,
            DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
            DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
        ]
    ).stack(0)
    stacked_demand_history_df.index.names = [
        DEMAND_POINT_INDEX_COLUMN_NAME,
        DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
        DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
        YEAR_INDEX_COLUMN_NAME,
    ]
    stacked_demand_history_df.name = VALUE_COLUMN_NAME
    stacked_demand_history_df = stacked_demand_history_df.to_frame()
    two_year_index = list(
        map(
            lambda index_tuple: (index_tuple[0], index_tuple[1], index_tuple[2], str(int(index_tuple[3]) + 2)),
            stacked_demand_history_df.index.values,
        )
    )
    two_year_shifted = stacked_demand_history_df.copy()
    two_year_shifted.index = pd.MultiIndex.from_tuples(two_year_index, names=stacked_demand_history_df.index.names)
    two_year_shifted.columns = ["value_2_years_ago"]

    last_year_index = list(
        map(
            lambda index_tuple: (index_tuple[0], index_tuple[1], index_tuple[2], str(int(index_tuple[3]) + 1)),
            stacked_demand_history_df.index.values,
        )
    )
    one_year_shifted = stacked_demand_history_df.copy()
    one_year_shifted.index = pd.MultiIndex.from_tuples(last_year_index, names=stacked_demand_history_df.index.names)
    one_year_shifted.columns = ["value_1_years_ago"]

    full_stacked_df = pd.concat([stacked_demand_history_df, one_year_shifted, two_year_shifted], axis=1)
    if mode == "training":
        full_stacked_df = full_stacked_df.dropna(how="any")
    full_stacked_df = full_stacked_df.reset_index()
    full_stacked_df[YEAR_INDEX_COLUMN_NAME] = full_stacked_df[YEAR_INDEX_COLUMN_NAME].astype(float)

    # Compute top 3 min distance
    restricted_distance_matrix_df = distance_matrix_df[
        distance_matrix_df.index.isin(full_stacked_df[DEMAND_POINT_INDEX_COLUMN_NAME])
    ]
    distance_matrix_transpose_df = restricted_distance_matrix_df.transpose()

    top_3_min_per_demand_point_list = []
    for demand_point in restricted_distance_matrix_df.index:
        demand_point_top3_min = distance_matrix_transpose_df.nsmallest(3, columns=[demand_point])
        demand_point_top3_min.index = ["min", "min-1", "min-2"]
        demand_point_top3_min = demand_point_top3_min[[demand_point]].transpose()
        top_3_min_per_demand_point_list.append(demand_point_top3_min)

    top_3_min_per_demand_point_df = pd.concat(top_3_min_per_demand_point_list)

    full_stacked_with_distance_df = pd.concat([
        full_stacked_df.set_index(DEMAND_POINT_INDEX_COLUMN_NAME), top_3_min_per_demand_point_df], axis=1
    ).reset_index()

    return full_stacked_with_distance_df


def forecast(
    demand_df: pd.DataFrame, distance_matrix_df: pd.DataFrame, forecast_horizon: List[int], forecaster: TrainingResult
) -> pd.DataFrame:
    full_stacked_df = feature_engineering(demand_df, distance_matrix_df, mode="inference")

    full_forecast_stacked_df = full_stacked_df[full_stacked_df[YEAR_INDEX_COLUMN_NAME].isin(forecast_horizon)]

    prediction_feature_space = forecaster.predictors_transformation_pipeline.transform(full_forecast_stacked_df)

    predictions_values = forecaster.model_pipeline.predict(prediction_feature_space)
    predictions_df = pd.DataFrame(
        {
            DEMAND_POINT_INDEX_COLUMN_NAME: full_forecast_stacked_df[DEMAND_POINT_INDEX_COLUMN_NAME],
            YEAR_INDEX_COLUMN_NAME: full_forecast_stacked_df[YEAR_INDEX_COLUMN_NAME],
            VALUE_COLUMN_NAME: predictions_values,
        }
    )

    unstacked_predictions_df = (
        predictions_df.set_index([DEMAND_POINT_INDEX_COLUMN_NAME, YEAR_INDEX_COLUMN_NAME])
        .unstack([YEAR_INDEX_COLUMN_NAME])
        .reset_index()
    )

    unstacked_predictions_df.columns = [DEMAND_POINT_INDEX_COLUMN_NAME] + forecast_horizon

    return unstacked_predictions_df


def clusterize_demand_points(nb_cluster: int, mode: str = "normal") -> pd.DataFrame:
    years_column = [str(year) for year in range(2010, 2019)]
    X = demand_history_df[years_column]
    if mode == "normalized":
        X = X.apply(lambda x: x / max(x) if max(x) != 0 else 0, axis=1)

    km = TimeSeriesKMeans(n_clusters=nb_cluster, metric="dtw", max_iter=5, random_state=0).fit(X)
    group = km.predict(X)
    group_df = pd.DataFrame({
        DEMAND_POINT_INDEX_COLUMN_NAME: range(len(group)),
        f"{mode}_group": group
    })

    return group_df


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data"
    result_path = data_path / "result"

    # --------------------------------- #
    #            Clustering             #
    # --------------------------------- #
    demand_history_df = pd.read_csv(data_path / "Demand_History.csv")
    existing_infrastructure_df = pd.read_csv(data_path / "exisiting_EV_infrastructure_2018.csv")

    group_df = clusterize_demand_points(nb_cluster=3)
    # --------------------------------- #
    #             Training              #
    # --------------------------------- #
    training_forecast_horizon = [2017, 2018]
    distance_matrix = extract_distance_matrix(demand_history_df, existing_infrastructure_df, data_path)
    distance_matrix_df = pd.DataFrame.from_dict(distance_matrix, orient="index")
    distance_matrix_df.index = pd.MultiIndex.from_tuples(
        distance_matrix_df.index, names=[DEMAND_POINT_INDEX_COLUMN_NAME, SUPPLY_POINT_INDEX_COLUMN_NAME]
    )
    distance_matrix_df = distance_matrix_df.unstack(level=1)

    forecaster_training_results_dict = {}
    pred_list = []
    for cluster_id in group_df["normal_group"].unique():
        cluster_demand_point = group_df[group_df["normal_group"] == cluster_id][DEMAND_POINT_INDEX_COLUMN_NAME].values
        cluster_demand_history_df = demand_history_df[
            demand_history_df[DEMAND_POINT_INDEX_COLUMN_NAME].isin(cluster_demand_point)
        ]
        forecaster_training_results_dict[cluster_id], pred = train_model_with_cross_validation(
            cluster_demand_history_df, distance_matrix_df, training_forecast_horizon
        )
        pred_list.append(pred)

    full_pred = pd.concat(pred_list).sort_index()
    print("Whole MAE:", mean_absolute_error(full_pred['value'], full_pred['pred']))
    print("Whole MAPE:", mean_absolute_percentage_error(full_pred['value'], full_pred['pred']))

    # --------------------------------- #
    #              Predict              #
    # --------------------------------- #
    forecast_horizon = [2019, 2020]
    forecast_year_list = []
    demand_history_for_forecast_df = demand_history_df.copy()
    for forecast_year in forecast_horizon:
        demand_history_for_forecast_df.columns = demand_history_for_forecast_df.columns.astype(str)
        demand_to_predict_df = demand_history_for_forecast_df[
            ["demand_point_index", "x_coordinate", "y_coordinate"] + [str(forecast_year-year) for year in range(1, 3)]
            ]

        forecast_df_list = []
        for cluster_id in group_df["normal_group"].unique():
            cluster_demand_point = group_df[group_df["normal_group"] == cluster_id][DEMAND_POINT_INDEX_COLUMN_NAME].values
            cluster_demand_history_df = demand_to_predict_df[
                demand_to_predict_df[DEMAND_POINT_INDEX_COLUMN_NAME].isin(cluster_demand_point)
            ]
            cluster_forecast_df = forecast(
                cluster_demand_history_df, distance_matrix_df, [forecast_year], forecaster_training_results_dict[cluster_id]
            )
            forecast_df_list.append(cluster_forecast_df)

        forecast_df = pd.concat(forecast_df_list).sort_values(by=[DEMAND_POINT_INDEX_COLUMN_NAME]).reset_index(drop=True)

        forecast_year_list.append(forecast_df.set_index(DEMAND_POINT_INDEX_COLUMN_NAME))

        demand_history_for_forecast_df = pd.concat([
            demand_history_for_forecast_df.set_index(DEMAND_POINT_INDEX_COLUMN_NAME),
            forecast_df.set_index(DEMAND_POINT_INDEX_COLUMN_NAME),
        ], axis=1).reset_index()

    full_forecast_df = pd.concat(forecast_year_list, axis=1).sort_index().reset_index()

    # --------------------------------- #
    #               Optim               #
    # --------------------------------- #
    demand_history: pd.DataFrame = pd.read_csv(data_path / "Demand_History.csv")
    demand_forecast = full_forecast_df.copy()
    existing_infrastructure: pd.DataFrame = pd.read_csv(data_path / "exisiting_EV_infrastructure_2018.csv")

    data_model = create_data_model(demand_forecast, demand_history, existing_infrastructure, data_path)

    all_years_result = apply_scip_optimizer(data_model, show_output=True, gap_limit=0.0, limit_in_time=30 * 60 * 1000)

    # --------------------------------- #
    #             Submission            #
    # --------------------------------- #
    submission_dataframe = get_scoring_dataframe(all_years_result)

    submission_dataframe.to_csv(result_path / f"submission_dataframe_{evse.__version__}.csv", index=False)
