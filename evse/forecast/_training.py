from typing import Callable, Protocol

from sklearn.pipeline import Pipeline


class TrainingResult(Protocol):
    predictors_transformation_pipeline: Pipeline
    model_pipeline: Pipeline
    target_transformation: Callable


class ForecasterTrainingResult:
    def __init__(
        self,
        predictors_transformation_pipeline: Pipeline,
        model_pipeline: Pipeline,
        target_transformation: Callable,
    ):
        self.predictors_transformation_pipeline = predictors_transformation_pipeline
        self.model_pipeline = model_pipeline
        self.target_transformation = target_transformation

    def __repr__(self):
        return f"{self.__class__.__name__}()"
