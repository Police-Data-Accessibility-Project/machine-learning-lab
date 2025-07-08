import numpy as np

from src.shared.bases.pydantic.arbitrary_base_model import ArbitraryBaseModel


class PredictionOutputs(ArbitraryBaseModel):
    pred: np.ndarray
    probability: np.ndarray
