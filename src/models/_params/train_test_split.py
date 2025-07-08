import numpy as np
from scipy.sparse import csr_matrix

from src.shared.bases.pydantic.arbitrary_base_model import ArbitraryBaseModel


class TrainTestSplit(ArbitraryBaseModel):
    x_train: csr_matrix
    x_test: csr_matrix
    y_train: np.ndarray
    y_test: np.ndarray