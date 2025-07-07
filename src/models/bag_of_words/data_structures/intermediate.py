import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from src.shared.bases.pydantic.arbitrary_base_model import ArbitraryBaseModel


class BagOfWordsIntermediate(ArbitraryBaseModel):
    sparse_matrix: csr_matrix
    urls_ids: npt.NDArray[np.int_]
    terms_ids: npt.NDArray[np.int_]
    y_relevant: pl.Series
    y_fine: pl.Series
    y_coarse: pl.Series
    url_encoder: LabelEncoder
    term_encoder: LabelEncoder