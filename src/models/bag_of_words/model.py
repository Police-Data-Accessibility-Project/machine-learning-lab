from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.shared.bases.pydantic.arbitrary_base_model import ArbitraryBaseModel


class BagOfWordsModelContainer(ArbitraryBaseModel):
    model: LogisticRegression
    term_label_encoder: LabelEncoder
    permitted_terms: list[str]
