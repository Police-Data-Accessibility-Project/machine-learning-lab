from dataclasses import dataclass

from src.models.types import FitPredictor


@dataclass
class BagOfWordsFullModel:
    model: FitPredictor
    term_label_encoder: FitPredictor
    permitted_terms: list[str]
