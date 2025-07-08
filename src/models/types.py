from typing import TypeVar, Protocol

X = TypeVar("X")
Y = TypeVar("Y")


class FitPredictor(Protocol[X, Y]):
    def fit(self, x: X, y: Y) -> "FitPredictor":
        ...
    def predict(self, x: X) -> Y:
        ...
    def predict_proba(self, x: X) -> Y:
        ...
