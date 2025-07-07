import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from src.models._params.train_test_split import TrainTestSplit
from src.models.data_containers.prediction_outputs import PredictionOutputs
from src.models.types import FitPredictor


def apply_train_test_split(
    x: csr_matrix,
    y: np.ndarray,
    test_size: float = 0.5,
    random_state: int = 42
) -> TrainTestSplit:

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return TrainTestSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )


def report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probability_estimates: np.ndarray
) -> None:
    print(f"Mean: {probability_estimates.mean()}")
    print(f"Variance: {probability_estimates.var()}")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")

    precision, recall, thresholds = precision_recall_curve(
        y_true=y_test,
        y_score=probability_estimates
    )

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.show()

    # Distribution of predicted probabilities
    plt.hist(probability_estimates, bins=50)
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Probabilities")
    plt.show()


def fit_and_predict(
    clf: FitPredictor,
    x_train: np.ndarray,
    x_test: csr_matrix,
    y_train: np.ndarray,
) -> PredictionOutputs:
    start = time.perf_counter()

    clf.fit(x_train, y_train)

    end = time.perf_counter()

    print(f"Training time: {end - start:.4f} seconds")

    y_pred = clf.predict(x_test)
    probability_estimates = clf.predict_proba(x_test)[:, 1]
    return PredictionOutputs(
        pred=y_pred,
        probability=probability_estimates
    )
