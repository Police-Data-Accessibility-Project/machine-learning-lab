import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src import export
from src.load import from_parquet
from src.models.bag_of_words.constants import HF_URL
from src.models.bag_of_words.data_structures.dataframe import BagOfWordsDataFrame
from src.models.bag_of_words.format import format_bag_of_words
from src.models.bag_of_words.model import BagOfWordsModelContainer
from src.models.helpers import apply_train_test_split, report, fit_and_predict

LOGISTIC_REGRESSION_MODEL = LogisticRegression(
    C=10.0,
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000
)

def run():

    bow_df = BagOfWordsDataFrame(from_parquet(HF_URL))

    permitted_terms = bow_df.term.unique().to_list()

    intermediate = format_bag_of_words(bow_df)

    x = intermediate.sparse_matrix
    y = intermediate.y_relevant.to_numpy()


    tts = apply_train_test_split(x=x,y=y)

    x_train_resampled = tts.x_train
    y_train_resampled = tts.y_train

    x_train_resampled, y_train_resampled = apply_undersampling(x_train_resampled, y_train_resampled)

    outputs = fit_and_predict(
        clf=LOGISTIC_REGRESSION_MODEL,
        x_train=x_train_resampled,
        x_test=tts.x_test,
        y_train=y_train_resampled
    )

    cal_clf = CalibratedClassifierCV(
        estimator=LOGISTIC_REGRESSION_MODEL,
        method='sigmoid',
        cv=5
    )
    # outputs = fit_and_predict(
    #     clf=cal_clf,
    #     x_train=x_train_resampled,
    #     x_test=tts.x_test,
    #     y_train=y_train_resampled
    # )

    # cal_clf.fit(x_train_resampled, y_train_resampled)
    # y_pred_cal = cal_clf.predict(tts.x_test)

    # y_pred_custom = probability_estimates > 0.05

    spot_check(
        y_test=tts.y_test,
        y_pred=outputs.pred,
        probability_estimates=outputs.probability,
        url_ids=tts.x_test.indices,
        url_encoder=intermediate.url_encoder,
        term_encoder=intermediate.term_encoder,
        sparse_matrix=tts.x_test
    )

    report(
        y_test=tts.y_test,
        y_pred=outputs.pred,
        probability_estimates=outputs.probability
    )

    path = export.to_joblib(
        obj=BagOfWordsModelContainer(
            model=LOGISTIC_REGRESSION_MODEL,
            term_label_encoder=intermediate.term_encoder,
            permitted_terms=permitted_terms
        ),
        filename="logistic_regression_bag_of_words"
    )
    export.upload_to_huggingface(
        path_in_repo=path,
        model_name="logistic_regression_bag_of_words",
    )

def get_indices(
    mask: np.ndarray
):
    return np.where(mask)[0]

def random_sample(
    indices: np.ndarray,
    size: int = 20
):
    return np.random.choice(
        indices,
        size=min(len(indices), size),
        replace=False
    )

def spot_check(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probability_estimates: np.ndarray,
    url_ids: np.ndarray,
    url_encoder: LabelEncoder,
    term_encoder: LabelEncoder,
    sparse_matrix: csr_matrix
):
    # masks
    tp_mask = (y_test == 1) & (y_pred == 1)
    tn_mask = (y_test == 0) & (y_pred == 0)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    # Least confident predictions
    uncertainty = np.abs(probability_estimates - 0.5)
    least_confident_indices = np.argsort(uncertainty)[:5]

    tp_indices = get_indices(tp_mask)
    tn_indices = get_indices(tn_mask)
    fp_indices = get_indices(fp_mask)
    fn_indices = get_indices(fn_mask)

    tp_sample = random_sample(tp_indices)
    tn_sample = random_sample(tn_indices)
    fp_sample = random_sample(fp_indices)
    fn_sample = random_sample(fn_indices)

    least_confident_sample = random_sample(least_confident_indices)

    print_header("TRUE POSITIVES")
    for i in tp_sample:
        print_example(i, y_test[i], y_pred[i], probability_estimates[i], url_encoder, term_encoder, url_ids, sparse_matrix)
    print_header("TRUE NEGATIVES")
    for i in tn_sample:
        print_example(i, y_test[i], y_pred[i], probability_estimates[i], url_encoder, term_encoder, url_ids, sparse_matrix)
    print_header("FALSE POSITIVES")
    for i in fp_sample:
        print_example(i, y_test[i], y_pred[i], probability_estimates[i], url_encoder, term_encoder, url_ids, sparse_matrix)
    print_header("FALSE NEGATIVES")
    for i in fn_sample:
        print_example(i, y_test[i], y_pred[i], probability_estimates[i], url_encoder, term_encoder, url_ids, sparse_matrix)
    print_header("LEAST CONFIDENT")
    for i in least_confident_sample:
        print_example(i, y_test[i], y_pred[i], probability_estimates[i], url_encoder, term_encoder, url_ids, sparse_matrix)

def print_header(val):
    print("-" * 50)
    print(val)
    print("-" * 50)


def print_example(
    idx: int,
    label: bool,
    prediction: bool,
    probability: float,
    url_encoder: LabelEncoder,
    term_encoder: LabelEncoder,
    url_ids: np.ndarray,
    sparse_matrix: csr_matrix
):
    url_idx = url_ids[idx]
    url = url_encoder.inverse_transform([url_idx])[0]
    print(f"URL: {url}")
    print(f"Correct Label: {label}, Prediction: {prediction} (Confidence: {probability:.3f})")
    # Top 5 terms
    row = sparse_matrix.getrow(idx)
    nonzero_term_idxs = row.indices
    nonzero_term_scores = row.data
    sorted_indices = np.argsort(nonzero_term_scores)[::-1]  # highest first
    top_term_idxs = nonzero_term_idxs[sorted_indices[:5]]
    top_terms = term_encoder.inverse_transform(top_term_idxs)
    print("Top Terms: ", top_terms)
    print("-" * 50)


def apply_undersampling(x_train_resampled, y_train_resampled):
    rus = RandomUnderSampler(random_state=42)
    x_train_resampled, y_train_resampled = rus.fit_resample(
        x_train_resampled,
        y_train_resampled
    )
    return x_train_resampled, y_train_resampled


if __name__ == "__main__":
    run()