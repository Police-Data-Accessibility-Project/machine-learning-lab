from sklearn.preprocessing import LabelEncoder

from src.models._params.csr_matrix import CSRMatrixParams
from src.models.bag_of_words.data_structures.dataframe import BagOfWordsDataFrame
from src.models.bag_of_words.data_structures.intermediate import BagOfWordsIntermediate

import polars as pl

def format_bag_of_words(bow_df: BagOfWordsDataFrame) -> BagOfWordsIntermediate:
    # Drop url column


    url_encoder = LabelEncoder()
    term_encoder = LabelEncoder()


    url_index_lable = "url_idx"
    term_index_lable = "term_idx"

    url_indices = pl.Series(
        name=url_index_lable,
        values=url_encoder.fit_transform(bow_df.url)
    )
    term_indices = pl.Series(
        name=term_index_lable,
        values=term_encoder.fit_transform(bow_df.term)
    )

    bow_df.df = bow_df.df.with_columns(
        url_indices,
        term_indices
    )

    # Convert tf_idf to one-hot encode
    one_hot_encoded = (bow_df.tf_idf > 0).cast(pl.UInt8)




    params = CSRMatrixParams(
        data=one_hot_encoded.to_numpy(),
        row=bow_df.df[url_index_lable].to_numpy(),
        col=bow_df.df[term_index_lable].to_numpy(),
        shape=(
            bow_df.df[url_index_lable].n_unique(),
            bow_df.df[term_index_lable].n_unique()
        )
    )

    sparse_matrix = params.to_csr()

    # Extract labels
    bow_df.df = bow_df.df.unique(subset=[
        url_index_lable
    ]).sort(url_index_lable)

    return BagOfWordsIntermediate(
        sparse_matrix=sparse_matrix,
        urls_ids=url_encoder.inverse_transform(range(len(url_encoder.classes_))),
        terms_ids=term_encoder.inverse_transform(range(len(term_encoder.classes_))),
        y_relevant=bow_df.relevant,
        y_fine=bow_df.record_type_fine,
        y_coarse=bow_df.record_type_coarse,
        url_encoder=url_encoder,
        term_encoder=term_encoder
    )