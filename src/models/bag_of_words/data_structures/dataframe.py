
import polars as pl

class BagOfWordsDataFrame:

    def __init__(self, df: pl.DataFrame):
        self.df = df

    @property
    def url(self) -> pl.Series:
        return self.df["url"]

    @property
    def term(self) -> pl.Series:
        return self.df["term"]

    @property
    def tf_idf(self) -> pl.Series:
        return self.df["tf_idf"]

    @property
    def relevant(self) -> pl.Series:
        return self.df["relevant"]

    @property
    def record_type_fine(self) -> pl.Series:
        return self.df["record_type_fine"]

    @property
    def record_type_coarse(self) -> pl.Series:
        return self.df["record_type_coarse"]