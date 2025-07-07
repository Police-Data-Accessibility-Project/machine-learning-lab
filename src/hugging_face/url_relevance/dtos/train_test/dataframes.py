from pydantic import BaseModel, ConfigDict
import polars as pl

class TrainTestDataframes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    test: pl.DataFrame
    train: pl.DataFrame