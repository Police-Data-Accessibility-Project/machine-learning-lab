import pandas as pd
from pydantic import BaseModel, ConfigDict


class TrainTestDataframes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    test: pd.DataFrame
    train: pd.DataFrame