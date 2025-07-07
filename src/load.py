import polars as pl
from joblib import Memory, expires_after

memory = Memory("cache_dir", verbose=0)


@memory.cache(
    cache_validation_callback=expires_after(
        days=1
    )
)
def from_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)