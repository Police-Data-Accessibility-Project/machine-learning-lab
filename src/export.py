from joblib import dump
import datetime

from src.util.path import get_data_joblib_path


def to_joblib(obj, filename: str):
    # Get timestamp string
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_filename = f"{filename}_{timestamp}.joblib"
    path = get_data_joblib_path() / full_filename
    dump(obj, path)
    print(f"Saved {filename} ({obj.__class__.__name__}) to {path}")
