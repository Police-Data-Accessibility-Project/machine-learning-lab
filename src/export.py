from pathlib import Path

from environs import Env
from huggingface_hub import upload_file
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
    return path

def upload_to_huggingface(
    path_in_repo: Path,
    model_name: str,
):
    env = Env()
    env.read_env()

    # Get version timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = f"models/{model_name}/{timestamp}/model.joblib"

    upload_file(
        path_or_fileobj=path_in_repo,
        path_in_repo=full_path,
        repo_id="PDAP/url-relevance-models",
        repo_type="model",
        token=env.str("HUGGINGFACE_TOKEN")
    )