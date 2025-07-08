from pathlib import Path


def find_repo_root(start_path: Path = Path.cwd()) -> Path:
    """Finds the root of the repo by locating the nearest pyproject.toml."""
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("No pyproject.toml found in any parent directories.")

def get_data_joblib_path() -> Path:
    repo_root = find_repo_root()
    return repo_root / "data" / "joblib"