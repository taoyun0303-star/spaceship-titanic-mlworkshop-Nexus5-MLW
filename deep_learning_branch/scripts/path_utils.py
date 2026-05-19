from pathlib import Path


def find_project_root(start=None):
    """Find the repository/project root containing the Kaggle data files."""
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate project root. Run from the project root or pass an explicit path."
    )


def get_dl_dir(project_root=None):
    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    dl_dir = root / "深度学习"
    if not dl_dir.exists():
        raise FileNotFoundError(f"Deep learning directory not found: {dl_dir}")
    return dl_dir


def default_rerun_dir(version, project_root=None):
    return get_dl_dir(project_root) / "outputs" / "rerun" / version
