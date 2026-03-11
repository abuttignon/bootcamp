from pathlib import Path


def get_project_root() -> Path:
    return Path.cwd().resolve()


def get_external_dir(root_dir: Path | None = None) -> Path:
    if root_dir is None:
        root_dir = get_project_root()
    return root_dir / "data" / "external"


def get_interim_dir(mlflow_pipeline_id: str, root_dir: Path | None = None) -> Path:
    if root_dir is None:
        root_dir = get_project_root()
    return root_dir / "data" / "interim" / mlflow_pipeline_id


def get_processed_dir(mlflow_pipeline_id: str, root_dir: Path | None = None) -> Path:
    if root_dir is None:
        root_dir = get_project_root()
    return root_dir / "data" / "processed" / mlflow_pipeline_id
