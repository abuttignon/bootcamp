# converts txt files to
from pathlib import Path

import click
import mlflow

from ml_ops_experiment.ingestion.common.converters import (
    build_doc_id, folder_context_for, _write_normalized
)
from ml_ops_experiment.common.paths_getter import (
    get_project_root, get_external_dir, get_interim_dir
)
from ml_ops_experiment.ingestion.common.data_models import NormalizedDocument, TextSegment


def load_text_document(file_path: Path, root_dir: Path) -> NormalizedDocument:
    content = file_path.read_text(encoding="utf-8")
    return NormalizedDocument(
        doc_id=build_doc_id(file_path, root_dir),
        title=file_path.stem.replace("_", " ").strip() or file_path.stem,
        source_path=file_path.relative_to(root_dir).as_posix(),
        source_type="txt",
        folder_context=folder_context_for(file_path, root_dir),
        segments=[TextSegment(text=content, section_path=[file_path.stem])],
    )


@click.command()
@click.option("--mlflow_pipeline_id", type=str, required=True)
def main(mlflow_pipeline_id: str):
    root_dir = get_project_root()
    external_dir = get_external_dir(root_dir)
    interim_dir = get_interim_dir(mlflow_pipeline_id, root_dir)

    if not external_dir.exists():
        raise FileNotFoundError(f"External data directory not found: '{external_dir}'")

    txt_files = sorted(external_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No Text files found in '{external_dir}'")

    # Log parameters
    mlflow.log_param("input_dir", str(external_dir))
    mlflow.log_param("output_dir", str(interim_dir))
    mlflow.log_param("file_count", len(txt_files))

    total_chars = 0
    total_segments = 0

    for input_txt in txt_files:
        doc = load_text_document(input_txt, root_dir)
        total_segments += len(doc.segments)
        total_chars += sum(len(seg.text) for seg in doc.segments)
        _write_normalized(doc, interim_dir)

    # Log metrics
    mlflow.log_metric("total_txt_files", len(txt_files))
    mlflow.log_metric("total_segments", total_segments)
    mlflow.log_metric("total_characters", total_chars)
    mlflow.log_metric("avg_chars_per_file", total_chars / len(txt_files) if txt_files else 0)

    # Log artifacts
    mlflow.log_artifacts(str(interim_dir), artifact_path="normalized_txts")

if __name__ == "__main__":
   with mlflow.start_run(run_name="normalize_txt_to_json") as active_run:
       main()
