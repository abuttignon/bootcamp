import click
import json
import mlflow

from pathlib import Path

from ml_ops_experiment.ingestion.common.chunker import StructureAwareChunker
from ml_ops_experiment.common.paths_getter import get_project_root, get_interim_dir, get_processed_dir
from ml_ops_experiment.ingestion.common.data_models import ProcessedChunk


def _write_chunks(doc_id: str, chunks: list[ProcessedChunk], processed_path: Path) -> None:
    processed_path.mkdir(parents=True, exist_ok=True)
    output_file = processed_path / f"{doc_id}.jsonl"
    lines = [json.dumps(chunk.model_dump(mode="json"), ensure_ascii=True) for chunk in chunks]
    output_file.write_text("\n".join(lines), encoding="utf-8")


def _process_document_file(document_path: Path, chunker: StructureAwareChunker, processed_dir: Path) -> None:
    with open(document_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    chunks = chunker.chunk_document(document)
    _write_chunks(document["doc_id"], chunks, processed_dir)

@click.command()
@click.option("--mlflow_pipeline_id", type=str, required=True)
def main(mlflow_pipeline_id: str):
    root_dir = get_project_root()
    interim_dir = get_interim_dir(mlflow_pipeline_id, root_dir)
    processed_dir = get_processed_dir(mlflow_pipeline_id, root_dir)
    chunker = StructureAwareChunker()

    if not interim_dir.exists():
        raise FileNotFoundError(f"Interim directory not found: '{interim_dir}'")

    input_files = sorted(interim_dir.rglob("*.json"))
    if not input_files:
        raise FileNotFoundError(f"No normalized JSON files found in '{interim_dir}'")

    # Log parameters
    mlflow.log_param("input_dir", str(interim_dir))
    mlflow.log_param("output_dir", str(processed_dir))
    mlflow.log_param("file_count", len(input_files))
    mlflow.log_param("chunk_size", chunker.chunk_size)
    mlflow.log_param("chunk_overlap", chunker.chunk_overlap)

    total_chunks = 0

    for json_file in input_files:
        with open(json_file, "r", encoding="utf-8") as f:
            document = json.load(f)
        chunks = chunker.chunk_document(document)
        total_chunks += len(chunks)
        _write_chunks(document["doc_id"], chunks, processed_dir)

    # Log metrics
    mlflow.log_metric("total_documents", len(input_files))
    mlflow.log_metric("total_chunks", total_chunks)
    mlflow.log_metric("avg_chunks_per_doc", total_chunks / len(input_files) if input_files else 0)

    # Log artifacts
    mlflow.log_artifacts(str(processed_dir), artifact_path="processed_chunks")


if __name__ == "__main__":
   with mlflow.start_run(run_name="normalized_to_chunks") as active_run:
       main()
