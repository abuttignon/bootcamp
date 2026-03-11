from pathlib import Path

import click
import fitz
import mlflow

from ml_ops_experiment.common.paths_getter import get_project_root, get_external_dir, get_interim_dir
from ml_ops_experiment.ingestion.common.converters import _extract_chapter_hint, build_doc_id, _safe_relative_path, \
    folder_context_for, _write_normalized
from ml_ops_experiment.ingestion.common.data_models import NormalizedDocument, TextSegment


def load_pdf_document(file_path: Path, root_dir: Path) -> NormalizedDocument:
    with fitz.open(file_path) as pdf_doc:
        segments: list[TextSegment] = []
        chapter_hints: list[str] = []
        for idx, page in enumerate(pdf_doc, start=1):
            text = page.get_text("text")
            if text.strip():
                chapter_hint = _extract_chapter_hint(text)
                if chapter_hint and chapter_hint not in chapter_hints:
                    chapter_hints.append(chapter_hint)

                section_path = [chapter_hint] if chapter_hint else [f"page-{idx}"]
                segments.append(TextSegment(text=text, section_path=section_path, page=idx))

        return NormalizedDocument(
            doc_id=build_doc_id(file_path, root_dir),
            title=file_path.stem.replace("_", " ").strip() or file_path.stem,
            source_path=_safe_relative_path(file_path, root_dir),
            source_type="pdf",
            folder_context=folder_context_for(file_path, root_dir),
            chapter_hints=chapter_hints,
            page_count=pdf_doc.page_count,
            segments=segments,
        )

@click.command()
@click.option("--mlflow_pipeline_id", type=str, required=True)
def main(mlflow_pipeline_id: str):
    root_dir = get_project_root()
    external_dir = get_external_dir(root_dir)
    interim_dir = get_interim_dir(mlflow_pipeline_id, root_dir)

    if not external_dir.exists():
        raise FileNotFoundError(f"External data directory not found: '{external_dir}'")

    pdf_files = sorted(external_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{external_dir}'")

    # Log parameters
    mlflow.log_param("input_dir", str(external_dir))
    mlflow.log_param("output_dir", str(interim_dir))
    mlflow.log_param("file_count", len(pdf_files))

    total_pages = 0
    total_segments = 0

    for input_pdf in pdf_files:
        doc = load_pdf_document(input_pdf, root_dir)
        total_pages += doc.page_count or 0
        total_segments += len(doc.segments)
        _write_normalized(doc, interim_dir)

    # Log metrics
    mlflow.log_metric("total_pdf_files", len(pdf_files))
    mlflow.log_metric("total_pages", total_pages)
    mlflow.log_metric("total_segments", total_segments)
    mlflow.log_metric("avg_pages_per_file", total_pages / len(pdf_files) if pdf_files else 0)

    # Log artifacts
    mlflow.log_artifacts(str(interim_dir), artifact_path="normalized_pdfs")


if __name__ == "__main__":
    with mlflow.start_run(run_name="normalize_pdf_to_json") as active_run:
        main()
