import hashlib
import json
import re
from pathlib import Path

from ml_ops_experiment.ingestion.common.data_models import NormalizedDocument


def _safe_relative_path(file_path: Path, root_dir: Path) -> str:
    try:
        return file_path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return file_path.name


def build_doc_id(file_path: Path, root_dir: Path) -> str:
    relative = _safe_relative_path(file_path, root_dir)
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
    safe_name = relative.replace("/", "_").replace(".", "_")
    return f"{safe_name}_{digest}"


def _extract_chapter_hint(text: str) -> str | None:
    for line in text.splitlines()[:15]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(chapter|ch\.)\s+\d+", stripped, flags=re.IGNORECASE):
            return stripped
    return None


def folder_context_for(file_path: Path, root_dir: Path) -> list[str]:
    relative = _safe_relative_path(file_path, root_dir)
    if "/" not in relative:
        return []
    return list(Path(relative).parts[:-1])

def _write_normalized(document: NormalizedDocument, processed_folder: Path) -> None:
    processed_folder.mkdir(parents=True, exist_ok=True)
    with open(processed_folder / f"{document.doc_id}.json", "w", encoding="utf-8") as f:
        json.dump(document.model_dump(mode="json"), f)
