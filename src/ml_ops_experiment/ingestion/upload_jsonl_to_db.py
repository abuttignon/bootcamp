import click
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

from ml_ops_experiment.common.paths_getter import get_project_root, get_processed_dir
from ml_ops_experiment.ingestion.common.data_models import ProcessedChunk


@dataclass
class IngestStats:
    files_seen: int = 0
    lines_read: int = 0
    valid_docs: int = 0
    inserted: int = 0
    updated: int = 0
    parse_errors: int = 0


def _iter_jsonl_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.jsonl"))


def _flush_batch(
    collection: Collection, operations: list[UpdateOne], stats: IngestStats
) -> None:
    if not operations:
        return
    result = collection.bulk_write(operations, ordered=False)
    stats.inserted += result.upserted_count
    stats.updated += result.modified_count
    operations.clear()


def _to_operation(chunk: ProcessedChunk) -> UpdateOne:
    now = datetime.now(timezone.utc)
    doc = chunk.model_dump(mode="json")
    doc["_id"] = chunk.chunk_id
    doc["ingested_at"] = now
    doc["pipeline_stage"] = "preprocessing"
    return UpdateOne({"_id": chunk.chunk_id}, {"$set": doc}, upsert=True)


def upload_jsonl_dir_to_mongo(
    input_dir: Path, collection: Collection, batch_size: int = 500
) -> IngestStats:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    stats = IngestStats()
    files = _iter_jsonl_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No JSONL files found in '{input_dir}'")

    stats.files_seen = len(files)
    operations: list[UpdateOne] = []

    collection.create_index("doc_id")
    collection.create_index("source_type")
    collection.create_index([("doc_id", 1), ("chunk_index", 1)])

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                stats.lines_read += 1
                try:
                    payload = json.loads(line)
                    chunk = ProcessedChunk.model_validate(payload)
                    operations.append(_to_operation(chunk))
                    stats.valid_docs += 1
                except (json.JSONDecodeError, ValueError) as exc:
                    stats.parse_errors += 1
                    print(f"[warn] Skipping {file_path}:{line_no} ({exc})")
                    continue

                if len(operations) >= batch_size:
                    _flush_batch(collection, operations, stats)

    _flush_batch(collection, operations, stats)
    return stats


@click.command()
@click.option("--mlflow_pipeline_id", type=str, required=True)
@click.option("--batch-size", type=int, default=500, help="Bulk upsert batch size")
def main(mlflow_pipeline_id: str, batch_size: int):
    load_dotenv()
    root_dir = get_project_root()
    input_dir = get_processed_dir(mlflow_pipeline_id, root_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: '{input_dir}'")

    mongo_url = os.getenv("MONGO_DB_URL")
    if not mongo_url:
        raise EnvironmentError("Missing required environment variable: MONGO_DB_URL")

    db_name = os.getenv("MONGO_DB_NAME", "bootcamp")
    collection_name = os.getenv("MONGO_COLLECTION_NAME_RAG", "curated_chunks")

    client = MongoClient(mongo_url)
    collection = client[db_name][collection_name]

    # Log parameters
    mlflow.log_param("input_dir", str(input_dir))
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("db_name", db_name)
    mlflow.log_param("collection_name", collection_name)

    stats = upload_jsonl_dir_to_mongo(
        input_dir=input_dir, collection=collection, batch_size=batch_size
    )

    print("Upload completed.")
    print(f"Collection: {db_name}.{collection_name}")
    print(f"Files seen: {stats.files_seen}")
    print(f"Lines read: {stats.lines_read}")
    print(f"Valid docs: {stats.valid_docs}")
    print(f"Inserted: {stats.inserted}")
    print(f"Updated: {stats.updated}")
    print(f"Parse errors: {stats.parse_errors}")

    # Log metrics
    mlflow.log_metric("files_seen", stats.files_seen)
    mlflow.log_metric("lines_read", stats.lines_read)
    mlflow.log_metric("valid_docs", stats.valid_docs)
    mlflow.log_metric("inserted", stats.inserted)
    mlflow.log_metric("updated", stats.updated)
    mlflow.log_metric("parse_errors", stats.parse_errors)
    mlflow.log_metric("success_rate", stats.valid_docs / stats.lines_read if stats.lines_read > 0 else 0)

    if stats.valid_docs == 0:
        raise RuntimeError("No valid chunk documents were uploaded")


if __name__ == "__main__":
    with mlflow.start_run(run_name="chunks_to_db"):
        main()
