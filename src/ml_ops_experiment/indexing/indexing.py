import os
from pathlib import Path

import click
import mlflow
import pymongo
from openai import OpenAI
from pymongo import UpdateOne
from tqdm import tqdm
from dotenv import load_dotenv

from ml_ops_experiment.common.mlflow_utils import save_artifact


VECTOR_INDEX_CONFIG = {
    "name": "vector_index",
    "definition": {
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {
                    "type": "knnVector",
                    "dimensions": 1536,
                    "similarity": "dotProduct",
                }
            },
        }
    },
}


def get_embedding(text, openAIClient, embedding_model):
    embedding = (
        openAIClient.embeddings.create(input=text, model=embedding_model)
        .data[0]
        .embedding
    )
    return embedding


@click.command()
@click.option("--mlflow_pipeline_id", type=str, required=True)
@click.option(
    "--embedding-model",
    type=str,
    default="text-embedding-3-small",
    help="OpenAI embedding model",
)
@click.option(
    "--batch-size", type=int, default=100, help="Batch size for bulk operations"
)
def main(mlflow_pipeline_id: str, embedding_model: str, batch_size: int):
    load_dotenv()

    # Initialize clients
    open_ai_client = OpenAI()
    embedding_model = os.getenv("EMBEDDING_MODEL", embedding_model)
    db_url = os.getenv("MONGO_DB_URL")
    db_name = os.getenv("MONGO_DB_NAME", "bootcamp")
    collection_name = os.getenv("MONGO_COLLECTION_NAME_RAG", "curated_chunks")

    if not db_url:
        raise EnvironmentError("Missing required environment variable: MONGO_DB_URL")

    db_client = pymongo.MongoClient(db_url)
    collection = db_client[db_name][collection_name]

    # Log parameters
    mlflow.log_param("embedding_model", embedding_model)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("db_name", db_name)
    mlflow.log_param("collection_name", collection_name)

    # Find documents without embeddings
    total_docs = collection.count_documents({})
    docs_without_embedding = collection.count_documents(
        {"embedding": {"$exists": False}}
    )

    print(f"Total documents: {total_docs}")
    print(f"Documents without embeddings: {docs_without_embedding}")

    mlflow.log_metric("total_documents", total_docs)
    mlflow.log_metric("docs_without_embedding", docs_without_embedding)

    # Process documents
    documents = collection.find({"embedding": {"$exists": False}})
    operations = []
    processed_count = 0
    error_count = 0
    error_log = []
    sample_embeddings = []

    for doc in tqdm(
        documents, total=docs_without_embedding, desc="Generating embeddings"
    ):
        try:
            # Generate embeddings for this document
            text = doc.get("text", "")
            if not text:
                error_count += 1
                error_log.append(
                    {"doc_id": str(doc.get("_id")), "error": "Empty text field"}
                )
                continue

            embedding = get_embedding(
                text, openAIClient=open_ai_client, embedding_model=embedding_model
            )

            # Store first 3 embeddings as samples
            if len(sample_embeddings) < 3:
                sample_embeddings.append(
                    {
                        "doc_id": str(doc.get("_id")),
                        "text_preview": text[:100] + "..." if len(text) > 100 else text,
                        "embedding_dims": len(embedding),
                        "embedding_sample": embedding[:5],  # First 5 dimensions
                    }
                )

            # Add the update operation to the list
            operations.append(
                UpdateOne({"_id": doc["_id"]}, {"$set": {"embedding": embedding}})
            )
            processed_count += 1

            # Execute batch operation
            if len(operations) >= batch_size:
                collection.bulk_write(operations)
                operations.clear()

        except Exception as e:
            print(f"Error processing document {doc.get('_id')}: {e}")
            error_count += 1
            error_log.append({"doc_id": str(doc.get("_id")), "error": str(e)})
            continue

    # Execute remaining operations
    if operations:
        collection.bulk_write(operations)
        operations.clear()

    # Create vector search index if it doesn't exist
    index_created = False
    try:
        existing_indexes = list(collection.list_search_indexes())
        index_exists = any(
            idx.get("name") == VECTOR_INDEX_CONFIG["name"] for idx in existing_indexes
        )

        if not index_exists:
            print("Creating vector search index...")
            collection.create_search_index(VECTOR_INDEX_CONFIG)
            print("Vector search index created successfully.")
            index_created = True
        else:
            print("Vector search index already exists.")
    except Exception as e:
        print(f"Note: Could not create/check search index: {e}")

    mlflow.log_param("index_created", index_created)

    # Log final metrics
    mlflow.log_metric("processed_count", processed_count)
    mlflow.log_metric("error_count", error_count)
    mlflow.log_metric(
        "success_rate",
        processed_count / docs_without_embedding if docs_without_embedding > 0 else 1.0,
    )

    print(f"Indexing completed.")
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")

    # Create artifacts directory and save artifacts
    artifacts_dir = (
        Path("mlflow_artifacts") / "ml_ops_experiment.indexing" / mlflow_pipeline_id
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Log artifacts
    if error_log:
        mlflow.log_artifact(
            str(save_artifact(artifacts_dir, "error_log.json", error_log))
        )

    if sample_embeddings:
        mlflow.log_artifact(
            str(
                save_artifact(
                    artifacts_dir, "sample_embeddings.json", sample_embeddings
                )
            )
        )

    mlflow.log_artifact(
        str(
            save_artifact(
                artifacts_dir, "vector_index_config.json", VECTOR_INDEX_CONFIG
            )
        )
    )


if __name__ == "__main__":
    with mlflow.start_run(run_name="ml_ops_experiment.indexing"):
        main()
