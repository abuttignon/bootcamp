import click
import json
import os
from pathlib import Path
import pymongo
from dotenv import load_dotenv
from openai import OpenAI
import mlflow


def create_vector_search_pipeline(query_embedding, num_results=3):
    return [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": num_results,
            }
        },
        {
            "$addFields": {
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]


def get_embedding(text, open_ai_client, embedding_model):
    embedding = (
        open_ai_client.embeddings.create(input=text, model=embedding_model)
        .data[0]
        .embedding
    )
    return embedding


def query_database(query, collection, open_ai_client, embedding_model, num_results):
    query_embedding = get_embedding(query, open_ai_client, embedding_model)
    pipeline = create_vector_search_pipeline(query_embedding, num_results=num_results)
    return list(collection.aggregate(pipeline))


def write_retrieved_results(
    mlflow_pipeline_id: str, query: str, results: list[dict]
) -> Path:
    root_dir = Path.cwd().resolve()
    output_dir = root_dir / "data" / "retrieved" / mlflow_pipeline_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "retrieved_results.json"

    payload = {
        "mlflow_pipeline_id": mlflow_pipeline_id,
        "query": query,
        "result_count": len(results),
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    return output_file


@click.command()
@click.option("--mlflow_pipeline_id", type=str, required=True)
@click.option(
    "--embedding-model",
    type=str,
    default="text-embedding-3-small",
    help="OpenAI embedding model",
)
@click.option("--query", type=str, default=None, help="User query text")
@click.option(
    "--num-results", type=int, default=3, help="Number of vector search results"
)
def main(
    mlflow_pipeline_id: str, embedding_model: str, query: str | None, num_results: int
):
    load_dotenv()

    # Initialize clients
    if query is None:
        query = os.getenv("RETRIEVAL_QUERY", "")

    if not query.strip():
        raise ValueError("Query must not be empty")

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
    mlflow.log_param("mlflow_pipeline_id", mlflow_pipeline_id)
    mlflow.log_param("embedding_model", embedding_model)
    mlflow.log_param("db_name", db_name)
    mlflow.log_param("collection_name", collection_name)
    mlflow.log_param("num_results", num_results)
    mlflow.log_param("query_length", len(query))

    results = query_database(
        query=query,
        collection=collection,
        open_ai_client=open_ai_client,
        embedding_model=embedding_model,
        num_results=num_results,
    )

    result_count = len(results)
    top_score = results[0].get("score", 0.0) if results else 0.0

    mlflow.log_metric("result_count", result_count)
    mlflow.log_metric("top_score", top_score)

    output_file = write_retrieved_results(mlflow_pipeline_id, query, results)
    mlflow.log_param("retrieved_output_file", str(output_file))
    mlflow.log_artifact(str(output_file), artifact_path="retrieved_results")

    print(f"Retrieved {result_count} result(s)")
    if results:
        print(f"Top score: {top_score}")
    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    with mlflow.start_run(run_name="retrieval_query"):
        main()
