import click
import os

import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import mlflow_tags
from mlflow.utils.logging_utils import eprint


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = MlflowClient()
    all_runs = reversed(client.search_runs([experiment_id]))
    for run in all_runs:
        tags = run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run.info.to_proto().status != RunStatus.FINISHED:
            eprint(
                (
                    "Run matched, but is not FINISHED, so skipping (run_id={}, status={})"
                ).format(run.info.run_id, run.info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                "Run matched, but has a different source version, so skipping "
                f"(found={previous_version}, expected={git_commit})"
            )
            continue
        return client.get_run(run.info.run_id)
    eprint("No matching run has been found.")
    return None


def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    """Get an existing run or launch a new one for the given entrypoint."""
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print(
            f"Found existing run for entrypoint={entrypoint} and parameters={parameters}"
        )
        return existing_run
    print(f"Launching new run for entrypoint={entrypoint} and parameters={parameters}")
    submitted_run = mlflow.run(
        ".",
        entrypoint,
        parameters=parameters,
        env_manager="local",
        experiment_id=_get_experiment_id(),
    )
    return MlflowClient().get_run(submitted_run.run_id)


@click.command()
@click.option(
    "--use-cache", default=True, type=bool, help="Use cached runs if available"
)
def workflow(use_cache):
    """Main workflow to orchestrate the data curation pipeline."""
    with mlflow.start_run(run_name="data_curation_pipeline") as active_run:
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        mlflow_pipeline_id = active_run.info.run_id

        # Step 1a: Convert raw PDFs to normalized JSON format
        print("=" * 60)
        print("Step 1a: Converting raw PDFs to normalized JSON format")
        print("=" * 60)
        pdf_to_normalized_json = _get_or_run(
            "normalize_pdf_to_json",
            {"mlflow_pipeline_id": mlflow_pipeline_id},
            git_commit,
            use_cache,
        )

        # Step 1b: Convert raw TXT files to normalized JSON format
        print("=" * 60)
        print("Step 1b: Converting raw TXT files to normalized JSON format")
        print("=" * 60)
        txt_to_normalized_json = _get_or_run(
            "normalize_txt_to_json",
            {"mlflow_pipeline_id": mlflow_pipeline_id},
            git_commit,
            use_cache,
        )

        # Step 2: Convert normalized JSON to JSONL chunks
        print("=" * 60)
        print("Step 2: Converting JSON to JSONL chunks")
        print("=" * 60)
        normalized_to_chunks_run = _get_or_run(
            "normalized_to_chunks",
            {"mlflow_pipeline_id": mlflow_pipeline_id},
            git_commit,
            use_cache,
        )

        # Step 3: Upload JSONL chunks to database
        print("=" * 60)
        print("Step 3: Uploading JSONL chunks to database")
        print("=" * 60)
        chunks_to_db_run = _get_or_run(
            "chunks_to_db",
            {"mlflow_pipeline_id": mlflow_pipeline_id},
            git_commit,
            use_cache,
        )

        # Step 4: Create embeddings and vector index
        print("=" * 60)
        print("Step 4: Creating embeddings and vector index")
        print("=" * 60)
        create_embeddings_run = _get_or_run(
            "indexing",
            {"mlflow_pipeline_id": mlflow_pipeline_id},
            git_commit,
            use_cache,
        )

        # # Step 5: Run ml_ops_experiment.retrieval query
        # print("=" * 60)
        # print("Step 5: Running retrieval query")
        # print("=" * 60)
        # retrieval_query = "What are the main topics in the indexed documents?"
        # os.environ["RETRIEVAL_QUERY"] = retrieval_query
        # retrieval_run = _get_or_run(
        #     "retrieval",
        #     {
        #         "mlflow_pipeline_id": mlflow_pipeline_id,
        #     },
        #     git_commit,
        #     use_cache,
        # )

        # Step 5: Register prompts to MLflow Prompt Registry
        print("=" * 60)
        print("Step 6: Registering prompts to MLflow Prompt Registry")
        print("=" * 60)
        register_prompts_run = _get_or_run(
            "register_prompts",
            {
                "mlflow_pipeline_id": mlflow_pipeline_id,
            },
            git_commit,
            use_cache,
        )

        # Step 5: Evaluate prompts using evaluation dataset
        print("=" * 60)
        print("Step 7: Evaluating prompts with evaluation dataset")
        print("=" * 60)
        evaluate_prompts_run = _get_or_run(
            "evaluate_prompts",
            {
                "mlflow_pipeline_id": mlflow_pipeline_id,
                "dataset_path": "data/eval/python_eval_dataset.json",
                "prompts_to_evaluate": "python_assistant_v1,python_assistant_v2,python_assistant_v3,python_assistant_v4,python_assistant_v5,python_assistant_v6",
                "max_cases": 10,  # Configurable: 1-50 evaluation cases
            },
            git_commit,
            use_cache,
        )

        # End of the pipeline
        print("=" * 60)
        print("Pipeline completed successfully!")
        print(f"Pipeline run ID: {mlflow_pipeline_id}")
        print("=" * 60)


if __name__ == "__main__":
    mlflow.set_experiment("Bootcamp")
    workflow()
