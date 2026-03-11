"""
Evaluate prompts using MLflow's evaluation framework.

This module evaluates registered prompts against the evaluation dataset
following MLflow best practices for prompt evaluation.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import click
import mlflow
import mlflow.data
import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from ml_ops_experiment.common.mlflow_utils import save_artifact


def load_evaluation_dataset(dataset_path: str, max_cases: int = 10) -> pd.DataFrame:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])

    # Validate and limit max_cases
    max_cases = max(1, min(max_cases, 50))  # Clamp between 1 and 50

    eval_data = []
    for case in cases[:max_cases]:
        eval_data.append(
            {
                "id": case["id"],
                "topic": case.get(
                    "topic", "general"
                ),  # e.g., 'numpy', 'pandas', 'syntax'
                "difficulty": case.get("difficulty", "Unknown"),
                "question": case.get("question", ""),
                "expected_route": case.get("expected_route", "docs"),
                "requires_citation": case.get("requires_citation", True),
                "expected_output": case.get("gold_answer", ""),
                "must_include": case.get(
                    "must_include", []
                ),  # Key terms that must appear
                "forbidden": case.get("forbidden", []),  # Terms that should not appear
            }
        )

    return pd.DataFrame(eval_data)


def initialize_llm(model_name: str, temperature: float = 0.0) -> BaseChatModel:
    if model_name.startswith("gpt-") or model_name.startswith("o1-"):
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.startswith("claude-"):
        return ChatAnthropic(model_name=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def generate_answer(question: str, system_prompt: str, llm: BaseChatModel) -> str:
    # Use direct message construction to avoid template variable parsing
    # The system_prompt may contain curly braces that should be literal
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

    chain = llm | StrOutputParser()
    return chain.invoke(messages)


def evaluate_answer(
    answer: str, must_include: List[str], forbidden: List[str]
) -> Dict[str, Any]:
    answer_lower = answer.lower()

    # Check must_include terms
    must_include_found = [term for term in must_include if term.lower() in answer_lower]
    must_include_missing = [
        term for term in must_include if term.lower() not in answer_lower
    ]
    must_include_score = (
        len(must_include_found) / len(must_include) if must_include else 1.0
    )

    # Check forbidden terms
    forbidden_found = [term for term in forbidden if term.lower() in answer_lower]
    forbidden_score = 0.0 if forbidden_found else 1.0

    # Combined score (weighted average)
    combined_score = must_include_score * 0.7 + forbidden_score * 0.3

    return {
        "must_include_score": must_include_score,
        "must_include_found": must_include_found,
        "must_include_missing": must_include_missing,
        "forbidden_score": forbidden_score,
        "forbidden_found": forbidden_found,
        "combined_score": combined_score,
    }


def evaluate_prompt_version(
    prompt_name: str,
    eval_df: pd.DataFrame,
    mlflow_pipeline_id: str,
    model_name: str = None,
) -> Dict[str, Any]:
    try:
        # Load the prompt from registry
        prompt = mlflow.genai.load_prompt(
            name_or_uri=prompt_name,
        )

        print(f"Evaluating {prompt_name}")
        print(f"Dataset size: {len(eval_df)} cases")

        # Get model name from environment if not provided
        if model_name is None:
            model_name = os.environ.get("MODEL_NAME", "gpt-5.2")

        # Initialize LLM for evaluation
        llm = initialize_llm(model_name, temperature=0.0)
        print(f"Using model: {model_name}")

        # Evaluate each case
        case_results = []
        total_combined_score = 0.0
        total_must_include_score = 0.0
        total_forbidden_score = 0.0

        for idx, row in eval_df.iterrows():
            try:
                # Generate answer
                answer = generate_answer(
                    question=row["question"], system_prompt=prompt.template, llm=llm
                )

                # Evaluate answer
                eval_metrics = evaluate_answer(
                    answer=answer,
                    must_include=row["must_include"],
                    forbidden=row["forbidden"],
                )

                case_result = {
                    "case_id": row["id"],
                    "topic": row["topic"],
                    "difficulty": row["difficulty"],
                    "question": row["question"],
                    "generated_answer": answer,
                    "expected_output": row["expected_output"],
                    **eval_metrics,
                }
                case_results.append(case_result)

                total_combined_score += eval_metrics["combined_score"]
                total_must_include_score += eval_metrics["must_include_score"]
                total_forbidden_score += eval_metrics["forbidden_score"]

                print(
                    f"  ✓ Case {row['id']}: combined={eval_metrics['combined_score']:.2f}"
                )

            except Exception as e:
                print(f"  ✗ Case {row['id']} failed: {e}")
                case_results.append(
                    {
                        "case_id": row["id"],
                        "topic": row["topic"],
                        "difficulty": row["difficulty"],
                        "question": row["question"],
                        "error": str(e),
                        "combined_score": 0.0,
                    }
                )

        # Compute aggregate metrics
        num_cases = len(eval_df)
        avg_combined_score = total_combined_score / num_cases if num_cases > 0 else 0.0
        avg_must_include_score = (
            total_must_include_score / num_cases if num_cases > 0 else 0.0
        )
        avg_forbidden_score = (
            total_forbidden_score / num_cases if num_cases > 0 else 0.0
        )

        results = {
            "prompt_name": prompt_name,
            "model_name": model_name,
            "dataset_size": num_cases,
            "evaluation_time": datetime.now().isoformat(),
            "avg_combined_score": avg_combined_score,
            "avg_must_include_score": avg_must_include_score,
            "avg_forbidden_score": avg_forbidden_score,
            "case_results": case_results,
            "status": "completed",
        }

        # Log aggregate metrics to MLflow
        mlflow.log_metric(f"{prompt_name}_avg_combined_score", avg_combined_score)
        mlflow.log_metric(
            f"{prompt_name}_avg_must_include_score", avg_must_include_score
        )
        mlflow.log_metric(f"{prompt_name}_avg_forbidden_score", avg_forbidden_score)
        mlflow.log_metric(f"{prompt_name}_dataset_size", num_cases)

        print(f"\n  Average scores:")
        print(f"    Combined: {avg_combined_score:.3f}")
        print(f"    Must-include: {avg_must_include_score:.3f}")
        print(f"    Forbidden: {avg_forbidden_score:.3f}")

        return results

    except Exception as e:
        print(f"Error evaluating {prompt_name}: {e}")
        return {
            "prompt_name": prompt_name,
            "status": "failed",
            "error": str(e),
        }


@click.command()
@click.option("--mlflow_pipeline_id", type=str, required=True)
@click.option(
    "--dataset-path",
    type=str,
    default="data/eval/python_eval_dataset.json",
    help="Path to evaluation dataset",
)
@click.option(
    "--prompts-to-evaluate",
    type=str,
    default="python_assistant_v1,python_assistant_v2,python_assistant_v3,python_assistant_v4,python_assistant_v5,python_assistant_v6",
    help="Comma-separated list of prompt names to evaluate",
)
@click.option(
    "--model-name",
    type=str,
    default=None,
    help="Model to use for evaluation (defaults to MODEL_NAME env var)",
)
@click.option(
    "--max-cases",
    type=int,
    default=10,
    help="Maximum number of evaluation cases to run (1-50)",
)
def main(
    mlflow_pipeline_id: str,
    dataset_path: str,
    prompts_to_evaluate: str,
    model_name: str,
    max_cases: int,
):
    """
    Evaluate registered prompts against evaluation dataset.

    - Load prompts from registry using load_prompt()
    - Log evaluation dataset as MLflow Dataset
    - Evaluate against structured Q&A dataset
    - Log evaluation metrics and results
    - Compare versions using MLflow's tracking
    """
    # Log parameters
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("prompts_to_evaluate", prompts_to_evaluate)
    mlflow.log_param("max_cases", max_cases)
    if model_name:
        mlflow.log_param("model_name", model_name)

    # Load evaluation dataset
    print(f"\n{'=' * 60}")
    print(f"Loading evaluation dataset from {dataset_path}")
    print(f"Max cases: {max_cases}")
    print(f"{'=' * 60}\n")

    try:
        eval_df = load_evaluation_dataset(dataset_path, max_cases=max_cases)
        print(f"✓ Loaded {len(eval_df)} evaluation cases")
        mlflow.log_metric("total_eval_cases", len(eval_df))

        # Create and log dataset to MLflow
        dataset = mlflow.data.from_pandas(
            eval_df,
            source=dataset_path,
            name="evaluation_dataset",
            targets="expected_output",
        )
        mlflow.log_input(dataset, context="evaluation")

        # Register in dataset registry
        client = mlflow.MlflowClient()
        experiment_id = mlflow.active_run().info.experiment_id

        try:
            existing = client.search_datasets(
                filter_string=f"name = 'evaluation_dataset'"
            )
            if not existing:
                client.create_dataset(
                    name="evaluation_dataset",
                    experiment_id=[experiment_id],
                    tags={"source": dataset_path, "cases": str(len(eval_df))},
                )
                print(f"✓ Registered dataset in MLflow")
            else:
                client.add_dataset_to_experiments(
                    dataset_id=existing[0].id, experiment_ids=[experiment_id]
                )
                print(f"✓ Dataset associated with experiment")
        except Exception as e:
            print(f"  Note: Dataset registry (experimental): {e}")

        print(f"✓ Logged dataset to MLflow")

    except Exception as e:
        print(f"✗ Failed to load evaluation dataset: {e}")
        raise

    # Parse prompts to evaluate
    prompt_names = [p.strip() for p in prompts_to_evaluate.split(",")]

    evaluation_results = []

    for prompt_name in prompt_names:
        try:
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {prompt_name}")
            print(f"{'=' * 60}\n")

            result = evaluate_prompt_version(
                prompt_name=prompt_name,
                eval_df=eval_df,
                mlflow_pipeline_id=mlflow_pipeline_id,
                model_name=model_name,
            )

            evaluation_results.append(result)

        except Exception as e:
            print(f"✗ Error evaluating {prompt_name}: {e}")
            evaluation_results.append(
                {
                    "prompt_name": prompt_name,
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Compute summary metrics
    successful = sum(1 for r in evaluation_results if r.get("status") == "completed")
    failed = len(evaluation_results) - successful

    mlflow.log_metric("prompts_evaluated", successful)
    mlflow.log_metric("prompts_failed_eval", failed)

    # Save artifacts
    artifacts_dir = Path("mlflow_artifacts") / "prompt_evaluation" / mlflow_pipeline_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation results
    results_path = save_artifact(
        artifacts_dir,
        "evaluation_results.json",
        {
            "results": evaluation_results,
            "summary": {
                "total_prompts": len(prompt_names),
                "successful": successful,
                "failed": failed,
                "timestamp": datetime.now().isoformat(),
            },
        },
    )
    mlflow.log_artifact(str(results_path))

    # Save evaluation dataset sample
    dataset_sample_path = save_artifact(
        artifacts_dir, "evaluation_dataset_sample.json", eval_df.head(5)
    )
    mlflow.log_artifact(str(dataset_sample_path))

    print(f"\n{'=' * 60}")
    print(f"Prompt evaluation completed")
    print(f"Successfully evaluated: {successful}/{len(prompt_names)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    with mlflow.start_run(run_name="evaluate_prompts"):
        load_dotenv()
        main()
