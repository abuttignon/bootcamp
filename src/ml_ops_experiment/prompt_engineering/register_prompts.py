"""Register Python assistant prompt variants in the MLflow Prompt Registry.

Prompts are loaded from ``data.raw.prompts_python`` and registered as
``python_assistant_v1`` through ``python_assistant_v6``.
"""

import click
import mlflow
from pathlib import Path
from datetime import datetime

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from data.raw.prompts_python import (
    SYSTEM_INSTRUCTIONS_1,
    SYSTEM_INSTRUCTIONS_2,
    SYSTEM_INSTRUCTIONS_3,
    SYSTEM_INSTRUCTIONS_4,
    SYSTEM_INSTRUCTIONS_5,
    SYSTEM_INSTRUCTIONS_6,
)
from ml_ops_experiment.common.mlflow_utils import save_artifact


PROMPT_TEMPLATES = {
    "python_assistant_v1": {
        "template": SYSTEM_INSTRUCTIONS_1,
        "description": "Basic Python assistant - answers questions from Python 3.13 docs and Data Science Handbook",
        "commit_message": "Initial version: Basic Python and data science Q&A assistant",
    },
    "python_assistant_v2": {
        "template": SYSTEM_INSTRUCTIONS_2,
        "description": "Precise Python tutor - strictly context-based answers with explicit source validation",
        "commit_message": "Add strict context validation and version hints for Python 3.13",
    },
    "python_assistant_v3": {
        "template": SYSTEM_INSTRUCTIONS_3,
        "description": "Data Science engineer - practical code examples with step-by-step explanations",
        "commit_message": "Add practical code examples with documentation references",
    },
    "python_assistant_v4": {
        "template": SYSTEM_INSTRUCTIONS_4,
        "description": "Self-validating assistant - automatic 3-step verification of answers",
        "commit_message": "Add self-validation loop with context support and syntax checks",
    },
    "python_assistant_v5": {
        "template": SYSTEM_INSTRUCTIONS_5,
        "description": "Two-variant assistant - best effort answer and honest gap identification",
        "commit_message": "Add dual-variant output for insufficient context scenarios",
    },
    "python_assistant_v6": {
        "template": SYSTEM_INSTRUCTIONS_6,
        "description": "Professional Python assistant - JSON schema output with comprehensive validation",
        "commit_message": "Add strict JSON schema with context verification and code examples",
    },
}


@click.command()
@click.option("--mlflow_pipeline_id", type=str, required=True)
def main(mlflow_pipeline_id: str):
    """
    Register all prompt versions to MLflow Prompt Registry.

    Following MLflow best practices:
    - Each prompt version is registered with immutable versioning
    - Metadata includes descriptions and commit messages
    """
    # Log parameters
    mlflow.log_param("num_prompt_versions", len(PROMPT_TEMPLATES))

    registration_summary = []

    for prompt_name, prompt_config in PROMPT_TEMPLATES.items():
        try:
            print(f"\n{'=' * 60}")
            print(f"Registering: {prompt_name}")
            print(f"{'=' * 60}")

            # Register prompt with MLflow
            # This creates an immutable version in the registry
            prompt = mlflow.genai.register_prompt(
                name=prompt_name,
                template=prompt_config["template"],
                commit_message=prompt_config["commit_message"],
                tags={
                    "description": prompt_config["description"],
                    "version_type": prompt_name.split("_")[-1],  # e.g., "v6"
                    "registration_time": datetime.now().isoformat(),
                    "pipeline_id": mlflow_pipeline_id,
                },
            )

            print(f"✓ Registered {prompt_name} as version {prompt.version}")

            registration_summary.append(
                {
                    "prompt_name": prompt_name,
                    "version": prompt.version,
                    "description": prompt_config["description"],
                }
            )

        except Exception as e:
            print(f"✗ Error registering {prompt_name}: {e}")
            registration_summary.append(
                {
                    "prompt_name": prompt_name,
                    "error": str(e),
                }
            )

    # Log metrics
    successful = sum(1 for item in registration_summary if "error" not in item)
    mlflow.log_metric("prompts_registered", successful)
    mlflow.log_metric("prompts_failed", len(registration_summary) - successful)

    # Save artifacts
    artifacts_dir = Path("mlflow_artifacts") / "prompt_registry" / mlflow_pipeline_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = save_artifact(
        artifacts_dir,
        "registration_summary.json",
        {
            "summary": registration_summary,
            "total_prompts": len(PROMPT_TEMPLATES),
            "successful": successful,
            "failed": len(registration_summary) - successful,
            "timestamp": datetime.now().isoformat(),
        },
    )

    mlflow.log_artifact(str(artifact_path))

    print(f"\n{'=' * 60}")
    print(f"Prompt registration completed")
    print(f"Successfully registered: {successful}/{len(PROMPT_TEMPLATES)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    with mlflow.start_run(run_name="register_prompts"):
        main()
