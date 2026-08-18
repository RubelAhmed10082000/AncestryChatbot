import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "extract": ["scripts.run_extract"],
    "hard-negatives": ["scripts.add_hard_negative_candidates"],
    "transform": ["scripts.run_transform"],
    "create-cases": ["scripts.create_evaluation_cases"],
    "evaluate-retrieval": ["scripts.run_retrieval_evaluation"],
    "evaluate-tree": ["scripts.run_tree_evaluation"],
    "search": ["app.retrieval.candidate_retrieval"],
    "confidence": ["app.scoring.confidence_scoring"],
    "tree": ["app.tree.generate_family_tree"],
    "serve": ["uvicorn", "app.api.main:app"],
    "test": ["pytest"],
}

PIPELINE = [
    ("transform", []),
    ("create-cases", []),
    ("evaluate-retrieval", ["--output-dir", "data/evaluation/final"]),
    ("evaluate-tree", []),
]


def show_help():
    print("Usage: python run.py <command> [options]\n")
    print("Available commands:")

    print("pipeline")
    for command in COMMANDS:
        print(f"{command}")

    print("\nExamples:")
    print("python run.py pipeline")
    print("python run.py transform")
    print("python run.py evaluate-retrieval")
    print("python run.py evaluate-tree")
    print("python run.py serve --reload")
    print("python run.py test -q")


def run_command(command_name, options=None):
    if options is None:
        options = []

    command = [
        sys.executable,
        "-m",
        *COMMANDS[command_name],
        *options,
    ]

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    ).returncode


def run_pipeline():
    for command_name, options in PIPELINE:
        print(f"\nRunning {command_name}...", flush=True)

        result = run_command(command_name, options)
        if result != 0:
            print(f"Pipeline stopped at {command_name}.")
            return result

    print("\nPipeline complete.")
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"help", "-h", "--help"}:
        show_help()
        return 0

    command_name = sys.argv[1]

    if command_name == "pipeline":
        return run_pipeline()

    if command_name not in COMMANDS:
        print(f"Unknown command: {command_name}\n")
        show_help()
        return 1

    return run_command(command_name, sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
