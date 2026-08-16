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


def show_help():
    print("Usage: python run.py <command> [options]\n")
    print("Available commands:")

    for command in COMMANDS:
        print(f"  {command}")

    print("\nExamples:")
    print("  python run.py transform")
    print("  python run.py evaluate-retrieval")
    print("  python run.py evaluate-tree")
    print("  python run.py serve --reload")
    print("  python run.py test -q")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"help", "-h", "--help"}:
        show_help()
        return 0

    command_name = sys.argv[1]

    if command_name not in COMMANDS:
        print(f"Unknown command: {command_name}\n")
        show_help()
        return 1

    command = [
        sys.executable,
        "-m",
        *COMMANDS[command_name],
        *sys.argv[2:],
    ]

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
