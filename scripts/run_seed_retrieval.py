from __future__ import annotations
import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd



DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")
DEFAULT_SEED_PROFILES_PATH = Path("data/wikitree_test/seed_profiles.csv")
DEFAULT_OUTPUT_DIR = Path("data/evaluation")
DEFAULT_CANDIDATE_MODULE_PATH = Path("app/retrieval/candidate_retrieval.py")


SEED_FIGURES = [
    {
        "label": "Samuel Langhorne Clemens / Mark Twain",
        "first_name": "Samuel",
        "last_name": "Clemens",
        "birth_date": "1835-11-30",
        "known_wikitree_id": "Clemens-1",
    },
    {
        "label": "Aretha Franklin",
        "first_name": "Aretha",
        "last_name": "Franklin",
        "birth_date": "1942-03-25",
        "known_wikitree_id": "Franklin-10478",
    },
    {
        "label": "Charles Darwin",
        "first_name": "Charles",
        "last_name": "Darwin",
        "birth_date": "1809-02-12",
        "known_wikitree_id": None,
    },
    {
        "label": "Jane Austen",
        "first_name": "Jane",
        "last_name": "Austen",
        "birth_date": "1775-12-16",
        "known_wikitree_id": None,
    },
    {
        "label": "Isaac Newton",
        "first_name": "Isaac",
        "last_name": "Newton",
        "birth_date": "1643-01-04",
        "known_wikitree_id": None,
    },
    {
        "label": "William Shakespeare",
        "first_name": "William",
        "last_name": "Shakespeare",
        "birth_date": "1564-04-26",
        "known_wikitree_id": None,
    },
    {
        "label": "Florence Nightingale",
        "first_name": "Florence",
        "last_name": "Nightingale",
        "birth_date": "1820-05-12",
        "known_wikitree_id": None,
    },
    {
        "label": "Winston Churchill",
        "first_name": "Winston",
        "last_name": "Churchill",
        "birth_date": "1874-11-30",
        "known_wikitree_id": None,
    },
    {
        "label": "Ada Lovelace",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "birth_date": "1815-12-10",
        "known_wikitree_id": None,
    },
    {
        "label": "Isambard Kingdom Brunel",
        "first_name": "Isambard",
        "last_name": "Brunel",
        "birth_date": "1806-04-09",
        "known_wikitree_id": None,
    },
]



EXPECTED_ID_FALLBACKS = {
    "Charles Darwin": "Darwin-15",
    "Jane Austen": "Austen-489",
    "William Shakespeare": "Shakespeare-1",
    "Florence Nightingale": "Nightingale-64",
    "Winston Churchill": "Churchill-4",
    "Isambard Kingdom Brunel": "Brunel-8",
}

@dataclass(frozen=True)
class SeedCase:
    label: str
    first_name: str
    last_name: str
    birth_date: str
    birth_year: int | None
    expected_wikitree_id: str | None
    expected_source: str


def parse_birth_year(birth_date):
    if not birth_date:
        return None
    try:
        return int(str(birth_date)[:4])
    except ValueError:
        return None


def clean_text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def load_candidate_retriever(module_path: Path):
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(
            f"Could not find {module_path}. Put evaluate_seed_retrieval.py next to candidate_retrieval.py "
            "or pass --candidate-module-path."
        )

    spec = importlib.util.spec_from_file_location("candidate_retrieval", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_retrieval"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "CandidateRetriever"):
        raise ImportError(f"{module_path} does not define CandidateRetriever")

    return module.CandidateRetriever


def load_seed_profile_expected_ids(seed_profiles_path: Path):
    if not seed_profiles_path.exists():
        return {}

    df = pd.read_csv(seed_profiles_path, dtype=str, keep_default_na=False)
    if "seed_label" not in df.columns or "wikitree_id" not in df.columns:
        return {}

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        label = clean_text(row.get("seed_label"))
        wikitree_id = clean_text(row.get("wikitree_id"))
        if label and wikitree_id:
            mapping[label] = wikitree_id
    return mapping


def build_seed_cases(seed_profiles_path: Path):
    inferred_ids = load_seed_profile_expected_ids(seed_profiles_path)

    cases = []
    for seed in SEED_FIGURES:
        label = str(seed["label"])
        known_id = clean_text(seed.get("known_wikitree_id"))
        inferred_id = inferred_ids.get(label)
        fallback_id = EXPECTED_ID_FALLBACKS.get(label)

        if known_id:
            expected_id = known_id
            expected_source = "known_wikitree_id"
        elif inferred_id:
            expected_id = inferred_id
            expected_source = "seed_profiles_csv"
        elif fallback_id:
            expected_id = fallback_id
            expected_source = "fallback_expected_id"
        else:
            expected_id = None
            expected_source = "unavailable"

        cases.append(
            SeedCase(
                label=label,
                first_name=str(seed["first_name"]),
                last_name=str(seed["last_name"]),
                birth_date=str(seed["birth_date"]),
                birth_year=parse_birth_year(str(seed["birth_date"])),
                expected_wikitree_id=expected_id,
                expected_source=expected_source,
            )
        )

    return cases


def reciprocal_rank(rank):
    if rank is None or rank <= 0:
        return 0.0
    return round(1.0 / rank, 6)


def make_empty_result_row(case, status, notes):
    return {
        "label": case.label,
        "query_first_name": case.first_name,
        "query_last_name": case.last_name,
        "query_birth_year": case.birth_year,
        "expected_wikitree_id": case.expected_wikitree_id,
        "expected_source": case.expected_source,
        "expected_rank": None,
        "top_1_hit": False,
        "top_3_hit": False,
        "top_5_hit": False,
        "reciprocal_rank": 0.0,
        "top_candidate_wikitree_id": None,
        "top_candidate_name": None,
        "top_candidate_score": None,
        "status": status,
        "notes": notes,
    }


def evaluate_cases(*, retriever, cases, top_k):
    evaluation_rows = []
    candidate_rows = []

    for case in cases:
        if not case.expected_wikitree_id:
            evaluation_rows.append(
                make_empty_result_row(
                    case,
                    status="skipped",
                    notes="No expected WikiTree ID available. This usually means the seed was not extracted successfully.",
                )
            )
            continue

        candidates = retriever.find_candidates(
            first_name=case.first_name,
            last_name=case.last_name,
            birth_year=case.birth_year,
            birth_location=None,
            gender=None,
            top_k=top_k,
            min_score=0.0,
        )

        if candidates.empty:
            evaluation_rows.append(
                make_empty_result_row(
                    case,
                    status="failed",
                    notes="Retriever returned no candidates.",
                )
            )
            continue

        candidates = candidates.copy()
        candidates.insert(0, "label", case.label)
        candidates.insert(1, "expected_wikitree_id", case.expected_wikitree_id)
        candidates.insert(2, "query_first_name", case.first_name)
        candidates.insert(3, "query_last_name", case.last_name)
        candidates.insert(4, "query_birth_year", case.birth_year)
        candidates["is_expected_candidate"] = candidates["wikitree_id"] == case.expected_wikitree_id
        candidate_rows.extend(candidates.to_dict(orient="records"))

        match_rows = candidates[candidates["wikitree_id"] == case.expected_wikitree_id]
        expected_rank = int(match_rows.iloc[0]["rank"]) if not match_rows.empty else None

        top_candidate = candidates.iloc[0]
        found_expected_somewhere = expected_rank is not None

        if found_expected_somewhere:
            status = "passed" if expected_rank == 1 else "partial"
            notes = f"Expected candidate found at rank {expected_rank}."
        else:
            status = "failed"
            notes = f"Expected candidate not found in top {top_k}."

        evaluation_rows.append(
            {
                "label": case.label,
                "query_first_name": case.first_name,
                "query_last_name": case.last_name,
                "query_birth_year": case.birth_year,
                "expected_wikitree_id": case.expected_wikitree_id,
                "expected_source": case.expected_source,
                "expected_rank": expected_rank,
                "top_1_hit": expected_rank == 1,
                "top_3_hit": expected_rank is not None and expected_rank <= 3,
                "top_5_hit": expected_rank is not None and expected_rank <= 5,
                "reciprocal_rank": reciprocal_rank(expected_rank),
                "top_candidate_wikitree_id": top_candidate.get("wikitree_id"),
                "top_candidate_name": top_candidate.get("full_name"),
                "top_candidate_score": top_candidate.get("rank_score"),
                "status": status,
                "notes": notes,
            }
        )

    evaluation_df = pd.DataFrame(evaluation_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    summary_df = build_summary(evaluation_df)
    return evaluation_df, candidate_df, summary_df


def build_summary(evaluation_df):
    eligible = evaluation_df[evaluation_df["status"] != "skipped"].copy()
    total_cases = len(evaluation_df)
    eligible_cases = len(eligible)
    skipped_cases = int((evaluation_df["status"] == "skipped").sum())

    def rate(column):
        if eligible_cases == 0:
            return 0.0
        return round(float(eligible[column].mean()), 4)

    mrr = round(float(eligible["reciprocal_rank"].mean()), 4) if eligible_cases else 0.0

    rows = [
        {"metric": "total_seed_cases", "value": total_cases},
        {"metric": "eligible_seed_cases", "value": eligible_cases},
        {"metric": "skipped_seed_cases", "value": skipped_cases},
        {"metric": "top_1_accuracy", "value": rate("top_1_hit")},
        {"metric": "top_3_accuracy", "value": rate("top_3_hit")},
        {"metric": "top_5_accuracy", "value": rate("top_5_hit")},
        {"metric": "mean_reciprocal_rank", "value": mrr},
        {"metric": "passed_cases", "value": int((evaluation_df["status"] == "passed").sum())},
        {"metric": "partial_cases", "value": int((evaluation_df["status"] == "partial").sum())},
        {"metric": "failed_cases", "value": int((evaluation_df["status"] == "failed").sum())},
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate retrieval on known WikiTree seed figures.")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR, help="Directory containing person.csv, names.csv, event.csv.")
    parser.add_argument("--seed-profiles", type=Path, default=DEFAULT_SEED_PROFILES_PATH, help="Path to seed_profiles.csv from extract.py.")
    parser.add_argument("--candidate-module-path", type=Path, default=DEFAULT_CANDIDATE_MODULE_PATH, help="Path to candidate_retrieval.py.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for evaluation outputs.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of candidates retrieved per seed case.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    CandidateRetriever = load_candidate_retriever(args.candidate_module_path)
    retriever = CandidateRetriever(schema_dir=args.schema_dir)
    cases = build_seed_cases(args.seed_profiles)

    evaluation_df, candidate_df, summary_df = evaluate_cases(retriever=retriever, cases=cases, top_k=args.top_k)

    evaluation_path = args.output_dir / "seed_retrieval_evaluation.csv"
    candidate_path = args.output_dir / "seed_candidate_results.csv"
    summary_path = args.output_dir / "seed_retrieval_summary.csv"

    evaluation_df.to_csv(evaluation_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Seed retrieval evaluation complete.")
    print(f"Evaluation results: {evaluation_path}")
    print(f"Candidate results:  {candidate_path}")
    print(f"Summary metrics:    {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("\nCase results:")
    display_cols = [
        "label",
        "expected_wikitree_id",
        "expected_rank",
        "top_candidate_wikitree_id",
        "top_candidate_score",
        "status",
        "notes",
    ]
    print(evaluation_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
