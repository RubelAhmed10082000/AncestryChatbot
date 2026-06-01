"""
Evaluate confidence_scoring.py on the WikiTree seed figures.

Inputs expected:
    data/wikitree_schema/person.csv
    data/wikitree_schema/names.csv
    data/wikitree_schema/event.csv

Optional input, used to infer expected WikiTree IDs where they are not hard-coded:
    data/wikitree_test/seed_profiles.csv

Requires:
    candidate_retrieval.py
    confidence_scoring.py

Outputs:
    data/evaluation/seed_confidence_evaluation.csv
    data/evaluation/seed_confidence_candidate_results.csv
    data/evaluation/seed_confidence_summary.csv

Run from your project root:
    python evaluate_seed_confidence.py

Useful options:
    python evaluate_seed_confidence.py --top-k 10
    python evaluate_seed_confidence.py --schema-dir data/wikitree_schema
    python evaluate_seed_confidence.py --output-dir data/evaluation

Purpose:
    This evaluates whether the confidence layer behaves sensibly on known seed cases:
    - correct top-ranked candidates should usually receive High confidence;
    - ambiguous or lower-ranked expected candidates should receive lower confidence;
    - skipped cases are treated as dataset/extraction gaps rather than retrieval failures.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")
DEFAULT_SEED_PROFILES_PATH = Path("data/wikitree_test/seed_profiles.csv")
DEFAULT_OUTPUT_DIR = Path("data/evaluation")
DEFAULT_CANDIDATE_MODULE_PATH = Path("candidate_retrieval.py")
DEFAULT_CONFIDENCE_MODULE_PATH = Path("confidence_scoring.py")


SEED_FIGURES: list[dict[str, str | None]] = [
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


# These are used only if seed_profiles.csv does not provide an expected ID.
# Newton and Ada are intentionally not hard-coded here because your current data makes them messy:
# - Newton selected profile has missing birth date.
# - Ada has not been extracted reliably.
EXPECTED_ID_FALLBACKS = {
    "Charles Darwin": "Darwin-15",
    "Jane Austen": "Austen-489",
    "William Shakespeare": "Shakespeare-1",
    "Florence Nightingale": "Nightingale-64",
    "Winston Churchill": "Churchill-4",
    "Isambard Kingdom Brunel": "Brunel-8",
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedCase:
    label: str
    first_name: str
    last_name: str
    birth_date: str
    birth_year: int | None
    expected_wikitree_id: str | None
    expected_source: str


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def parse_birth_year(birth_date: str | None) -> int | None:
    if not birth_date:
        return None
    try:
        return int(str(birth_date)[:4])
    except ValueError:
        return None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def load_python_module(module_path: Path, module_name: str):
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_seed_profile_expected_ids(seed_profiles_path: Path) -> dict[str, str]:
    """Load label -> wikitree_id from seed_profiles.csv if available."""
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


def build_seed_cases(seed_profiles_path: Path) -> list[SeedCase]:
    """Create test cases, using hard-coded IDs first, then seed_profiles.csv, then fallbacks."""
    inferred_ids = load_seed_profile_expected_ids(seed_profiles_path)

    cases: list[SeedCase] = []
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


def confidence_is_reasonable(*, status: str, expected_rank: int | None, expected_band: str | None) -> bool | None:
    """
    A light sanity check for confidence behaviour.

    This is not a statistical calibration test. It checks whether confidence bands are directionally sensible:
    - passed rank-1 cases should normally be High;
    - partial cases should normally be Moderate or lower;
    - failed/skipped cases have no expected candidate confidence to assess.
    """
    if expected_rank is None or expected_band is None:
        return None

    if status == "passed":
        return expected_band == "High"

    if status == "partial":
        return expected_band in {"Moderate", "Low", "Very low"}

    return None


def make_skipped_result_row(case: SeedCase, notes: str) -> dict[str, Any]:
    return {
        "label": case.label,
        "query_first_name": case.first_name,
        "query_last_name": case.last_name,
        "query_birth_year": case.birth_year,
        "expected_wikitree_id": case.expected_wikitree_id,
        "expected_source": case.expected_source,
        "expected_rank": None,
        "expected_confidence_score": None,
        "expected_confidence_band": None,
        "top_candidate_wikitree_id": None,
        "top_candidate_name": None,
        "top_candidate_rank_score": None,
        "top_candidate_confidence_score": None,
        "top_candidate_confidence_band": None,
        "top_1_hit": False,
        "top_3_hit": False,
        "top_5_hit": False,
        "status": "skipped",
        "confidence_reasonable": None,
        "notes": notes,
    }


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate_confidence_cases(
    *,
    retriever: Any,
    add_confidence_scores: Any,
    cases: list[SeedCase],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluation_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for case in cases:
        if not case.expected_wikitree_id:
            evaluation_rows.append(
                make_skipped_result_row(
                    case,
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
                {
                    **make_skipped_result_row(case, notes="Retriever returned no candidates."),
                    "status": "failed",
                }
            )
            continue

        candidates = add_confidence_scores(candidates)
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

        if expected_rank is None:
            status = "failed"
            notes = f"Expected candidate not found in top {top_k}."
            expected_confidence_score = None
            expected_confidence_band = None
        else:
            status = "passed" if expected_rank == 1 else "partial"
            notes = f"Expected candidate found at rank {expected_rank}."
            expected_confidence_score = float(match_rows.iloc[0]["confidence_score"])
            expected_confidence_band = str(match_rows.iloc[0]["confidence_band"])

        top_candidate = candidates.iloc[0]
        reasonable = confidence_is_reasonable(
            status=status,
            expected_rank=expected_rank,
            expected_band=expected_confidence_band,
        )

        evaluation_rows.append(
            {
                "label": case.label,
                "query_first_name": case.first_name,
                "query_last_name": case.last_name,
                "query_birth_year": case.birth_year,
                "expected_wikitree_id": case.expected_wikitree_id,
                "expected_source": case.expected_source,
                "expected_rank": expected_rank,
                "expected_confidence_score": expected_confidence_score,
                "expected_confidence_band": expected_confidence_band,
                "top_candidate_wikitree_id": top_candidate.get("wikitree_id"),
                "top_candidate_name": top_candidate.get("full_name"),
                "top_candidate_rank_score": top_candidate.get("rank_score"),
                "top_candidate_confidence_score": top_candidate.get("confidence_score"),
                "top_candidate_confidence_band": top_candidate.get("confidence_band"),
                "top_1_hit": expected_rank == 1,
                "top_3_hit": expected_rank is not None and expected_rank <= 3,
                "top_5_hit": expected_rank is not None and expected_rank <= 5,
                "status": status,
                "confidence_reasonable": reasonable,
                "notes": notes,
            }
        )

    evaluation_df = pd.DataFrame(evaluation_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    summary_df = build_summary(evaluation_df, candidate_df)
    return evaluation_df, candidate_df, summary_df


def build_summary(evaluation_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    eligible = evaluation_df[evaluation_df["status"] != "skipped"].copy()
    evaluated_confidence = evaluation_df[evaluation_df["expected_confidence_score"].notna()].copy()

    total_cases = len(evaluation_df)
    eligible_cases = len(eligible)
    skipped_cases = int((evaluation_df["status"] == "skipped").sum())

    def rate(column: str) -> float:
        if eligible_cases == 0:
            return 0.0
        return round(float(eligible[column].mean()), 4)

    def mean_or_zero(series: pd.Series) -> float:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return 0.0
        return round(float(clean.mean()), 4)

    expected_band_counts = {}
    if not evaluated_confidence.empty:
        expected_band_counts = evaluated_confidence["expected_confidence_band"].value_counts().to_dict()

    top_band_counts = {}
    if not evaluation_df.empty:
        top_band_counts = evaluation_df["top_candidate_confidence_band"].dropna().value_counts().to_dict()

    confidence_reasonable_rate = 0.0
    reasonableness_rows = evaluation_df[evaluation_df["confidence_reasonable"].notna()].copy()
    if not reasonableness_rows.empty:
        confidence_reasonable_rate = round(float(reasonableness_rows["confidence_reasonable"].mean()), 4)

    rows = [
        {"metric": "total_seed_cases", "value": total_cases},
        {"metric": "eligible_seed_cases", "value": eligible_cases},
        {"metric": "skipped_seed_cases", "value": skipped_cases},
        {"metric": "passed_cases", "value": int((evaluation_df["status"] == "passed").sum())},
        {"metric": "partial_cases", "value": int((evaluation_df["status"] == "partial").sum())},
        {"metric": "failed_cases", "value": int((evaluation_df["status"] == "failed").sum())},
        {"metric": "top_1_accuracy", "value": rate("top_1_hit")},
        {"metric": "top_3_accuracy", "value": rate("top_3_hit")},
        {"metric": "top_5_accuracy", "value": rate("top_5_hit")},
        {"metric": "mean_expected_confidence_score", "value": mean_or_zero(evaluation_df["expected_confidence_score"])},
        {"metric": "mean_top_candidate_confidence_score", "value": mean_or_zero(evaluation_df["top_candidate_confidence_score"])},
        {"metric": "confidence_reasonable_rate", "value": confidence_reasonable_rate},
        {"metric": "expected_high_confidence_cases", "value": int(expected_band_counts.get("High", 0))},
        {"metric": "expected_moderate_confidence_cases", "value": int(expected_band_counts.get("Moderate", 0))},
        {"metric": "expected_low_confidence_cases", "value": int(expected_band_counts.get("Low", 0))},
        {"metric": "expected_very_low_confidence_cases", "value": int(expected_band_counts.get("Very low", 0))},
        {"metric": "top_candidate_high_confidence_cases", "value": int(top_band_counts.get("High", 0))},
        {"metric": "top_candidate_moderate_confidence_cases", "value": int(top_band_counts.get("Moderate", 0))},
        {"metric": "top_candidate_low_confidence_cases", "value": int(top_band_counts.get("Low", 0))},
        {"metric": "top_candidate_very_low_confidence_cases", "value": int(top_band_counts.get("Very low", 0))},
        {"metric": "candidate_rows_written", "value": len(candidate_df)},
    ]

    return pd.DataFrame(rows, columns=["metric", "value"])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate confidence scoring on known WikiTree seed figures.")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR, help="Directory containing person.csv, names.csv, event.csv.")
    parser.add_argument("--seed-profiles", type=Path, default=DEFAULT_SEED_PROFILES_PATH, help="Path to seed_profiles.csv from extract.py.")
    parser.add_argument("--candidate-module-path", type=Path, default=DEFAULT_CANDIDATE_MODULE_PATH, help="Path to candidate_retrieval.py.")
    parser.add_argument("--confidence-module-path", type=Path, default=DEFAULT_CONFIDENCE_MODULE_PATH, help="Path to confidence_scoring.py.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for evaluation outputs.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of candidates retrieved per seed case.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_module = load_python_module(args.candidate_module_path, "candidate_retrieval")
    confidence_module = load_python_module(args.confidence_module_path, "confidence_scoring")

    if not hasattr(candidate_module, "CandidateRetriever"):
        raise ImportError(f"{args.candidate_module_path} does not define CandidateRetriever")
    if not hasattr(confidence_module, "add_confidence_scores"):
        raise ImportError(f"{args.confidence_module_path} does not define add_confidence_scores")

    retriever = candidate_module.CandidateRetriever(schema_dir=args.schema_dir)
    cases = build_seed_cases(args.seed_profiles)

    evaluation_df, candidate_df, summary_df = evaluate_confidence_cases(
        retriever=retriever,
        add_confidence_scores=confidence_module.add_confidence_scores,
        cases=cases,
        top_k=args.top_k,
    )

    evaluation_path = args.output_dir / "seed_confidence_evaluation.csv"
    candidate_path = args.output_dir / "seed_confidence_candidate_results.csv"
    summary_path = args.output_dir / "seed_confidence_summary.csv"

    evaluation_df.to_csv(evaluation_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Seed confidence evaluation complete.")
    print(f"Evaluation results: {evaluation_path}")
    print(f"Candidate results:  {candidate_path}")
    print(f"Summary metrics:    {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    display_cols = [
        "label",
        "expected_wikitree_id",
        "expected_rank",
        "expected_confidence_score",
        "expected_confidence_band",
        "top_candidate_wikitree_id",
        "top_candidate_confidence_score",
        "top_candidate_confidence_band",
        "status",
        "confidence_reasonable",
        "notes",
    ]

    print("\nCase results:")
    print(evaluation_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
