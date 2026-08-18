from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.candidate_retrieval import CandidateRetriever
from app.scoring.confidence_scoring import add_confidence_scores


DEFAULT_CASES_PATH = Path("data/evaluation/evaluation_cases.csv")
DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")
DEFAULT_OUTPUT_DIR = Path("data/evaluation")

# Setting up evaluation definition
TOP_K = 5 # Only evaluating retrieval  with top-5 ranking
MIN_SCORE = 0.0
TOP_SCORE_TIE_TOLERANCE = 0.001 # Scores are considered tied if they differ by this value
AMBIGUITY_MARGIN_THRESHOLD = 5.0 # Candidate is flagged if score is within 5.0 of another candidate 

RESULTS_FILENAME = "evaluation_results.csv"
SUMMARY_FILENAME = "evaluation_summary.csv"
FAILURES_FILENAME = "failure_cases.csv"
AMBIGUITY_CASES_FILENAME = "ambiguity_cases.csv"
CONFIDENCE_SUMMARY_FILENAME = "confidence_summary.csv"

# Listing expected test conditions
EXPECTED_CONDITIONS = (
    "full_profile",
    "name_year",
    "name_only",
    "noisy_input",
    "wrong_year",
    "wrong_location",
    "wrong_gender",
)

# Requried fields for evaluation
REQUIRED_CASE_COLUMNS = {
    "case_id",
    "seed_label",
    "condition",
    "expected_wikitree_id",
    "first_name",
    "last_name",
    "birth_year",
    "birth_location",
    "gender",
}

# Listing scoring columns
SCORE_COLUMNS = (
    "first_name_score",
    "last_name_score",
    "birth_year_score",
    "birth_location_score",
    "gender_score",
)


def clean_optional_text(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    text = str(value).strip()
    return text or None


def parse_optional_year(value):
    text = clean_optional_text(value)

    if text is None:
        return None

    try:
        numeric_year = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid birth year in evaluation case: {value!r}") from error

    if not numeric_year.is_integer():
        raise ValueError(f"Birth year must be a whole number: {value!r}")

    return int(numeric_year)


def value_or_none(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def load_evaluation_cases(path = DEFAULT_CASES_PATH):

    # Checks file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Missing evaluation cases: {path}. "
            "Run scripts/create_evaluation_cases.py first."
        )

    # loading case file and checking if required columns exist
    cases = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_CASE_COLUMNS - set(cases.columns)

    # raising error if mssing columns required for evaluation 
    if missing_columns:
        raise ValueError(
            "evaluation_cases.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # checking if cases are empty
    if cases.empty:
        raise ValueError("evaluation_cases.csv contains no cases.")

    # checking for duplicated case_id
    if cases["case_id"].duplicated().any():
        duplicate_ids = cases.loc[
            cases["case_id"].duplicated(keep=False),
            "case_id",
        ].unique()
        raise ValueError(f"Duplicate evaluation case IDs: {sorted(duplicate_ids)}")

    # Checking for missing expepected ids
    missing_expected_ids = cases["expected_wikitree_id"].map(clean_optional_text).isna()

    if missing_expected_ids.any():
        invalid_cases = cases.loc[missing_expected_ids, "case_id"].tolist()
        raise ValueError(
            "Evaluation cases are missing expected WikiTree IDs: "
            f"{invalid_cases}"
        )

    # CHecking for missing condition values
    actual_conditions = set(cases["condition"])
    expected_conditions = set(EXPECTED_CONDITIONS)

    if actual_conditions != expected_conditions:
        raise ValueError(
            "Evaluation conditions do not match the required set. "
            f"Expected {sorted(expected_conditions)}, got {sorted(actual_conditions)}."
        )

    # CHecking for mismatching condition counts
    condition_counts = cases.groupby("expected_wikitree_id")["condition"].nunique()
    incomplete_profiles = condition_counts[condition_counts != len(EXPECTED_CONDITIONS)]

    if not incomplete_profiles.empty:
        raise ValueError(
            "Every expected profile must have exactly the seven evaluation conditions. "
            f"Invalid profiles: {incomplete_profiles.to_dict()}"
        )

    return cases


def reciprocal_rank(rank):
    if rank is None or rank <= 0:
        return 0.0

    # Returning rank divided by position 
    return round(1.0 / rank, 6)


def failure_reason(expected_rank, candidate_count):
    if expected_rank == 1:
        return None
    if candidate_count == 0:
        return "no_candidates_returned"
    if expected_rank is None:
        return "expected_candidate_outside_top_5"

    return "expected_candidate_ranked_below_first"


def top_score_diagnostics(candidates):
    # building diagnostics dict
    diagnostics = {
        "second_candidate_wikitree_id": None,
        "second_candidate_rank_score": None,
        "score_margin": None,
        "top_score_tie": False,
        "top_score_tie_size": 0,
    }

    if candidates.empty:
        return diagnostics

    # retrieving candidate rank score
    rank_scores = pd.to_numeric(candidates["rank_score"], errors="coerce")
    # retrieving top scoring candidate
    top_score = value_or_none(rank_scores.iloc[0])

    # Finding all scores that are within the Tie Tolerance from the top_score
    if top_score is not None:
        diagnostics["top_score_tie_size"] = int(
            (rank_scores.sub(float(top_score)).abs() <= TOP_SCORE_TIE_TOLERANCE).sum()
        )

    # Returning candidates that are within the tie tolerane of top score
    if len(candidates) < 2:
        return diagnostics

    # getting second top ranked candidate
    second_candidate = candidates.iloc[1]
    second_score = value_or_none(rank_scores.iloc[1])
    diagnostics["second_candidate_wikitree_id"] = clean_optional_text(
        second_candidate.get("wikitree_id")
    )

    
    diagnostics["second_candidate_rank_score"] = second_score

    if top_score is not None and second_score is not None:
        # calculating margin between top scorer and second top scorer
        score_margin = round(float(top_score) - float(second_score), 6)
        diagnostics["score_margin"] = score_margin
        diagnostics["top_score_tie"] = (
            score_margin <= TOP_SCORE_TIE_TOLERANCE
        )

    return diagnostics


def evaluate_cases(retriever, cases, confidence_scorer = add_confidence_scores):

    result_rows = []

    # Extracting fields from cases
    for _, case in cases.iterrows():
        first_name = clean_optional_text(case.get("first_name"))
        last_name = clean_optional_text(case.get("last_name"))
        birth_year = parse_optional_year(case.get("birth_year"))
        birth_location = clean_optional_text(case.get("birth_location"))
        gender = clean_optional_text(case.get("gender"))
        expected_wikitree_id = clean_optional_text(case.get("expected_wikitree_id"))

        candidates = retriever.find_candidates(
            first_name=first_name,
            last_name=last_name,
            birth_year=birth_year,
            birth_location=birth_location,
            gender=gender,
            top_k=TOP_K,
            min_score=MIN_SCORE,
        )

        # Counting candidates, vaidating scores and providing confidence scores
        candidates = confidence_scorer(candidates)
        candidate_count = len(candidates)
        score_diagnostics = top_score_diagnostics(candidates)

        if candidates.empty:
            expected_rank = None
            top_candidate = None
        else:
            candidate_ids = candidates["wikitree_id"].map(clean_optional_text)
            expected_matches = candidates[candidate_ids == expected_wikitree_id]
            expected_rank = (
                int(expected_matches.iloc[0]["rank"])
                if not expected_matches.empty
                else None
            )
            top_candidate = candidates.iloc[0]

        top_1_correct = expected_rank == 1
        top_3_correct = expected_rank is not None and expected_rank <= 3
        top_5_correct = expected_rank is not None and expected_rank <= 5
        unique_top_1_correct = (
            top_1_correct
            and score_diagnostics["top_score_tie_size"] == 1
        )

        result_row = {
            "case_id": clean_optional_text(case.get("case_id")),
            "seed_label": clean_optional_text(case.get("seed_label")),
            "condition": clean_optional_text(case.get("condition")),
            "expected_wikitree_id": expected_wikitree_id,
            "first_name": first_name,
            "last_name": last_name,
            "birth_year": birth_year,
            "birth_location": birth_location,
            "gender": gender,
            "notes": clean_optional_text(case.get("notes")),
            "candidate_count": candidate_count,
            "expected_rank": expected_rank,
            "top_1_correct": top_1_correct,
            "unique_top_1_correct": unique_top_1_correct,
            "top_3_correct": top_3_correct,
            "top_5_correct": top_5_correct,
            "reciprocal_rank": reciprocal_rank(expected_rank),
            "retrieval_failed": expected_rank is None,
            "returned_top_wikitree_id": None,
            "returned_top_name": None,
            "top_rank_score": None,
            "top_confidence_score": None,
            "confidence_interpretation": None,
            "failure_reason": failure_reason(expected_rank, candidate_count),
            **score_diagnostics,
        }

        for column in SCORE_COLUMNS:
            result_row[column] = None

        if top_candidate is not None:
            result_row.update(
                {
                    "returned_top_wikitree_id": clean_optional_text(
                        top_candidate.get("wikitree_id")
                    ),
                    "returned_top_name": clean_optional_text(
                        top_candidate.get("full_name")
                    ),
                    "top_rank_score": value_or_none(top_candidate.get("rank_score")),
                    "top_confidence_score": value_or_none(
                        top_candidate.get("confidence_score")
                    ),
                    "confidence_interpretation": clean_optional_text(
                        top_candidate.get("confidence_interpretation")
                    ),
                }
            )

            for column in SCORE_COLUMNS:
                result_row[column] = value_or_none(top_candidate.get(column))

        result_rows.append(result_row)

    results = pd.DataFrame(result_rows)
    results["expected_rank"] = results["expected_rank"].astype("Int64")
    return results


def mean_or_none(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return None

    return round(float(values.mean()), 6)


def minimum_or_none(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return None

    return round(float(values.min()), 6)


def rate_or_none(series):
    if series.empty:
        return None

    return round(float(series.mean()), 6)


def result_groups(results):
    yield "overall", "all", results

    for condition in EXPECTED_CONDITIONS:
        yield "condition", condition, results[results["condition"] == condition]


def build_evaluation_summary(results):
    summary_rows = []

    for scope, condition, group in result_groups(results):
        correct_top = group[group["top_1_correct"]]
        incorrect_top = group[~group["top_1_correct"]]
        exact_tie_count = int(group["top_score_tie"].sum())

        summary_rows.append(
            {
                "scope": scope,
                "condition": condition,
                "total_cases": len(group),
                "top_1_accuracy": rate_or_none(group["top_1_correct"]),
                "unique_top_1_accuracy": rate_or_none(
                    group["unique_top_1_correct"]
                ),
                "top_3_accuracy": rate_or_none(group["top_3_correct"]),
                "top_5_accuracy": rate_or_none(group["top_5_correct"]),
                "mean_reciprocal_rank": mean_or_none(group["reciprocal_rank"]),
                "exact_tie_count": exact_tie_count,
                "exact_tie_rate": (
                    round(exact_tie_count / len(group), 6)
                    if len(group)
                    else None
                ),
                "mean_score_margin": mean_or_none(group["score_margin"]),
                "minimum_score_margin": minimum_or_none(group["score_margin"]),
                "mean_confidence": mean_or_none(group["top_confidence_score"]),
                "mean_confidence_correct_top_1": mean_or_none(
                    correct_top["top_confidence_score"]
                ),
                "mean_confidence_incorrect_top_1": mean_or_none(
                    incorrect_top["top_confidence_score"]
                ),
                "failed_retrieval_count": int(group["retrieval_failed"].sum()),
                "no_candidates_count": int((group["candidate_count"] == 0).sum()),
            }
        )

    return pd.DataFrame(summary_rows)


def build_failure_cases(results):
    return results.loc[~results["top_1_correct"]].reset_index(drop=True)


def ambiguity_reasons(row):
    reasons = []

    if not row["top_1_correct"]:
        reasons.append("incorrect_top_1")

    if not row["unique_top_1_correct"]:
        reasons.append("non_unique_top_1")

    score_margin = pd.to_numeric(row.get("score_margin"), errors="coerce")

    if pd.notna(score_margin) and score_margin < AMBIGUITY_MARGIN_THRESHOLD:
        reasons.append("score_margin_below_5")

    return ";".join(reasons)


def build_ambiguity_cases(results):

    score_margins = pd.to_numeric(results["score_margin"], errors="coerce")
    analysis_mask = (
        (~results["top_1_correct"])
        | (~results["unique_top_1_correct"])
        | (score_margins < AMBIGUITY_MARGIN_THRESHOLD)
    )
    analysis_cases = results.loc[analysis_mask].copy()
    analysis_cases["retrieval_analysis_class"] = (
        "ambiguous_or_fragile_retrieval"
    )
    analysis_cases["ambiguity_reasons"] = analysis_cases.apply(
        ambiguity_reasons,
        axis=1,
    )

    return analysis_cases.reset_index(drop=True)


def confidence_statistics(group):
    scores = pd.to_numeric(group["top_confidence_score"], errors="coerce").dropna()

    if scores.empty:
        return {
            "cases_with_confidence": 0,
            "mean_confidence": None,
            "median_confidence": None,
            "minimum_confidence": None,
            "maximum_confidence": None,
        }

    return {
        "cases_with_confidence": len(scores),
        "mean_confidence": round(float(scores.mean()), 6),
        "median_confidence": round(float(scores.median()), 6),
        "minimum_confidence": round(float(scores.min()), 6),
        "maximum_confidence": round(float(scores.max()), 6),
    }


def build_confidence_summary(results):
    summary_rows = []

    for scope, condition, group in result_groups(results):
        outcome_groups = (
            ("all", group),
            ("correct", group[group["top_1_correct"]]),
            ("incorrect", group[~group["top_1_correct"]]),
        )

        for top_1_outcome, outcome_group in outcome_groups:
            summary_rows.append(
                {
                    "scope": scope,
                    "condition": condition,
                    "top_1_outcome": top_1_outcome,
                    "case_count": len(outcome_group),
                    **confidence_statistics(outcome_group),
                }
            )

    return pd.DataFrame(summary_rows)


def write_outputs(results, output_dir = DEFAULT_OUTPUT_DIR):

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "results": output_dir / RESULTS_FILENAME,
        "summary": output_dir / SUMMARY_FILENAME,
        "failures": output_dir / FAILURES_FILENAME,
        "ambiguity_cases": output_dir / AMBIGUITY_CASES_FILENAME,
        "confidence_summary": output_dir / CONFIDENCE_SUMMARY_FILENAME,
    }

    results.to_csv(outputs["results"], index=False)
    build_evaluation_summary(results).to_csv(outputs["summary"], index=False)
    build_failure_cases(results).to_csv(outputs["failures"], index=False)
    build_ambiguity_cases(results).to_csv(
        outputs["ambiguity_cases"],
        index=False,
    )
    build_confidence_summary(results).to_csv(
        outputs["confidence_summary"],
        index=False,
    )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen 63-case retrieval and confidence evaluation."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to evaluation_cases.csv.",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=DEFAULT_SCHEMA_DIR,
        help="Directory containing the transformed WikiTree schema CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the four evaluation output CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_evaluation_cases(args.cases)

    # Construct once so every case uses the same frozen in-memory search index.
    retriever = CandidateRetriever(schema_dir=args.schema_dir)
    results = evaluate_cases(retriever, cases)
    outputs = write_outputs(results, args.output_dir)
    overall = build_evaluation_summary(results).iloc[0]

    print("Retrieval evaluation complete.")
    print(f"Cases evaluated: {len(results)}")
    print(f"Top-1 accuracy: {overall['top_1_accuracy']:.4f}")
    print(
        "Unique Top-1 accuracy: "
        f"{overall['unique_top_1_accuracy']:.4f}"
    )
    print(f"Top-3 accuracy: {overall['top_3_accuracy']:.4f}")
    print(f"Top-5 accuracy: {overall['top_5_accuracy']:.4f}")
    print(f"Mean reciprocal rank: {overall['mean_reciprocal_rank']:.4f}")
    print(f"Exact top-score ties: {int(overall['exact_tie_count'])}")
    print(f"Mean score margin: {overall['mean_score_margin']:.4f}")
    print(f"Failed retrievals: {int(overall['failed_retrieval_count'])}")
    print("Outputs:")

    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
