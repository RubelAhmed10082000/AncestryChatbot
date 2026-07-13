import argparse
import importlib.util
import sys
from pathlib import Path
import pandas as pd
from typing import Any

DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")
DEFAULT_CANDIDATE_MODULE_PATH = Path("app/retrieval/candidate_retrieval.py")


SCORE_COLUMNS = [
    "first_name_score",
    "last_name_score",
    "birth_year_score",
    "birth_location_score",
    "gender_score",
]


FRONT_COLUMNS = [
    "rank",
    "rank_score",
    "confidence_score",
    "wikitree_id",
    "full_name",
    "birth_year",
    "birth_location",
    "confidence_explanation",
    "confidence_interpretation",
]


def safe_float(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evidence_coverage(row: pd) -> float:
    """
    Calcuates how many rows in dataframe can be used to calculate confidence
    Args -  
        row (pd): row of data containing confidence scores of candidate profile
    Returns -
        float: ratio of strong column values compared to total available column values
    """
    supplied = 0

    for columns in SCORE_COLUMNS:
        value = safe_float(row.get(columns))

        if value is not None:
            supplied += 1

    return supplied / len(SCORE_COLUMNS)

def strong_evidence_ratio(row: pd.Series) -> float:
    """
    Returns the amount of evidence field that strongly agree
    with the candidate.
    """
    supplied_values = []

    for column in SCORE_COLUMNS:
        value = safe_float(row.get(column))

        if value is not None:
            supplied_values.append(value)

    if not supplied_values:
        return 0.0

    strong_matches = sum(
        value >= 0.8
        for value in supplied_values
    )

    return strong_matches / len(supplied_values)


def birth_date_quality(row: pd) -> str:
    """
    Provides synopsis for birth year match
    """
    score = safe_float(row.get("birth_year_score"))

    if score is None:
        return "birth year not supplied"
    if score >= 1.0:
        return "exact birth-year match"
    if score >= 0.75:
        return "close birth-year match"
    if score > 0:
        return "weak birth-year match"

    return "birth year missing or mismatch"


def name_quality(row: pd) -> str:
    """
    Provides synopsis for name match
    """
    first = safe_float(row.get("first_name_score")) or 0.0
    last = safe_float(row.get("last_name_score")) or 0.0

    if first >= 0.95 and last >= 0.95:
        return "exact/near-exact name match"
    if first >= 0.8 and last >= 0.8:
        return "strong name match"
    if first >= 0.6 and last >= 0.6:
        return "moderate name match"
    if first >= 0.6 or last >= 0.6:
        return "partial name match"

    return "weak name match"


def ambiguity_penalty(row: pd, candidates: pd) -> float:
    """
    Calculates penalty for profiles that multiple close candidate matches
    Args - 
        row(pd): row of data containing similarity of candidates ranked
        candidates(Any): containes candidate data
    Returns -
        float: penalty 0 - 15
    """
    rank = safe_float(row.get("rank"))

    if rank != 1 or len(candidates) < 2:
        return 0.0

    top_score = safe_float(candidates.iloc[0].get("rank_score")) or 0.0
    second_score = safe_float(candidates.iloc[1].get("rank_score")) or 0.0
    margin = top_score - second_score
    
    # Calculting penalties depending on rank closeness of first place and second place
    if margin >= 20:
        return 0.0
    if margin >= 10:
        return 5.0
    if margin >= 5:
        return 10.0
    
    return 15.0

def calculate_confidence_score(row: pd, candidates: pd) -> float:
    """
    Convert rank and score into confidence scores
    """
    confidence = safe_float(row.get("rank_score")) or 0.0

    coverage = evidence_coverage(row)
    evidence_strength = strong_evidence_ratio(row)


    # Confidence increases if coverage score is high, penalised if low
    if coverage >= 0.8 and evidence_strength >= 0.8:
        confidence += 5
    elif coverage <= 0.4:
        confidence -= 20
    elif coverage <= 0.6:
        confidence -= 10

    first_name_score = safe_float(row.get("first_name_score"))
    last_name_score = safe_float(row.get("last_name_score"))
    birth_year_score = safe_float(row.get("birth_year_score"))

    # Confidence score penalty based on low first and last name score
    if first_name_score is not None and first_name_score < 0.6:
        confidence -= 15

    if last_name_score is not None and last_name_score < 0.6:
        confidence -= 15

    # Confidence penalty or boost based on birth_year_score
    if birth_year_score is not None:
        if birth_year_score == 0:
            confidence -= 10
        elif birth_year_score >= 1.0:
            confidence += 5

    # Adding ambiguity penalty to confidence score
    confidence -= ambiguity_penalty(row, candidates)
    confidence = max(0.0, min(100.0, confidence))

    return round(confidence, 2)

def confidence_interpretation(confidence_score):
    """
    Placeholder interpertation for confidence score bands
    """
    score = safe_float(confidence_score) or 0.0

    if score >= 90:
        return "Strong candidate based on close agreement across key fields. Still requires source verification."
    if score >= 70:
        return "Plausible candidate, but at least one important field is missing, weak, or ambiguous."
    if score >= 50:
        return "Weak candidate. Treat as exploratory unless supported by additional evidence."

    return "Very weak candidate. Likely not reliable without substantial extra evidence."


def build_confidence_explanation(row: pd) -> str:
    """
    Placeholder explanation for each attribute score
    """
    parts = [name_quality(row), birth_date_quality(row)]

    location_score = safe_float(row.get("birth_location_score"))
    
    if location_score is not None:
        if location_score >= 0.5:
            parts.append("strong birth-location match")
        elif location_score > 0:
            parts.append("partial birth-location match")
        else:
            parts.append("birth location missing or mismatch")

    gender_score = safe_float(row.get("gender_score"))

    if gender_score is not None:
        if gender_score >= 1.0:
            parts.append("gender match")
        else:
            parts.append("gender mismatch or missing")
    
    coverage = evidence_coverage(row)

    if coverage >= 0.8:
        parts.append("high evidence coverage")
    elif coverage >= 0.6:
        parts.append("moderate evidence coverage")
    else:
        parts.append("limited evidence coverage")
        
    return "; ".join(parts) + "."


def add_confidence_scores(candidates: pd) -> pd:
    """
    adding confidence scores and explanation to pandas dataframe
    """
    if candidates.empty:
        return candidates

    df = candidates.copy()
    scores = []

    for _, row in df.iterrows():
        scores.append(calculate_confidence_score(row, df))

    df["confidence_score"] = scores
    df["confidence_interpretation"] = df["confidence_score"].apply(confidence_interpretation)
    df["confidence_explanation"] = df.apply(build_confidence_explanation, axis=1)

    front_cols = []
    for col in FRONT_COLUMNS:
        if col in df.columns:
            front_cols.append(col)

    other_cols = []
    for col in df.columns:
        if col not in front_cols:
            other_cols.append(col)

    return df[front_cols + other_cols]


def load_candidate_retriever(module_path):
    """
    loads candidate retriever claas from retrieval/candidate_retrieval.py module
    """
    module_path = Path(module_path)

    if not module_path.exists() and module_path == DEFAULT_CANDIDATE_MODULE_PATH:
        old_layout_path = Path("candidate_retrieval.py")
        if old_layout_path.exists():
            module_path = old_layout_path

    module_path = module_path.resolve()

    if not module_path.exists():
        raise FileNotFoundError(f"Could not find {module_path}")

    spec = importlib.util.spec_from_file_location("candidate_retrieval", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_retrieval"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "CandidateRetriever"):
        raise ImportError(f"{module_path} does not define CandidateRetriever")

    return module.CandidateRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve candidates and add confidence scores.")

    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--candidate-module-path", type=Path, default=DEFAULT_CANDIDATE_MODULE_PATH)
    parser.add_argument("--first-name", type=str, default=None)
    parser.add_argument("--last-name", type=str, default=None)
    parser.add_argument("--birth-year", type=int, default=None)
    parser.add_argument("--birth-location", type=str, default=None)
    parser.add_argument("--gender", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    CandidateRetriever = load_candidate_retriever(args.candidate_module_path)
    retriever = CandidateRetriever(schema_dir=args.schema_dir)

    candidates = retriever.find_candidates(
        first_name=args.first_name,
        last_name=args.last_name,
        birth_year=args.birth_year,
        birth_location=args.birth_location,
        gender=args.gender,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    results = add_confidence_scores(candidates)

    if results.empty:
        print("No candidates found.")
        return

    display_cols = [
        "rank",
        "confidence_score",
        "confidence_band",
        "rank_score",
        "wikitree_id",
        "full_name",
        "birth_year",
        "birth_location",
        "confidence_explanation",
    ]
    display_cols = [col for col in display_cols if col in results.columns]

    print(results[display_cols].to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"\nSaved confidence results to: {args.output}")


if __name__ == "__main__":
    main()
