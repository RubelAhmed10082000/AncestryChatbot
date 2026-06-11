import pandas as pd
import pytest
import app.scoring.confidence_scoring as confidence_scoring

### Testing safe_float ####

def test_safe_float_converts_valid_numbers():
    assert confidence_scoring.safe_float("92.5") == 92.5
    assert confidence_scoring.safe_float(80) == 80.0
    assert confidence_scoring.safe_float(0.75) == 0.75


def test_safe_float_returns_none_for_missing_or_invalid_values():
    assert confidence_scoring.safe_float(None) is None
    assert confidence_scoring.safe_float("") is None
    assert confidence_scoring.safe_float("not-a-number") is None

### Testing evidence_coverage ###

def test_evidence_coverage_all_strong_scores_returns_one():
    row = pd.Series(
        {
            "first_name_score": 1.0,
            "last_name_score": 0.95,
            "birth_year_score": 1.0,
            "birth_location_score": 0.85,
            "gender_score": 1.0,
        }
    )

    assert confidence_scoring.evidence_coverage(row) == 1.0


def test_evidence_coverage_ratio():
    row = pd.Series(
        {
            "first_name_score": 1.0,
            "last_name_score": 0.7,
            "birth_year_score": 1.0,
            "birth_location_score": 0.4,
            "gender_score": 1.0,
        }
    )

    assert confidence_scoring.evidence_coverage(row) == pytest.approx(3 / 5)


def test_evidence_coverage_ignores_missing_scores():
    row = pd.Series(
        {
            "first_name_score": 1.0,
            "last_name_score": None,
            "birth_year_score": 0.5,
            "birth_location_score": None,
            "gender_score": 1.0,
        }
    )

    assert confidence_scoring.evidence_coverage(row) == pytest.approx(2 / 3)


def test_evidence_coverage_no_scores():
    row = pd.Series(
        {
            "first_name_score": None,
            "last_name_score": None,
            "birth_year_score": None,
            "birth_location_score": None,
            "gender_score": None,
        }
    )

    assert confidence_scoring.evidence_coverage(row) == 0.0

### Testing Birth_Date_Quality ###

def test_birth_date_quality_exact_match():
    row = pd.Series({"birth_year_score": 1.0})

    assert confidence_scoring.birth_date_quality(row) == "exact birth-year match"


def test_birth_date_quality_close_match():
    row = pd.Series({"birth_year_score": 0.8})

    assert confidence_scoring.birth_date_quality(row) == "close birth-year match"


def test_birth_date_quality_weak_match():
    row = pd.Series({"birth_year_score": 0.3})

    assert confidence_scoring.birth_date_quality(row) == "weak birth-year match"


def test_birth_date_quality_mismatch():
    row = pd.Series({"birth_year_score": 0.0})

    assert confidence_scoring.birth_date_quality(row) == "birth year missing or mismatch"


def test_birth_date_quality_missing_query():
    row = pd.Series({"birth_year_score": None})

    assert confidence_scoring.birth_date_quality(row) == "birth year not supplied"

### Testing Name_Quality ###

def test_name_quality_exact_or_near_exact():
    row = pd.Series(
        {
            "first_name_score": 1.0,
            "last_name_score": 0.98,
        }
    )

    assert confidence_scoring.name_quality(row) == "exact/near-exact name match"


def test_name_quality_strong_match():
    row = pd.Series(
        {
            "first_name_score": 0.85,
            "last_name_score": 0.82,
        }
    )

    assert confidence_scoring.name_quality(row) == "strong name match"


def test_name_quality_moderate_match():
    row = pd.Series(
        {
            "first_name_score": 0.65,
            "last_name_score": 0.7,
        }
    )

    assert confidence_scoring.name_quality(row) == "moderate name match"


def test_name_quality_partial_match():
    row = pd.Series(
        {
            "first_name_score": 0.7,
            "last_name_score": 0.2,
        }
    )

    assert confidence_scoring.name_quality(row) == "partial name match"


def test_name_quality_weak_match():
    row = pd.Series(
        {
            "first_name_score": 0.3,
            "last_name_score": 0.2,
        }
    )

    assert confidence_scoring.name_quality(row) == "weak name match"

### Testing Ambiguity_Penalty ###

def test_ambiguity_penalty_no_penalty_when_not_rank_one():
    candidates = pd.DataFrame(
        [
            {"rank": 1, "rank_score": 95},
            {"rank": 2, "rank_score": 90},
        ]
    )

    row = pd.Series({"rank": 2, "rank_score": 90})

    assert confidence_scoring.ambiguity_penalty(row, candidates) == 0.0


def test_ambiguity_penalty_no_penalty_when_margin_at_least_20():
    candidates = pd.DataFrame(
        [
            {"rank": 1, "rank_score": 95},
            {"rank": 2, "rank_score": 70},
        ]
    )

    row = candidates.iloc[0]

    assert confidence_scoring.ambiguity_penalty(row, candidates) == 0.0


def test_ambiguity_penalty_five_when_margin_at_least_10():
    candidates = pd.DataFrame(
        [
            {"rank": 1, "rank_score": 95},
            {"rank": 2, "rank_score": 84},
        ]
    )

    row = candidates.iloc[0]

    assert confidence_scoring.ambiguity_penalty(row, candidates) == 5.0


def test_ambiguity_penalty_ten_when_margin_at_least_5():
    candidates = pd.DataFrame(
        [
            {"rank": 1, "rank_score": 95},
            {"rank": 2, "rank_score": 88},
        ]
    )

    row = candidates.iloc[0]

    assert confidence_scoring.ambiguity_penalty(row, candidates) == 10.0

def test_ambiguity_penalty_max_penalty_when_margin_under_5():
    candidates = pd.DataFrame(
        [
            {"rank": 1, "rank_score": 95},
            {"rank": 2, "rank_score": 92},
        ]
    )

    row = candidates.iloc[0]

    assert confidence_scoring.ambiguity_penalty(row, candidates) == 15.0

### Testing Calculate_Confidence_Scoring ###

def test_calculate_confidence_score_strong_candidate():
    candidates = pd.DataFrame(
        [
            {
                "rank": 1,
                "rank_score": 90,
                "first_name_score": 1.0,
                "last_name_score": 1.0,
                "birth_year_score": 1.0,
                "birth_location_score": 0.9,
                "gender_score": 1.0,
            }
        ]
    )

    row = candidates.iloc[0]

    assert confidence_scoring.calculate_confidence_score(row, candidates) == 100.0


def test_calculate_confidence_score_penalises_weak_name_scores():
    candidates = pd.DataFrame(
        [
            {
                "rank": 1,
                "rank_score": 80,
                "first_name_score": 0.4,
                "last_name_score": 0.5,
                "birth_year_score": 1.0,
                "birth_location_score": 0.9,
                "gender_score": 1.0,
            }
        ]
    )

    row = candidates.iloc[0]

    assert confidence_scoring.calculate_confidence_score(row, candidates) == 55.0


def test_calculate_confidence_score_penalises_birth_year_mismatch():
    candidates = pd.DataFrame(
        [
            {
                "rank": 1,
                "rank_score": 80,
                "first_name_score": 1.0,
                "last_name_score": 1.0,
                "birth_year_score": 0.0,
                "birth_location_score": 0.9,
                "gender_score": 1.0,
            }
        ]
    )

    row = candidates.iloc[0]

    assert confidence_scoring.calculate_confidence_score(row, candidates) == 75.0


def test_calculate_confidence_score_penalises_ambiguity_for_close_second_candidate():
    candidates = pd.DataFrame(
        [
            {
                "rank": 1,
                "rank_score": 90,
                "first_name_score": 1.0,
                "last_name_score": 1.0,
                "birth_year_score": 1.0,
                "birth_location_score": 0.9,
                "gender_score": 1.0,
            },
            {
                "rank": 2,
                "rank_score": 82,
                "first_name_score": 1.0,
                "last_name_score": 1.0,
                "birth_year_score": 0.8,
                "birth_location_score": 0.8,
                "gender_score": 1.0,
            },
        ]
    )

    row = candidates.iloc[0]

    assert confidence_scoring.calculate_confidence_score(row, candidates) == 90.0


def test_calculate_confidence_score_clamps_to_zero():
    candidates = pd.DataFrame(
        [
            {
                "rank": 1,
                "rank_score": 5,
                "first_name_score": 0.1,
                "last_name_score": 0.1,
                "birth_year_score": 0.0,
                "birth_location_score": 0.0,
                "gender_score": 0.0,
            }
        ]
    )

    row = candidates.iloc[0]

    assert confidence_scoring.calculate_confidence_score(row, candidates) == 0.0

### Testing Confidence_Interpertation ###

def test_confidence_interpretation_strong_candidate():
    result = confidence_scoring.confidence_interpretation(95)

    assert result == (
        "Strong candidate based on close agreement across key fields. "
        "Still requires source verification."
    )


def test_confidence_interpretation_plausible_candidate():
    result = confidence_scoring.confidence_interpretation(75)

    assert result == (
        "Plausible candidate, but at least one important field is missing, weak, or ambiguous."
    )


def test_confidence_interpretation_weak_candidate():
    result = confidence_scoring.confidence_interpretation(55)

    assert result == (
        "Weak candidate. Treat as exploratory unless supported by additional evidence."
    )


def test_confidence_interpretation_very_weak_candidate():
    result = confidence_scoring.confidence_interpretation(30)

    assert result == (
        "Very weak candidate. Likely not reliable without substantial extra evidence."
    )

### Testing Build_Confidence_Explanation ###

def test_build_confidence_explanation_strong_candidate():
    row = pd.Series(
        {
            "first_name_score": 1.0,
            "last_name_score": 1.0,
            "birth_year_score": 1.0,
            "birth_location_score": 0.9,
            "gender_score": 1.0,
        }
    )

    result = confidence_scoring.build_confidence_explanation(row)

    assert result == (
        "exact/near-exact name match; "
        "exact birth-year match; "
        "strong birth-location match; "
        "gender match."
    )


def test_build_confidence_explanation_partial_candidate():
    row = pd.Series(
        {
            "first_name_score": 0.7,
            "last_name_score": 0.3,
            "birth_year_score": 0.4,
            "birth_location_score": 0.3,
            "gender_score": 0.0,
        }
    )

    result = confidence_scoring.build_confidence_explanation(row)

    assert result == (
        "partial name match; "
        "weak birth-year match; "
        "partial birth-location match; "
        "gender mismatch or missing."
    )

### Testing Add_Confidence_Scores ###

def test_add_confidence_scores():
    candidates = pd.DataFrame(
        [
            {
                "rank": 1,
                "rank_score": 90,
                "wikitree_id": "Austen-489",
                "full_name": "Jane Austen",
                "birth_year": "1775",
                "birth_location": "Steventon, Hampshire, England",
                "first_name_score": 1.0,
                "last_name_score": 1.0,
                "birth_year_score": 1.0,
                "birth_location_score": 1.0,
                "gender_score": 1.0,
            }
        ]
    )

    results = confidence_scoring.add_confidence_scores(candidates)

    assert "confidence_score" in results.columns
    assert "confidence_interpretation" in results.columns
    assert "confidence_explanation" in results.columns

    row = results.iloc[0]

    assert row["confidence_score"] == 100.0
    assert row["wikitree_id"] == "Austen-489"
    assert row["full_name"] == "Jane Austen"


def test_add_confidence_score_empty_dataframe():
    candidates = pd.DataFrame()

    results = confidence_scoring.add_confidence_scores(candidates)

    assert results.empty