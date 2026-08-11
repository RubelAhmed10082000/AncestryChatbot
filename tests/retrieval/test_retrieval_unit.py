import math
import pandas as pd
import pytest

import app.retrieval.candidate_retrieval as candidate_retrieval

### Testing Text Cleaning ###

def test_clean_text_unknown_values():
    assert candidate_retrieval.clean_text(None) is None
    assert candidate_retrieval.clean_text("") is None
    assert candidate_retrieval.clean_text("Unknown") is None
    assert candidate_retrieval.clean_text("  Jane   Austen  ") == "Jane Austen"


def test_lowercases_and_removes_punctuation():
    result = candidate_retrieval.normalise_for_matching("  Jane-Austen, Esq. ")

    assert result == "jane austen esq"


def test_token_set():
    result = candidate_retrieval.token_set("Shrewsbury, Shropshire, England")

    assert result == {"shrewsbury", "shropshire", "england"}

### Testing String Similarity ###

def test_sequence_similarity():
    assert candidate_retrieval.sequence_similarity("Jane", "Jane") == 1.0


def test_sequence_similarity_case_and_punctuation_insensitive():
    assert candidate_retrieval.sequence_similarity("Jane-Austen", "jane austen") == 1.0


def test_sequence_similarity_missing_value_is_zero():
    assert candidate_retrieval.sequence_similarity("", "Jane") == 0.0
    assert candidate_retrieval.sequence_similarity("Jane", "") == 0.0


def test_best_string_similarity_returns_best_candidate_value():
    score = candidate_retrieval.best_string_similarity(
        "Samuel",
        ["Sam", "Samuel", "Samuell"],
    )

    assert score == 1.0


def test_best_string_similarity_missing_query():
    score = candidate_retrieval.best_string_similarity(None, ["Jane"])

    assert math.isnan(score)


def test_best_string_similarity_returns_zero():
    score = candidate_retrieval.best_string_similarity("Jane", ["", None])

    assert score == 0.0

def test_token_overlap_similarity():
    score = candidate_retrieval.token_overlap_similarity(
        "Shrewsbury Shropshire",
        "Shrewsbury England",
    )

    assert score == pytest.approx(1 / 3)

### Testing Location Similarity ###

def test_best_location_similarity_exact_location_is_one():
    score = candidate_retrieval.best_location_similarity(
        "Shrewsbury, Shropshire, England",
        "Shrewsbury, Shropshire, England",
    )

    assert score == 1.0


def test_best_location_similarity_partial_location_has_positive_score():
    score = candidate_retrieval.best_location_similarity(
        "Shrewsbury",
        "Shrewsbury, Shropshire, England",
    )

    assert score > 0


def test_best_location_similarity_missing_query_returns_nan():
    score = candidate_retrieval.best_location_similarity(
        None,
        "Shrewsbury, Shropshire, England",
    )

    assert math.isnan(score)


def test_best_location_similarity_missing_candidate_returns_zero():
    score = candidate_retrieval.best_location_similarity(
        "Shrewsbury",
        None,
    )

    assert score == 0.0

### Testing Year Similarity ###

def test_year_similarity_exact_match_is_one():
    assert candidate_retrieval.year_similarity(1775, 1775) == 1.0


def test_year_similarity_declines_within_window():
    assert candidate_retrieval.year_similarity(1775, 1785) == 0.5


def test_year_similarity_twenty():
    assert candidate_retrieval.year_similarity(1775, 1795) == 0.0


def test_year_similarity_bad_candidate_year():
    assert candidate_retrieval.year_similarity(1775, "not-a-year") == 0.0


def test_year_similarity_missing_query():
    score = candidate_retrieval.year_similarity(None, 1775)

    assert math.isnan(score)

### Testing Gender Similarity ###

def test_gender_similarity_first_letter():
    assert candidate_retrieval.gender_similarity("Female", "F") == 1.0
    assert candidate_retrieval.gender_similarity("Male", "M") == 1.0


def test_gender_similarity_mismatch():
    assert candidate_retrieval.gender_similarity("Female", "Male") == 0.0


def test_gender_similarity_missing_query():
    score = candidate_retrieval.gender_similarity(None, "Female")

    assert math.isnan(score)


def test_gender_similarity_missing_candidate():
    assert candidate_retrieval.gender_similarity("Female", None) == 0.0

### Testing Score Adjustment ###

def test_adjust_score_boosts():
    scores = {
        "first_name_score": 1.0,
        "last_name_score": 1.0,
        "birth_year_score": 1.0,
        "birth_location_score": 1.0,
    }

    result = candidate_retrieval.adjust_score(
        score=80.0,
        scores=scores,
        query_birth_year=1775,
        candidate_birth_year=1775,
    )

    assert result == 96.0


def test_adjust_score_penalises_large_year_gap():
    scores = {
        "first_name_score": 1.0,
        "last_name_score": 1.0,
        "birth_year_score": 0.0,
        "birth_location_score": 1.0,
    }

    result = candidate_retrieval.adjust_score(
        score=80.0,
        scores=scores,
        query_birth_year=1775,
        candidate_birth_year=1820,
    )

    assert result == 55.0


def test_adjust_score_penalises_weak_first_name():
    scores = {
        "first_name_score": 0.4,
        "last_name_score": 1.0,
        "birth_year_score": 1.0,
        "birth_location_score": 1.0,
    }

    result = candidate_retrieval.adjust_score(
        score=80.0,
        scores=scores,
        query_birth_year=None,
        candidate_birth_year=1775,
    )

    assert result == 60.0


def test_adjust_score_is_clamped_between_zero_and_100():
    scores = {
        "first_name_score": 0.0,
        "last_name_score": 0.0,
        "birth_year_score": 0.0,
        "birth_location_score": 0.0,
    }

    result = candidate_retrieval.adjust_score(
        score=5.0,
        scores=scores,
        query_birth_year=1775,
        candidate_birth_year=1900,
    )

    assert result == 0.0

### Testing Rounding ###

def test_round_or_none_rounds_valid_score():
    assert candidate_retrieval.round_or_none(0.98765) == 0.988


def test_round_or_none_returns_none_for_nan():
    assert candidate_retrieval.round_or_none(math.nan) is None

### Testing Explanation ### 

def test_explain_matches_strong_and_partial():
    scores = {
        "first_name_score": 1.0,
        "last_name_score": 0.7,
        "birth_year_score": 0.0,
        "birth_location_score": math.nan,
        "gender_score": 1.0,
    }

    explanation = candidate_retrieval.explain_matches(scores)

    assert explanation == "Strong match on first name, gender; partial match on last name."


def test_explain_matches_no_match():
    scores = {
        "first_name_score": 0.0,
        "last_name_score": 0.0,
        "birth_year_score": 0.0,
    }

    assert candidate_retrieval.explain_matches(scores) == "No strong field-level match."