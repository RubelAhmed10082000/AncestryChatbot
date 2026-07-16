import pandas as pd
import pytest

from scripts import run_retrieval_evaluation as evaluation


class FakeRetriever:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def find_candidates(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def fake_add_confidence_scores(candidates):
    if candidates.empty:
        return candidates

    results = candidates.copy()
    results["confidence_score"] = [88.0 - index for index in range(len(results))]
    results["confidence_interpretation"] = "Test interpretation"
    return results


def make_cases():
    return pd.DataFrame(
        [
            {
                "case_id": "Austen-489__name_year",
                "seed_label": "Jane Austen",
                "condition": "name_year",
                "expected_wikitree_id": "Austen-489",
                "first_name": "Jane",
                "last_name": "Austen",
                "birth_year": "1775.0",
                "birth_location": "",
                "gender": "",
                "perturbation_notes": "Names and year supplied.",
            },
            {
                "case_id": "Missing-1__name_year",
                "seed_label": "Missing Person",
                "condition": "name_year",
                "expected_wikitree_id": "Missing-1",
                "first_name": "Missing",
                "last_name": "Person",
                "birth_year": "1900",
                "birth_location": "",
                "gender": "",
                "perturbation_notes": "No matching candidate.",
            },
        ]
    )


def candidate_rows():
    score_values = {
        "first_name_score": 1.0,
        "last_name_score": 1.0,
        "birth_year_score": 1.0,
        "birth_location_score": None,
        "gender_score": None,
    }

    return pd.DataFrame(
        [
            {
                "rank": 1,
                "rank_score": 90.0,
                "wikitree_id": "Other-1",
                "full_name": "Jane Other",
                **score_values,
            },
            {
                "rank": 2,
                "rank_score": 85.0,
                "wikitree_id": "Austen-489",
                "full_name": "Jane Austen",
                **score_values,
            },
        ]
    )


def test_evaluate_cases_uses_frozen_parameters_and_calculates_rank():
    retriever = FakeRetriever([candidate_rows(), pd.DataFrame()])

    results = evaluation.evaluate_cases(
        retriever,
        make_cases(),
        confidence_scorer=fake_add_confidence_scores,
    )

    assert len(retriever.calls) == 2
    assert all(call["top_k"] == 5 for call in retriever.calls)
    assert all(call["min_score"] == 0.0 for call in retriever.calls)
    assert retriever.calls[0]["birth_year"] == 1775
    assert retriever.calls[0]["birth_location"] is None
    assert retriever.calls[0]["gender"] is None

    ranked_below_first = results.iloc[0]
    assert ranked_below_first["expected_rank"] == 2
    assert not ranked_below_first["top_1_correct"]
    assert ranked_below_first["top_3_correct"]
    assert ranked_below_first["reciprocal_rank"] == 0.5
    assert ranked_below_first["returned_top_wikitree_id"] == "Other-1"
    assert ranked_below_first["top_confidence_score"] == 88.0
    assert ranked_below_first["failure_reason"] == "expected_candidate_ranked_below_first"

    missing = results.iloc[1]
    assert pd.isna(missing["expected_rank"])
    assert missing["retrieval_failed"]
    assert missing["failure_reason"] == "no_candidates_returned"


def test_build_evaluation_summary_calculates_required_metrics():
    results = pd.DataFrame(
        [
            {
                "condition": "full_profile",
                "top_1_correct": True,
                "top_3_correct": True,
                "top_5_correct": True,
                "reciprocal_rank": 1.0,
                "top_confidence_score": 90.0,
                "retrieval_failed": False,
                "candidate_count": 5,
            },
            {
                "condition": "full_profile",
                "top_1_correct": False,
                "top_3_correct": True,
                "top_5_correct": True,
                "reciprocal_rank": 0.5,
                "top_confidence_score": 80.0,
                "retrieval_failed": False,
                "candidate_count": 5,
            },
            {
                "condition": "full_profile",
                "top_1_correct": False,
                "top_3_correct": False,
                "top_5_correct": False,
                "reciprocal_rank": 0.0,
                "top_confidence_score": 70.0,
                "retrieval_failed": True,
                "candidate_count": 5,
            },
        ]
    )

    overall = evaluation.build_evaluation_summary(results).iloc[0]

    assert overall["total_cases"] == 3
    assert overall["top_1_accuracy"] == pytest.approx(1 / 3, abs=1e-6)
    assert overall["top_3_accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert overall["top_5_accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert overall["mean_reciprocal_rank"] == 0.5
    assert overall["mean_confidence"] == 80.0
    assert overall["mean_confidence_correct_top_1"] == 90.0
    assert overall["mean_confidence_incorrect_top_1"] == 75.0
    assert overall["failed_retrieval_count"] == 1


def test_failure_cases_include_every_incorrect_top_rank():
    results = pd.DataFrame(
        {
            "case_id": ["correct", "rank_two", "absent"],
            "top_1_correct": [True, False, False],
            "first_name_score": [1.0, 0.8, 0.3],
        }
    )

    failures = evaluation.build_failure_cases(results)

    assert failures["case_id"].tolist() == ["rank_two", "absent"]
    assert failures["first_name_score"].tolist() == [0.8, 0.3]
