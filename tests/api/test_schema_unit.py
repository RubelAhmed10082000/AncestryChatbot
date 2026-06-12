import pytest 
from pydantic import ValidationError

from app.api.schema import CandidateSearchRequest

def test_candidate_search_request():
    request = CandidateSearchRequest(
        first_name="Jane",
        last_name="Austen",
    )

    assert request.first_name == "Jane"
    assert request.last_name == "Austen"
    assert request.top_k == 5
    assert request.min_score == 0.0

def test_candidate_search_request_low_top_k():
    with pytest.raises(ValidationError):
        CandidateSearchRequest(top_k=0)

def test_candidate_search_request_high_top_k():
    with pytest.raises(ValidationError):
        CandidateSearchRequest(top_k=21)

def test_candidate_search_request_invalid_score():
    with  pytest.raises(ValidationError):
        CandidateSearchRequest(min_score=101)