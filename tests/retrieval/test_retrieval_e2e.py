import pytest

import app.retrieval.candidate_retrieval as candidate_retrieval


@pytest.mark.e2e
def test_candidate_retrievaljane_austen():
    retriever = candidate_retrieval.CandidateRetriever(
        schema_dir=candidate_retrieval.DEFAULT_SCHEMA_DIR
    )

    results = retriever.find_candidates(
        first_name="Jane",
        last_name="Austen",
        birth_year=1775,
        birth_location="Steventon, Hampshire, England",
        gender="Female",
        top_k=5,
    )

    assert not results.empty
    assert results.iloc[0]["wikitree_id"] == "Austen-489"

@pytest.mark.e2e
def test_candidate_retrieval_charles_darwin():
    retriever = candidate_retrieval.CandidateRetriever(
        schema_dir=candidate_retrieval.DEFAULT_SCHEMA_DIR
    )

    results = retriever.find_candidates(
        first_name="Charles",
        last_name="Darwin",
        birth_year=1809,
        birth_location="Shrewsbury, Shropshire, England",
        gender="Male",
        top_k=5,
    )

    assert not results.empty
    assert results.iloc[0]["wikitree_id"] == "Darwin-15"