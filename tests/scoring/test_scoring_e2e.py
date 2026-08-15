import pytest

import app.scoring.confidence_scoring as confidence_scoring
from app.retrieval.candidate_retrieval import CandidateRetriever


@pytest.mark.e2e
def test_confidence_real_candidate_retrieval_for_jane_austen():

    retriever = CandidateRetriever(
    schema_dir=confidence_scoring.DEFAULT_SCHEMA_DIR
)

    candidates = retriever.find_candidates(
        first_name="Jane",
        last_name="Austen",
        birth_year=1775,
        birth_location="Steventon, Hampshire, England",
        gender="Female",
        top_k=5,
    )

    results = confidence_scoring.add_confidence_scores(candidates)

    assert not results.empty

    top = results.iloc[0]

    assert top["wikitree_id"] == "Austen-489"
    assert top["confidence_score"] >= 80
    assert "confidence_explanation" in results.columns