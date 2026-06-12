import pandas as pd
import app.api.services as services

### Testing dataframe_to_records() ###
def test_dataframe_to_records():
    df = pd.DataFrame(
        [
            {
                "name": "Jane Austen",
                "birth_year": float("nan"),
                "score": 95.5,
            }
        ]
    )

    records = services.dataframe_to_records(df)

    assert records == [
        {
            "name": "Jane Austen",
            "birth_year": None,
            "score": 95.5,
        }
    ]


def test_dataframe_to_records_empty_dataframe():
    df = pd.DataFrame()

    records = services.dataframe_to_records(df)

    assert records == []

### Testing search_candidate() ###

def test_search_candidate_calls_retriever_and_confidence_scoring(monkeypatch):
    class FakeRetriever:
        def __init__(self, schema_dir):
            self.schema_dir = schema_dir

        def find_candidates(
            self,
            first_name=None,
            last_name=None,
            birth_year=None,
            birth_location=None,
            gender=None,
            top_k=5,
            min_score=0.0,
        ):
            assert first_name == "Jane"
            assert last_name == "Austen"
            assert birth_year == 1775
            assert birth_location == "Hampshire, England"
            assert gender == "Female"
            assert top_k == 5
            assert min_score == 0.0

            return pd.DataFrame(
                [
                    {
                        "rank": 1,
                        "rank_score": 100.0,
                        "wikitree_id": "Austen-489",
                        "full_name": "Jane Austen",
                    }
                ]
            )

    def fake_add_confidence_scores(candidates):
        candidates = candidates.copy()
        candidates["confidence_score"] = 100.0
        return candidates

    monkeypatch.setattr(services, "CandidateRetriever", FakeRetriever)
    monkeypatch.setattr(services, "add_confidence_scores", fake_add_confidence_scores)

    results = services.search_candidate(
        first_name="Jane",
        last_name="Austen",
        birth_year=1775,
        birth_location="Hampshire, England",
        gender="Female",
        top_k=5,
        min_score=0.0,
    )

    assert len(results) == 1
    assert results[0]["wikitree_id"] == "Austen-489"
    assert results[0]["confidence_score"] == 100.0