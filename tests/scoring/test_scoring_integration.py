import pandas as pd
import pytest

import app.scoring.confidence_scoring as confidence_scoring


class FakeCandidateRetriever:
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
        return pd.DataFrame(
            [
                {
                    "rank": 1,
                    "rank_score": 92,
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


@pytest.mark.e2e
def test_confidence_main_writes_output_csv(tmp_path, monkeypatch):
    output_path = tmp_path / "confidence_results.csv"

    monkeypatch.setattr(
        confidence_scoring,
        "load_candidate_retriever",
        lambda module_path: FakeCandidateRetriever,
    )

    monkeypatch.setattr(
        confidence_scoring,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "schema_dir": tmp_path,
                "candidate_module_path": tmp_path / "candidate_retrieval.py",
                "first_name": "Jane",
                "last_name": "Austen",
                "birth_year": 1775,
                "birth_location": "Steventon, Hampshire, England",
                "gender": "Female",
                "top_k": 5,
                "min_score": 0.0,
                "output": output_path,
            },
        )(),
    )

    confidence_scoring.main()

    assert output_path.exists()

    output = pd.read_csv(output_path)

    assert len(output) == 1
    assert output.iloc[0]["wikitree_id"] == "Austen-489"
    assert output.iloc[0]["confidence_score"] == 100.0