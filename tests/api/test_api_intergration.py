from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import app.api.main as api_main

client = TestClient(api_main.app)

@pytest.mark.skipif(
    not Path("data/wikitree_schema/person.csv").exists(),
    reason="Transformed schema CSVs not found. Run transform.py first.",
)
def test_candidate_search_jane_austen_real_data():
    response = client.post(
        "/api/candidates/search",
        json={
            "first_name": "Jane",
            "last_name": "Austen",
            "birth_year": 1775,
            "birth_location": "Hampshire, England",
            "gender": "Female",
            "top_k": 5,
            "min_score": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["count"] >= 1

    wikitree_ids = [candidate["wikitree_id"] for candidate in data["candidates"]]
    assert "Austen-489" in wikitree_ids