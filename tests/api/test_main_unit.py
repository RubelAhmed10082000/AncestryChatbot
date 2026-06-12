from fastapi.testclient import TestClient

import app.api.main as api_main


client = TestClient(api_main.app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_candidates_endpoint_returns_candidates(monkeypatch):
    def fake_search_candidate(**kwargs):
        assert kwargs["first_name"] == "Jane"
        assert kwargs["last_name"] == "Austen"
        assert kwargs["birth_year"] == 1775
        assert kwargs["birth_location"] == "Hampshire, England"
        assert kwargs["gender"] == "Female"
        assert kwargs["top_k"] == 5
        assert kwargs["min_score"] == 0.0

        return [
            {
                "rank": 1,
                "rank_score": 100.0,
                "confidence_score": 100.0,
                "wikitree_id": "Austen-489",
                "full_name": "Jane Austen",
                "birth_year": "1775",
                "birth_location": "Steventon, Hampshire, England",
                "confidence_explanation": "exact/near-exact name match; exact birth-year match.",
                "confidence_interpretation": "Strong candidate based on close agreement across key fields.",
                "profile_url": "https://www.wikitree.com/wiki/Austen-489",
            }
        ]

    monkeypatch.setattr(api_main, "search_candidate", fake_search_candidate)

    response = client.post(
        "/api/candidates/search",
        json={
            "first_name": "Jane",
            "last_name": "Austen",
            "birth_year": 1775,
            "birth_location": "Hampshire, England",
            "gender": "Female",
            "top_k": 5,
            "min_score": 0.0,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["count"] == 1
    assert data["candidates"][0]["wikitree_id"] == "Austen-489"
    assert data["candidates"][0]["confidence_score"] == 100.0


def test_search_candidates_endpoint_rejects_invalid_top_k():
    response = client.post(
        "/api/candidates/search",
        json={
            "first_name": "Jane",
            "last_name": "Austen",
            "top_k": 999,
        },
    )

    assert response.status_code == 422


def test_search_candidates_endpoint_returns_500_when_service_file_missing(monkeypatch):
    def fake_search_candidate(**kwargs):
        raise FileNotFoundError("Missing required file: person.csv")

    monkeypatch.setattr(api_main, "search_candidate", fake_search_candidate)

    response = client.post(
        "/api/candidates/search",
        json={
            "first_name": "Jane",
            "last_name": "Austen",
        },
    )

    assert response.status_code == 500
    assert "Missing required file" in response.json()["detail"]


def test_get_tree_by_person_id_returns_tree(monkeypatch):
    def fake_tree(**kwargs):
        assert kwargs["person_id"] == "person-jane"
        assert kwargs["generations"] == 3
        assert kwargs["include_missing_stubs"] is False

        return {
            "root_person_id": "person-jane",
            "generations": 3,
            "summary": [{"metric": "node_count", "value": 3}],
            "nodes": [
                {
                    "person_id": "person-jane",
                    "wikitree_id": "Austen-489",
                    "full_name": "Jane Austen",
                    "generation": 0,
                }
            ],
            "edges": [],
        }

    monkeypatch.setattr(api_main, "tree", fake_tree)

    response = client.get("/api/tree/person-jane?generations=3")

    assert response.status_code == 200

    data = response.json()
    assert data["root_person_id"] == "person-jane"
    assert data["nodes"][0]["wikitree_id"] == "Austen-489"


def test_get_tree_by_person_id_returns_404_for_unknown_person(monkeypatch):
    def fake_tree(**kwargs):
        raise ValueError("No person found with Person_ID=unknown")

    monkeypatch.setattr(api_main, "tree", fake_tree)

    response = client.get("/api/tree/unknown")

    assert response.status_code == 404
    assert "No person found" in response.json()["detail"]


def test_get_tree_by_wikitree_id_returns_tree(monkeypatch):
    def fake_tree(**kwargs):
        assert kwargs["wikitree_id"] == "Austen-489"
        assert kwargs["generations"] == 3

        return {
            "root_person_id": "person-jane",
            "generations": 3,
            "summary": [{"metric": "node_count", "value": 3}],
            "nodes": [
                {
                    "person_id": "person-jane",
                    "wikitree_id": "Austen-489",
                    "full_name": "Jane Austen",
                    "generation": 0,
                }
            ],
            "edges": [],
        }

    monkeypatch.setattr(api_main, "tree", fake_tree)

    response = client.get("/api/tree/by-wikitree/Austen-489?generations=3")

    assert response.status_code == 200

    data = response.json()
    assert data["root_person_id"] == "person-jane"
    assert data["nodes"][0]["wikitree_id"] == "Austen-489"