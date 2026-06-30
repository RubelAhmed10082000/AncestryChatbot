from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_homepage_loads():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Ancestry Chatbot" in response.text
    assert "chat-window" in response.text
    assert "chat-form" in response.text


def test_static_app_js_loads():

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "questions" in response.text
    assert "searchCandidates" in response.text


def test_static_style_css_loads():
    response = client.get("/static/style.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    assert ".chat-window" in response.text
    assert ".candidate-card" in response.text


def test_candidate_search_from_ui_payload():
    payload = {
        "first_name": "Jane",
        "last_name": "Austen",
        "birth_year": 1775,
        "birth_location": "Hampshire, England",
        "gender": "Female",
        "top_k": 5,
        "min_score": 0,
    }

    response = client.post("/api/candidates/search", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["query"]["first_name"] == "Jane"
    assert data["query"]["last_name"] == "Austen"
    assert data["count"] >= 1
    assert isinstance(data["candidates"], list)

    top_candidate = data["candidates"][0]

    assert "rank" in top_candidate
    assert "rank_score" in top_candidate
    assert "confidence_score" in top_candidate
    assert "wikitree_id" in top_candidate
    assert "full_name" in top_candidate