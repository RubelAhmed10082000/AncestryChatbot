from app.data_pipeline import extract

def test_process_seed_extract(monkeypatch):
    monkeypatch.setattr(extract.time, "sleep", lambda seconds: None)

    seed = {
        "label": "Jane Austen",
        "first_name": "Jane",
        "last_name": "Austen",
        "birth_date": "1775-12-16",
    }

    search_response = [
        {
            "matches": [
                {
                    "Id": 5688919,
                    "Name": "Austen-489",
                    "FirstName": "Jane",
                    "LastNameAtBirth": "Austen",
                    "BirthDate": "1775-12-16",
                    "BirthLocation": "Steventon, Hampshire, England",
                    "Father": 1124609,
                    "Mother": 1124608,
                }
            ]
        }
    ]

    ancestor_response = [
        {
            "ancestors": [
                {
                    "Id": 1124609,
                    "Name": "Austen-109",
                    "FirstName": "George",
                    "LastNameAtBirth": "Austen",
                    "Father": 5688972,
                    "Mother": 5688808,
                },
                {
                    "Id": 1124608,
                    "Name": "Leigh-138",
                    "FirstName": "Cassandra",
                    "LastNameAtBirth": "Leigh",
                    "Father": 1124600,
                    "Mother": 1124601,
                },
            ]
        }
    ]

    monkeypatch.setattr(extract, "search_person", lambda seed: search_response)
    monkeypatch.setattr(extract, "get_ancestors", lambda profile_key, depth: ancestor_response)

    raw_search_results = {}
    raw_ancestor_results = {}
    selected_seeds = []
    people_by_wikitree_id = {}
    relationships = []

    extract.process_seed(
        seed,
        raw_search_results,
        raw_ancestor_results,
        selected_seeds,
        people_by_wikitree_id,
        relationships,
    )

    assert "Jane Austen" in raw_search_results
    assert "Austen-489" in raw_ancestor_results

    assert selected_seeds[0]["seed_label"] == "Jane Austen"
    assert selected_seeds[0]["wikitree_id"] == "Austen-489"

    assert "Austen-489" in people_by_wikitree_id
    assert "Austen-109" in people_by_wikitree_id
    assert "Leigh-138" in people_by_wikitree_id

    relationship_types = {row["relationship_type"] for row in relationships}
    assert "father_of" in relationship_types
    assert "mother_of" in relationship_types