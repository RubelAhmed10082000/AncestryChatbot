import pytest 
import pandas
import json
from app.data_pipeline import extract

class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.raise_for_status_called = False
    
    def raise_for_status(self):
        self.raise_for_status_called = True
    
    def json(self):
        return self.payload

def test_wiki_tree(monkeypatch):
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse({"status": 0})
    
    monkeypatch.setattr(extract.requests, "get", fake_get)

    result = extract.call_wikitree({"action":'searchPerson'})
    
    assert result == {"status": 0}
    assert captured["url"] == extract.BASE_URL
    assert captured["params"]["action"] == "searchPerson" 
    assert captured["params"]["appId"] == extract.APP_ID
    assert captured["timeout"] == 30

def test_birth_date_included_search_person(monkeypatch):
    captured = {}

    def fake_call_wikitree(params):
        captured.update(params)
        return {"ok":True}

    monkeypatch.setattr(extract, "call_wikitree",fake_call_wikitree)

    seed = {
        "first_name": "Jane",
        "last_name": "Austen",
        "birth_date": "1775-12-16",
    }

    result = extract.search_person(seed)

    assert result == {"ok": True}
    assert captured["action"] == "searchPerson"
    assert captured["FirstName"] == "Jane"
    assert captured["LastName"] == "Austen"
    assert captured["BirthDate"] == "1775-12-16"
    assert captured["limit"] == 10
    assert captured["fields"] == extract.FIELDS

def test_search_person_ommited(monkeypatch):
    captured = {}

    def fake_call_wikitree(params):
        captured.update(params)
        return {"ok": True}
    
    monkeypatch.setattr(extract, "call_wikitree", fake_call_wikitree)

    seed = {
        "first_name": "Jane",
        "last_name": "Austen",
    }

    extract.search_person(seed)

    assert "BirthDate" not in captured
    
def test_get_ancestors(monkeypatch):
    captured = {}

    def fake_call_wikitree(params):
        captured.update(params)
        return [{"ancestors": []}]
    
    monkeypatch.setattr(extract, "call_wikitree", fake_call_wikitree)
    
    result = extract.get_ancestors("Austen-489", depth=3)

    assert result == [{"ancestors": []}]
    assert captured["action"] == "getPeople&ancestors"
    assert captured["key"] == "Austen-489"
    assert captured["depth"] == 3
    assert captured["fields"] == extract.FIELDS
    assert captured["resolveRedirect"] == "1"

def test_profile_extraction():
    response = [
        {
            "profile": {"Id": 1, "Name": "Root-1"},
            "matches": [
                {"Id": 2, "Name": "Match-1"},
                {"bad": "ignored"},
            ],
            "ancestors": [
                {"Id": 3, "Name": "Ancestor-1"},
            ],
        }
    ]

    profiles = extract.flatten_api_profiles(response)

    assert {"Id": 1, "Name": "Root-1"} in profiles
    assert {"Id": 2, "Name": "Match-1"} in profiles
    assert {"Id": 3, "Name": "Ancestor-1"} in profiles
    assert {"bad": "ignored"} not in profiles

def test_flatten_api_profiles_rejects_invalid_input():
    assert extract.flatten_api_profiles(None) == []
    assert extract.flatten_api_profiles("bad input") == []

def test_choose_best_search_match_uses_wikitree_id():
    seed = {
        "label": "Mark Twain",
        "known_wikitree_id": "Clemens-1"
    }

    selected = extract.choose_best_search_match(seed, search_response = [])

    assert selected["Name"] == "Clemens-1"
    assert selected["SeedLabel"] == "Mark Twain"
    assert selected["SelectionMethod"] == "known_wikitree_id"

def test_choose_best_search_match():
    seed = {
        "label": "Jane Austen",
        "birth_date": "1775-12-16",
    }

    search_response = [
        {
            "matches": [
                {"Id": 1, "Name": "Austen-999", "BirthDate": "1775-00-00"},
            ]
        }
    ]

    selected = extract.choose_best_search_match(seed, search_response)

    assert selected["Name"] == "Austen-999"
    assert selected["SelectionMethod"] == "first_search_result"

def test_choose_best_search_none():
    seed = {"label": "Unknown", "birth_date": "1900-01-01"}

    selected = extract.choose_best_search_match(seed, [])

    assert selected is None 

def test_normalize_person():
    profile = {
        "Id": 5688919,
        "Name": "Austen-489",
        "FirstName": "Jane",
        "MiddleName": "",
        "LastNameAtBirth": "Austen",
        "LastNameCurrent": "Austen",
        "BirthDate": "1775-12-16",
        "BirthLocation": "Steventon, Hampshire, England",
        "DeathDate": "1817-07-18",
        "DeathLocation": "Winchester, Hampshire, England",
        "Gender": "Female",
        "Father": 1124609,
        "Mother": 1124608,
        "Privacy": 60,
        "DataStatus": {"BirthDate": "certain"},
    }

    row = extract.normalise_person(profile)

    assert row["person_id"] == 5688919
    assert row["wikitree_id"] == "Austen-489"
    assert row["first_name"] == "Jane"
    assert row["father_id"] == 1124609
    assert row["mother_id"] == 1124608
    assert json.loads(row["data_status"]) == {"BirthDate": "certain"}

def test_normalize_person_missing_date():
    row = extract.normalise_person({"Id": 1, "Name": "Test-1"})

    assert row["person_id"] == 1
    assert row["wikitree_id"] == 'Test-1'
    assert row["data_status"] is None

def test_extract_relationships():
    profile = {
        "Id": 5688919,
        "Name": "Austen-489",
        "Father": 1124609,
        "Mother": 1124608,
    }

    rows = extract.extract_relationships(profile)

    assert {
        "parent_id": 1124609,
        "child_id": 5688919,
        "child_wikitree_id": "Austen-489",
        "relationship_type": "father_of",
    } in rows

    assert {
        "parent_id": 1124608,
        "child_id": 5688919,
        "child_wikitree_id": "Austen-489",
        "relationship_type": "mother_of",
    } in rows

def test_extract_relationship_child_missing():
    rows = extract.extract_relationships({
        "Name": "Austen-489",
        "Father": 1124609,
        "Mother": 1124608,
    })

    assert rows == []