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
    

