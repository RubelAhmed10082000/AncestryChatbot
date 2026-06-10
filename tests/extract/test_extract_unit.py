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
