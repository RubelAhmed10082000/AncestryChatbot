import pytest
import app.data_pipeline.extract as extract

@pytest.mark.e2e
def test_live_wikitree_search():
    seed = {
        "label": "Jane Austen",
        "first_name": "Jane",
        "last_name": "Austen",
        "birth_date": "1775-12-16",
    }

    response = extract.search_person(seed)
    selected = extract.choose_best_search_match(seed, response)

    assert selected is not None
    assert selected["Name"] == 'Austen-489' 