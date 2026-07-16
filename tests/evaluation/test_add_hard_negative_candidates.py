import pandas as pd

from scripts import add_hard_negative_candidates as hard_negatives


def test_merge_hard_negatives_adds_search_matches_but_not_seed_roots():
    people = pd.DataFrame(
        [
            {
                "person_id": "5688919",
                "wikitree_id": "Austen-489",
                "first_name": "Jane",
                "last_name_at_birth": "Austen",
            }
        ]
    )
    seeds = pd.DataFrame(
        [{"seed_label": "Jane Austen", "wikitree_id": "Austen-489"}]
    )
    raw_search_results = {
        "Jane Austen": [
            {
                "matches": [
                    {
                        "Id": 5688919,
                        "Name": "Austen-489",
                        "FirstName": "Jane",
                        "LastNameAtBirth": "Austen",
                    },
                    {
                        "Id": 28677854,
                        "Name": "Austen-1465",
                        "FirstName": "Jane",
                        "LastNameAtBirth": "Austen",
                        "BirthDate": "1775-00-00",
                    },
                ]
            }
        ]
    }

    merged, manifest = hard_negatives.merge_hard_negative_candidates(
        people,
        seeds,
        raw_search_results,
    )

    assert set(merged["wikitree_id"]) == {"Austen-489", "Austen-1465"}
    assert manifest["wikitree_id"].tolist() == ["Austen-1465"]
    assert not manifest.iloc[0]["was_already_searchable"]
    assert "Austen-489" not in set(manifest["wikitree_id"])
