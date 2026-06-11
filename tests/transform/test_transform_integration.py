import app.data_pipeline.transform as transform 
import json
import pandas as pd
import pytest

def test_transform_pipeline():
    raw_people = pd.DataFrame(
        [
            {
                "person_id": "5688919",
                "wikitree_id": "Austen-489",
                "first_name": "Jane",
                "middle_name": "",
                "last_name_at_birth": "Austen",
                "last_name_current": "Austen",
                "birth_date": "1775-12-16",
                "birth_location": "Steventon, Hampshire, England",
                "death_date": "1817-07-18",
                "death_location": "Winchester, Hampshire, England",
                "gender": "Female",
                "father_id": "1124609",
                "mother_id": "1124608",
                "privacy": "60",
                "data_status": json.dumps(
                    {
                        "BirthDate": "certain",
                        "DeathDate": "certain",
                        "BirthLocation": "certain",
                    }
                ),
            },
            {
                "person_id": "1124609",
                "wikitree_id": "Austen-109",
                "first_name": "George",
                "middle_name": "",
                "last_name_at_birth": "Austen",
                "last_name_current": "Austen",
                "birth_date": "1731-05-01",
                "birth_location": "Royal Tunbridge Wells, Kent, England",
                "death_date": "1805-01-21",
                "death_location": "Bath, Somerset, England",
                "gender": "Male",
                "father_id": "",
                "mother_id": "",
                "privacy": "60",
                "data_status": "{}",
            },
            {
                "person_id": "1124608",
                "wikitree_id": "Leigh-138",
                "first_name": "Cassandra",
                "middle_name": "",
                "last_name_at_birth": "Leigh",
                "last_name_current": "Austen",
                "birth_date": "1739-09-26",
                "birth_location": "Harpsden, Oxfordshire, England",
                "death_date": "1827-01-17",
                "death_location": "Chawton, Hampshire, England",
                "gender": "Female",
                "father_id": "",
                "mother_id": "",
                "privacy": "60",
                "data_status": "{}",
            },
        ]
    )

    raw_relationships = pd.DataFrame(
        [
            {
                "parent_id": "1124609",
                "child_id": "5688919",
                "child_wikitree_id": "Austen-489",
                "relationship_type": "father_of",
            },
            {
                "parent_id": "1124608",
                "child_id": "5688919",
                "child_wikitree_id": "Austen-489",
                "relationship_type": "mother_of",
            },
        ]
    )

    people = transform.prepare_people(raw_people)
    relationships = transform.prepare_relationships(raw_relationships)

    person_table = transform.build_person_table(people, relationships)
    names_table = transform.build_names_table(people)
    event_table, rejections = transform.build_event_table(people, relationships)
    crosswalk = transform.build_id_crosswalk(people)
    quality_report = transform.build_quality_report(
        raw_people,
        people,
        raw_relationships,
        relationships,
        person_table,
        names_table,
        event_table,
        rejections,
    )

    assert len(person_table) == 3
    assert len(names_table) == 3
    assert len(crosswalk) == 3
    assert rejections.empty

    event_types = event_table["Event_Type"].value_counts().to_dict()

    assert event_types["birth"] == 3
    assert event_types["death"] == 3
    assert event_types["father_of"] == 1
    assert event_types["mother_of"] == 1

    metrics = dict(zip(quality_report["metric"], quality_report["value"]))

    assert metrics["raw_people_rows"] == 3
    assert metrics["schema_people_rows"] == 3
    assert metrics["schema_names_rows"] == 3
    assert metrics["raw_relationship_rows"] == 2
    assert metrics["clean_relationship_rows"] == 2
    assert metrics["relationship_rejections"] == 0