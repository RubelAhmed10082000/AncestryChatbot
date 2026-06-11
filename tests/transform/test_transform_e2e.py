import json
import pandas as pd
import pytest
import app.data_pipeline.transform as transform


@pytest.mark.e2e
def test_transform_main_writes_schema_outputs(tmp_path, monkeypatch):
    input_dir = tmp_path / "wikitree_test"
    output_dir = tmp_path / "wikitree_schema"

    input_dir.mkdir()
    output_dir.mkdir()

    people_path = input_dir / "people.csv"
    relationships_path = input_dir / "relationships.csv"

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
                "data_status": json.dumps({"BirthDate": "certain"}),
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
        ]
    )

    raw_relationships = pd.DataFrame(
        [
            {
                "parent_id": "1124609",
                "child_id": "5688919",
                "child_wikitree_id": "Austen-489",
                "relationship_type": "father_of",
            }
        ]
    )

    raw_people.to_csv(people_path, index=False)
    raw_relationships.to_csv(relationships_path, index=False)

    monkeypatch.setattr(transform, "INPUT_DIR", input_dir)
    monkeypatch.setattr(transform, "OUTPUT_DIR", output_dir)

    monkeypatch.setattr(transform, "PEOPLE_INPUT", people_path)
    monkeypatch.setattr(transform, "RELATIONSHIPS_INPUT", relationships_path)

    monkeypatch.setattr(transform, "PERSON_OUTPUT", output_dir / "person.csv")
    monkeypatch.setattr(transform, "NAMES_OUTPUT", output_dir / "names.csv")
    monkeypatch.setattr(transform, "EVENT_OUTPUT", output_dir / "event.csv")
    monkeypatch.setattr(transform, "ID_CROSSWALK_OUTPUT", output_dir / "id_crosswalk.csv")
    monkeypatch.setattr(transform, "RELATIONSHIP_REJECTIONS_OUTPUT", output_dir / "relationship_rejections.csv")
    monkeypatch.setattr(transform, "QUALITY_REPORT_OUTPUT", output_dir / "transform_quality_report.csv")

    transform.main()

    assert (output_dir / "person.csv").exists()
    assert (output_dir / "names.csv").exists()
    assert (output_dir / "event.csv").exists()
    assert (output_dir / "id_crosswalk.csv").exists()
    assert (output_dir / "relationship_rejections.csv").exists()
    assert (output_dir / "transform_quality_report.csv").exists()

    person_df = pd.read_csv(output_dir / "person.csv")
    names_df = pd.read_csv(output_dir / "names.csv")
    event_df = pd.read_csv(output_dir / "event.csv")
    report_df = pd.read_csv(output_dir / "transform_quality_report.csv")

    assert len(person_df) == 2
    assert len(names_df) == 2
    assert "birth" in set(event_df["Event_Type"])
    assert "father_of" in set(event_df["Event_Type"])

    metrics = dict(zip(report_df["metric"], report_df["value"]))
    assert metrics["schema_people_rows"] == 2