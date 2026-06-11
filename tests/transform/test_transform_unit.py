import app.data_pipeline.transform as transform 
import json
import pandas as pd
import pytest

### Testing Cleaning Functions ###

def test_clean_unknown_to_none():
    assert transform.clean_text("") is None
    assert transform.clean_text(" ") is None
    assert transform.clean_text("Unknown") is None
    assert transform.clean_text("0000-00-00") is None
    assert transform.clean_text("0") is None
    assert transform.clean_text(None) is None

def test_clean_text_strips_whitespace():
    assert transform.clean_text("  Jane  Austen  ") == "Jane Austen"

def test_clean_id_removes__float():
    assert transform.clean_id("5688919.0") == "5688919"
    assert transform.clean_id("5688919") == "5688919"
    assert transform.clean_id("0") is None

def test_normalize_gender():
    assert transform.normalise_gender("Male") == "Male"
    assert transform.normalise_gender("M") == "Male"
    assert transform.normalise_gender("Female") == "Female"
    assert transform.normalise_gender("F") == "Female"
    assert transform.normalise_gender("") is None
    assert transform.normalise_gender(None) is None
    assert transform.normalise_gender("other") == "Unknown"

def test_clean_location_normalises_commas_and_spacing():
    raw = "Steventon ,  Hampshire,England"
    assert transform.clean_location(raw) == "Steventon, Hampshire, England"

### Testing Date Parse Functions ### 

def test_parse_full_iso_date():
    raw, year, month, day  = transform.parse_date_parts("1775-12-16")

    assert raw == "1775-12-16"
    assert year == 1775
    assert month == 12
    assert day == 16

def test_parse_partial_dates():
    raw, year, month, day = transform.parse_date_parts("1775-00-00")

    assert raw == "1775-00-00"
    assert year == 1775
    assert month is None
    assert day is None


def test_parse_date_string():
    raw, year, month, day = transform.parse_date_parts("about 1835")

    assert raw == "about 1835"
    assert year == 1835
    assert month is None
    assert day is None

### Testing Data Status ###

def test_parse_data_status_valid_json_dict():
    raw = json.dumps({"BirthDate": "certain", "DeathDate": "guess"})

    result = transform.parse_data_status(raw)

    assert result == {"BirthDate": "certain", "DeathDate": "guess"}


def test_parse_data_status_invalid_json_returns_empty_dict():
    assert transform.parse_data_status("not json") == {}


def test_parse_data_status_non_dict_json_returns_empty_dict():
    assert transform.parse_data_status('["BirthDate", "certain"]') == {}


def test_status_for_field_returns_cleaned_status():
    row = {
        "_parsed_data_status": {
            "BirthDate": " certain ",
            "DeathDate": "",
        }
    }

    assert transform.status_for_field(row, "BirthDate") == "certain"
    assert transform.status_for_field(row, "DeathDate") is None

### Testing UUID ###

def test_stable_uuid():
    first = transform.stable_uuid("person", "Austen-489")
    second = transform.stable_uuid("person", "Austen-489")

    assert first == second

def test_stable_uuid_different_fields():
    person_id = transform.stable_uuid("person", "Austen-489")
    name_id = transform.stable_uuid("name", "Austen-489")

    assert person_id != name_id

def test_profile_url_builds_wikitree_url():
    assert (
        transform.profile_url("Austen-489")
        == "https://www.wikitree.com/wiki/Austen-489"
    )

def test_profile_url_missing_returns_none():
    assert transform.profile_url(None) is None

### Testing Prepare Function ####

def test_prepare_people():
    raw_people = pd.DataFrame(
        [
            {
                "person_id": "5688919.0",
                "wikitree_id": "Austen-489",
                "first_name": " Jane ",
                "middle_name": "",
                "last_name_at_birth": "Austen",
                "last_name_current": "Austen",
                "birth_date": "1775-12-16",
                "birth_location": "Steventon , Hampshire, England",
                "death_date": "1817-07-18",
                "death_location": "Winchester, Hampshire, England",
                "gender": "F",
                "father_id": "1124609.0",
                "mother_id": "1124608.0",
                "privacy": "60",
                "data_status": json.dumps({"BirthDate": "certain"}),
            },
            # Duplicate by person_id should be dropped
            {
                "person_id": "5688919",
                "wikitree_id": "Austen-Duplicate",
                "first_name": "Duplicate",
            },
            # Missing person_id should be dropped
            {
                "person_id": "",
                "wikitree_id": "Bad-1",
                "first_name": "Bad",
            },
        ]
    )

    people = transform.prepare_people(raw_people)

    assert len(people) == 1
    row = people.iloc[0]
    assert row["person_id"] == "5688919"
    assert row["wikitree_id"] == "Austen-489"
    assert row["first_name"] == "Jane"
    assert row["gender"] == "Female"
    assert row["father_id"] == "1124609"
    assert row["mother_id"] == "1124608"
    assert row["birth_location"] == "Steventon, Hampshire, England"
    assert row["schema_person_id"] == transform.stable_uuid("person", "Austen-489")
    assert row["schema_name_id"] == transform.stable_uuid("name", "Austen-489")

def test_prepare_relationships_removes_invalid_rows():
    raw_relationships = pd.DataFrame(
        [
            {
                "parent_id": "1124609.0",
                "child_id": "5688919.0",
                "child_wikitree_id": "Austen-489",
                "relationship_type": "father_of",
            },
            {
                "parent_id": "1124609",
                "child_id": "5688919",
                "child_wikitree_id": "Austen-489",
                "relationship_type": "father_of",
            },
            {
                "parent_id": "",
                "child_id": "5688919",
                "child_wikitree_id": "Austen-489",
                "relationship_type": "mother_of",
            },
        ]
    )

    relationships = transform.prepare_relationships(raw_relationships)

    assert len(relationships) == 1

    row = relationships.iloc[0]
    assert row["parent_id"] == "1124609"
    assert row["child_id"] == "5688919"
    assert row["relationship_type"] == "father_of"

### Testing Schema Builders ###

def test_build_person_has_children():
    people = pd.DataFrame(
        [
            {
                "person_id": "1124609",
                "wikitree_id": "Austen-109",
                "gender": "Male",
                "schema_person_id": "person-george",
            },
            {
                "person_id": "5688919",
                "wikitree_id": "Austen-489",
                "gender": "Female",
                "schema_person_id": "person-jane",
            },
        ]
    )

    relationships = pd.DataFrame(
        [
            {
                "parent_id": "1124609",
                "child_id": "5688919",
                "relationship_type": "father_of",
            }
        ]
    )

    person_table = transform.build_person_table(people, relationships)

    assert list(person_table.columns) == transform.PERSON_COLUMNS

    george = person_table[person_table["Wikitree_ID"] == "Austen-109"].iloc[0]
    jane = person_table[person_table["Wikitree_ID"] == "Austen-489"].iloc[0]

    assert george["Has_Children"] == True
    assert jane["Has_Children"] == False
    assert george["Profile_URL"] == "https://www.wikitree.com/wiki/Austen-109"

def test_build_tables_names_initials():
    people = pd.DataFrame(
        [
            {
                "schema_name_id": "name-samuel",
                "schema_person_id": "person-samuel",
                "first_name": "Samuel",
                "middle_name": "Langhorne",
                "last_name_at_birth": "Clemens",
                "last_name_current": "Clemens",
            },
            {
                "schema_name_id": "name-john",
                "schema_person_id": "person-john",
                "first_name": "John",
                "middle_name": "Marshall Henry",
                "last_name_at_birth": "Clemens",
                "last_name_current": "Clemens",
            },
        ]
    )
    
    names = transform.build_names_table(people)
    assert list(names.columns) == transform.NAME_COLUMNS

    samuel = names[names["First_Name"] == "Samuel"].iloc[0]
    john = names[names["First_Name"] == "John"].iloc[0]

    assert samuel["Middle_Initials"] == "L"
    assert john["Middle_Initials"] == "MH"
    assert samuel["Last_Name_At_Birth"] == "Clemens"
    
def test_build_event_table_relationship_events():
    people = pd.DataFrame(
        [
            {
                "person_id": "5688919",
                "wikitree_id": "Austen-489",
                "schema_person_id": "person-jane",
                "birth_date": "1775-12-16",
                "birth_location": "Steventon, Hampshire, England",
                "death_date": "1817-07-18",
                "death_location": "Winchester, Hampshire, England",
                "_parsed_data_status": {
                    "BirthDate": "certain",
                    "DeathDate": "certain",
                },
            },
            {
                "person_id": "1124609",
                "wikitree_id": "Austen-109",
                "schema_person_id": "person-george",
                "birth_date": "1731-05-01",
                "birth_location": "Royal Tunbridge Wells, Kent, England",
                "death_date": "1805-01-21",
                "death_location": "Bath, Somerset, England",
                "_parsed_data_status": {},
            },
        ]
    )

    relationships = pd.DataFrame(
        [
            {
                "parent_id": "1124609",
                "child_id": "5688919",
                "child_wikitree_id": "Austen-489",
                "relationship_type": "father_of",
            }
        ]
    )

    event_table, rejections = transform.build_event_table(people, relationships)

    assert list(event_table.columns) == transform.EVENT_COLUMNS
    assert rejections.empty

    event_types = set(event_table["Event_Type"])
    assert "birth" in event_types
    assert "death" in event_types
    assert "father_of" in event_types

    jane_birth = event_table[
        (event_table["Person_ID_1"] == "person-jane")
        & (event_table["Event_Type"] == "birth")
    ].iloc[0]

    assert jane_birth["Event_Raw_Date"] == "1775-12-16"
    assert jane_birth["Event_Year"] == 1775
    assert jane_birth["Event_Month"] == 12
    assert jane_birth["Event_Day"] == 16
    assert jane_birth["Event_Location"] == "Steventon, Hampshire, England"
    assert jane_birth["Data Status"] == "certain"

    father_event = event_table[event_table["Event_Type"] == "father_of"].iloc[0]
    assert father_event["Person_ID_1"] == "person-george"
    assert father_event["Person_ID_2"] == "person-jane"

def test_build_event_table_rejects_relationship():
    people = pd.DataFrame(
        [
            {
                "person_id": "5688919",
                "wikitree_id": "Austen-489",
                "schema_person_id": "person-jane",
                "birth_date": "1775-12-16",
                "birth_location": "Steventon, Hampshire, England",
                "death_date": None,
                "death_location": None,
                "_parsed_data_status": {},
            }
        ]
    )

    relationships = pd.DataFrame(
        [
            {
                "parent_id": "1124609",
                "child_id": "5688919",
                "child_wikitree_id": "Austen-489",
                "relationship_type": "father_of",
            }
        ]
    )

    event_table, rejections = transform.build_event_table(people, relationships)

    assert "father_of" not in set(event_table["Event_Type"])
    assert len(rejections) == 1

    rejection = rejections.iloc[0]
    assert rejection["parent_id"] == "1124609"
    assert rejection["child_id"] == "5688919"
    assert rejection["relationship_type"] == "father_of"
    assert rejection["reason"] == "parent_or_child_not_present_in_people_csv"

def test_build_id_crosswalkd():
    people = pd.DataFrame(
        [
            {
                "person_id": "5688919",
                "wikitree_id": "Austen-489",
                "schema_person_id": "person-jane",
                "schema_name_id": "name-jane",
                "first_name": "Jane",
                "middle_name": None,
                "last_name_at_birth": "Austen",
                "last_name_current": "Austen",
            }
        ]
    )

    crosswalk = transform.build_id_crosswalk(people)

    row = crosswalk.iloc[0]

    assert row["source_wikitree_numeric_id"] == "5688919"
    assert row["Wikitree_ID"] == "Austen-489"
    assert row["Person_ID"] == "person-jane"
    assert row["Name_ID"] == "name-jane"

