import json
import pandas as pd
import pytest

import app.tree.generate_family_tree as generate_family_tree


@pytest.fixture
def mini_family_data():
    person = pd.DataFrame(
        [
            {
                "Person_ID": "person-jane",
                "Wikitree_ID": "Austen-489",
                "Gender": "Female",
                "Profile_URL": "https://www.wikitree.com/wiki/Austen-489",
                "Has_Children": "False",
            },
            {
                "Person_ID": "person-george",
                "Wikitree_ID": "Austen-109",
                "Gender": "Male",
                "Profile_URL": "https://www.wikitree.com/wiki/Austen-109",
                "Has_Children": "True",
            },
            {
                "Person_ID": "person-cassandra",
                "Wikitree_ID": "Leigh-138",
                "Gender": "Female",
                "Profile_URL": "https://www.wikitree.com/wiki/Leigh-138",
                "Has_Children": "True",
            },
            {
                "Person_ID": "person-william",
                "Wikitree_ID": "Austen-499",
                "Gender": "Male",
                "Profile_URL": "https://www.wikitree.com/wiki/Austen-499",
                "Has_Children": "True",
            },
        ]
    )

    names = pd.DataFrame(
        [
            {
                "Person_ID": "person-jane",
                "First_Name": "Jane",
                "Middle_Name": "",
                "Last_Name_At_Birth": "Austen",
                "Last_Name_Current": "Austen",
                "Nicknames": "",
            },
            {
                "Person_ID": "person-george",
                "First_Name": "George",
                "Middle_Name": "",
                "Last_Name_At_Birth": "Austen",
                "Last_Name_Current": "Austen",
                "Nicknames": "",
            },
            {
                "Person_ID": "person-cassandra",
                "First_Name": "Cassandra",
                "Middle_Name": "",
                "Last_Name_At_Birth": "Leigh",
                "Last_Name_Current": "Austen",
                "Nicknames": "",
            },
            {
                "Person_ID": "person-william",
                "First_Name": "William",
                "Middle_Name": "",
                "Last_Name_At_Birth": "Austen",
                "Last_Name_Current": "Austen",
                "Nicknames": "",
            },
        ]
    )

    event = pd.DataFrame(
        [
            {
                "Marriage_ID": "event-jane-birth",
                "Person_ID_1": "person-jane",
                "Person_ID_2": "",
                "Event_Type": "birth",
                "Event_Raw_Date": "1775-12-16",
                "Event_Year": "1775",
                "Event_Location": "Steventon, Hampshire, England",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-george-birth",
                "Person_ID_1": "person-george",
                "Person_ID_2": "",
                "Event_Type": "birth",
                "Event_Raw_Date": "1731-05-01",
                "Event_Year": "1731",
                "Event_Location": "Royal Tunbridge Wells, Kent, England",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-cassandra-birth",
                "Person_ID_1": "person-cassandra",
                "Person_ID_2": "",
                "Event_Type": "birth",
                "Event_Raw_Date": "1739-09-26",
                "Event_Year": "1739",
                "Event_Location": "Harpsden, Oxfordshire, England",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-william-birth",
                "Person_ID_1": "person-william",
                "Person_ID_2": "",
                "Event_Type": "birth",
                "Event_Raw_Date": "1700-02-03",
                "Event_Year": "1700",
                "Event_Location": "Broadford, Kent, England",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-george-father-jane",
                "Person_ID_1": "person-george",
                "Person_ID_2": "person-jane",
                "Event_Type": "father_of",
                "Event_Raw_Date": "",
                "Event_Year": "",
                "Event_Location": "",
                "Data Status": "",
            },
            {
                "Marriage_ID": "event-cassandra-mother-jane",
                "Person_ID_1": "person-cassandra",
                "Person_ID_2": "person-jane",
                "Event_Type": "mother_of",
                "Event_Raw_Date": "",
                "Event_Year": "",
                "Event_Location": "",
                "Data Status": "",
            },
            {
                "Marriage_ID": "event-william-father-george",
                "Person_ID_1": "person-william",
                "Person_ID_2": "person-george",
                "Event_Type": "father_of",
                "Event_Raw_Date": "",
                "Event_Year": "",
                "Event_Location": "",
                "Data Status": "",
            },
        ]
    )

    return person, names, event

### Test build_people_index ### 

def test_build_people_index(mini_family_data):
    person, names, event = mini_family_data

    people_index = generate_family_tree.build_people_index(person, names, event)

    jane = people_index[people_index["Person_ID"] == "person-jane"].iloc[0]

    assert jane["Full_Name"] == "Jane Austen"
    assert jane["Birth_Year"] == "1775"
    assert jane["Birth_Location"] == "Steventon, Hampshire, England"

### Test collect_ancestor_subgraph one generation ###

def test_collect_ancestor_subgraph_one_generation(mini_family_data):
    person, names, event = mini_family_data

    people_index = generate_family_tree.build_people_index(person, names, event)
    parent_edges = generate_family_tree.build_parent_edges(event)

    nodes, edges = generate_family_tree.collect_ancestor_subgraph(
        root_person_id="person-jane",
        person=people_index,
        parent_edges=parent_edges,
        max_generations=1,
        include_missing_stubs=False,
    )

    assert len(nodes) == 3
    assert len(edges) == 2

    generations = dict(zip(nodes["person_id"], nodes["generation"]))

    assert generations["person-jane"] == 0
    assert generations["person-george"] == 1
    assert generations["person-cassandra"] == 1

    assert set(edges["relationship_type"]) == {"father_of", "mother_of"}

### Test collect_ancestor_subgraph two generation ###

def test_collect_ancestor_subgraph_two_generations(mini_family_data):
    person, names, event = mini_family_data

    people_index = generate_family_tree.build_people_index(person, names, event)
    parent_edges = generate_family_tree.build_parent_edges(event)

    nodes, edges = generate_family_tree.collect_ancestor_subgraph(
        root_person_id="person-jane",
        person=people_index,
        parent_edges=parent_edges,
        max_generations=2,
        include_missing_stubs=False,
    )

    assert len(nodes) == 4
    assert len(edges) == 3

    generations = dict(zip(nodes["person_id"], nodes["generation"]))

    assert generations["person-jane"] == 0
    assert generations["person-george"] == 1
    assert generations["person-cassandra"] == 1
    assert generations["person-william"] == 2

### Testing missing stubs ###

def test_collect_ancestor_subgraph_excludes_missing_stubs_when_disabled(mini_family_data):
    person, names, event = mini_family_data

    missing_parent_event = pd.DataFrame(
        [
            {
                "Marriage_ID": "event-missing-parent",
                "Person_ID_1": "person-missing",
                "Person_ID_2": "person-jane",
                "Event_Type": "parent_of",
                "Event_Raw_Date": "",
                "Event_Year": "",
                "Event_Location": "",
                "Data Status": "",
            }
        ]
    )

    event = pd.concat([event, missing_parent_event], ignore_index=True)

    people_index = generate_family_tree.build_people_index(person, names, event)
    parent_edges = generate_family_tree.build_parent_edges(event)

    nodes, edges = generate_family_tree.collect_ancestor_subgraph(
        root_person_id="person-jane",
        person=people_index,
        parent_edges=parent_edges,
        max_generations=1,
        include_missing_stubs=False,
    )

    assert "person-missing" not in set(nodes["person_id"])
    assert "person-missing" not in set(edges["parent_person_id"])

def test_collect_ancestor_subgraph_includes_missing_stubs_when_enabled(mini_family_data):
    person, names, event = mini_family_data

    missing_parent_event = pd.DataFrame(
        [
            {
                "Marriage_ID": "event-missing-parent",
                "Person_ID_1": "person-missing",
                "Person_ID_2": "person-jane",
                "Event_Type": "parent_of",
                "Event_Raw_Date": "",
                "Event_Year": "",
                "Event_Location": "",
                "Data Status": "",
            }
        ]
    )

    event = pd.concat([event, missing_parent_event], ignore_index=True)

    people_index = generate_family_tree.build_people_index(person, names, event)
    parent_edges = generate_family_tree.build_parent_edges(event)

    nodes, edges = generate_family_tree.collect_ancestor_subgraph(
        root_person_id="person-jane",
        person=people_index,
        parent_edges=parent_edges,
        max_generations=1,
        include_missing_stubs=True,
    )

    missing = nodes[nodes["person_id"] == "person-missing"].iloc[0]

    assert missing["full_name"] == "Missing linked profile"
    assert bool(missing["is_stub"]) is True
    assert missing["generation"] == 1

### Testing duplicate edges are removed ### 

def test_collect_ancestor_subgraph_deduplicates_edges(mini_family_data):
    person, names, event = mini_family_data

    duplicate = event[event["Marriage_ID"] == "event-george-father-jane"].copy()
    duplicate["Marriage_ID"] = "event-george-father-jane-duplicate"

    event = pd.concat([event, duplicate], ignore_index=True)

    people_index = generate_family_tree.build_people_index(person, names, event)
    parent_edges = generate_family_tree.build_parent_edges(event)

    nodes, edges = generate_family_tree.collect_ancestor_subgraph(
        root_person_id="person-jane",
        person=people_index,
        parent_edges=parent_edges,
        max_generations=1,
        include_missing_stubs=False,
    )

    george_edges = edges[
        (edges["parent_person_id"] == "person-george")
        & (edges["child_person_id"] == "person-jane")
        & (edges["relationship_type"] == "father_of")
    ]

    assert len(george_edges) == 1

### Testing file_loading ### 

def test_load_schema_reads_person_names_and_event_files(tmp_path, mini_family_data):
    person, names, event = mini_family_data

    person.to_csv(tmp_path / generate_family_tree.PERSON_FILE, index=False)
    names.to_csv(tmp_path / generate_family_tree.NAMES_FILE, index=False)
    event.to_csv(tmp_path / generate_family_tree.EVENT_FILE, index=False)

    loaded_person, loaded_names, loaded_event = generate_family_tree.load_schema(tmp_path)

    assert len(loaded_person) == len(person)
    assert len(loaded_names) == len(names)
    assert len(loaded_event) == len(event)


def test_load_schema_adds_missing_optional_columns(tmp_path):
    person = pd.DataFrame([{"Person_ID": "person-jane"}])
    names = pd.DataFrame([{"Person_ID": "person-jane"}])
    event = pd.DataFrame([{"Event_Type": "birth"}])

    person.to_csv(tmp_path / generate_family_tree.PERSON_FILE, index=False)
    names.to_csv(tmp_path / generate_family_tree.NAMES_FILE, index=False)
    event.to_csv(tmp_path / generate_family_tree.EVENT_FILE, index=False)

    loaded_person, loaded_names, loaded_event = generate_family_tree.load_schema(tmp_path)

    assert "Wikitree_ID" in loaded_person.columns
    assert "Gender" in loaded_person.columns
    assert "First_Name" in loaded_names.columns
    assert "Person_ID_1" in loaded_event.columns
    assert "Data Status" in loaded_event.columns





