import json
import pandas as pd
import pytest
import app.tree.generate_family_tree as generate_family_tree

### Testing text_clean() ###

def test_clean_text():
    assert generate_family_tree.clean_text(None) is None
    assert generate_family_tree.clean_text("") is None
    assert generate_family_tree.clean_text("nan") is None
    assert generate_family_tree.clean_text("NULL") is None


def test_clean_text_strip():
    assert generate_family_tree.clean_text("  Jane Austen  ") == "Jane Austen"

### Testing safe_int() ### 

def test_safe_int():
    assert generate_family_tree.safe_int("1775") == 1775
    assert generate_family_tree.safe_int("1775.0") == 1775


def test_safe_int_missing_value():
    assert generate_family_tree.safe_int(None) is None
    assert generate_family_tree.safe_int("") is None

### Testing build_full_names() ###

def test_build_full_name_uses_first_middle_and_birth_last_name():
    row = pd.Series(
        {
            "First_Name": "Charles",
            "Middle_Name": "Robert",
            "Last_Name_At_Birth": "Darwin",
            "Last_Name_Current": "Darwin",
        }
    )

    assert generate_family_tree.build_full_name(row) == "Charles Robert Darwin"


def test_build_full_name_falls_back_to_current_last_name():
    row = pd.Series(
        {
            "First_Name": "Cassandra",
            "Middle_Name": "",
            "Last_Name_At_Birth": "",
            "Last_Name_Current": "Austen",
        }
    )

    assert generate_family_tree.build_full_name(row) == "Cassandra Austen"


def test_build_full_name_falls_back_to_wikitree_id():
    row = pd.Series(
        {
            "First_Name": "",
            "Middle_Name": "",
            "Last_Name_At_Birth": "",
            "Last_Name_Current": "",
            "Wikitree_ID": "Austen-489",
            "Person_ID": "person-jane",
        }
    )

    assert generate_family_tree.build_full_name(row) == "Austen-489"


def test_build_full_name_falls_back_to_unknown_person():
    row = pd.Series(
        {
            "First_Name": "",
            "Middle_Name": "",
            "Last_Name_At_Birth": "",
            "Last_Name_Current": "",
            "Wikitree_ID": "",
            "Person_ID": "",
        }
    )

    assert generate_family_tree.build_full_name(row) == "Unknown person"

### Testing read_required_csv ###

def test_read_required_csv(tmp_path):
    path = tmp_path / "person.csv"
    path.write_text("Person_ID,Wikitree_ID\nperson-jane,Austen-489\n", encoding="utf-8")

    df = generate_family_tree.read_required_csv(path)

    assert len(df) == 1
    assert df.iloc[0]["Wikitree_ID"] == "Austen-489"


def test_read_required_csv_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Missing required file"):
        generate_family_tree.read_required_csv(missing_path)

### Testing build_parent_edges ###

def test_build_parent_edges_keeps_only_parent_events():
    event = pd.DataFrame(
        [
            {
                "Marriage_ID": "event-father",
                "Person_ID_1": "person-george",
                "Person_ID_2": "person-jane",
                "Event_Type": "father_of",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-mother",
                "Person_ID_1": "person-cassandra",
                "Person_ID_2": "person-jane",
                "Event_Type": "mother_of",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-birth",
                "Person_ID_1": "person-jane",
                "Person_ID_2": "",
                "Event_Type": "birth",
                "Data Status": "certain",
            },
        ]
    )

    edges = generate_family_tree.build_parent_edges(event)

    assert len(edges) == 2
    assert set(edges["relationship_type"]) == {"father_of", "mother_of"}
    assert set(edges["parent_person_id"]) == {"person-george", "person-cassandra"}
    assert set(edges["child_person_id"]) == {"person-jane"}


def test_build_parent_edges_drops_missing_parent_or_child():
    event = pd.DataFrame(
        [
            {
                "Marriage_ID": "event-bad",
                "Person_ID_1": None,
                "Person_ID_2": "person-jane",
                "Event_Type": "father_of",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-good",
                "Person_ID_1": "person-george",
                "Person_ID_2": "person-jane",
                "Event_Type": "father_of",
                "Data Status": "certain",
            },
        ]
    )

    edges = generate_family_tree.build_parent_edges(event)

    assert len(edges) == 1
    assert edges.iloc[0]["parent_person_id"] == "person-george"

### Testing summarise_tree ### 

def test_summarise_tree():
    nodes = pd.DataFrame(
        [
            {"person_id": "person-jane", "generation": 0, "is_stub": False},
            {"person_id": "person-george", "generation": 1, "is_stub": False},
            {"person_id": "person-cassandra", "generation": 1, "is_stub": False},
        ]
    )

    edges = pd.DataFrame(
        [
            {"parent_person_id": "person-george", "child_person_id": "person-jane", "relationship_type": "father_of"},
            {"parent_person_id": "person-cassandra", "child_person_id": "person-jane", "relationship_type": "mother_of"},
        ]
    )

    summary = generate_family_tree.summarise_tree(nodes, edges)
    metrics = dict(zip(summary["metric"], summary["value"]))

    assert metrics["node_count"] == 3
    assert metrics["edge_count"] == 2
    assert metrics["max_generation"] == 1
    assert metrics["stub_node_count"] == 0
    assert metrics["father_edges"] == 1
    assert metrics["mother_edges"] == 1
    assert metrics["generation_0_nodes"] == 1
    assert metrics["generation_1_nodes"] == 2

### Testing tree_json ###

def test_tree_to_json():
    nodes = pd.DataFrame(
        [
            {"person_id": "person-jane", "full_name": "Jane Austen"},
        ]
    )

    edges = pd.DataFrame(
        [
            {"parent_person_id": "person-george", "child_person_id": "person-jane"},
        ]
    )

    result = generate_family_tree.tree_to_json(nodes, edges)

    assert result == {
        "nodes": [{"person_id": "person-jane", "full_name": "Jane Austen"}],
        "edges": [{"parent_person_id": "person-george", "child_person_id": "person-jane"}],
    }

### Testing html_escape ###

def test_html_escape_escapes_special_characters():
    assert generate_family_tree.html_escape("<Jane & George>") == "&lt;Jane &amp; George&gt;"

### Testing node_label ###

def test_node_label_includes_name_dates_and_wikitree_id():
    row = pd.Series(
        {
            "full_name": "Jane Austen",
            "wikitree_id": "Austen-489",
            "birth_year": "1775",
            "death_year": "1817",
        }
    )

    result = generate_family_tree.node_label(row)

    assert result == "Jane Austen (1775–1817)\nAusten-489"


def test_node_label_handles_missing_dates():
    row = pd.Series(
        {
            "full_name": "Jane Austen",
            "wikitree_id": "Austen-489",
            "birth_year": None,
            "death_year": None,
        }
    )

    result = generate_family_tree.node_label(row)

    assert result == "Jane Austen\nAusten-489"


