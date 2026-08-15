import pytest

import app.tree.generate_family_tree as generate_family_tree


@pytest.mark.e2e
def test_generate_family_tree_real_schema_jane_austen():
    person, names, event = generate_family_tree.load_schema(
        generate_family_tree.DEFAULT_SCHEMA_DIR
    )

    people_index = generate_family_tree.build_people_index(person, names, event)

    jane_match = people_index[people_index["Wikitree_ID"] == "Austen-489"]
    assert not jane_match.empty

    root_person_id = jane_match.iloc[0]["Person_ID"]

    parent_edges = generate_family_tree.build_parent_edges(event)

    nodes, edges = generate_family_tree.collect_ancestor_subgraph(
        root_person_id=root_person_id,
        people=people_index,
        parent_edges=parent_edges,
        max_generations=2,
        include_missing_stubs=False,
    )

    assert not nodes.empty
    assert not edges.empty

    assert "Austen-489" in set(nodes["wikitree_id"])

    relationship_types = set(edges["relationship_type"])
    assert relationship_types & {"father_of", "mother_of", "parent_of"}

import json
import pytest

import app.tree.generate_family_tree as generate_family_tree


@pytest.mark.e2e
def test_generate_family_tree_real_schema_writes_json_and_html(tmp_path):
    person, names, event = generate_family_tree.load_schema(
        generate_family_tree.DEFAULT_SCHEMA_DIR
    )

    people_index = generate_family_tree.build_people_index(person, names, event)

    jane_match = people_index[people_index["Wikitree_ID"] == "Austen-489"]
    assert not jane_match.empty

    root_person_id = jane_match.iloc[0]["Person_ID"]
    parent_edges = generate_family_tree.build_parent_edges(event)

    nodes, edges = generate_family_tree.collect_ancestor_subgraph(
        root_person_id=root_person_id,
        people=people_index,
        parent_edges=parent_edges,
        max_generations=2,
        include_missing_stubs=False,
    )

    tree_json = generate_family_tree.tree_to_json(nodes, edges)

    json_path = tmp_path / "jane_austen_tree.json"

    json_path.write_text(json.dumps(tree_json, indent=2), encoding="utf-8")


    assert json_path.exists()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert "nodes" in loaded
    assert "edges" in loaded
    assert len(loaded["nodes"]) == len(nodes)
    assert len(loaded["edges"]) == len(edges)
