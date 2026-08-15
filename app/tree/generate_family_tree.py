"""
Generate ancestor trees from schema.

loads Person, Name and Event tables, builds people
index, extracts recorded parent-child relationships, and traverses
relationships from root profile.

traversal uses breadth-first search so generation depth can be
limited. The resulting nodes and edges can be returned through the
API or exported as CSV, JSON and a standalone HTML visualisation.

It does not infer or create unsupported family relationships.
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from app.data_pipeline.transform import clean_text

import pandas as pd


DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")
DEFAULT_OUTPUT_DIR = Path("data/family_trees")

PERSON_FILE = "person.csv"
NAMES_FILE = "names.csv"
EVENT_FILE = "event.csv"

PARENT_EVENT_TYPES = {"father_of", "mother_of", "parent_of"}

BAD_VALUES = {"", "nan", "NaN", "None", "none", "NULL", "null"}


def read_required_csv(path):
    """
    Reads csv file path
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def safe_int(value: any) -> float | None:
    """
    Converts numeric values to integer otherwise returns None
    """
    text = clean_text(value)
    if text is None:
        return None
    return int(float(text))
   


def build_full_name(row: pd.Series) -> str:
    """
    Builds full name for display using full name, middle name and birth surname

    Prefers birth surname over current surname

    If names unavalable then fallback on WikiTreeID, PersonID then generic label
    """
    parts = []

    # Extracting  first_name and middle_name
    first_name = clean_text(row.get("First_Name"))
    middle_name = clean_text(row.get("Middle_Name"))
    # Extracting last name at birth if available or current last name if not
    last_name = clean_text(row.get("Last_Name_At_Birth")) or clean_text(row.get("Last_Name_Current"))

    # combining names together to get full name
    for part in [first_name, middle_name, last_name]:
        if part:
            parts.append(part)

    if parts:
        return " ".join(parts)

    # If names not avaiable then falling back on WikiTreeID then PersonID then label
    return clean_text(row.get("Wikitree_ID")) or clean_text(row.get("Person_ID")) or "Unknown person"


def load_schema(schema_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Person, Name and Event tables.

    Optional columns are added with missing values when absent so 
    tree generation logic can operate against a consistent schema.

    Args -
        schema_dir(str): Directory containing the transformed schema CSV files

    Returns -
        tuple containing the Person, Name and Event DataFrames.
    """

    schema_dir = Path(schema_dir)

    # Extracting  tables from csv files
    person = read_required_csv(schema_dir / PERSON_FILE)
    names = read_required_csv(schema_dir / NAMES_FILE)
    event = read_required_csv(schema_dir / EVENT_FILE)

    # Extracting relevant columns from person, name and event tables
    person_cols = ["Person_ID", "Wikitree_ID", "Gender", "Profile_URL", "Has_Children"]
    name_cols = [
        "Person_ID",
        "First_Name",
        "Middle_Name",
        "Last_Name_At_Birth",
        "Last_Name_Current",
        "Nicknames",
    ]
    event_cols = [
        "Marriage_ID",
        "Person_ID_1",
        "Person_ID_2",
        "Event_Type",
        "Event_Raw_Date",
        "Event_Year",
        "Event_Location",
        "Data Status",
    ]

    # Filling in any optional columns with None values
    for col in person_cols:
        if col not in person.columns:
            person[col] = None

    for col in name_cols:
        if col not in names.columns:
            names[col] = None

    for col in event_cols:
        if col not in event.columns:
            event[col] = None

    return person, names, event


def build_people_index(
    person: pd.DataFrame,
    names: pd.DataFrame,
    event: pd.DataFrame,
) -> pd.DataFrame:
    
    """
    Builds row, one row per person, with person, name and event attributes
    
    Args - 
        person(pd): dataframe with person attribute
        names(pd): dataframe with name attribute for each person
        event(pd): dataframe with event attribute which is experienced by one or more person(s)
    Return - 
        pd: dataframe with names, events and persons merged
    """
    # Merging name and person table
    people = person.merge(names, on="Person_ID", how="left", suffixes=("", "_name"))

    # Extracting, deduplicating and renaming birth event table
    birth = event[event["Event_Type"] == "birth"].copy()
    birth = birth.drop_duplicates(subset=["Person_ID_1"], keep="first")
    birth = birth.rename(
        columns={
            "Person_ID_1": "Person_ID",
            "Event_Raw_Date": "Birth_Date",
            "Event_Year": "Birth_Year",
            "Event_Location": "Birth_Location",
            "Data Status": "Birth_Data_Status",
        }
    )

    # Extracting, deduplicating and renaming death event table
    death = event[event["Event_Type"] == "death"].copy()
    death = death.drop_duplicates(subset=["Person_ID_1"], keep="first")
    death = death.rename(
        columns={
            "Person_ID_1": "Person_ID",
            "Event_Raw_Date": "Death_Date",
            "Event_Year": "Death_Year",
            "Event_Location": "Death_Location",
            "Data Status": "Death_Data_Status",
        }
    )


    birth_cols = ["Person_ID", "Birth_Date", "Birth_Year", "Birth_Location", "Birth_Data_Status"]
    death_cols = ["Person_ID", "Death_Date", "Death_Year", "Death_Location", "Death_Data_Status"]

    # Merging people table with  death and birth event table
    people = people.merge(birth[birth_cols], on="Person_ID", how="left")
    people = people.merge(death[death_cols], on="Person_ID", how="left")
    people["Full_Name"] = people.apply(build_full_name, axis=1)

    return people


def resolve_root_person_id(person: pd, person_id: str | None,
                           wikitree_id: str | None) -> str:
    """
    resolves entity match using either wikitree_id or person_id

    PersonID takes priority, otherwise WikiTreeID is used as a fallback

    Args -
        person(pd): pandas dataframe containing person attribute
        person_id(str | None): synthetic ID identifying person within person dataframe 
        wikitree_id(str | None): ID given by wikitree

    Returns - 
        str: ID of person entity
    """

    # Matching on person_id
    if person_id:
        match = person[person["Person_ID"] == person_id]
        if not match.empty:
            return str(match.iloc[0]["Person_ID"])
        raise ValueError(f"No person found with Person_ID={person_id}")
    
    # Matching on wikitree_id if no person_id match found
    if wikitree_id:
        match = person[person["Wikitree_ID"] == wikitree_id]
        if not match.empty:
            return str(match.iloc[0]["Person_ID"])
        raise ValueError(f"No person found with Wikitree_ID={wikitree_id}")

    raise ValueError("Provide either --person-id or --wikitree-id.")


def build_parent_edges(event: pd.DataFrame) -> pd.DataFrame:
    """Extract parent child edges from the relationship event table

    Only father_of, mother_of, child_of relationship retained
    Args - 
        event(pd.DataFrame): pandas dataframe containing event data involving one or more person entity
    Returns -   
       dataframe that will contain edges(relations) between person entites
    """

    # retaining only event types elating to parent child relationships and renaming them 
    edges = event[event["Event_Type"].isin(PARENT_EVENT_TYPES)].copy()
    edges = edges.rename(
        columns={
            "Marriage_ID": "relationship_event_id",
            "Person_ID_1": "parent_person_id",
            "Person_ID_2": "child_person_id",
            "Event_Type": "relationship_type",
            "Data Status": "relationship_data_status",
        }
    )

    keep_cols = [
        "relationship_event_id",
        "parent_person_id",
        "child_person_id",
        "relationship_type",
        "relationship_data_status",
    ]

    for col in keep_cols:
        if col not in edges.columns:
            edges[col] = None

    edges = edges[keep_cols]
    edges = edges.dropna(subset=["parent_person_id", "child_person_id"])

    return edges


def collect_ancestor_subgraph(root_person_id: str, people: pd.DataFrame, parent_edges:pd.DataFrame, 
                              max_generations: int, 
                              include_missing_stubs: bool) -> tuple[pd.DataFrame,pd.DataFrame]:
    """
    "Collect the ancestors for a root person.

    Breadth first search from child to parent. Rroot is
    generation 0, its parents are generation 1, and traversal continues until
    `max_generations` is reached.

    A visited map prevents repeated visits while still.
    Missing linked profiles can optionally be represented as stub nodes.

    Args - 
        root_person_id(str): ID of person that traversal will start from
        person(pd.DataFrame): dataframe containing attribute of people, will be used in traversal
        parent_edges(pd.DataFrame): dataframe recording relations between child and parent 
        max_generations(int): maximum depth of traversal

    Returns - 
        tuple[df, df]: a tuple that contains both nodes and edges between nodes
    """
    # Setting person_id
    people_ids = set(people["Person_ID"].astype(str))
    parent_lookup = defaultdict(list)

    # collecting child of every person
    for _, edge in parent_edges.iterrows():
        child_id = str(edge["child_person_id"])
        parent_lookup[child_id].append(edge.to_dict())

    # Using a cache to make sure we don't retraverse already visited nodes
    visited = {root_person_id: 0}
    found_edges = []
    queue = deque([(root_person_id, 0)])

    # Using queue so we respect maximum depth
    while queue:
        current_id, generation = queue.popleft()

        if generation >= max_generations:
            continue

        # Collecting id of both parent and child
        for edge in parent_lookup.get(current_id, []):
            parent_id = str(edge["parent_person_id"])
            child_id = str(edge["child_person_id"])

            # parent and child need to exist first before being added as an edge
            parent_exists = parent_id in people_ids
            child_exists = child_id in people_ids

            if not include_missing_stubs and (not parent_exists or not child_exists):
                continue
            

            # Creating new edge if edge not already recorded
            new_edge = dict(edge)
            # Adding parent and child to edges
            new_edge["parent_exists_in_people"] = parent_exists
            new_edge["child_exists_in_people"] = child_exists
            new_edge["parent_generation"] = generation + 1
            new_edge["child_generation"] = generation
            found_edges.append(new_edge)

            # Adding parent_id to visited then traversing one generation lower
            if parent_id not in visited or visited[parent_id] > generation + 1:
                visited[parent_id] = generation + 1
                if parent_exists:
                    queue.append((parent_id, generation + 1))

    
    people_by_id = {}
    for _, row in people.iterrows():
        people_by_id[str(row["Person_ID"])] = row

    # Organising people by generations
    node_rows = []
    for person_id, generation in sorted(visited.items(), key=lambda item: (item[1], item[0])):
        person_row = people_by_id.get(person_id)

        if person_row is None:
            if not include_missing_stubs:
                continue
        
        # Appending to nodes person attribute fields
            node_rows.append(
                {
                    "person_id": person_id,
                    "wikitree_id": None,
                    "full_name": "Missing linked profile",
                    "generation": generation,
                    "gender": None,
                    "birth_year": None,
                    "birth_date": None,
                    "birth_location": None,
                    "death_year": None,
                    "death_date": None,
                    "death_location": None,
                    "profile_url": None,
                    "is_root": person_id == root_person_id,
                    "is_stub": True,
                }
            )
            continue
        # Appending to each node the attribute values of the person the node represents
        node_rows.append(
            {
                "person_id": person_id,
                "wikitree_id": clean_text(person_row.get("Wikitree_ID")),
                "full_name": clean_text(person_row.get("Full_Name")) or "Unknown person",
                "generation": generation,
                "gender": clean_text(person_row.get("Gender")),
                "birth_year": safe_int(person_row.get("Birth_Year")),
                "birth_date": clean_text(person_row.get("Birth_Date")),
                "birth_location": clean_text(person_row.get("Birth_Location")),
                "death_year": safe_int(person_row.get("Death_Year")),
                "death_date": clean_text(person_row.get("Death_Date")),
                "death_location": clean_text(person_row.get("Death_Location")),
                "profile_url": clean_text(person_row.get("Profile_URL")),
                "is_root": person_id == root_person_id,
                "is_stub": False,
            }
        )

    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(found_edges)

    # Filtering edges 
    if not edges.empty:
        edges = edges.drop_duplicates(subset=["parent_person_id", "child_person_id", "relationship_type"], keep="first")
        edges = edges[edges["parent_person_id"].isin(nodes["person_id"])]
        edges = edges[edges["child_person_id"].isin(nodes["person_id"])]

    return nodes.reset_index(drop=True), edges.reset_index(drop=True)


def summarise_tree(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd:
    """Summarises data collected from tree traversal

    Includes node and edge count, maximum generation, stub count and relationship type counts

    Args - 
        nodes(pd.DataFrame): Generated tree nodes
        edges(pd.DataFrame): Generated tree edges
    
    Returns - 
        Dataframe describing the tree
    """
    rows = [
        {"metric": "node_count", "value": len(nodes)},
        {"metric": "edge_count", "value": len(edges)},
        {"metric": "max_generation", "value": int(nodes["generation"].max()) if not nodes.empty else 0},
        {"metric": "stub_node_count", "value": int(nodes["is_stub"].sum()) if "is_stub" in nodes.columns else 0},
        {"metric": "father_edges", "value": int((edges["relationship_type"] == "father_of").sum()) if not edges.empty else 0},
        {"metric": "mother_edges", "value": int((edges["relationship_type"] == "mother_of").sum()) if not edges.empty else 0},
    ]

    if not nodes.empty:
        generation_counts = nodes["generation"].value_counts().sort_index().to_dict()
        for generation, count in generation_counts.items():
            rows.append({"metric": f"generation_{generation}_nodes", "value": int(count)})

    return pd.DataFrame(rows)


def tree_to_json(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict[str, list[dict]]:    
    """Convert tree node and edge into a JSON dictionary."""

    return {
        "nodes": nodes.to_dict(orient="records"),
        "edges": edges.to_dict(orient="records"),
    }

def build_output_dir(base_output_dir, root_wikitree_id, root_person_id):
    """Build directory for a generated tree."""
    folder_name = root_wikitree_id or root_person_id
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in folder_name)
    return Path(base_output_dir) / safe_name


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a preliminary family tree from transformed WikiTree schema CSVs.")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wikitree-id", type=str, default=None)
    parser.add_argument("--person-id", type=str, default=None)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--include-missing-stubs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.generations < 0:
        raise ValueError("--generations must be 0 or greater.")

    person, names, event = load_schema(args.schema_dir)
    people = build_people_index(person, names, event)
    parent_edges = build_parent_edges(event)

    root_person_id = resolve_root_person_id(people, args.person_id, args.wikitree_id)
    root_row = people[people["Person_ID"] == root_person_id].iloc[0]
    root_wikitree_id = clean_text(root_row.get("Wikitree_ID"))
    root_label = clean_text(root_row.get("Full_Name")) or root_wikitree_id or root_person_id

    nodes, edges = collect_ancestor_subgraph(
        root_person_id=root_person_id,
        people=people,
        parent_edges=parent_edges,
        max_generations=args.generations,
        include_missing_stubs=args.include_missing_stubs,
    )

    summary = summarise_tree(nodes, edges)

    output_dir = build_output_dir(args.output_dir, root_wikitree_id, root_person_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = output_dir / "family_tree_nodes.csv"
    edges_path = output_dir / "family_tree_edges.csv"
    summary_path = output_dir / "family_tree_summary.csv"
    json_path = output_dir / "family_tree.json"

    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(tree_to_json(nodes, edges), indent=2, ensure_ascii=False), encoding="utf-8")

    print("Family tree generated.")
    print(f"Root:          {root_label} ({root_wikitree_id or root_person_id})")
    print(f"Generations:   {args.generations}")
    print(f"Nodes:         {len(nodes)}")
    print(f"Edges:         {len(edges)}")
    print(f"Output folder: {output_dir}")
    print(f"Nodes CSV:     {nodes_path}")
    print(f"Edges CSV:     {edges_path}")
    print(f"JSON:          {json_path}")


if __name__ == "__main__":
    main()
