from pathlib import Path
import math
import pandas as pd

from app.retrieval.candidate_retrieval import CandidateRetriever 
from app.scoring.confidence_scoring import add_confidence_scores
from app.tree.generate_family_tree import (
    load_schema,
    build_people_index,
    build_parent_edges,
    collect_ancestor_subgraph,
    summarise_tree,
    resolve_root_person_id
)

DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")

def dataframe_to_records(df: pd) -> list[dict]:
    """
    Converts dataframe to list of dict
    Args - 
        df(pd): Dataframe to be turned into list of dict
    """
    if df.empty:
        return []
    
    records = df.to_dict(orient="records")
    cleaned = []

    for row in records:
        cleaned_row = {}
        for key, value in row.items():
            if value is None:
                cleaned_row[key] = None
            elif isinstance(value, float) and math.isnan(value):
                cleaned_row[key] = None
            else:
                cleaned_row[key] = value
        cleaned.append(cleaned_row)
    return cleaned

def search_candidate(
    first_name=None,
    last_name=None,
    birth_year=None,
    birth_location=None,
    gender=None,
    top_k=5,
    min_score=0.0,
    schema_dir=DEFAULT_SCHEMA_DIR,
):
    """
    Retrieves candidates
    """
    retriever = CandidateRetriever(schema_dir=schema_dir)

    candidates = retriever.find_candidates(
        first_name=first_name,
        last_name=last_name,
        birth_year=birth_year,
        birth_location=birth_location,
        gender=gender,
        top_k=top_k,
        min_score=min_score
    )

    results = add_confidence_scores(candidates)
    return dataframe_to_records(results)

def tree(
    person_id=None,
    wikitree_id=None,
    generations=3,
    include_missing_stubs=False,
    schema_dir=DEFAULT_SCHEMA_DIR,
):
    """
    builds ancestor tree
    """
    person,names,event = load_schema(schema_dir)
    people = build_people_index(person, names, event)
    parent_edges = build_parent_edges(event)

    root_person_id = resolve_root_person_id(
        person=people,
        person_id=person_id,
        wikitree_id=wikitree_id,
    )

    nodes, edges = collect_ancestor_subgraph(
        root_person_id=root_person_id,
        person = people,
        parent_edges=parent_edges,
        max_generations=generations,
        include_missing_stubs=include_missing_stubs
    )

    summary = summarise_tree(nodes, edges)

    return {
        "root_person_id": root_person_id,
        "generations": generations,
        "summary": dataframe_to_records(summary),
        "nodes": dataframe_to_records(nodes),
        "edges": dataframe_to_records(edges),
    } 