"""
Connects API layer to candidate retrieval, confidence scoring
and family tree generation modules.
"""

from pathlib import Path
import math
import pandas as pd
from typing import Any

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

def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame into dictionaries.

    NaN values converted to None so FastAPI can serialise missing
    values as JSON null.

    Args -
        df(pd.DataFrame) DataFrame to convert.

    Returns -
        One dictionary per row, or an empty list for an empty
        DataFrame.
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
    first_name: str | None = None,
    last_name: str | None = None,
    birth_year: int | None = None,
    birth_location: str | None = None,
    gender: str | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
    schema_dir: str | Path = DEFAULT_SCHEMA_DIR,
) -> list[dict[str, Any]]:
    """Retrieve ranked candidates and attach confidence information.

    Args - 
        first_name(str): Query first name.
        last_name(str): Query surname.
        birth_year(int): Optional query birth year.
        birth_location(str): Optional query birth location.
        gender(str): Optional query gender.
        top_k(int): Maximum number of candidates to return.
        min_score(float): Minimum adjusted retrieval score.
        schema_dir(str): Directory containing the transformed schema files.

    Return - 
        Ranked candidate records including confidence metadata.
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
    """Generate an ancestor tree for a profile

    Args - 
        person_id: Optional internal Person ID of the root.
        wikitree_id: Optional WikiTree ID of the root.
        generations: Maximum ancestor generation to include.
        include_missing_stubs: Whether unresolved linked profiles may appear as
            placeholder nodes.
        schema_dir: Directory containing the transformed schema files.

    Returns - 
        Dictionary containing root metadata, a tree summary, nodes and edges.

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
        people=people,
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