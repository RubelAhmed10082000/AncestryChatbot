"""
Evaluate ancestor trees against WikiTree source.

Expected and actual trees are compared using node and edge precision, recall and
F1, generation accuracy, direct-parent recall and structural integrity checks.

It does not verify if the WikiTree relationships are historically correct.
"""


from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from app.api.services import tree

from app.data_pipeline.transform import clean_text, read_csv_required

RAW_DIR = Path("data/wikitree_test")
OUTPUT_DIR = Path("data/evaluation/final")

PEOPLE_FILE = RAW_DIR / "people.csv"
RELATIONSHIPS_FILE = RAW_DIR / "relationships.csv"
SEEDS_FILE = RAW_DIR / "seed_profiles.csv"

RESULTS_FILE = OUTPUT_DIR / "tree_evaluation_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "tree_evaluation_summary.csv"
DISCREPANCIES_FILE = OUTPUT_DIR / "tree_discrepancies.csv"

GENERATIONS = 3


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Divides two values while handling empty values.

    When both numerator and denominator are zero, the comparison is treated as
    perfect agreement and returns 1.0. A above zero numerator with a zero
    denominator returns 0.0.
    """
    
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return numerator / denominator


def harmonic_mean(
    precision: float,
    recall: float,
) -> float:
    """Return the harmonic mean of precision and recall"""
    
    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def load_source_data():
    """Load and validate inputs used for tree evaluation

    The evaluation requires extracted people, parent-child relationships and
    seed profiles

    Returns -
        A tuple containing people, relationships and seed-profile DataFrames.
    """
    people = read_csv_required(PEOPLE_FILE)
    relationships = read_csv_required(
        RELATIONSHIPS_FILE
    )
    seeds = read_csv_required(SEEDS_FILE)

    required_people = {
        "person_id",
        "wikitree_id",
    }

    required_relationships = {
        "parent_id",
        "child_id",
        "relationship_type",
    }

    required_seeds = {
        "seed_label",
        "person_id",
        "wikitree_id",
    }

    if not required_people.issubset(people.columns):
        raise ValueError(
            "people.csv is missing required columns."
        )

    if not required_relationships.issubset(
        relationships.columns
    ):
        raise ValueError(
            "relationships.csv is missing required columns."
        )

    if not required_seeds.issubset(seeds.columns):
        raise ValueError(
            "seed_profiles.csv is missing required columns."
        )

    return people, relationships, seeds


def build_expected_tree(
    root_person_id: str,
    people: pd.DataFrame,
    relationships: pd.DataFrame,
    max_generations: int,
) -> tuple[dict[str, int], set[tuple[str, str, str]]]:
    """
    Construct expected nodes and edges directly from raw
    files.

    Evaluates if transformation and traversal preserve
    the source relationships. Does not verify
    if WikiTree is historically correct.

    Args - 
        root_person_id(str): WikiTree ID of profile
        people(pd.DataFrame): people record
        relationships(pd.DataFrame): parent-child relationshipa
        max_generations(pd.DataFrame): Maximum ancestor generation to include

    Returns - 
        Mapping of WikiTree ID to expected generation and set of expected
        parent child relationship edges.
    """
    source_to_wikitree = {
        str(row["person_id"]): clean_text(
            row["wikitree_id"]
        )
        for _, row in people.iterrows()
        if clean_text(row.get("person_id"))
        and clean_text(row.get("wikitree_id"))
    }

    available_ids = set(source_to_wikitree)

    parent_lookup = defaultdict(list)

    for _, relationship in relationships.iterrows():
        child_id = clean_text(
            relationship.get("child_id")
        )
        parent_id = clean_text(
            relationship.get("parent_id")
        )
        relationship_type = clean_text(
            relationship.get("relationship_type")
        )

        if not child_id or not parent_id:
            continue

        parent_lookup[child_id].append(
            {
                "parent_id": parent_id,
                "child_id": child_id,
                "relationship_type": (
                    relationship_type or "parent_of"
                ),
            }
        )

    root_person_id = str(root_person_id)

    visited = {
        root_person_id: 0,
    }

    queue = deque(
        [(root_person_id, 0)]
    )

    expected_edges = set()

    while queue:
        child_id, generation = queue.popleft()

        if generation >= max_generations:
            continue

        for relationship in parent_lookup.get(
            child_id,
            [],
        ):
            parent_id = relationship["parent_id"]

            if (
                parent_id not in available_ids
                or child_id not in available_ids
            ):
                continue

            parent_wikitree_id = source_to_wikitree[
                parent_id
            ]
            child_wikitree_id = source_to_wikitree[
                child_id
            ]

            expected_edges.add(
                (
                    parent_wikitree_id,
                    child_wikitree_id,
                    relationship[
                        "relationship_type"
                    ],
                )
            )

            parent_generation = generation + 1

            if (
                parent_id not in visited
                or visited[parent_id]
                > parent_generation
            ):
                visited[parent_id] = parent_generation
                queue.append(
                    (
                        parent_id,
                        parent_generation,
                    )
                )

    expected_generations = {
        source_to_wikitree[person_id]: generation
        for person_id, generation in visited.items()
        if person_id in source_to_wikitree
    }

    return expected_generations, expected_edges


def parse_actual_tree(
    payload: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
    set[tuple[str, str, str]],
]:
     
    nodes = pd.DataFrame(
        payload.get("nodes", [])
    )

    edges = pd.DataFrame(
        payload.get("edges", [])
    )

    if nodes.empty:
        return (
            nodes,
            edges,
            {},
            set(),
        )

    node_id_to_wikitree = {}

    actual_generations = {}

    for _, node in nodes.iterrows():
        person_id = clean_text(
            node.get("person_id")
        )
        wikitree_id = clean_text(
            node.get("wikitree_id")
        )

        if not person_id:
            continue

        node_id_to_wikitree[person_id] = (
            wikitree_id
        )

        if wikitree_id:
            actual_generations[wikitree_id] = int(
                float(node.get("generation", 0))
            )

    actual_edges = set()

    if not edges.empty:
        for _, edge in edges.iterrows():
            parent_person_id = clean_text(
                edge.get("parent_person_id")
            )
            child_person_id = clean_text(
                edge.get("child_person_id")
            )

            parent_wikitree_id = (
                node_id_to_wikitree.get(
                    parent_person_id
                )
            )

            child_wikitree_id = (
                node_id_to_wikitree.get(
                    child_person_id
                )
            )

            relationship_type = (
                clean_text(
                    edge.get("relationship_type")
                )
                or "parent_of"
            )

            if (
                parent_wikitree_id
                and child_wikitree_id
            ):
                actual_edges.add(
                    (
                        parent_wikitree_id,
                        child_wikitree_id,
                        relationship_type,
                    )
                )

    return (
        nodes,
        edges,
        actual_generations,
        actual_edges,
    )


def count_structural_errors(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict:
    """Counting integrity problems in a generated tree.

    Include duplicate nodes and edges, missing edge endpoints,
    loopss, incorrect generation differences etc.

    Args -
        nodes(pd.DataFrame): Generated tree nodes
        edges(pd.DataFrame): Generated tree edges

    Returns - 
        Counts for each structural integrity check
    """
    if nodes.empty:
        return {
            "duplicate_node_count": 0,
            "duplicate_edge_count": 0,
            "orphan_edge_count": 0,
            "self_loop_count": 0,
            "invalid_generation_edge_count": 0,
            "root_node_count": 0,
            "stub_node_count": 0,
        }

    node_ids = set(
        nodes["person_id"]
        .astype(str)
        .tolist()
    )

    duplicate_node_count = int(
        nodes["person_id"]
        .astype(str)
        .duplicated()
        .sum()
    )

    root_node_count = int(
        nodes.get(
            "is_root",
            pd.Series(dtype=bool),
        )
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
        .sum()
    )

    stub_node_count = int(
        nodes.get(
            "is_stub",
            pd.Series(dtype=bool),
        )
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
        .sum()
    )

    node_generation = {
        str(row["person_id"]): int(
            float(row["generation"])
        )
        for _, row in nodes.iterrows()
    }

    duplicate_edge_count = 0
    orphan_edge_count = 0
    self_loop_count = 0
    invalid_generation_edge_count = 0

    if not edges.empty:
        edge_key_columns = [
            "parent_person_id",
            "child_person_id",
            "relationship_type",
        ]

        duplicate_edge_count = int(
            edges[edge_key_columns]
            .astype(str)
            .duplicated()
            .sum()
        )

        for _, edge in edges.iterrows():
            parent_id = str(
                edge["parent_person_id"]
            )
            child_id = str(
                edge["child_person_id"]
            )

            if (
                parent_id not in node_ids
                or child_id not in node_ids
            ):
                orphan_edge_count += 1
                continue

            if parent_id == child_id:
                self_loop_count += 1

            parent_generation = (
                node_generation.get(parent_id)
            )

            child_generation = (
                node_generation.get(child_id)
            )

            if (
                parent_generation is None
                or child_generation is None
                or parent_generation
                != child_generation + 1
            ):
                invalid_generation_edge_count += 1

    return {
        "duplicate_node_count": (
            duplicate_node_count
        ),
        "duplicate_edge_count": (
            duplicate_edge_count
        ),
        "orphan_edge_count": (
            orphan_edge_count
        ),
        "self_loop_count": self_loop_count,
        "invalid_generation_edge_count": (
            invalid_generation_edge_count
        ),
        "root_node_count": root_node_count,
        "stub_node_count": stub_node_count,
    }


def add_set_discrepancies(
    rows: list[dict],
    seed_label: str,
    wikitree_id: str,
    expected_generations: dict[str, int],
    actual_generations: dict[str, int],
    expected_edges: set[tuple[str, str, str]],
    actual_edges: set[tuple[str, str, str]],
) -> None:
    """Record differences between expected and generated tree sets.

    Discrepancies recorded for missing or unexpected nodes, generation
    mismatches and missing or unexpected relationship edges

    Args:
        rows(list): Mutable list receiving discrepancy records.
        seed_label(str): Human-readable evaluation seed label.
        wikitree_id(str): WikiTree ID of the evaluated root.
        expected_generations(dict): Expected generation for each source profile.
        actual_generations(dict): Generated generation for each source profile.
        expected_edges(set): Expected parent-child relationship set.
        actual_edges(set): Generated parent-child relationship set.
    """
    expected_nodes = set(expected_generations)
    actual_nodes = set(actual_generations)

    for missing_node in sorted(
        expected_nodes - actual_nodes
    ):
        rows.append(
            {
                "seed_label": seed_label,
                "root_wikitree_id": wikitree_id,
                "discrepancy_type": "missing_node",
                "details": missing_node,
            }
        )

    for unexpected_node in sorted(
        actual_nodes - expected_nodes
    ):
        rows.append(
            {
                "seed_label": seed_label,
                "root_wikitree_id": wikitree_id,
                "discrepancy_type": (
                    "unexpected_node"
                ),
                "details": unexpected_node,
            }
        )

    for node_id in sorted(
        expected_nodes & actual_nodes
    ):
        expected_generation = (
            expected_generations[node_id]
        )

        actual_generation = (
            actual_generations[node_id]
        )

        if expected_generation != actual_generation:
            rows.append(
                {
                    "seed_label": seed_label,
                    "root_wikitree_id": (
                        wikitree_id
                    ),
                    "discrepancy_type": (
                        "generation_mismatch"
                    ),
                    "details": (
                        f"{node_id}: expected "
                        f"{expected_generation}, "
                        f"actual {actual_generation}"
                    ),
                }
            )

    for missing_edge in sorted(
        expected_edges - actual_edges
    ):
        rows.append(
            {
                "seed_label": seed_label,
                "root_wikitree_id": wikitree_id,
                "discrepancy_type": "missing_edge",
                "details": repr(missing_edge),
            }
        )

    for unexpected_edge in sorted(
        actual_edges - expected_edges
    ):
        rows.append(
            {
                "seed_label": seed_label,
                "root_wikitree_id": wikitree_id,
                "discrepancy_type": (
                    "unexpected_edge"
                ),
                "details": repr(unexpected_edge),
            }
        )


def evaluate_tree(
    seed: pd.Series,
    people: pd.DataFrame,
    relationships: pd.DataFrame,
    discrepancy_rows: list[dict],
) -> dict:
    """Evaluate ancestor tree against expectations.

        Node sets, edge sets, generation assignments and structural
        properties are compared.

        Args  -
            seed(pd.Series): Evaluation root containing its label and source identifiers.
            people(pd.DataFrame): Frozen raw people records.
            relationships(pd.DataFrame): Frozen raw relationship records.
            discrepancy_rows(list): Shared collection for detailed evaluation failures.

        Returns - 
            Per-tree evaluation metrics and structural diagnostic counts.
    """
    seed_label = clean_text(
        seed["seed_label"]
    )

    root_person_id = clean_text(
        seed["person_id"]
    )

    root_wikitree_id = clean_text(
        seed["wikitree_id"]
    )

    expected_generations, expected_edges = (
        build_expected_tree(
            root_person_id=root_person_id,
            people=people,
            relationships=relationships,
            max_generations=GENERATIONS,
        )
    )

    try:
        payload = tree(
            wikitree_id=root_wikitree_id,
            generations=GENERATIONS,
            include_missing_stubs=False,
        )

        (
            nodes,
            edges,
            actual_generations,
            actual_edges,
        ) = parse_actual_tree(payload)

        structural = count_structural_errors(
            nodes,
            edges,
        )

        expected_nodes = set(
            expected_generations
        )

        actual_nodes = set(
            actual_generations
        )

        node_true_positive = len(
            expected_nodes & actual_nodes
        )

        edge_true_positive = len(
            expected_edges & actual_edges
        )

        node_precision = safe_divide(
            node_true_positive,
            len(actual_nodes),
        )

        node_recall = safe_divide(
            node_true_positive,
            len(expected_nodes),
        )

        edge_precision = safe_divide(
            edge_true_positive,
            len(actual_edges),
        )

        edge_recall = safe_divide(
            edge_true_positive,
            len(expected_edges),
        )

        matching_nodes = (
            expected_nodes & actual_nodes
        )

        correct_generations = sum(
            expected_generations[node_id]
            == actual_generations[node_id]
            for node_id in matching_nodes
        )

        generation_accuracy = safe_divide(
            correct_generations,
            len(matching_nodes),
        )

        expected_direct_parents = {
            node_id
            for node_id, generation
            in expected_generations.items()
            if generation == 1
        }

        actual_direct_parents = {
            node_id
            for node_id, generation
            in actual_generations.items()
            if generation == 1
        }

        direct_parent_recall = safe_divide(
            len(
                expected_direct_parents
                & actual_direct_parents
            ),
            len(expected_direct_parents),
        )

        root_correct = (
            actual_generations.get(
                root_wikitree_id
            )
            == 0
        )

        actual_max_generation = (
            max(actual_generations.values())
            if actual_generations
            else None
        )

        generation_limit_respected = (
            actual_max_generation is not None
            and actual_max_generation
            <= GENERATIONS
        )

        exact_node_set = (
            expected_nodes == actual_nodes
        )

        exact_edge_set = (
            expected_edges == actual_edges
        )

        no_structural_errors = all(
            structural[key] == 0
            for key in [
                "duplicate_node_count",
                "duplicate_edge_count",
                "orphan_edge_count",
                "self_loop_count",
                "invalid_generation_edge_count",
                "stub_node_count",
            ]
        )

        exact_tree_match = (
            root_correct
            and structural["root_node_count"] == 1
            and exact_node_set
            and exact_edge_set
            and generation_accuracy == 1.0
            and generation_limit_respected
            and no_structural_errors
        )

        add_set_discrepancies(
            rows=discrepancy_rows,
            seed_label=seed_label,
            wikitree_id=root_wikitree_id,
            expected_generations=(
                expected_generations
            ),
            actual_generations=(
                actual_generations
            ),
            expected_edges=expected_edges,
            actual_edges=actual_edges,
        )

        return {
            "seed_label": seed_label,
            "root_wikitree_id": (
                root_wikitree_id
            ),
            "tree_generated": True,
            "error_message": None,
            "root_correct": root_correct,
            "root_node_count": (
                structural["root_node_count"]
            ),
            "expected_node_count": (
                len(expected_nodes)
            ),
            "actual_node_count": (
                len(actual_nodes)
            ),
            "node_true_positive": (
                node_true_positive
            ),
            "node_precision": round(
                node_precision,
                4,
            ),
            "node_recall": round(
                node_recall,
                4,
            ),
            "node_f1": round(
                harmonic_mean(
                    node_precision,
                    node_recall,
                ),
                4,
            ),
            "expected_edge_count": (
                len(expected_edges)
            ),
            "actual_edge_count": (
                len(actual_edges)
            ),
            "edge_true_positive": (
                edge_true_positive
            ),
            "edge_precision": round(
                edge_precision,
                4,
            ),
            "edge_recall": round(
                edge_recall,
                4,
            ),
            "edge_f1": round(
                harmonic_mean(
                    edge_precision,
                    edge_recall,
                ),
                4,
            ),
            "direct_parent_recall": round(
                direct_parent_recall,
                4,
            ),
            "generation_accuracy": round(
                generation_accuracy,
                4,
            ),
            "expected_max_generation": (
                max(
                    expected_generations.values()
                )
                if expected_generations
                else 0
            ),
            "actual_max_generation": (
                actual_max_generation
            ),
            "generation_limit_respected": (
                generation_limit_respected
            ),
            "exact_node_set": exact_node_set,
            "exact_edge_set": exact_edge_set,
            "exact_tree_match": (
                exact_tree_match
            ),
            **structural,
        }

    except Exception as exc:
        discrepancy_rows.append(
            {
                "seed_label": seed_label,
                "root_wikitree_id": (
                    root_wikitree_id
                ),
                "discrepancy_type": (
                    "generation_error"
                ),
                "details": str(exc),
            }
        )

        return {
            "seed_label": seed_label,
            "root_wikitree_id": (
                root_wikitree_id
            ),
            "tree_generated": False,
            "error_message": str(exc),
            "root_correct": False,
            "root_node_count": 0,
            "expected_node_count": (
                len(expected_generations)
            ),
            "actual_node_count": 0,
            "node_true_positive": 0,
            "node_precision": 0.0,
            "node_recall": 0.0,
            "node_f1": 0.0,
            "expected_edge_count": (
                len(expected_edges)
            ),
            "actual_edge_count": 0,
            "edge_true_positive": 0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "edge_f1": 0.0,
            "direct_parent_recall": 0.0,
            "generation_accuracy": 0.0,
            "expected_max_generation": (
                GENERATIONS
            ),
            "actual_max_generation": None,
            "generation_limit_respected": (
                False
            ),
            "exact_node_set": False,
            "exact_edge_set": False,
            "exact_tree_match": False,
            "duplicate_node_count": 0,
            "duplicate_edge_count": 0,
            "orphan_edge_count": 0,
            "self_loop_count": 0,
            "invalid_generation_edge_count": 0,
            "stub_node_count": 0,
        }


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate tree results into overall evaluation metrics.

    Node and edge precision, recall and F1 are calculated using counts
    across all evaluated trees. 

    Args -
        results (pd.DataFrame): Per-tree evaluation results.

    Returns -
        Metric/value DataFrame for the final tree-evaluation summary.
    """
    trees_attempted = len(results)

    trees_generated = int(
        results["tree_generated"].sum()
    )

    exact_matches = int(
        results["exact_tree_match"].sum()
    )

    root_correct = int(
        results["root_correct"].sum()
    )

    node_true_positive = int(
        results["node_true_positive"].sum()
    )

    expected_nodes = int(
        results["expected_node_count"].sum()
    )

    actual_nodes = int(
        results["actual_node_count"].sum()
    )

    edge_true_positive = int(
        results["edge_true_positive"].sum()
    )

    expected_edges = int(
        results["expected_edge_count"].sum()
    )

    actual_edges = int(
        results["actual_edge_count"].sum()
    )

    micro_node_precision = safe_divide(
        node_true_positive,
        actual_nodes,
    )

    micro_node_recall = safe_divide(
        node_true_positive,
        expected_nodes,
    )

    micro_edge_precision = safe_divide(
        edge_true_positive,
        actual_edges,
    )

    micro_edge_recall = safe_divide(
        edge_true_positive,
        expected_edges,
    )

    rows = [
        {
            "metric": "trees_attempted",
            "value": trees_attempted,
        },
        {
            "metric": "trees_generated",
            "value": trees_generated,
        },
        {
            "metric": "tree_generation_rate",
            "value": safe_divide(
                trees_generated,
                trees_attempted,
            ),
        },
        {
            "metric": "root_resolution_accuracy",
            "value": safe_divide(
                root_correct,
                trees_attempted,
            ),
        },
        {
            "metric": "exact_tree_match_rate",
            "value": safe_divide(
                exact_matches,
                trees_attempted,
            ),
        },
        {
            "metric": "micro_node_precision",
            "value": micro_node_precision,
        },
        {
            "metric": "micro_node_recall",
            "value": micro_node_recall,
        },
        {
            "metric": "micro_node_f1",
            "value": harmonic_mean(
                micro_node_precision,
                micro_node_recall,
            ),
        },
        {
            "metric": "micro_edge_precision",
            "value": micro_edge_precision,
        },
        {
            "metric": "micro_edge_recall",
            "value": micro_edge_recall,
        },
        {
            "metric": "micro_edge_f1",
            "value": harmonic_mean(
                micro_edge_precision,
                micro_edge_recall,
            ),
        },
        {
            "metric": "mean_direct_parent_recall",
            "value": results[
                "direct_parent_recall"
            ].mean(),
        },
        {
            "metric": "mean_generation_accuracy",
            "value": results[
                "generation_accuracy"
            ].mean(),
        },
        {
            "metric": "total_duplicate_nodes",
            "value": int(
                results[
                    "duplicate_node_count"
                ].sum()
            ),
        },
        {
            "metric": "total_duplicate_edges",
            "value": int(
                results[
                    "duplicate_edge_count"
                ].sum()
            ),
        },
        {
            "metric": "total_orphan_edges",
            "value": int(
                results[
                    "orphan_edge_count"
                ].sum()
            ),
        },
        {
            "metric": "total_self_loops",
            "value": int(
                results[
                    "self_loop_count"
                ].sum()
            ),
        },
        {
            "metric": (
                "total_invalid_generation_edges"
            ),
            "value": int(
                results[
                    "invalid_generation_edge_count"
                ].sum()
            ),
        },
        {
            "metric": "total_stub_nodes",
            "value": int(
                results[
                    "stub_node_count"
                ].sum()
            ),
        },
        {
            "metric": "expected_nodes_across_trees",
            "value": expected_nodes,
        },
        {
            "metric": "actual_nodes_across_trees",
            "value": actual_nodes,
        },
        {
            "metric": "expected_edges_across_trees",
            "value": expected_edges,
        },
        {
            "metric": "actual_edges_across_trees",
            "value": actual_edges,
        },
    ]

    summary = pd.DataFrame(rows)

    summary["value"] = summary["value"].apply(
        lambda value: (
            round(float(value), 4)
            if isinstance(
                value,
                (float, int),
            )
            else value
        )
    )

    return summary


def main() -> None:
    people, relationships, seeds = (
        load_source_data()
    )

    results = []
    discrepancy_rows = []

    for _, seed in seeds.iterrows():
        result = evaluate_tree(
            seed=seed,
            people=people,
            relationships=relationships,
            discrepancy_rows=discrepancy_rows,
        )

        results.append(result)

        status = (
            "PASS"
            if result["exact_tree_match"]
            else "REVIEW"
        )

        print(
            f"{status}: "
            f"{result['seed_label']} — "
            f"{result['actual_node_count']} nodes, "
            f"{result['actual_edge_count']} edges"
        )

    results_df = pd.DataFrame(results)

    summary_df = build_summary(
        results_df
    )

    discrepancies_df = pd.DataFrame(
        discrepancy_rows,
        columns=[
            "seed_label",
            "root_wikitree_id",
            "discrepancy_type",
            "details",
        ],
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    discrepancies_df.to_csv(
        DISCREPANCIES_FILE,
        index=False,
    )

    print("\nTree evaluation complete.")
    print(f"Trees attempted: {len(results_df)}")
    print(
        "Exact matches: "
        f"{int(results_df['exact_tree_match'].sum())}"
    )
    print(
        "Discrepancies: "
        f"{len(discrepancies_df)}"
    )
    print(f"Results: {RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(
        f"Discrepancies: {DISCREPANCIES_FILE}"
    )


if __name__ == "__main__":
    main()