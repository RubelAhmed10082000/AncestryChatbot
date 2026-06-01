"""
Batch-generate preliminary family trees for all WikiTree seed figures.

This wraps generate_family_tree.py and creates one tree per seed profile.

Inputs expected:
    data/wikitree_schema/person.csv
    data/wikitree_schema/names.csv
    data/wikitree_schema/event.csv

Optional input:
    data/wikitree_test/seed_profiles.csv

Requires:
    generate_family_tree.py in the same directory, or pass --tree-module-path.

Outputs:
    data/family_trees/<Wikitree_ID>/family_tree_nodes.csv
    data/family_trees/<Wikitree_ID>/family_tree_edges.csv
    data/family_trees/<Wikitree_ID>/family_tree.json
    data/family_trees/<Wikitree_ID>/family_tree.html
    data/family_trees/<Wikitree_ID>/family_tree_summary.csv

Batch outputs:
    data/family_trees/seed_family_tree_batch_summary.csv
    data/family_trees/seed_family_tree_batch_manifest.csv

Run:
    python generate_seed_family_trees.py --generations 3

Useful options:
    python generate_seed_family_trees.py --generations 3 --include-missing-stubs
    python generate_seed_family_trees.py --output-dir data/family_trees_demo
    python generate_seed_family_trees.py --only Clemens-1 Austen-489 Franklin-10478
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")
DEFAULT_SEED_PROFILES_PATH = Path("data/wikitree_test/seed_profiles.csv")
DEFAULT_OUTPUT_DIR = Path("data/family_trees")
DEFAULT_TREE_MODULE_PATH = Path("generate_family_tree.py")


SEED_FIGURES: list[dict[str, str | None]] = [
    {
        "label": "Samuel Langhorne Clemens / Mark Twain",
        "first_name": "Samuel",
        "last_name": "Clemens",
        "birth_date": "1835-11-30",
        "known_wikitree_id": "Clemens-1",
    },
    {
        "label": "Aretha Franklin",
        "first_name": "Aretha",
        "last_name": "Franklin",
        "birth_date": "1942-03-25",
        "known_wikitree_id": "Franklin-10478",
    },
    {
        "label": "Charles Darwin",
        "first_name": "Charles",
        "last_name": "Darwin",
        "birth_date": "1809-02-12",
        "known_wikitree_id": None,
    },
    {
        "label": "Jane Austen",
        "first_name": "Jane",
        "last_name": "Austen",
        "birth_date": "1775-12-16",
        "known_wikitree_id": None,
    },
    {
        "label": "Isaac Newton",
        "first_name": "Isaac",
        "last_name": "Newton",
        "birth_date": "1643-01-04",
        "known_wikitree_id": None,
    },
    {
        "label": "William Shakespeare",
        "first_name": "William",
        "last_name": "Shakespeare",
        "birth_date": "1564-04-26",
        "known_wikitree_id": None,
    },
    {
        "label": "Florence Nightingale",
        "first_name": "Florence",
        "last_name": "Nightingale",
        "birth_date": "1820-05-12",
        "known_wikitree_id": None,
    },
    {
        "label": "Winston Churchill",
        "first_name": "Winston",
        "last_name": "Churchill",
        "birth_date": "1874-11-30",
        "known_wikitree_id": None,
    },
    {
        "label": "Ada Lovelace",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "birth_date": "1815-12-10",
        "known_wikitree_id": None,
    },
    {
        "label": "Isambard Kingdom Brunel",
        "first_name": "Isambard",
        "last_name": "Brunel",
        "birth_date": "1806-04-09",
        "known_wikitree_id": None,
    },
]


# Used only if seed_profiles.csv is missing or incomplete.
# Newton/Ada are intentionally not hard-coded because your current dataset makes them messy/unavailable.
EXPECTED_ID_FALLBACKS = {
    "Charles Darwin": "Darwin-15",
    "Jane Austen": "Austen-489",
    "William Shakespeare": "Shakespeare-1",
    "Florence Nightingale": "Nightingale-64",
    "Winston Churchill": "Churchill-4",
    "Isambard Kingdom Brunel": "Brunel-8",
}


@dataclass(frozen=True)
class SeedTreeCase:
    label: str
    first_name: str
    last_name: str
    birth_date: str
    wikitree_id: str | None
    id_source: str


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def load_python_module(module_path: Path, module_name: str):
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(
            f"Could not find {module_path}. Put generate_seed_family_trees.py next to "
            "generate_family_tree.py or pass --tree-module-path."
        )

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_seed_profile_ids(seed_profiles_path: Path) -> dict[str, str]:
    """Load seed_label -> wikitree_id from seed_profiles.csv when available."""
    if not seed_profiles_path.exists():
        return {}

    df = pd.read_csv(seed_profiles_path, dtype=str, keep_default_na=False)
    if "seed_label" not in df.columns or "wikitree_id" not in df.columns:
        return {}

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        label = clean_text(row.get("seed_label"))
        wikitree_id = clean_text(row.get("wikitree_id"))
        if label and wikitree_id:
            mapping[label] = wikitree_id
    return mapping


def build_seed_cases(seed_profiles_path: Path) -> list[SeedTreeCase]:
    inferred_ids = load_seed_profile_ids(seed_profiles_path)
    cases: list[SeedTreeCase] = []

    for seed in SEED_FIGURES:
        label = str(seed["label"])
        known_id = clean_text(seed.get("known_wikitree_id"))
        inferred_id = inferred_ids.get(label)
        fallback_id = EXPECTED_ID_FALLBACKS.get(label)

        if known_id:
            wikitree_id = known_id
            source = "known_wikitree_id"
        elif inferred_id:
            wikitree_id = inferred_id
            source = "seed_profiles_csv"
        elif fallback_id:
            wikitree_id = fallback_id
            source = "fallback_expected_id"
        else:
            wikitree_id = None
            source = "unavailable"

        cases.append(
            SeedTreeCase(
                label=label,
                first_name=str(seed["first_name"]),
                last_name=str(seed["last_name"]),
                birth_date=str(seed["birth_date"]),
                wikitree_id=wikitree_id,
                id_source=source,
            )
        )

    return cases


def normalise_filter_values(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


def metric_value(summary_df: pd.DataFrame, metric_name: str) -> Any:
    if summary_df.empty or "metric" not in summary_df.columns or "value" not in summary_df.columns:
        return None
    match = summary_df[summary_df["metric"] == metric_name]
    if match.empty:
        return None
    return match.iloc[0]["value"]


def safe_folder_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def generate_one_tree(
    *,
    tree_module: Any,
    people: pd.DataFrame,
    parent_edges: pd.DataFrame,
    case: SeedTreeCase,
    output_dir: Path,
    generations: int,
    include_missing_stubs: bool,
) -> dict[str, Any]:
    """Generate one tree and return one manifest row."""
    if not case.wikitree_id:
        return {
            "label": case.label,
            "wikitree_id": None,
            "id_source": case.id_source,
            "status": "skipped",
            "reason": "No WikiTree ID available for this seed.",
            "output_folder": None,
            "html_path": None,
            "node_count": 0,
            "edge_count": 0,
            "max_generation": 0,
            "stub_node_count": 0,
        }

    try:
        root_person_id = tree_module.resolve_root_person_id(people, person_id=None, wikitree_id=case.wikitree_id)
        root_row = people[people["Person_ID"] == root_person_id].iloc[0]
        root_label = tree_module.clean_text(root_row.get("Full_Name")) or case.wikitree_id

        nodes, edges = tree_module.collect_ancestor_subgraph(
            root_person_id=root_person_id,
            people=people,
            parent_edges=parent_edges,
            max_generations=generations,
            include_missing_stubs=include_missing_stubs,
        )
        summary = tree_module.summarise_tree(nodes, edges)

        tree_output_dir = output_dir / safe_folder_name(case.wikitree_id)
        tree_output_dir.mkdir(parents=True, exist_ok=True)

        nodes_path = tree_output_dir / "family_tree_nodes.csv"
        edges_path = tree_output_dir / "family_tree_edges.csv"
        summary_path = tree_output_dir / "family_tree_summary.csv"
        json_path = tree_output_dir / "family_tree.json"
        html_path = tree_output_dir / "family_tree.html"

        nodes.to_csv(nodes_path, index=False)
        edges.to_csv(edges_path, index=False)
        summary.to_csv(summary_path, index=False)
        json_path.write_text(
            json.dumps(tree_module.tree_to_json(nodes, edges), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tree_module.generate_html(nodes, edges, root_label, html_path)

        return {
            "label": case.label,
            "wikitree_id": case.wikitree_id,
            "id_source": case.id_source,
            "status": "generated",
            "reason": None,
            "output_folder": str(tree_output_dir),
            "html_path": str(html_path),
            "nodes_csv": str(nodes_path),
            "edges_csv": str(edges_path),
            "json_path": str(json_path),
            "summary_path": str(summary_path),
            "node_count": int(metric_value(summary, "node_count") or 0),
            "edge_count": int(metric_value(summary, "edge_count") or 0),
            "max_generation": int(metric_value(summary, "max_generation") or 0),
            "stub_node_count": int(metric_value(summary, "stub_node_count") or 0),
            "father_edges": int(metric_value(summary, "father_edges") or 0),
            "mother_edges": int(metric_value(summary, "mother_edges") or 0),
        }

    except Exception as exc:  # Keep batch processing going even if one seed fails.
        return {
            "label": case.label,
            "wikitree_id": case.wikitree_id,
            "id_source": case.id_source,
            "status": "failed",
            "reason": str(exc),
            "output_folder": None,
            "html_path": None,
            "node_count": 0,
            "edge_count": 0,
            "max_generation": 0,
            "stub_node_count": 0,
        }


def build_batch_summary(manifest_df: pd.DataFrame, generations: int, include_missing_stubs: bool) -> pd.DataFrame:
    generated = manifest_df[manifest_df["status"] == "generated"].copy()
    skipped = manifest_df[manifest_df["status"] == "skipped"].copy()
    failed = manifest_df[manifest_df["status"] == "failed"].copy()

    rows = [
        {"metric": "requested_seed_cases", "value": len(manifest_df)},
        {"metric": "generated_trees", "value": len(generated)},
        {"metric": "skipped_cases", "value": len(skipped)},
        {"metric": "failed_cases", "value": len(failed)},
        {"metric": "generations_requested", "value": generations},
        {"metric": "include_missing_stubs", "value": include_missing_stubs},
        {"metric": "total_nodes", "value": int(pd.to_numeric(generated.get("node_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())},
        {"metric": "total_edges", "value": int(pd.to_numeric(generated.get("edge_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())},
        {"metric": "total_stub_nodes", "value": int(pd.to_numeric(generated.get("stub_node_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())},
        {"metric": "mean_nodes_per_tree", "value": round(float(pd.to_numeric(generated.get("node_count", pd.Series(dtype=float)), errors="coerce").mean()), 2) if not generated.empty else 0.0},
        {"metric": "mean_edges_per_tree", "value": round(float(pd.to_numeric(generated.get("edge_count", pd.Series(dtype=float)), errors="coerce").mean()), 2) if not generated.empty else 0.0},
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate family trees for all WikiTree seed figures.")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR, help="Directory containing person.csv, names.csv, event.csv.")
    parser.add_argument("--seed-profiles", type=Path, default=DEFAULT_SEED_PROFILES_PATH, help="Path to seed_profiles.csv from extract.py.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for generated family trees.")
    parser.add_argument("--tree-module-path", type=Path, default=DEFAULT_TREE_MODULE_PATH, help="Path to generate_family_tree.py.")
    parser.add_argument("--generations", type=int, default=3, help="Number of ancestor generations to include.")
    parser.add_argument("--include-missing-stubs", action="store_true", help="Include placeholder nodes for missing linked people.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset by WikiTree ID or label text, e.g. --only Clemens-1 Austen-489.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generations < 0:
        raise ValueError("--generations must be 0 or greater.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tree_module = load_python_module(args.tree_module_path, "generate_family_tree")

    person, names, event = tree_module.load_schema(args.schema_dir)
    people = tree_module.build_people_index(person, names, event)
    parent_edges = tree_module.build_parent_edges(event)

    cases = build_seed_cases(args.seed_profiles)
    filters = normalise_filter_values(args.only)
    if filters:
        cases = [
            case
            for case in cases
            if (case.wikitree_id and case.wikitree_id.lower() in filters)
            or case.label.lower() in filters
            or any(fragment in case.label.lower() for fragment in filters)
        ]

    manifest_rows = []
    for case in cases:
        row = generate_one_tree(
            tree_module=tree_module,
            people=people,
            parent_edges=parent_edges,
            case=case,
            output_dir=args.output_dir,
            generations=args.generations,
            include_missing_stubs=args.include_missing_stubs,
        )
        manifest_rows.append(row)
        print(f"{row['status'].upper():9} {case.label} -> {row.get('wikitree_id') or 'N/A'}")

    manifest_df = pd.DataFrame(manifest_rows)
    summary_df = build_batch_summary(manifest_df, args.generations, args.include_missing_stubs)

    manifest_path = args.output_dir / "seed_family_tree_batch_manifest.csv"
    summary_path = args.output_dir / "seed_family_tree_batch_summary.csv"

    manifest_df.to_csv(manifest_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSeed family tree batch generation complete.")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    generated = manifest_df[manifest_df["status"] == "generated"]
    if not generated.empty:
        print("\nGenerated HTML files:")
        for path in generated["html_path"].dropna().tolist():
            print(f"- {path}")


if __name__ == "__main__":
    main()
