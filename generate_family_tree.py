"""
Generate a preliminary family tree from the transformed WikiTree schema dataset.

Inputs expected from transform.py / transform_wikitree_to_schema.py:
    data/wikitree_schema/person.csv
    data/wikitree_schema/names.csv
    data/wikitree_schema/event.csv

Core assumption:
    Parent-child relationships are stored in event.csv as:
        Event_Type = father_of or mother_of
        Person_ID_1 = parent
        Person_ID_2 = child

Outputs:
    data/family_trees/<root_wikitree_id>/family_tree_nodes.csv
    data/family_trees/<root_wikitree_id>/family_tree_edges.csv
    data/family_trees/<root_wikitree_id>/family_tree.json
    data/family_trees/<root_wikitree_id>/family_tree.html
    data/family_trees/<root_wikitree_id>/family_tree_summary.csv

Example:
    python generate_family_tree.py --wikitree-id Clemens-1 --generations 3
    python generate_family_tree.py --wikitree-id Austen-489 --generations 3 --include-missing-stubs
    python generate_family_tree.py --person-id <schema-person-uuid> --generations 2

Notes:
    - This is a preliminary graph generator for dissertation prototyping.
    - It does not claim genealogical certainty; it visualises relationships already present
      in the transformed local dataset.
    - Use --include-missing-stubs to include placeholder nodes when a relationship references
      a parent/child that is not present in person.csv. This is useful for showing data gaps.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")
DEFAULT_OUTPUT_DIR = Path("data/family_trees")

PERSON_FILE = "person.csv"
NAMES_FILE = "names.csv"
EVENT_FILE = "event.csv"

PARENT_EVENT_TYPES = {"father_of", "mother_of", "parent_of"}


def clean_text(value: Any) -> str | None:
    """Return stripped text or None for blank/NaN-like values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text in {"", "nan", "NaN", "None", "none", "NULL", "null"}:
        return None
    return text


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def safe_int(value: Any) -> int | None:
    text = clean_text(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def build_full_name(row: pd.Series) -> str:
    parts = [
        clean_text(row.get("First_Name")),
        clean_text(row.get("Middle_Name")),
        clean_text(row.get("Last_Name_At_Birth")) or clean_text(row.get("Last_Name_Current")),
    ]
    parts = [part for part in parts if part]
    if parts:
        return " ".join(parts)
    return clean_text(row.get("Wikitree_ID")) or clean_text(row.get("Person_ID")) or "Unknown person"


def load_schema(schema_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    person = read_required_csv(schema_dir / PERSON_FILE)
    names = read_required_csv(schema_dir / NAMES_FILE)
    event = read_required_csv(schema_dir / EVENT_FILE)

    for col in ["Person_ID", "Wikitree_ID", "Gender", "Profile_URL", "Has_Children"]:
        if col not in person.columns:
            person[col] = None

    for col in [
        "Person_ID",
        "First_Name",
        "Middle_Name",
        "Last_Name_At_Birth",
        "Last_Name_Current",
        "Nicknames",
    ]:
        if col not in names.columns:
            names[col] = None

    for col in [
        "Marriage_ID",
        "Person_ID_1",
        "Person_ID_2",
        "Event_Type",
        "Event_Raw_Date",
        "Event_Year",
        "Event_Location",
        "Data Status",
    ]:
        if col not in event.columns:
            event[col] = None

    return person, names, event


def build_people_index(person: pd.DataFrame, names: pd.DataFrame, event: pd.DataFrame) -> pd.DataFrame:
    """Create one person row with name, birth, death, and profile data."""
    people = person.merge(names, on="Person_ID", how="left", suffixes=("", "_name"))

    birth = (
        event[event["Event_Type"] == "birth"]
        .drop_duplicates(subset=["Person_ID_1"], keep="first")
        .rename(
            columns={
                "Person_ID_1": "Person_ID",
                "Event_Raw_Date": "Birth_Date",
                "Event_Year": "Birth_Year",
                "Event_Location": "Birth_Location",
                "Data Status": "Birth_Data_Status",
            }
        )
    )

    death = (
        event[event["Event_Type"] == "death"]
        .drop_duplicates(subset=["Person_ID_1"], keep="first")
        .rename(
            columns={
                "Person_ID_1": "Person_ID",
                "Event_Raw_Date": "Death_Date",
                "Event_Year": "Death_Year",
                "Event_Location": "Death_Location",
                "Data Status": "Death_Data_Status",
            }
        )
    )

    birth_cols = ["Person_ID", "Birth_Date", "Birth_Year", "Birth_Location", "Birth_Data_Status"]
    death_cols = ["Person_ID", "Death_Date", "Death_Year", "Death_Location", "Death_Data_Status"]

    people = people.merge(birth[birth_cols], on="Person_ID", how="left")
    people = people.merge(death[death_cols], on="Person_ID", how="left")
    people["Full_Name"] = people.apply(build_full_name, axis=1)

    return people


def resolve_root_person_id(people: pd.DataFrame, person_id: str | None, wikitree_id: str | None) -> str:
    if person_id:
        match = people[people["Person_ID"] == person_id]
        if not match.empty:
            return str(match.iloc[0]["Person_ID"])
        raise ValueError(f"No person found with Person_ID={person_id}")

    if wikitree_id:
        match = people[people["Wikitree_ID"] == wikitree_id]
        if not match.empty:
            return str(match.iloc[0]["Person_ID"])
        raise ValueError(f"No person found with Wikitree_ID={wikitree_id}")

    raise ValueError("Provide either --person-id or --wikitree-id.")


def build_parent_edges(event: pd.DataFrame) -> pd.DataFrame:
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
    return edges[keep_cols].dropna(subset=["parent_person_id", "child_person_id"])


def collect_ancestor_subgraph(
    *,
    root_person_id: str,
    people: pd.DataFrame,
    parent_edges: pd.DataFrame,
    max_generations: int,
    include_missing_stubs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk upward from root and collect ancestors up to max_generations."""
    person_ids_in_people = set(people["Person_ID"].astype(str))
    parent_lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for _, edge in parent_edges.iterrows():
        child_id = str(edge["child_person_id"])
        parent_lookup[child_id].append(edge.to_dict())

    visited_depth: dict[str, int] = {root_person_id: 0}
    output_edges: list[dict[str, Any]] = []
    queue: deque[tuple[str, int]] = deque([(root_person_id, 0)])

    while queue:
        current_id, generation = queue.popleft()
        if generation >= max_generations:
            continue

        for edge in parent_lookup.get(current_id, []):
            parent_id = str(edge["parent_person_id"])
            child_id = str(edge["child_person_id"])

            parent_exists = parent_id in person_ids_in_people
            child_exists = child_id in person_ids_in_people

            if not include_missing_stubs and (not parent_exists or not child_exists):
                continue

            edge_record = dict(edge)
            edge_record["parent_exists_in_people"] = parent_exists
            edge_record["child_exists_in_people"] = child_exists
            edge_record["parent_generation"] = generation + 1
            edge_record["child_generation"] = generation
            output_edges.append(edge_record)

            if parent_id not in visited_depth or visited_depth[parent_id] > generation + 1:
                visited_depth[parent_id] = generation + 1
                if parent_exists:
                    queue.append((parent_id, generation + 1))

    # Build node table.
    node_rows: list[dict[str, Any]] = []
    people_by_id = {str(row["Person_ID"]): row for _, row in people.iterrows()}

    for pid, generation in sorted(visited_depth.items(), key=lambda item: (item[1], item[0])):
        person_row = people_by_id.get(pid)

        if person_row is None:
            if not include_missing_stubs:
                continue
            node_rows.append(
                {
                    "person_id": pid,
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
                    "is_root": pid == root_person_id,
                    "is_stub": True,
                }
            )
            continue

        node_rows.append(
            {
                "person_id": pid,
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
                "is_root": pid == root_person_id,
                "is_stub": False,
            }
        )

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(output_edges).drop_duplicates(
        subset=["parent_person_id", "child_person_id", "relationship_type"], keep="first"
    )

    if not edges_df.empty:
        edges_df = edges_df[edges_df["parent_person_id"].isin(nodes_df["person_id"]) & edges_df["child_person_id"].isin(nodes_df["person_id"])]

    return nodes_df.reset_index(drop=True), edges_df.reset_index(drop=True)


def summarise_tree(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    generation_counts = nodes["generation"].value_counts().sort_index().to_dict() if not nodes.empty else {}
    rows = [
        {"metric": "node_count", "value": len(nodes)},
        {"metric": "edge_count", "value": len(edges)},
        {"metric": "max_generation", "value": int(nodes["generation"].max()) if not nodes.empty else 0},
        {"metric": "stub_node_count", "value": int(nodes["is_stub"].sum()) if "is_stub" in nodes.columns else 0},
        {"metric": "father_edges", "value": int((edges["relationship_type"] == "father_of").sum()) if not edges.empty else 0},
        {"metric": "mother_edges", "value": int((edges["relationship_type"] == "mother_of").sum()) if not edges.empty else 0},
    ]

    for generation, count in generation_counts.items():
        rows.append({"metric": f"generation_{generation}_nodes", "value": int(count)})

    return pd.DataFrame(rows)


def tree_to_json(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    return {
        "nodes": nodes.to_dict(orient="records"),
        "edges": edges.to_dict(orient="records"),
    }


def html_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return html.escape(text)


def node_label(row: pd.Series) -> str:
    name = clean_text(row.get("full_name")) or "Unknown person"
    wt = clean_text(row.get("wikitree_id"))
    birth_year = clean_text(row.get("birth_year"))
    death_year = clean_text(row.get("death_year"))

    dates = ""
    if birth_year or death_year:
        dates = f" ({birth_year or '?'}–{death_year or '?'})"

    return f"{name}{dates}" + (f"\n{wt}" if wt else "")


def generate_html(nodes: pd.DataFrame, edges: pd.DataFrame, root_label: str, output_path: Path) -> None:
    """Generate a self-contained static SVG HTML family tree."""
    if nodes.empty:
        output_path.write_text("<html><body><h1>No tree data found</h1></body></html>", encoding="utf-8")
        return

    # Layout: generations as columns, nodes within generation stacked vertically.
    generation_groups: dict[int, list[str]] = defaultdict(list)
    for _, row in nodes.sort_values(by=["generation", "full_name"]).iterrows():
        generation_groups[int(row["generation"])].append(row["person_id"])

    x_gap = 330
    y_gap = 120
    box_w = 240
    box_h = 74
    margin_x = 60
    margin_y = 70

    positions: dict[str, tuple[int, int]] = {}
    max_rows = max(len(ids) for ids in generation_groups.values())
    max_generation = max(generation_groups.keys())

    for generation, ids in generation_groups.items():
        total_height = (len(ids) - 1) * y_gap
        start_y = margin_y + max(0, (max_rows - len(ids)) * y_gap // 2)
        for idx, person_id in enumerate(ids):
            x = margin_x + generation * x_gap
            y = start_y + idx * y_gap
            positions[person_id] = (x, y)

    width = margin_x * 2 + (max_generation + 1) * x_gap + box_w
    height = margin_y * 2 + max_rows * y_gap + box_h

    nodes_by_id = {row["person_id"]: row for _, row in nodes.iterrows()}

    edge_svg = []
    for _, edge in edges.iterrows():
        parent = str(edge["parent_person_id"])
        child = str(edge["child_person_id"])
        if parent not in positions or child not in positions:
            continue
        px, py = positions[parent]
        cx, cy = positions[child]
        # Parent is to the right of child in ancestor tree.
        x1 = px
        y1 = py + box_h / 2
        x2 = cx + box_w
        y2 = cy + box_h / 2
        rel = clean_text(edge.get("relationship_type")) or "parent_of"
        stroke_dasharray = "" if rel == "father_of" else " stroke-dasharray='6 4'"
        edge_svg.append(
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' class='edge'{stroke_dasharray} />"
        )

    node_svg = []
    for person_id, (x, y) in positions.items():
        row = nodes_by_id[person_id]
        is_root = bool(row.get("is_root"))
        is_stub = bool(row.get("is_stub"))
        gender = clean_text(row.get("gender")) or "Unknown"
        css_class = "node root" if is_root else "node stub" if is_stub else "node"
        label_lines = node_label(row).split("\n")
        title = html_escape(
            f"{row.get('full_name')} | {row.get('wikitree_id')} | Born: {row.get('birth_date')} | Died: {row.get('death_date')}"
        )
        profile_url = clean_text(row.get("profile_url"))

        text_lines = []
        for i, line in enumerate(label_lines[:3]):
            font_size = 13 if i == 0 else 11
            dy = 22 + i * 18
            text_lines.append(
                f"<text x='{x + 12}' y='{y + dy}' font-size='{font_size}'>{html_escape(line)}</text>"
            )

        if profile_url:
            node_svg.append(f"<a href='{html_escape(profile_url)}' target='_blank'>")

        node_svg.append(f"<g class='{css_class}'>")
        node_svg.append(f"<title>{title}</title>")
        node_svg.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='{box_h}' rx='10' />")
        node_svg.extend(text_lines)
        node_svg.append(f"<text x='{x + 12}' y='{y + box_h - 10}' font-size='10' class='meta'>{html_escape(gender)}</text>")
        node_svg.append("</g>")

        if profile_url:
            node_svg.append("</a>")

    legend = """
    <div class="legend">
      <strong>Legend:</strong>
      <span class="legend-box root-box"></span> root person
      <span class="legend-box normal-box"></span> extracted person
      <span class="legend-line solid-line"></span> father_of / parent link
      <span class="legend-line dashed-line"></span> mother_of link
    </div>
    """

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Family Tree - {html_escape(root_label)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8f9fb; color: #1f2937; }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #4b5563; margin-bottom: 16px; }}
    .canvas {{ background: white; border: 1px solid #d1d5db; border-radius: 12px; overflow: auto; padding: 12px; }}
    svg {{ min-width: 100%; }}
    .edge {{ stroke: #6b7280; stroke-width: 2; }}
    .node rect {{ fill: #ffffff; stroke: #374151; stroke-width: 1.5; }}
    .node.root rect {{ fill: #eef2ff; stroke: #3730a3; stroke-width: 2.5; }}
    .node.stub rect {{ fill: #fef2f2; stroke: #991b1b; stroke-width: 1.5; }}
    .node text {{ fill: #111827; pointer-events: none; }}
    .node .meta {{ fill: #6b7280; }}
    .legend {{ margin: 12px 0 18px; color: #374151; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
    .legend-box {{ width: 18px; height: 12px; display: inline-block; border-radius: 3px; margin-right: -8px; }}
    .root-box {{ background: #eef2ff; border: 1px solid #3730a3; }}
    .normal-box {{ background: #fff; border: 1px solid #374151; }}
    .legend-line {{ width: 36px; height: 0; display: inline-block; border-top: 2px solid #6b7280; margin-right: -8px; }}
    .dashed-line {{ border-top-style: dashed; }}
    .table-link {{ margin-top: 16px; font-size: 14px; color: #4b5563; }}
  </style>
</head>
<body>
  <h1>Preliminary Family Tree</h1>
  <div class="subtitle">Root: <strong>{html_escape(root_label)}</strong>. Ancestors are shown left-to-right by generation.</div>
  {legend}
  <div class="canvas">
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
      {''.join(edge_svg)}
      {''.join(node_svg)}
    </svg>
  </div>
  <p class="table-link">This is a preliminary visualisation from local transformed WikiTree data. It should be treated as a starting point for verification, not as final genealogical proof.</p>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def build_output_dir(base_output_dir: Path, root_wikitree_id: str | None, root_person_id: str) -> Path:
    folder_name = root_wikitree_id or root_person_id
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in folder_name)
    return base_output_dir / safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a preliminary family tree from transformed WikiTree schema CSVs.")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR, help="Directory containing person.csv, names.csv, event.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base output directory for generated family trees.")
    parser.add_argument("--wikitree-id", type=str, default=None, help="Root WikiTree ID, e.g. Clemens-1.")
    parser.add_argument("--person-id", type=str, default=None, help="Root internal schema Person_ID UUID.")
    parser.add_argument("--generations", type=int, default=3, help="Number of ancestor generations to include.")
    parser.add_argument("--include-missing-stubs", action="store_true", help="Include placeholder nodes for missing linked people.")
    return parser.parse_args()


def main() -> None:
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
    html_path = output_dir / "family_tree.html"

    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(tree_to_json(nodes, edges), indent=2, ensure_ascii=False), encoding="utf-8")
    generate_html(nodes, edges, root_label, html_path)

    print("Family tree generated.")
    print(f"Root:          {root_label} ({root_wikitree_id or root_person_id})")
    print(f"Generations:   {args.generations}")
    print(f"Nodes:         {len(nodes)}")
    print(f"Edges:         {len(edges)}")
    print(f"Output folder: {output_dir}")
    print(f"HTML:          {html_path}")
    print(f"Nodes CSV:     {nodes_path}")
    print(f"Edges CSV:     {edges_path}")
    print(f"JSON:          {json_path}")


if __name__ == "__main__":
    main()
