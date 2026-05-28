"""
Transform extracted WikiTree sample data into the project's relational schema V0.1.

Inputs expected from extract.py:
    data/wikitree_test/people.csv
    data/wikitree_test/relationships.csv

Outputs:
    data/wikitree_schema/person.csv
    data/wikitree_schema/names.csv
    data/wikitree_schema/event.csv
    data/wikitree_schema/id_crosswalk.csv
    data/wikitree_schema/relationship_rejections.csv
    data/wikitree_schema/transform_quality_report.csv

Run from your project root:
    python transform_wikitree_to_schema.py

Notes:
    - Your V0.1 schema names the Event primary key as Marriage_ID. This script keeps that
      column name for schema compatibility, but uses it as a generic event UUID for birth,
      death, father_of, and mother_of events.
    - Person_ID and Name_ID are generated as deterministic UUIDv5 values so rerunning the
      script produces stable IDs.
    - Wikitree_ID is preserved as the external source identifier.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

INPUT_DIR = Path("data/wikitree_test")
OUTPUT_DIR = Path("data/wikitree_schema")

PEOPLE_INPUT = INPUT_DIR / "people.csv"
RELATIONSHIPS_INPUT = INPUT_DIR / "relationships.csv"

PERSON_OUTPUT = OUTPUT_DIR / "person.csv"
NAMES_OUTPUT = OUTPUT_DIR / "names.csv"
EVENT_OUTPUT = OUTPUT_DIR / "event.csv"
ID_CROSSWALK_OUTPUT = OUTPUT_DIR / "id_crosswalk.csv"
RELATIONSHIP_REJECTIONS_OUTPUT = OUTPUT_DIR / "relationship_rejections.csv"
QUALITY_REPORT_OUTPUT = OUTPUT_DIR / "transform_quality_report.csv"

WIKITREE_PROFILE_BASE_URL = "https://www.wikitree.com/wiki/"

# A fixed namespace makes generated UUIDs stable across reruns.
PROJECT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "from-conversation-to-family-trees/wikitree-schema-v0.1")

UNKNOWN_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "None",
    "none",
    "NULL",
    "null",
    "Unknown",
    "unknown",
    "UNKNOWN",
    "0000-00-00",
    "0000",
    "0",
    "0.0",
}


# -----------------------------------------------------------------------------
# Generic cleaning helpers
# -----------------------------------------------------------------------------


def clean_text(value: Any) -> str | None:
    """Return a stripped string or None for empty/unknown-like values."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text in UNKNOWN_VALUES:
        return None

    text = re.sub(r"\s+", " ", text)
    return text or None



def clean_id(value: Any) -> str | None:
    """Clean numeric/string IDs loaded from CSV, including 123.0-style values."""
    text = clean_text(value)
    if text is None:
        return None

    if text.endswith(".0"):
        text = text[:-2]

    return text



def normalise_gender(value: Any) -> str | None:
    """Normalise WikiTree gender values into a small enum-like set."""
    text = clean_text(value)
    if text is None:
        return None

    lower = text.lower()
    if lower in {"male", "m"}:
        return "Male"
    if lower in {"female", "f"}:
        return "Female"
    return "Unknown"



def parse_date_parts(value: Any) -> tuple[str | None, int | None, int | None, int | None]:
    """
    Parse WikiTree-style dates into raw date, year, month, day.

    Handles examples such as:
        1835-11-30
        1773-00-00
        1858-05-00
        0000-00-00
    """
    raw = clean_text(value)
    if raw is None:
        return None, None, None, None

    # Extract first plausible year.
    year_match = re.search(r"(\d{3,4})", raw)
    year = int(year_match.group(1)) if year_match else None

    month = None
    day = None

    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if iso_match:
        year = int(iso_match.group(1))
        raw_month = int(iso_match.group(2))
        raw_day = int(iso_match.group(3))
        month = raw_month if 1 <= raw_month <= 12 else None
        day = raw_day if 1 <= raw_day <= 31 else None

    if year is not None and year <= 0:
        year = None

    return raw, year, month, day



def clean_location(value: Any) -> str | None:
    """Clean location strings while preserving meaningful detail."""
    text = clean_text(value)
    if text is None:
        return None

    text = text.replace(" ,", ",")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text or None



def parse_data_status(value: Any) -> dict[str, Any]:
    """Parse the JSON-ish data_status column produced by extract.py."""
    text = clean_text(value)
    if text is None:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}



def status_for_field(row: pd.Series, field_name: str) -> str | None:
    """Get WikiTree DataStatus value for a given field, if available."""
    status = row.get("_parsed_data_status")
    if isinstance(status, dict):
        value = status.get(field_name)
        return clean_text(value)
    return None



def stable_uuid(entity_type: str, natural_key: str) -> str:
    """Generate stable UUIDv5 IDs from deterministic natural keys."""
    return str(uuid.uuid5(PROJECT_NAMESPACE, f"{entity_type}:{natural_key}"))



def profile_url(wikitree_id: str | None) -> str | None:
    if not wikitree_id:
        return None
    return f"{WIKITREE_PROFILE_BASE_URL}{wikitree_id}"



def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------


def prepare_people(raw_people: pd.DataFrame) -> pd.DataFrame:
    """Clean the extracted people.csv into a stable intermediate dataframe."""
    df = raw_people.copy()

    expected_columns = [
        "person_id",
        "wikitree_id",
        "first_name",
        "middle_name",
        "last_name_at_birth",
        "last_name_current",
        "birth_date",
        "birth_location",
        "death_date",
        "death_location",
        "gender",
        "father_id",
        "mother_id",
        "privacy",
        "data_status",
    ]

    for column in expected_columns:
        if column not in df.columns:
            df[column] = None

    for column in ["person_id", "father_id", "mother_id"]:
        df[column] = df[column].apply(clean_id)

    for column in [
        "wikitree_id",
        "first_name",
        "middle_name",
        "last_name_at_birth",
        "last_name_current",
        "birth_date",
        "birth_location",
        "death_date",
        "death_location",
        "privacy",
        "data_status",
    ]:
        df[column] = df[column].apply(clean_text)

    df["gender"] = df["gender"].apply(normalise_gender)
    df["birth_location"] = df["birth_location"].apply(clean_location)
    df["death_location"] = df["death_location"].apply(clean_location)
    df["_parsed_data_status"] = df["data_status"].apply(parse_data_status)

    # Drop rows that cannot be identified.
    df = df[df["person_id"].notna() & df["wikitree_id"].notna()].copy()

    # Deduplicate by WikiTree numeric ID first, then WikiTree page ID.
    df = df.drop_duplicates(subset=["person_id"], keep="first")
    df = df.drop_duplicates(subset=["wikitree_id"], keep="first")

    # Stable schema UUIDs.
    df["schema_person_id"] = df["wikitree_id"].apply(lambda key: stable_uuid("person", key))
    df["schema_name_id"] = df["wikitree_id"].apply(lambda key: stable_uuid("name", key))

    return df.reset_index(drop=True)



def prepare_relationships(raw_relationships: pd.DataFrame) -> pd.DataFrame:
    """Clean raw relationship rows from extract.py."""
    df = raw_relationships.copy()

    expected_columns = ["parent_id", "child_id", "child_wikitree_id", "relationship_type"]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = None

    df["parent_id"] = df["parent_id"].apply(clean_id)
    df["child_id"] = df["child_id"].apply(clean_id)
    df["child_wikitree_id"] = df["child_wikitree_id"].apply(clean_text)
    df["relationship_type"] = df["relationship_type"].apply(clean_text)

    df = df[df["parent_id"].notna() & df["child_id"].notna()].copy()
    df = df.drop_duplicates(subset=["parent_id", "child_id", "relationship_type"], keep="first")

    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Schema table builders
# -----------------------------------------------------------------------------


def build_person_table(people: pd.DataFrame, relationships: pd.DataFrame) -> pd.DataFrame:
    """Build Person table matching the uploaded V0.1 schema."""
    parent_ids_with_children = set(relationships["parent_id"].dropna())

    records = []
    for _, row in people.iterrows():
        source_person_id = row["person_id"]
        wikitree_id = row["wikitree_id"]

        records.append(
            {
                "Person_ID": row["schema_person_id"],
                "Wikitree_ID": wikitree_id,
                "Gender": row["gender"],
                "Profile_URL": profile_url(wikitree_id),
                "Has_Children": source_person_id in parent_ids_with_children,
            }
        )

    return pd.DataFrame(
        records,
        columns=["Person_ID", "Wikitree_ID", "Gender", "Profile_URL", "Has_Children"],
    )



def build_names_table(people: pd.DataFrame) -> pd.DataFrame:
    """Build Names table matching the uploaded V0.1 schema."""
    records = []

    for _, row in people.iterrows():
        middle_name = clean_text(row.get("middle_name"))
        middle_initials = None
        if middle_name:
            middle_initials = "".join(part[0].upper() for part in middle_name.split() if part)

        records.append(
            {
                "Name_ID": row["schema_name_id"],
                "Person_ID": row["schema_person_id"],
                "Last_Name_Current": clean_text(row.get("last_name_current")),
                "Last_Name_At_Birth": clean_text(row.get("last_name_at_birth")),
                "Middle_Name": middle_name,
                "Middle_Initials": middle_initials,
                "First_Name": clean_text(row.get("first_name")),
                "Prefix": None,
                "Suffix": None,
                "Nicknames": None,
            }
        )

    return pd.DataFrame(
        records,
        columns=[
            "Name_ID",
            "Person_ID",
            "Last_Name_Current",
            "Last_Name_At_Birth",
            "Middle_Name",
            "Middle_Initials",
            "First_Name",
            "Prefix",
            "Suffix",
            "Nicknames",
        ],
    )



def event_row(
    *,
    event_key: str,
    person_id_1: str,
    person_id_2: str | None,
    event_type: str,
    raw_date: str | None = None,
    month: int | None = None,
    day: int | None = None,
    year: int | None = None,
    location: str | None = None,
    data_status: str | None = None,
) -> dict[str, Any]:
    """Create one event row using the V0.1 Event column names."""
    return {
        "Marriage_ID": stable_uuid("event", event_key),
        "Person_ID_1": person_id_1,
        "Person_ID_2": person_id_2,
        "Event_Type": event_type,
        "Event_Raw_Date": raw_date,
        "Event_Month": month,
        "Event_Day": day,
        "Event_Year": year,
        "Event_Location": location,
        "Data Status": data_status,
    }



def build_event_table(people: pd.DataFrame, relationships: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Event table from birth, death, and parent-child relationships.

    Returns:
        event_df, relationship_rejections_df
    """
    events: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    source_id_to_schema_id = dict(zip(people["person_id"], people["schema_person_id"]))
    source_id_to_wikitree_id = dict(zip(people["person_id"], people["wikitree_id"]))

    # Birth/death events.
    for _, row in people.iterrows():
        schema_person_id = row["schema_person_id"]
        wikitree_id = row["wikitree_id"]

        birth_raw, birth_year, birth_month, birth_day = parse_date_parts(row.get("birth_date"))
        if birth_raw or row.get("birth_location"):
            events.append(
                event_row(
                    event_key=f"{wikitree_id}:birth",
                    person_id_1=schema_person_id,
                    person_id_2=None,
                    event_type="birth",
                    raw_date=birth_raw,
                    month=birth_month,
                    day=birth_day,
                    year=birth_year,
                    location=row.get("birth_location"),
                    data_status=status_for_field(row, "BirthDate") or status_for_field(row, "BirthLocation"),
                )
            )

        death_raw, death_year, death_month, death_day = parse_date_parts(row.get("death_date"))
        if death_raw or row.get("death_location"):
            events.append(
                event_row(
                    event_key=f"{wikitree_id}:death",
                    person_id_1=schema_person_id,
                    person_id_2=None,
                    event_type="death",
                    raw_date=death_raw,
                    month=death_month,
                    day=death_day,
                    year=death_year,
                    location=row.get("death_location"),
                    data_status=status_for_field(row, "DeathDate") or status_for_field(row, "DeathLocation"),
                )
            )

    # Parent-child relationship events.
    for _, rel in relationships.iterrows():
        parent_source_id = rel.get("parent_id")
        child_source_id = rel.get("child_id")
        relationship_type = clean_text(rel.get("relationship_type")) or "parent_of"

        parent_schema_id = source_id_to_schema_id.get(parent_source_id)
        child_schema_id = source_id_to_schema_id.get(child_source_id)

        if not parent_schema_id or not child_schema_id:
            rejections.append(
                {
                    "parent_id": parent_source_id,
                    "child_id": child_source_id,
                    "child_wikitree_id": rel.get("child_wikitree_id"),
                    "relationship_type": relationship_type,
                    "reason": "parent_or_child_not_present_in_people_csv",
                }
            )
            continue

        parent_wikitree_id = source_id_to_wikitree_id.get(parent_source_id)
        child_wikitree_id = source_id_to_wikitree_id.get(child_source_id)
        event_key = f"{parent_wikitree_id}:{relationship_type}:{child_wikitree_id}"

        events.append(
            event_row(
                event_key=event_key,
                person_id_1=parent_schema_id,
                person_id_2=child_schema_id,
                event_type=relationship_type,
                raw_date=None,
                month=None,
                day=None,
                year=None,
                location=None,
                data_status=None,
            )
        )

    event_df = pd.DataFrame(
        events,
        columns=[
            "Marriage_ID",
            "Person_ID_1",
            "Person_ID_2",
            "Event_Type",
            "Event_Raw_Date",
            "Event_Month",
            "Event_Day",
            "Event_Year",
            "Event_Location",
            "Data Status",
        ],
    ).drop_duplicates(subset=["Marriage_ID"], keep="first")

    rejections_df = pd.DataFrame(
        rejections,
        columns=["parent_id", "child_id", "child_wikitree_id", "relationship_type", "reason"],
    )

    return event_df.reset_index(drop=True), rejections_df.reset_index(drop=True)



def build_id_crosswalk(people: pd.DataFrame) -> pd.DataFrame:
    """Build source-to-schema ID mapping for debugging and database loading."""
    return people[
        [
            "person_id",
            "wikitree_id",
            "schema_person_id",
            "schema_name_id",
            "first_name",
            "middle_name",
            "last_name_at_birth",
            "last_name_current",
        ]
    ].rename(
        columns={
            "person_id": "source_wikitree_numeric_id",
            "wikitree_id": "Wikitree_ID",
            "schema_person_id": "Person_ID",
            "schema_name_id": "Name_ID",
        }
    )



def build_quality_report(
    *,
    raw_people: pd.DataFrame,
    people: pd.DataFrame,
    raw_relationships: pd.DataFrame,
    relationships: pd.DataFrame,
    person_table: pd.DataFrame,
    names_table: pd.DataFrame,
    event_table: pd.DataFrame,
    rejections: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise transformation results."""
    event_counts = event_table["Event_Type"].value_counts(dropna=False).to_dict()

    metrics = [
        {"metric": "raw_people_rows", "value": len(raw_people)},
        {"metric": "schema_people_rows", "value": len(person_table)},
        {"metric": "schema_names_rows", "value": len(names_table)},
        {"metric": "raw_relationship_rows", "value": len(raw_relationships)},
        {"metric": "clean_relationship_rows", "value": len(relationships)},
        {"metric": "schema_event_rows", "value": len(event_table)},
        {"metric": "relationship_rejections", "value": len(rejections)},
        {"metric": "birth_event_rows", "value": int(event_counts.get("birth", 0))},
        {"metric": "death_event_rows", "value": int(event_counts.get("death", 0))},
        {"metric": "father_of_event_rows", "value": int(event_counts.get("father_of", 0))},
        {"metric": "mother_of_event_rows", "value": int(event_counts.get("mother_of", 0))},
        {"metric": "people_missing_gender", "value": int(person_table["Gender"].isna().sum())},
        {"metric": "people_with_children", "value": int(person_table["Has_Children"].sum())},
        {"metric": "names_missing_first_name", "value": int(names_table["First_Name"].isna().sum())},
        {"metric": "names_missing_last_name_at_birth", "value": int(names_table["Last_Name_At_Birth"].isna().sum())},
    ]

    return pd.DataFrame(metrics, columns=["metric", "value"])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_people = read_csv_required(PEOPLE_INPUT)
    raw_relationships = read_csv_required(RELATIONSHIPS_INPUT)

    people = prepare_people(raw_people)
    relationships = prepare_relationships(raw_relationships)

    person_table = build_person_table(people, relationships)
    names_table = build_names_table(people)
    event_table, rejections = build_event_table(people, relationships)
    id_crosswalk = build_id_crosswalk(people)

    quality_report = build_quality_report(
        raw_people=raw_people,
        people=people,
        raw_relationships=raw_relationships,
        relationships=relationships,
        person_table=person_table,
        names_table=names_table,
        event_table=event_table,
        rejections=rejections,
    )

    person_table.to_csv(PERSON_OUTPUT, index=False)
    names_table.to_csv(NAMES_OUTPUT, index=False)
    event_table.to_csv(EVENT_OUTPUT, index=False)
    id_crosswalk.to_csv(ID_CROSSWALK_OUTPUT, index=False)
    rejections.to_csv(RELATIONSHIP_REJECTIONS_OUTPUT, index=False)
    quality_report.to_csv(QUALITY_REPORT_OUTPUT, index=False)

    print("Transformation complete.")
    print(f"Person rows: {len(person_table)} -> {PERSON_OUTPUT}")
    print(f"Names rows: {len(names_table)} -> {NAMES_OUTPUT}")
    print(f"Event rows: {len(event_table)} -> {EVENT_OUTPUT}")
    print(f"Rejected relationship rows: {len(rejections)} -> {RELATIONSHIP_REJECTIONS_OUTPUT}")
    print(f"ID crosswalk: {ID_CROSSWALK_OUTPUT}")
    print(f"Quality report: {QUALITY_REPORT_OUTPUT}")
    print("\nQuality report:")
    print(quality_report.to_string(index=False))


if __name__ == "__main__":
    main()
