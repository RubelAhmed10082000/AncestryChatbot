"""
Transform WikiTree data into relational schema.

The module reads the people and relationship records produced by the
extraction module, cleans and normalises their values, and creates Person,
Name and Event tables.
"""

import json
import re
import uuid
from pathlib import Path
from typing import Any
import pandas as pd


INPUT_DIR = Path("data/wikitree_test")
OUTPUT_DIR = Path("data/wikitree_schema")

# Input  filepaths
PEOPLE_INPUT = INPUT_DIR / "people.csv"
RELATIONSHIPS_INPUT = INPUT_DIR / "relationships.csv"

# Output file paths
PERSON_OUTPUT = OUTPUT_DIR / "person.csv"
NAMES_OUTPUT = OUTPUT_DIR / "names.csv"
EVENT_OUTPUT = OUTPUT_DIR / "event.csv"
ID_CROSSWALK_OUTPUT = OUTPUT_DIR / "id_crosswalk.csv"
RELATIONSHIP_REJECTIONS_OUTPUT = OUTPUT_DIR / "relationship_rejections.csv"
QUALITY_REPORT_OUTPUT = OUTPUT_DIR / "transform_quality_report.csv"

WIKITREE_PROFILE_BASE_URL = "https://www.wikitree.com/wiki/"
PROJECT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "from-conversation-to-family-trees/wikitree-schema-v0.1")

# Unknown values allowing us to better clean the data
UNKNOWN_VALUES = {
    "", " ", "nan", "NaN", "None", "none", "NULL", "null",
    "Unknown", "unknown", "UNKNOWN", "0000-00-00", "0000", "0", "0.0",
}

# Listing attributes from schema
PERSON_COLUMNS = ["Person_ID", "Wikitree_ID", "Gender", "Profile_URL", "Has_Children"]

NAME_COLUMNS = [
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
]

EVENT_COLUMNS = [
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
]

PEOPLE_INPUT_COLUMNS = [
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

RELATIONSHIP_INPUT_COLUMNS = ["parent_id", "child_id", "child_wikitree_id", "relationship_type"]


def clean_text(value: Any) -> str | None:
    """Normalise a text value and convert unknown values to None.

    Removes whitespace.

    Args:
        value: Raw value to clean.

    Returns:
        A cleaned string, or None when the value is missing or unknown
    """
     
    if value is None:
        return None


    if pd.isna(value):
        return None

    # stripping irrelevant value from text such as whitespace or ellipsis
    text = str(value).strip()
    if text in UNKNOWN_VALUES:
        return None

    text = re.sub(r"\s+", " ", text)
    return text or None

# cleaning ID field
def clean_id(value: Any) -> str | None:
    """
    Turns source identifier into string and removes zeros
    """
    text = clean_text(value)
    if text is None:
        return None

    if text.endswith(".0"):
        text = text[:-2]

    return text


def normalise_gender(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None

    # Normalizing gender fields to either Male or Female values
    text = text.lower()
    if text in ["male", "m"]:
        return "Male"
    if text in ["female", "f"]:
        return "Female"

    return "Unknown"


def clean_location(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None

    # normalising location values
    text = text.replace(" ,", ",")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text or None


def parse_date_parts(
    value: Any,
) -> tuple[str | None, int | None, int | None, int | None]:
    """Extracts and parses available day, month and year from dates

    Can parse either complete or  incomplete ISO date formats using regex
    
    Args - 
        value: date value
    """
    raw = clean_text(value)
    if raw is None:
        return None, None, None, None

    year = None
    month = None
    day = None

    # Matching on either year or ISO formatted date
    year_match = re.search(r"(\d{3,4})", raw)
    if year_match:
        year = int(year_match.group(1))

    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if iso_match:
        year = int(iso_match.group(1))
        raw_month = int(iso_match.group(2))
        raw_day = int(iso_match.group(3))

        if 1 <= raw_month <= 12:
            month = raw_month
        if 1 <= raw_day <= 31:
            day = raw_day

    if year is not None and year <= 0:
        year = None

    return raw, year, month, day


def parse_data_status(value: Any) -> dict[str, Any]:
    """ Parses and records data-status metadata 
    """
    text = clean_text(value)
    if text is None:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if isinstance(data, dict):
        return data

    return {}


def status_for_field(
    row: pd.Series,
    field_name: str,
) -> str | None:
    """Parses metadata for fields  
    """
    # Cleaning data status field if exists
    data_status = row.get("_parsed_data_status")
    if not isinstance(data_status, dict):
        return None

    return clean_text(data_status.get(field_name))


def stable_uuid(entity_type: str, natural_key: str) -> str:
    """Generate UUID for person, relationships etc

    Uses entity type and natural key to create a stable and deterministic identifier
    survives transformation 

    Args -
        entity_type(str): Type of entity, such as person or event.
        natural_key(str): Stable value used to identify the entity e.g. WikiTreeID.
    Returns:
        UUID string.
    """
    return str(uuid.uuid5(PROJECT_NAMESPACE, f"{entity_type}:{natural_key}"))


def profile_url(wikitree_id: str | None) -> str | None:
    if not wikitree_id:
        return None
    return f"{WIKITREE_PROFILE_BASE_URL}{wikitree_id}"


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def add_missing_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:    
    """Ensure that all expected columns exist in DataFrame.
        
        Missing columns are added with None values
    """
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df


def prepare_people(raw_people: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare WikiTree people

    Ensures expected columns exist, cleans ID, normalise gender, 
    location, removes unusable records and duplicates and provides UUID 
    Args -  
        raw_people (list[str]): csv file containing data of people 
    Returns -
        pandas dataframe: dataframe of cleaned person 
    """
    # Ensuring expected columns exist and are fileld 
    df = raw_people.copy()
    df = add_missing_columns(df, PEOPLE_INPUT_COLUMNS)

    # Cleaning id's of person, mother and father
    for col in ["person_id", "father_id", "mother_id"]:
        df[col] = df[col].apply(clean_id)

    
    text_cols = [
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
    ]

    # Cleaning text
    for col in text_cols:
        df[col] = df[col].apply(clean_text)

    # Cleaning, normalizing and parsing gender, location and data status
    df["gender"] = df["gender"].apply(normalise_gender)
    df["birth_location"] = df["birth_location"].apply(clean_location)
    df["death_location"] = df["death_location"].apply(clean_location)
    df["_parsed_data_status"] = df["data_status"].apply(parse_data_status)

    # Only copying people wth both WikiTreeID and PersonID
    df = df[df["person_id"].notna() & df["wikitree_id"].notna()].copy()
    # Dropping rows with duplicate WikiTreeID or PersonID
    df = df.drop_duplicates(subset=["person_id"], keep="first")
    df = df.drop_duplicates(subset=["wikitree_id"], keep="first")

    # Generates ID for name and person
    df["schema_person_id"] = df["wikitree_id"].apply(lambda value: stable_uuid("person", value))
    df["schema_name_id"] = df["wikitree_id"].apply(lambda value: stable_uuid("name", value))

    return df.reset_index(drop=True)


def prepare_relationships(
    raw_relationships: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and prepare parent-child relationship 
    
    cleans ParentIDs, ChildIDs, relationship types and removing duplicate or incomplete relationship rows
    Args -
        raw_relationships(list[str]): csv file listing relationships between people
    Return - 
        pandas dataframe: cleaned dataframe 
    """
    df = raw_relationships.copy()
    # NaNing null values
    df = add_missing_columns(df, RELATIONSHIP_INPUT_COLUMNS)

    # Cleaning various columns
    df["parent_id"] = df["parent_id"].apply(clean_id)
    df["child_id"] = df["child_id"].apply(clean_id)
    df["child_wikitree_id"] = df["child_wikitree_id"].apply(clean_text)
    df["relationship_type"] = df["relationship_type"].apply(clean_text)

    # Only copying if rows has valid ParentID and ChildID
    df = df[df["parent_id"].notna() & df["child_id"].notna()].copy()
    # Dropping duplicates
    df = df.drop_duplicates(subset=["parent_id", "child_id", "relationship_type"], keep="first")

    return df.reset_index(drop=True)


def build_person_table(
    people: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    """Builds relational people table
    
    Each  WikiTree profile turned into  Person row.
    `Has_Children` comes from if profile is a
    parent in the cleaned relationship data.


    Args -
        people (pandas dataframe): cleaned dataframe of people
        relationships (pandas dataframe): cleaned dataframe of relationships
    Returns -
        pandas dataframe: people dataframe with gender and has_children appended
    """
    # Collecting unique parent ids
    parent_ids = set(relationships["parent_id"].dropna())
    rows = []

    # Building rows for persons table
    for _, row in people.iterrows():
        source_person_id = row["person_id"]
        wikitree_id = row["wikitree_id"]

        rows.append({
            "Person_ID": row["schema_person_id"],
            "Wikitree_ID": wikitree_id,
            "Gender": row["gender"],
            "Profile_URL": profile_url(wikitree_id),
            "Has_Children": source_person_id in parent_ids,
        })

    return pd.DataFrame(rows, columns=PERSON_COLUMNS)


def build_names_table(
    people: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds rows to `name` table
    Args -
        people (pd): pandas dataframe of people
    Returns -
        Pandas Dataframe: table with name data for each person in persons table
    """
    rows = []

    for _, row in people.iterrows():
        # Cleaning middle name
        middle_name = clean_text(row.get("middle_name"))
        middle_initials = None

        if middle_name:
            initials = []
            for part in middle_name.split():
                if part:
                    initials.append(part[0].upper())
            middle_initials = "".join(initials) or None

        # Appending rows related to name to table
        rows.append({
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
        })

    return pd.DataFrame(rows, columns=NAME_COLUMNS)


def make_event_row(event_key, person_id_1, person_id_2, event_type, raw_date=None, month=None, day=None, year=None, location=None, data_status=None):
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


def add_life_events(events: list[dict[str, Any]], people: pd.DataFrame) -> None:
    """
    Adding rows relating to life events of people
    Args - 
        events (list): list of events experience by a person(s)
        people (list): list of people experiencing event
    returns - 
        None 
    """
    for _, row in people.iterrows():
        schema_person_id = row["schema_person_id"]
        wikitree_id = row["wikitree_id"]

        # parsing and appending date of birth data
        birth_raw, birth_year, birth_month, birth_day = parse_date_parts(row.get("birth_date"))
        # Appending birth data
        if birth_raw or row.get("birth_location"):
            events.append(make_event_row(
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
            ))

        # Appending death data
        death_raw, death_year, death_month, death_day = parse_date_parts(row.get("death_date"))
        if death_raw or row.get("death_location"):
            events.append(make_event_row(
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
            ))

def add_relationship_events(events: list, rejections:list, people:list[str],
                             relationships:list[str]) -> None:
    """Convert relationship into event records

    WikiTree IDs  mapped to Person UUIDs. Relationship 
    only converted into an event only when both the parent and child exist in the
    transformed people dataset. Relationships with missing nodes are added
    to the rejection log 

    Args -
        events(list): Event records.
        rejections(list): Rejected relationship records.
        people(list): Cleaned people records.
        relationships(list): Cleaned parent-child relationships.
    """

    id_to_schema_id = dict(zip(people["person_id"], people["schema_person_id"]))
    id_to_wikitree_id = dict(zip(people["person_id"], people["wikitree_id"]))
    
    # Building parent-child relationship 
    for _, rel in relationships.iterrows():
        parent_source_id = rel.get("parent_id")
        child_source_id = rel.get("child_id")
        relationship_type = clean_text(rel.get("relationship_type")) or "parent_of"

        parent_schema_id = id_to_schema_id.get(parent_source_id)
        child_schema_id = id_to_schema_id.get(child_source_id)

        # Rejecting relationship if row cotains no child or parent ID
        if not parent_schema_id or not child_schema_id:
            rejections.append({
                "parent_id": parent_source_id,
                "child_id": child_source_id,
                "child_wikitree_id": rel.get("child_wikitree_id"),
                "relationship_type": relationship_type,
                "reason": "parent_or_child_not_present_in_people_csv",
            })
            continue

        parent_wikitree_id = id_to_wikitree_id.get(parent_source_id)
        child_wikitree_id = id_to_wikitree_id.get(child_source_id)
        event_key = f"{parent_wikitree_id}:{relationship_type}:{child_wikitree_id}"

        events.append(make_event_row(
            event_key=event_key,
            person_id_1=parent_schema_id,
            person_id_2=child_schema_id,
            event_type=relationship_type,
        ))


def build_event_table(people: list[str], 
                      relationships: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds relationship and life event records for event table

    Birth and death events generated from each person's available
    data. Parent child relationships converted into events
    if both exist in the transformed people dataset.

    Args - 
        people (list[str]): Data related to people
        relationships (list[str]): Data related to relationships
    Return -
        pandas dataframe: dataframe of events and rejected events
    """
    events = []
    rejections = []


    # Adding relationships and life events to events list 
    add_life_events(events, people)
    add_relationship_events(events, rejections, people, relationships)

    # Creating event pandas dataframe
    event_df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    if not event_df.empty:
        event_df = event_df.drop_duplicates(subset=["Marriage_ID"], keep="first")

    # Creating rejected dataframe
    rejection_cols = ["parent_id", "child_id", "child_wikitree_id", "relationship_type", "reason"]
    rejections_df = pd.DataFrame(rejections, columns=rejection_cols)

    return event_df.reset_index(drop=True), rejections_df.reset_index(drop=True)


def build_id_crosswalk(people):
    cols = [
        "person_id",
        "wikitree_id",
        "schema_person_id",
        "schema_name_id",
        "first_name",
        "middle_name",
        "last_name_at_birth",
        "last_name_current",
    ]

    df = people[cols].copy()
    df = df.rename(columns={
        "person_id": "source_wikitree_numeric_id",
        "wikitree_id": "Wikitree_ID",
        "schema_person_id": "Person_ID",
        "schema_name_id": "Name_ID",
    })

    return df


def build_quality_report(raw_people, people, raw_relationships, relationships, person_table, names_table, event_table, rejections):
    event_counts = {}
    if not event_table.empty and "Event_Type" in event_table.columns:
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


def save_outputs(person_table, names_table, event_table, id_crosswalk, rejections, quality_report):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    person_table.to_csv(PERSON_OUTPUT, index=False)
    names_table.to_csv(NAMES_OUTPUT, index=False)
    event_table.to_csv(EVENT_OUTPUT, index=False)
    id_crosswalk.to_csv(ID_CROSSWALK_OUTPUT, index=False)
    rejections.to_csv(RELATIONSHIP_REJECTIONS_OUTPUT, index=False)
    quality_report.to_csv(QUALITY_REPORT_OUTPUT, index=False)


def print_summary(person_table, names_table, event_table, rejections, quality_report):
    print("Transformation complete.")
    print(f"Person rows: {len(person_table)} -> {PERSON_OUTPUT}")
    print(f"Names rows: {len(names_table)} -> {NAMES_OUTPUT}")
    print(f"Event rows: {len(event_table)} -> {EVENT_OUTPUT}")
    print(f"Rejected relationship rows: {len(rejections)} -> {RELATIONSHIP_REJECTIONS_OUTPUT}")
    print(f"ID crosswalk: {ID_CROSSWALK_OUTPUT}")
    print(f"Quality report: {QUALITY_REPORT_OUTPUT}")
    print("\nQuality report:")
    print(quality_report.to_string(index=False))


def main():
    raw_people = read_csv_required(PEOPLE_INPUT)
    raw_relationships = read_csv_required(RELATIONSHIPS_INPUT)

    people = prepare_people(raw_people)
    relationships = prepare_relationships(raw_relationships)

    person_table = build_person_table(people, relationships)
    names_table = build_names_table(people)
    event_table, rejections = build_event_table(people, relationships)
    id_crosswalk = build_id_crosswalk(people)

    quality_report = build_quality_report(
        raw_people,
        people,
        raw_relationships,
        relationships,
        person_table,
        names_table,
        event_table,
        rejections,
    )

    save_outputs(person_table, names_table, event_table, id_crosswalk, rejections, quality_report)
    print_summary(person_table, names_table, event_table, rejections, quality_report)
