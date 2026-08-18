"""
Generate the retrieval cases used by evaluation.

Cases written to data/evaluation/evaluation_cases.csv and
are inputs to the retrieval evaluation script.
"""

from pathlib import Path

import pandas as pd
from typing import Any
from app.data_pipeline.transform import clean_text

INPUT_DIR = Path("data/wikitree_test")
OUTPUT_DIR = Path("data/evaluation")

SEED_PROFILES_FILE = INPUT_DIR / "seed_profiles.csv"
PEOPLE_FILE = INPUT_DIR / "people.csv"
OUTPUT_FILE = OUTPUT_DIR / "evaluation_cases.csv"

# Fields needed for evaluation
REQUIRED_ROOT_FIELDS = [
    "first_name",
    "last_name_at_birth",
    "birth_date",
    "birth_location",
    "gender",
]

def extract_birth_year(birth_date):
    birth_date = clean_text(birth_date)

    if birth_date is None:
        return None

    # Splitting DOB and extracting year by using "-" as a delimiter
    year_text = birth_date.split("-")[0]

    # turning year into Int
    try:
        year = int(year_text)
    except ValueError:
        return None

    if year <= 0:
        return None

    return year


def introduce_typo(value):

    value = clean_text(value)

    if value is None or len(value) < 3:
        return value

    characters = list(value)

    # finding index of transposed character
    index = max(1, (len(characters) // 2) - 1)

    if index >= len(characters) - 1:
        index = len(characters) - 2

    # tranposing character
    characters[index], characters[index + 1] = (
        characters[index + 1],
        characters[index],
    )

    return "".join(characters)


def shorten_location(value):

    value = clean_text(value)

    if value is None:
        return None

    # Splitting location into two components (e.g. city and county)
    components = [
        component.strip()
        for component in value.split(",")
        if component.strip()
    ]

    if len(components) <= 2:
        return value

    return ", ".join(components[-2:])


def opposite_gender(value):

    value = clean_text(value)

    if value == "Male":
        return "Female"

    if value == "Female":
        return "Male"

    return None


def make_case(seed_label, expected_wikitree_id, condition,first_name,
    last_name, birth_year, birth_location, gender, notes,
):

    return {
        "case_id": f"{expected_wikitree_id}:{condition}",
        "seed_label": seed_label,
        "condition": condition,
        "expected_wikitree_id": expected_wikitree_id,
        "first_name": first_name,
        "last_name": last_name,
        "birth_year": birth_year,
        "birth_location": birth_location,
        "gender": gender,
        "notes": notes,
    }


def build_cases_for_profile(row):

    # Extracting and cleaning gender, birth location, birth  year, 
    # first name, last name and WikiTreeID
    seed_label = clean_text(row["seed_label"])
    expected_wikitree_id = clean_text(row["wikitree_id"])

    first_name = clean_text(row["first_name"])
    # Birth surname preferred over current
    last_name = (
        clean_text(row["last_name_at_birth"])
        or clean_text(row["last_name_current"])
    )

    birth_year = extract_birth_year(row["birth_date"])
    birth_location = clean_text(row["birth_location"])
    gender = clean_text(row["gender"])

    # Creating cases
    cases = [
        make_case(
            seed_label=seed_label,
            expected_wikitree_id=expected_wikitree_id,
            condition="full_profile",
            first_name=first_name,
            last_name=last_name,
            birth_year=birth_year,
            birth_location=birth_location,
            gender=gender,
            notes="Full profile, all fields correct",
        ),
        make_case(
            seed_label=seed_label,
            expected_wikitree_id=expected_wikitree_id,
            condition="name_year",
            first_name=first_name,
            last_name=last_name,
            birth_year=birth_year,
            birth_location=None,
            gender=None,
            notes="Only first name, surname and birth year",
        ),
        make_case(
            seed_label=seed_label,
            expected_wikitree_id=expected_wikitree_id,
            condition="name_only",
            first_name=first_name,
            last_name=last_name,
            birth_year=None,
            birth_location=None,
            gender=None,
            notes="Only first name and surname",
        ),
        make_case(
            seed_label=seed_label,
            expected_wikitree_id=expected_wikitree_id,
            condition="noisy_input",
            first_name=first_name,
            last_name=introduce_typo(last_name),
            birth_year=birth_year,
            birth_location=shorten_location(birth_location),
            gender=gender,
            notes=(
                "One surname typo and a shortened "
                "birth location."
            ),
        ),
        make_case(
            seed_label=seed_label,
            expected_wikitree_id=expected_wikitree_id,
            condition="wrong_year",
            first_name=first_name,
            last_name=last_name,
            birth_year=birth_year + 20,
            birth_location=birth_location,
            gender=gender,
            notes="Birth year increased by 20 years",
        ),
        make_case(
            seed_label=seed_label,
            expected_wikitree_id=expected_wikitree_id,
            condition="wrong_location",
            first_name=first_name,
            last_name=last_name,
            birth_year=birth_year,
            birth_location="Tokyo, Japan",
            gender=gender,
            notes="Birth location replaced with Tokyo, Japan"

        ),
        make_case(
            seed_label=seed_label,
            expected_wikitree_id=expected_wikitree_id,
            condition="wrong_gender",
            first_name=first_name,
            last_name=last_name,
            birth_year=birth_year,
            birth_location=birth_location,
            gender=opposite_gender(gender),
            notes="Genders switched",
        ),
    ]

    return cases


def load_evaluation_roots():

    # Raising error if seed or people files do not exist
    if not SEED_PROFILES_FILE.exists():
        raise FileNotFoundError(
            f"Missing seed profiles: {SEED_PROFILES_FILE}"
        )

    if not PEOPLE_FILE.exists():
        raise FileNotFoundError(
            f"Missing people data: {PEOPLE_FILE}"
        )

    # loading seeds and people files
    seeds = pd.read_csv(
        SEED_PROFILES_FILE,
        dtype=str,
        keep_default_na=False,
    )

    people = pd.read_csv(
        PEOPLE_FILE,
        dtype=str,
        keep_default_na=False,
    )

    required_seed_columns = {
        "seed_label",
        "wikitree_id",
    }

    missing_seed_columns = required_seed_columns - set(seeds.columns)

    if missing_seed_columns:
        raise ValueError(
            "seed_profiles.csv is missing columns: "
            f"{sorted(missing_seed_columns)}"
        )

    # Merging seed leable and expected WikiTreeID with people table
    # Validating that merge is 1-1 cardinality 
    roots = seeds[
        ["seed_label", "wikitree_id"]
    ].merge(
        people,
        on="wikitree_id",
        how="left",
        validate="one_to_one",
    )

    # Checking if there are seeds with missing person
    missing_profiles = roots[
        roots["person_id"].map(clean_text).isna()
    ]

    if not missing_profiles.empty:
        missing_ids = missing_profiles["wikitree_id"].tolist()

        raise ValueError(
            "The following seed profiles are missing or incomplete "
            f"in people.csv: {missing_ids}"
        )

    return roots


def validate_roots(roots):

    errors = []

    for _, row in roots.iterrows():
        missing_fields = [
            field
            for field in REQUIRED_ROOT_FIELDS
            if clean_text(row.get(field)) is None
        ]

        if extract_birth_year(row.get("birth_date")) is None:
            missing_fields.append("valid_birth_year")

        if missing_fields:
            errors.append(
                {
                    "wikitree_id": row["wikitree_id"],
                    "missing_fields": missing_fields,
                }
            )

    if errors:
        raise ValueError(
            "Evaluation roots contain missing data:\n"
            + "\n".join(str(error) for error in errors)
        )


def main() -> None:
    roots = load_evaluation_roots()
    validate_roots(roots)

    evaluation_cases = []

    # Building evaluation cases from root profiles
    for _, row in roots.iterrows():
        evaluation_cases.extend(
            build_cases_for_profile(row)
        )

    # building cases DatFrame
    cases_df = pd.DataFrame(evaluation_cases)

    # Listing expected evaluation conditions
    expected_conditions = {
        "full_profile",
        "name_year",
        "name_only",
        "noisy_input",
        "wrong_year",
        "wrong_location",
        "wrong_gender",
    }

    # Seeing how  many conditions are possible with cases DataFrame
    actual_conditions = set(cases_df["condition"])

    # If they don't match then we raise an error
    if actual_conditions != expected_conditions:
        raise ValueError(
            "Evaluation conditions do not match the required set."
        )

    # Validating that case counts are expected
    expected_case_count = len(roots) * len(expected_conditions)

    if len(cases_df) != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} cases, "
            f"but generated {len(cases_df)}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Eligible profiles: {len(roots)}")
    print(f"Conditions per profile: {len(expected_conditions)}")
    print(f"Evaluation cases created: {len(cases_df)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nCases by condition:")
    print(
        cases_df["condition"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()