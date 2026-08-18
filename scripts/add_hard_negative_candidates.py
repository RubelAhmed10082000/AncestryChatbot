from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_pipeline.extract import add_search_candidates, flatten_api_profiles
from app.data_pipeline.transform import read_csv_required


INPUT_DIR = Path("data/wikitree_test")
RAW_SEARCH_RESULTS_FILE = INPUT_DIR / "raw_search_results.json"
PEOPLE_FILE = INPUT_DIR / "people.csv"
SEED_PROFILES_FILE = INPUT_DIR / "seed_profiles.csv"
MANIFEST_FILE = INPUT_DIR / "hard_negative_candidates.csv"

def load_raw_search_results(path) :
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

    with path.open(encoding="utf-8") as file:
        raw_search_results = json.load(file)

    if not isinstance(raw_search_results, dict):
        raise ValueError("raw_search_results.json needs to contain an object by seed label.")

    return raw_search_results


def merge_hard_negative_candidates(people, seeds, raw_search_results):

    required_people_columns = {"wikitree_id", "person_id"}
    required_seed_columns = {"seed_label", "wikitree_id"}

    # Raising error if people.csv or seed.csv missing required columns
    if missing := required_people_columns - set(people.columns):
        raise ValueError(f"people.csv is missing columns: {sorted(missing)}")

    if missing := required_seed_columns - set(seeds.columns):
        raise ValueError(f"seed_profiles.csv is missing columns: {sorted(missing)}")

    # Collects known seed_ids and WikiTreeID
    seed_ids = {
        str(value).strip()
        for value in seeds["wikitree_id"]
        if str(value).strip()
    }

    people_by_wikitree_id = {
        row["wikitree_id"]: row
        for row in people.to_dict(orient="records")
        if row.get("wikitree_id")
    }

    manifest_rows = []
    manifested_ids = set()
    initially_searchable_ids = set(people_by_wikitree_id)

    for source_seed_search, search_response in raw_search_results.items():
        add_search_candidates(
            search_response,
            people_by_wikitree_id,
            excluded_wikitree_ids=seed_ids,
        )

        for search_profile in flatten_api_profiles(search_response):
            wikitree_id = search_profile.get("Name")

            if (
                not wikitree_id
                or wikitree_id in seed_ids
                or wikitree_id in manifested_ids
            ):
                continue

            manifested_ids.add(wikitree_id)
            profile = people_by_wikitree_id[wikitree_id]
            manifest_rows.append(
                {
                    "source_seed_search": source_seed_search,
                    "wikitree_id": wikitree_id,
                    "person_id": profile.get("person_id"),
                    "first_name": profile.get("first_name"),
                    "last_name_at_birth": profile.get("last_name_at_birth"),
                    "last_name_current": profile.get("last_name_current"),
                    "birth_date": profile.get("birth_date"),
                    "birth_location": profile.get("birth_location"),
                    "gender": profile.get("gender"),
                    "was_already_searchable": (
                        wikitree_id in initially_searchable_ids
                    ),
                }
            )

    merged_people = pd.DataFrame(
        people_by_wikitree_id.values(),
        columns=people.columns,
    )
    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "source_seed_search",
            "wikitree_id",
            "person_id",
            "first_name",
            "last_name_at_birth",
            "last_name_current",
            "birth_date",
            "birth_location",
            "gender",
            "was_already_searchable",
        ],
    )

    if set(manifest.get("wikitree_id", [])) & seed_ids:
        raise ValueError("A seed root was incorrectly included as a hard negative.")

    return merged_people, manifest


def main() -> None:
    people = read_csv_required(PEOPLE_FILE)
    seeds = read_csv_required(SEED_PROFILES_FILE)
    raw_search_results = load_raw_search_results(RAW_SEARCH_RESULTS_FILE)
    merged_people, manifest = merge_hard_negative_candidates(
        people,
        seeds,
        raw_search_results,
    )

    merged_people.to_csv(PEOPLE_FILE, index=False)
    manifest.to_csv(MANIFEST_FILE, index=False)

    added_count = int(
        (~manifest["was_already_searchable"].astype(bool)).sum()
    )

    print(f"People before: {len(people)}")
    print(f"Hard negatives catalogued: {len(manifest)}")
    print(f"New hard negatives added: {added_count}")
    print(f"People after: {len(merged_people)}")
    print(f"Updated people: {PEOPLE_FILE}")
    print(f"Audit manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
