"""

import json
import time
from pathlib import Path
import pandas as pd
import requests


BASE_URL = "https://api.wikitree.com/api.php"
APP_ID = "AncestryChatbotMSc"

OUTPUT_DIR = Path("data/wikitree_test")

ANCESTOR_DEPTH = 3
REQUEST_DELAY_SECONDS = 1.0


TEST_FIGURES = [
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
    },
    {
        "label": "Jane Austen",
        "first_name": "Jane",
        "last_name": "Austen",
        "birth_date": "1775-12-16",
    },
    {
        "label": "Isaac Newton",
        "first_name": "Isaac",
        "last_name": "Newton",
        "birth_date": "1643-01-04",
    },
    {
        "label": "William Shakespeare",
        "first_name": "William",
        "last_name": "Shakespeare",
        "birth_date": "1564-04-26",
    },
    {
        "label": "Florence Nightingale",
        "first_name": "Florence",
        "last_name": "Nightingale",
        "birth_date": "1820-05-12",
    },
    {
        "label": "Winston Churchill",
        "first_name": "Winston",
        "last_name": "Churchill",
        "birth_date": "1874-11-30",
    },
    {
        "label": "Ada Lovelace",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "birth_date": "1815-12-10",
    },
    {
        "label": "Isambard Kingdom Brunel",
        "first_name": "Isambard",
        "last_name": "Brunel",
        "birth_date": "1806-04-09",
    },
]


FIELDS = ",".join(
    [
        "Id",
        "Name",
        "FirstName",
        "MiddleName",
        "LastNameAtBirth",
        "LastNameCurrent",
        "BirthDate",
        "BirthLocation",
        "DeathDate",
        "DeathLocation",
        "Father",
        "Mother",
        "Gender",
        "Privacy",
        "DataStatus",
    ]
)


def call_wikitree_api(params):

    params = dict(params)
    params["appId"] = APP_ID

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def search_person(seed):
    params = {
        "action": "searchPerson",
        "FirstName": seed["first_name"],
        "LastName": seed["last_name"],
        "fields": FIELDS,
        "limit": 10,
    }

    if seed.get("birth_date"):
        params["BirthDate"] = seed["birth_date"]

    return call_wikitree_api(params)


def get_ancestors(profile_key, depth: int = ANCESTOR_DEPTH):

    return call_wikitree_api(
        {
            "action": "getAncestors",
            "key": profile_key,
            "depth": depth,
            "fields": FIELDS,
            "resolveRedirect": "1",
        }
    )


def flatten_api_profiles(api_response):

    profiles = []

    def add_profile(obj):
        if isinstance(obj, dict):
            if "Id" in obj or "Name" in obj:
                profiles.append(obj)

    if isinstance(api_response, list):
        for item in api_response:
            if isinstance(item, dict):
                if "profile" in item:
                    add_profile(item["profile"])

                if "matches" in item and isinstance(item["matches"], list):
                    for match in item["matches"]:
                        add_profile(match)

                if "ancestors" in item and isinstance(item["ancestors"], list):
                    for ancestor in item["ancestors"]:
                        add_profile(ancestor)

                add_profile(item)

    elif isinstance(api_response, dict):
        if "profile" in api_response:
            add_profile(api_response["profile"])

        if "matches" in api_response and isinstance(api_response["matches"], list):
            for match in api_response["matches"]:
                add_profile(match)

        if "ancestors" in api_response and isinstance(api_response["ancestors"], list):
            for ancestor in api_response["ancestors"]:
                add_profile(ancestor)

        add_profile(api_response)

    return profiles


def choose_best_search_match(seed, search_response):
    if seed.get("known_wikitree_id"):
        return {
            "Name": seed["known_wikitree_id"],
            "SeedLabel": seed["label"],
            "SelectionMethod": "known_wikitree_id",
        }

    matches = flatten_api_profiles(search_response)

    if not matches:
        return None

    birth_date = seed.get("birth_date")

    if birth_date:
        for match in matches:
            if match.get("BirthDate") == birth_date:
                match["SeedLabel"] = seed["label"]
                match["SelectionMethod"] = "exact_birth_date_match"
                return match

    best = matches[0]
    best["SeedLabel"] = seed["label"]
    best["SelectionMethod"] = "first_search_result"
    return best


def normalise_person(profile) :

    return {
        "person_id": profile.get("Id"),
        "wikitree_id": profile.get("Name"),
        "first_name": profile.get("FirstName"),
        "middle_name": profile.get("MiddleName"),
        "last_name_at_birth": profile.get("LastNameAtBirth"),
        "last_name_current": profile.get("LastNameCurrent"),
        "birth_date": profile.get("BirthDate"),
        "birth_location": profile.get("BirthLocation"),
        "death_date": profile.get("DeathDate"),
        "death_location": profile.get("DeathLocation"),
        "gender": profile.get("Gender"),
        "father_id": profile.get("Father"),
        "mother_id": profile.get("Mother"),
        "privacy": profile.get("Privacy"),
        "data_status": json.dumps(profile.get("DataStatus"), ensure_ascii=False)
        if profile.get("DataStatus") is not None
        else None,
    }


def extract_relationships(profile):
    relationships = []

    child_id = profile.get("Id")
    child_wikitree_id = profile.get("Name")

    father_id = profile.get("Father")
    mother_id = profile.get("Mother")

    if father_id and child_id:
        relationships.append(
            {
                "parent_id": father_id,
                "child_id": child_id,
                "child_wikitree_id": child_wikitree_id,
                "relationship_type": "father_of",
            }
        )

    if mother_id and child_id:
        relationships.append(
            {
                "parent_id": mother_id,
                "child_id": child_id,
                "child_wikitree_id": child_wikitree_id,
                "relationship_type": "mother_of",
            }
        )

    return relationships


def main():
    raw_search_results = {}
    raw_ancestor_results = {}

    selected_seeds = []
    all_people_by_wikitree_id = {}
    all_relationships = []

    for figures in TEST_FIGURES:
        label = figures["label"]
        print(f"\nSearching seed: {label}")

        search_response = search_person(figures)
        raw_search_results[label] = search_response

        selected = choose_best_search_match(figures, search_response)

        if not selected:
            print(f"  No match found for {label}")
            continue

        profile_key = selected.get("Name")

        if not profile_key:
            print(f"Match found but no WikiTree Name/key for {label}")
            continue

        print(f" Selected WikiTree profile: {profile_key}")

        selected_seeds.append(
            {
                "seed_label": label,
                "wikitree_id": profile_key,
                "person_id": selected.get("Id"),
                "first_name": selected.get("FirstName"),
                "last_name_at_birth": selected.get("LastNameAtBirth"),
                "birth_date": selected.get("BirthDate"),
                "birth_location": selected.get("BirthLocation"),
                "selection_method": selected.get("SelectionMethod"),
            }
        )

        time.sleep(REQUEST_DELAY_SECONDS)

        print(f"  Fetching ancestors depth={ANCESTOR_DEPTH}")
        ancestor_response = get_ancestors(profile_key, depth=ANCESTOR_DEPTH)
        raw_ancestor_results[profile_key] = ancestor_response

        profiles = flatten_api_profiles(ancestor_response)

        if selected.get("Id") or selected.get("FirstName") or selected.get("BirthDate"):
            profiles.append(selected)

        for profile in profiles:
            wikitree_id = profile.get("Name")
            if not wikitree_id:
                continue

            new_person = normalise_person(profile)

            existing_person = all_people_by_wikitree_id.get(wikitree_id)

            if existing_person is None:
                all_people_by_wikitree_id[wikitree_id] = new_person
            else:
                existing_filled_fields = sum(value is not None and value != "" for value in existing_person.values())
                new_filled_fields = sum(value is not None and value != "" for value in new_person.values())

                if new_filled_fields > existing_filled_fields:
                    all_people_by_wikitree_id[wikitree_id] = new_person

            all_relationships.extend(extract_relationships(profile))

        time.sleep(REQUEST_DELAY_SECONDS)

    relationship_df = pd.DataFrame(all_relationships)
    if not relationship_df.empty:
        relationship_df = relationship_df.drop_duplicates()

    people_df = pd.DataFrame(all_people_by_wikitree_id.values())
    seed_df = pd.DataFrame(selected_seeds)

    with open(OUTPUT_DIR / "raw_search_results.json", "w", encoding="utf-8") as f:
        json.dump(raw_search_results, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "raw_ancestor_results.json", "w", encoding="utf-8") as f:
        json.dump(raw_ancestor_results, f, indent=2, ensure_ascii=False)

    seed_df.to_csv(OUTPUT_DIR / "seed_profiles.csv", index=False)
    people_df.to_csv(OUTPUT_DIR / "people.csv", index=False)
    relationship_df.to_csv(OUTPUT_DIR / "relationships.csv", index=False)

    print("\nDone.")
    print(f"Seed profiles saved: {len(seed_df)}")
    print(f"People saved: {len(people_df)}")
    print(f"Relationships saved: {len(relationship_df)}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
"""