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

SEED_FIGURES = [
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

FIELDS = ",".join([
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
])


def call_wikitree(params):
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

    return call_wikitree(params)


def get_ancestors(profile_key, depth=ANCESTOR_DEPTH):
    params = {
        "action": "getAncestors",
        "key": profile_key,
        "depth": depth,
        "fields": FIELDS,
        "resolveRedirect": "1",
    }
    return call_wikitree(params)


def flatten_api_profiles(api_response):
    profiles = []

    if isinstance(api_response, list):
        items = api_response
    elif isinstance(api_response, dict):
        items = [api_response]
    else:
        return profiles

    for item in items:
        if not isinstance(item, dict):
            continue

        if "profile" in item:
            add_profile(profiles, item.get("profile"))

        for key in ["matches", "ancestors"]:
            values = item.get(key)
            if isinstance(values, list):
                for value in values:
                    add_profile(profiles, value)

        add_profile(profiles, item)

    return profiles


def add_profile(profiles, value):
    if not isinstance(value, dict):
        return

    if "Id" in value or "Name" in value:
        profiles.append(value)


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


def normalise_person(profile):
    data_status = None
    if profile.get("DataStatus") is not None:
        data_status = json.dumps(profile.get("DataStatus"), ensure_ascii=False)

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
        "data_status": data_status,
    }


def extract_relationships(profile):
    rows = []

    child_id = profile.get("Id")
    child_wikitree_id = profile.get("Name")

    father_id = profile.get("Father")
    mother_id = profile.get("Mother")

    if father_id and child_id:
        rows.append({
            "parent_id": father_id,
            "child_id": child_id,
            "child_wikitree_id": child_wikitree_id,
            "relationship_type": "father_of",
        })

    if mother_id and child_id:
        rows.append({
            "parent_id": mother_id,
            "child_id": child_id,
            "child_wikitree_id": child_wikitree_id,
            "relationship_type": "mother_of",
        })

    return rows


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_outputs(raw_search_results, raw_ancestor_results, selected_seeds, people_by_wikitree_id, relationships):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_df = pd.DataFrame(selected_seeds)
    people_df = pd.DataFrame(people_by_wikitree_id.values())
    relationship_df = pd.DataFrame(relationships)

    if not relationship_df.empty:
        relationship_df = relationship_df.drop_duplicates()

    save_json(OUTPUT_DIR / "raw_search_results.json", raw_search_results)
    save_json(OUTPUT_DIR / "raw_ancestor_results.json", raw_ancestor_results)

    seed_df.to_csv(OUTPUT_DIR / "seed_profiles.csv", index=False)
    people_df.to_csv(OUTPUT_DIR / "people.csv", index=False)
    relationship_df.to_csv(OUTPUT_DIR / "relationships.csv", index=False)

    return seed_df, people_df, relationship_df


def process_seed(seed, raw_search_results, raw_ancestor_results, selected_seeds, people_by_wikitree_id, relationships):
    label = seed["label"]
    print(f"\nSearching seed: {label}")

    search_response = search_person(seed)
    raw_search_results[label] = search_response

    selected = choose_best_search_match(seed, search_response)
    if not selected:
        print(f"  No match found for {label}")
        return

    profile_key = selected.get("Name")
    if not profile_key:
        print(f"  Match found but no WikiTree Name/key for {label}")
        return

    print(f"  Selected WikiTree profile: {profile_key}")
    selected_seeds.append(make_seed_row(label, selected))

    time.sleep(REQUEST_DELAY_SECONDS)

    print(f"  Fetching ancestors depth={ANCESTOR_DEPTH}")
    ancestor_response = get_ancestors(profile_key, depth=ANCESTOR_DEPTH)
    raw_ancestor_results[profile_key] = ancestor_response

    profiles = flatten_api_profiles(ancestor_response)
    profiles.append(selected)

    for profile in profiles:
        wikitree_id = profile.get("Name")
        if not wikitree_id:
            continue

        people_by_wikitree_id[wikitree_id] = normalise_person(profile)
        relationships.extend(extract_relationships(profile))

    time.sleep(REQUEST_DELAY_SECONDS)


def make_seed_row(label, selected):
    return {
        "seed_label": label,
        "wikitree_id": selected.get("Name"),
        "person_id": selected.get("Id"),
        "first_name": selected.get("FirstName"),
        "last_name_at_birth": selected.get("LastNameAtBirth"),
        "birth_date": selected.get("BirthDate"),
        "birth_location": selected.get("BirthLocation"),
        "selection_method": selected.get("SelectionMethod"),
    }


def main():
    raw_search_results = {}
    raw_ancestor_results = {}
    selected_seeds = []
    people_by_wikitree_id = {}
    relationships = []

    for seed in SEED_FIGURES:
        process_seed(
            seed,
            raw_search_results,
            raw_ancestor_results,
            selected_seeds,
            people_by_wikitree_id,
            relationships,
        )

    seed_df, people_df, relationship_df = save_outputs(
        raw_search_results,
        raw_ancestor_results,
        selected_seeds,
        people_by_wikitree_id,
        relationships,
    )

    print("\nDone.")
    print(f"Seed profiles saved: {len(seed_df)}")
    print(f"People saved: {len(people_df)}")
    print(f"Relationships saved: {len(relationship_df)}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
