import json
import time
from pathlib import Path
import pandas as pd
import requests
from typing import Any 

# URL from which we will make our API requests to
BASE_URL = "https://api.wikitree.com/api.php"
# Making requests to WikiTree API requires an APPID
APP_ID = "AncestryChatbotMSc"
# We are going to upload key metrics from our call to this file 
OUTPUT_DIR = Path("data/wikitree_test")

# Specifies the amount of generations back from a particular figure we want to search for
# For now it will be three but it can be adjusted in the CLI
ANCESTOR_DEPTH = 3
# Delaying requests in order to limit rate limiting or throttling 
REQUEST_DELAY_SECONDS = 1.0

# Preliminary seed figures - these are some of the most famous and most well documented figures on WikiTree
# Will be used to validate our system
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
        "birth_date": "1642-12-25",
        "known_wikitree_id": "Newton-17",
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

# These are the fields we are going to request data form  
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


def call_wikitree(params: dict[str]) -> dict[str | Any]:
    """
    Retrieves data from WikiTree for a specific set of paramters 
    Designed to be passed in other functions
    Args -
        Params(dict): Fields that we want data from
    Returns - 
        response.json: JSON object with fields and values 
    """
    # Turning params into a dict just in case
    params = dict(params)
    # Adding APP_ID to our parameters to reduce rate limiting and gain other benefits
    params["appId"] = APP_ID

    # Making request using URL and params
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    # Returning JSON object with our fields and values
    return response.json()


def search_person(seed: dict[str | Any]) -> dict[str]:
    """
    Calls WikiAPI for one specific person using SearchPerson call
    Args -
        seed(dict): The person we want to search
    Returns -
        response.json: JSON fields and values
    """
    # Sets parameters to be used in call_wikitree function
    params = {
        "action": "searchPerson",
        "FirstName": seed["first_name"],
        "LastName": seed["last_name"],
        "fields": FIELDS,
        "limit": 10,
    }

    # If our seed figure already has a DOB then add it to our params 
    if seed.get("birth_date"):
        params["BirthDate"] = seed["birth_date"]

    # We call the call_wikitree function using the params we built
    return call_wikitree(params)


def get_ancestors(profile_key: str, depth: int=ANCESTOR_DEPTH):
    """
    Retrieves ancestors of a particular profile (person) using WikiTree getAncestors function

    Args -
        profile_key: name of the person you want ancestors for 
        depth: How far down the family tree you want to get ancestors for

    Returns -
        response.json: JSON object with fields and values of ancestor 
    """
    # Creating params for ancestor of profile
    params = {
        "action": "getPeople",
        "keys": profile_key,
        "ancestors": depth,
        "fields": FIELDS,
    }

    # Calling call_wikitree() function using params
    return call_wikitree(params)

def add_profile(profiles: list, value: dict) -> None:
    """
    Helper function that appends ID or name to profile

    Args -
        profiles (list): List of dicts - contians fields and values for specific person
        value (dict): value that will be appended to profiles if it contains identifier field
    
    Returns -
        None
    """
    if not isinstance(value, dict):
        return

    if "Id" in value or "Name" in value:
        profiles.append(value)

def flatten_api_profiles(api_response: dict[str | Any]) -> list:
    """
    Turns API response into flattened list
    
    Args - 
        api_reponse (json): json response object returned by call_wikitree()  
    Returns -
        list: flattened list of fields and values
    """
    # Instantiating list that will be returned
    profiles = []

    # Checking if the outer most nest is a list - if not we wrap api_response in a list data structure
    # We want profiles to be a list of dictionaries
    if isinstance(api_response, list):
        items = api_response
    elif isinstance(api_response, dict):
        items = [api_response]
    else:
        return profiles

    # Iterating over each item in our api reponse
    for item in items:
        # Skips any item that is not a dictionary type
        if not isinstance(item, dict):
            continue
        
        # Appending any outer nested profiles to profiles list
        if "profile" in item:
            add_profile(profiles, item.get("profile"))
        
        people = item.get("people")

        if isinstance(people, dict):
            for profile in people.values():
                add_profile(profiles, profile)

        # Appending to profile any inner nested profiles (ancestors) to profiles list
        for key in ["matches", "ancestors"]:
            values = item.get(key)
            if isinstance(values, list):
                for value in values:
                    add_profile(profiles, value)

        # calling add_profile() to the items 
        add_profile(profiles, item)

    return profiles

def choose_best_search_match(seed: dict, search_response: dict[str | Any]) -> None:
    """
    Selects the best complete WikiTree profile for a seed figure.
    Args - 
        seed(dict): seed profile chosen to be matched
        search_response(json): JSON respone of profile returned by call_wikitree()
    Returns - 
        List: Profile that is best match to seed profile
    """

    matches = flatten_api_profiles(search_response)
    known_wikitree_id = seed.get("known_wikitree_id")

    # Storing all profiles that match the WikiTreedID
    if known_wikitree_id:
        for match in matches:
            if match.get("Name") == known_wikitree_id:
                    selected = dict(match)
                    selected["SeedLabel"] = seed["label"]
                    selected["SelectionMethod"] = "known_wikitree_id"
                    return selected
        return {
            "Name": known_wikitree_id,
            "SeedLabel": seed["label"],
            "SelectionMethod": "known_wikitree_id",
        }

    if not matches: 
        return None
    
    birth_date = seed.get("birth_date")
    
    # if there is no match with names we can match based on date of birt
    if birth_date:
        for match in matches:
            if match.get("BirthDate") == birth_date:
                selected = dict(match)
                selected["SeedLabel"] = seed["label"]
                selected["SelectionMethod"] = "exact_birth_date_match"
                return selected
            
    selected = dict(matches[0])
    selected["SeedLabel"] = seed["label"]
    selected["SelectionMethod"] = "first_search_result"

    return selected

def enrich_selected_profile(
    selected: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Adds extra data to profiles that have missing data using ancestors 
    """
    profile_key = selected.get("Name")

    if not profile_key:
        return selected

    for profile in profiles:
        if profile.get("Name") == profile_key:
            enriched = dict(profile)
            enriched["SeedLabel"] = selected.get("SeedLabel")
            enriched["SelectionMethod"] = selected.get("SelectionMethod")
            return enriched

    return selected

def normalise_person(profile: list) -> dict:
    """
    Normalizing profile to fit schema

    Args:
        profile(list): Flattened profile of person
    Return:
        dict: Dictionary with fields and values of person
    """

    # Data status not alwas included in response so we set it to None by default
    # No way to impute values otherwise
    data_status = None
    # If there is a data status field in the response then we add it to the return dictionary
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


def add_search_candidates(
    search_response: dict[str | Any],
    people_by_wikitree_id: dict[str, dict],
    excluded_wikitree_ids: set[str] | None = None,
) -> list[str]:
    """Add non-root search matches as searchable hard-negative candidates.

    Existing people are never overwritten because ancestor responses usually
    contain more complete profiles than search responses.
    """
    excluded_wikitree_ids = excluded_wikitree_ids or set()
    added_wikitree_ids = []

    for profile in flatten_api_profiles(search_response):
        wikitree_id = profile.get("Name")

        if (
            not wikitree_id
            or wikitree_id in excluded_wikitree_ids
            or wikitree_id in people_by_wikitree_id
        ):
            continue

        people_by_wikitree_id[wikitree_id] = normalise_person(profile)
        added_wikitree_ids.append(wikitree_id)

    return added_wikitree_ids


def extract_relationships(profile: list) -> list:
    """
    Extracts relationships from profile

    Args -
        profile: Flattened list of historical profiles
    Returns -
        list: list of relationships
    """
    rows = []

    # Getting children of person profile
    child_id = profile.get("Id")
    child_wikitree_id = profile.get("Name")

    # Getting parents of profile
    father_id = profile.get("Father")
    mother_id = profile.get("Mother")

    # Appending mother, father and children id to rows 
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

def make_seed_row(label: str, selected: dict) -> dict:
    """
    Creates
    """
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



def process_seed(seed, raw_search_results, raw_ancestor_results, selected_seeds, people_by_wikitree_id, relationships):
    """
    Runs all extraction functions and saves output
    """
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

    time.sleep(REQUEST_DELAY_SECONDS)

    print(f"  Fetching ancestors depth={ANCESTOR_DEPTH}")
    ancestor_response = get_ancestors(
        profile_key,
        depth=ANCESTOR_DEPTH,
    )
    raw_ancestor_results[profile_key] = ancestor_response

    profiles = flatten_api_profiles(ancestor_response)


    selected = enrich_selected_profile(
        selected,
        profiles,
    )

    selected_seeds.append(
        make_seed_row(label, selected)
    )


    if not any(
        profile.get("Name") == profile_key
        for profile in profiles
    ):
        profiles.append(selected)

    for profile in profiles:
        wikitree_id = profile.get("Name")
        if not wikitree_id:
            continue

        people_by_wikitree_id[wikitree_id] = normalise_person(profile)
        relationships.extend(extract_relationships(profile))

    add_search_candidates(
        search_response,
        people_by_wikitree_id,
        excluded_wikitree_ids={profile_key},
    )

    time.sleep(REQUEST_DELAY_SECONDS)


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

