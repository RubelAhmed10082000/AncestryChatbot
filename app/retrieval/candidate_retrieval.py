import argparse
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd


DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")
PERSON_FILE = "person.csv"
NAMES_FILE = "names.csv"
EVENT_FILE = "event.csv"

UNKNOWN_VALUES = {
    "", " ", "nan", "NaN", "None", "none", "NULL", "null",
    "Unknown", "unknown", "UNKNOWN",
}

# Weigths given to attibutes if they are a match
DEFAULT_WEIGHTS = {
    "first_name_score": 0.25,
    "last_name_score": 0.25,
    "birth_year_score": 0.35,
    "birth_location_score": 0.15,
    "gender_score": 0.05,
}

# Final columns to be displayed 
DISPLAY_COLUMNS = [
    "rank",
    "rank_score",
    "wikitree_id",
    "full_name",
    "birth_year",
    "birth_location",
    "death_year",
    "match_explanation",
]


class CandidateRetriever:
    """
    
    """
    DEFAULT_WEIGHTS = DEFAULT_WEIGHTS

    def __init__(self, schema_dir=DEFAULT_SCHEMA_DIR, weights=None):
        self.schema_dir = Path(schema_dir)
        self.weights = weights or self.DEFAULT_WEIGHTS

        self.person_df = self._read_required_csv(PERSON_FILE)
        self.names_df = self._read_required_csv(NAMES_FILE)
        self.event_df = self._read_required_csv(EVENT_FILE)
        self.index_df = self._build_search_index()

    def _read_required_csv(self, filename):
        path = self.schema_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}. Run transform.py first.")

        return pd.read_csv(path, dtype=str, keep_default_na=False)

    def _build_search_index(self):

        people = self.person_df.copy()
        names = self.names_df.copy()
        events = self.event_df.copy()

        for col in ["Person_ID", "Wikitree_ID", "Gender", "Profile_URL", "Has_Children"]:
            if col not in people.columns:
                people[col] = None

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
            "Person_ID_1",
            "Person_ID_2",
            "Event_Type",
            "Event_Raw_Date",
            "Event_Year",
            "Event_Location",
            "Data Status",
        ]:
            if col not in events.columns:
                events[col] = None

        people = people.merge(names, on="Person_ID", how="left", suffixes=("", "_name"))

        birth_events = events[events["Event_Type"] == "birth"].copy()
        birth_events = birth_events.sort_values(by=["Person_ID_1", "Event_Year"], na_position="last")
        birth_events = birth_events.drop_duplicates(subset=["Person_ID_1"], keep="first")
        birth_events = birth_events.rename(
            columns={
                "Person_ID_1": "Person_ID",
                "Event_Raw_Date": "Birth_Raw_Date",
                "Event_Year": "Birth_Year",
                "Event_Location": "Birth_Location",
                "Data Status": "Birth_Data_Status",
            }
        )

        death_events = events[events["Event_Type"] == "death"].copy()
        death_events = death_events.sort_values(by=["Person_ID_1", "Event_Year"], na_position="last")
        death_events = death_events.drop_duplicates(subset=["Person_ID_1"], keep="first")
        death_events = death_events.rename(
            columns={
                "Person_ID_1": "Person_ID",
                "Event_Raw_Date": "Death_Raw_Date",
                "Event_Year": "Death_Year",
                "Event_Location": "Death_Location",
                "Data Status": "Death_Data_Status",
            }
        )

        birth_cols = [
            "Person_ID",
            "Birth_Raw_Date",
            "Birth_Year",
            "Birth_Location",
            "Birth_Data_Status",
        ]
        death_cols = [
            "Person_ID",
            "Death_Raw_Date",
            "Death_Year",
            "Death_Location",
            "Death_Data_Status",
        ]

        people = people.merge(birth_events[birth_cols], on="Person_ID", how="left")
        people = people.merge(death_events[death_cols], on="Person_ID", how="left")
        people["Full_Name"] = people.apply(self._build_full_name, axis=1)

        return people

    def _build_full_name(self, row: list) -> list | None:
        """
        Combines first, middle and last names
        Args -
            row (list): list of names
        Returns -
            list: list of full names
        """
        # Cleaning each name 
        parts = [
            clean_text(row.get("First_Name")),
            clean_text(row.get("Middle_Name")),
            clean_text(row.get("Last_Name_At_Birth")) or clean_text(row.get("Last_Name_Current")),
        ]

        # Adding names to the list
        parts = [part for part in parts if part]

        if parts:
            return " ".join(parts)

        return None

    def find_candidates(
        self,
        first_name=None,
        last_name=None,
        birth_year=None,
        birth_location=None,
        gender=None,
        top_k=5,
        min_score=0.0,
    ):
        """
        Matches candidates based on attribute similarity

        """
        rows = []

        # matching candidates based on similarity of attributes
        for _, candidate in self.index_df.iterrows():
            scores = {
                "first_name_score": best_string_similarity(first_name, [candidate.get("First_Name")]),
                "last_name_score": max(
                    best_string_similarity(last_name, [candidate.get("Last_Name_At_Birth")]),
                    0.5 * best_string_similarity(last_name, [candidate.get("Last_Name_Current")]),
                ),
                "birth_year_score": year_similarity(birth_year, candidate.get("Birth_Year")),
                "birth_location_score": best_location_similarity(birth_location, candidate.get("Birth_Location")),
                "gender_score": gender_similarity(gender, candidate.get("Gender")),
            }

            # creating scoring scoring weights
            rank_score = weighted_score(scores, self.weights)
            rank_score = adjust_score(rank_score, scores, birth_year, candidate.get("Birth_Year"))

            if rank_score < min_score:
                continue
            # Appending scores along with information
            rows.append(
                {
                    "rank_score": rank_score,
                    "person_id": candidate.get("Person_ID"),
                    "wikitree_id": candidate.get("Wikitree_ID"),
                    "profile_url": candidate.get("Profile_URL"),
                    "full_name": candidate.get("Full_Name"),
                    "first_name": candidate.get("First_Name"),
                    "middle_name": candidate.get("Middle_Name"),
                    "last_name_at_birth": candidate.get("Last_Name_At_Birth"),
                    "last_name_current": candidate.get("Last_Name_Current"),
                    "gender": candidate.get("Gender"),
                    "birth_year": candidate.get("Birth_Year"),
                    "birth_date": candidate.get("Birth_Raw_Date"),
                    "birth_location": candidate.get("Birth_Location"),
                    "death_year": candidate.get("Death_Year"),
                    "death_date": candidate.get("Death_Raw_Date"),
                    "death_location": candidate.get("Death_Location"),
                    "first_name_score": round_or_none(scores["first_name_score"]),
                    "last_name_score": round_or_none(scores["last_name_score"]),
                    "birth_year_score": round_or_none(scores["birth_year_score"]),
                    "birth_location_score": round_or_none(scores["birth_location_score"]),
                    "gender_score": round_or_none(scores["gender_score"]),
                    "match_explanation": explain_matches(scores),
                }
            )

        results = pd.DataFrame(rows)

        if results.empty:
            return results

        sort_cols = [
            "rank_score",
            "last_name_score",
            "first_name_score",
            "birth_year_score",
            "birth_location_score",
        ]

        results = results.sort_values(by=sort_cols, ascending=[False, False, False, False, False])
        results = results.reset_index(drop=True)
        results.insert(0, "rank", range(1, len(results) + 1))

        return results.head(top_k)


def clean_text(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    text = str(value).strip()

    if text in UNKNOWN_VALUES:
        return None

    text = re.sub(r"\s+", " ", text)
    return text or None


def normalise_for_matching(value):
    text = clean_text(value)

    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_set(value):
    text = normalise_for_matching(value)

    if not text:
        return set()

    return set(text.split())


def sequence_similarity(left, right):
    left = normalise_for_matching(left)
    right = normalise_for_matching(right)

    if not left or not right:
        return 0.0

    return SequenceMatcher(None, left, right).ratio()


def token_overlap_similarity(left, right):
    left_tokens = token_set(left)
    right_tokens = token_set(right)

    if not left_tokens or not right_tokens:
        return 0.0

    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def best_string_similarity(query_value, candidate_values):
    if clean_text(query_value) is None:
        return math.nan

    scores = []
    for value in candidate_values:
        if clean_text(value):
            scores.append(sequence_similarity(query_value, value))

    if not scores:
        return 0.0

    return max(scores)


def best_location_similarity(query_location, candidate_location):
    if clean_text(query_location) is None:
        return math.nan

    if clean_text(candidate_location) is None:
        return 0.0

    char_score = sequence_similarity(query_location, candidate_location)
    token_score = token_overlap_similarity(query_location, candidate_location)

    return max(char_score, token_score)


def year_similarity(query_year, candidate_year, max_difference=20):
    if query_year is None:
        return math.nan

    try:
        if pd.isna(candidate_year):
            return 0.0
    except TypeError:
        pass

    try:
        candidate_year = int(float(candidate_year))
        query_year = int(float(query_year))
    except (TypeError, ValueError):
        return 0.0

    diff = abs(query_year - candidate_year)

    if diff >= max_difference:
        return 0.0

    return 1.0 - (diff / max_difference)


def gender_similarity(query_gender, candidate_gender):
    query = clean_text(query_gender)

    if query is None:
        return math.nan

    candidate = clean_text(candidate_gender)

    if candidate is None:
        return 0.0

    return 1.0 if query.lower()[0] == candidate.lower()[0] else 0.0


def weighted_score(scores, weights):
    numerator = 0.0
    denominator = 0.0

    for col, weight in weights.items():
        score = scores.get(col, math.nan)

        if score is None:
            continue

        try:
            if math.isnan(score):
                continue
        except TypeError:
            continue

        numerator += score * weight
        denominator += weight

    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 2)


def adjust_score(score, scores, query_birth_year, candidate_birth_year):
    first = scores.get("first_name_score", math.nan)
    last = scores.get("last_name_score", math.nan)
    year = scores.get("birth_year_score", math.nan)

    if first == 1.0 and last == 1.0 and year == 1.0:
        score += 15

    if query_birth_year is not None:
        try:
            candidate_year = int(float(candidate_birth_year))
            query_year = int(float(query_birth_year))
            year_gap = abs(query_year - candidate_year)

            if year_gap > 20:
                score -= 25
            elif year_gap > 10:
                score -= 10

        except (TypeError, ValueError):
            score -= 5

    if not is_missing_number(first) and first < 0.6:
        score -= 20

    score = max(0.0, min(100.0, score))
    return round(score, 2)


def round_or_none(value):
    if is_missing_number(value):
        return None

    return round(value, 3)


def is_missing_number(value):
    if value is None:
        return True

    try:
        return math.isnan(value)
    except TypeError:
        return True


def explain_matches(scores):
    labels = {
        "first_name_score": "first name",
        "last_name_score": "last name",
        "birth_year_score": "birth year",
        "birth_location_score": "birth location",
        "gender_score": "gender",
    }

    strong = []
    partial = []

    for col, label in labels.items():
        score = scores.get(col, math.nan)

        if is_missing_number(score):
            continue

        if score >= 0.8:
            strong.append(label)
        elif score > 0:
            partial.append(label)

    if strong and partial:
        return f"Strong match on {', '.join(strong)}; partial match on {', '.join(partial)}."
    if strong:
        return f"Strong match on {', '.join(strong)}."
    if partial:
        return f"Partial match on {', '.join(partial)}."

    return "No strong field-level match."


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve ranked genealogy candidates from transformed WikiTree schema CSVs.")

    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--first-name", type=str, default=None)
    parser.add_argument("--last-name", type=str, default=None)
    parser.add_argument("--birth-year", type=int, default=None)
    parser.add_argument("--birth-location", type=str, default=None)
    parser.add_argument("--gender", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    retriever = CandidateRetriever(schema_dir=args.schema_dir)
    results = retriever.find_candidates(
        first_name=args.first_name,
        last_name=args.last_name,
        birth_year=args.birth_year,
        birth_location=args.birth_location,
        gender=args.gender,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if results.empty:
        print("No candidates found. Try lowering --min-score or providing more query fields.")
        return

    display_cols = []
    for col in DISPLAY_COLUMNS:
        if col in results.columns:
            display_cols.append(col)

    print(results[display_cols].to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
