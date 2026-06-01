
import argparse
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd


DEFAULT_SCHEMA_DIR = Path("data/wikitree_schema")
PERSON_FILE = "person.csv"
NAMES_FILE = "names.csv"
EVENT_FILE = "event.csv"


@dataclass(frozen=True)
class QueryProfile:

    first_name: str | None = None
    last_name: str | None = None
    birth_year: int | None = None
    birth_location: str | None = None
    gender: str | None = None

UNKNOWN_VALUES = {"", " ", "nan", "NaN", "None", "none", "NULL", "null", "Unknown", "unknown", "UNKNOWN"}


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
    return {token for token in text.split() if token}


def sequence_similarity(left, right):
    left_norm = normalise_for_matching(left)
    right_norm = normalise_for_matching(right)

    if not left_norm or not right_norm:
        return 0.0

    return SequenceMatcher(None, left_norm, right_norm).ratio()


def token_overlap_similarity(left, right):
    left_tokens = token_set(left)
    right_tokens = token_set(right)

    if not left_tokens or not right_tokens:
        return 0.0

    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def best_string_similarity(query_value, candidate_values):
    query_text = clean_text(query_value)
    if query_text is None:
        return math.nan

    scores = [sequence_similarity(query_text, value) for value in candidate_values if clean_text(value)]
    return max(scores) if scores else 0.0


def best_location_similarity(query_location, candidate_location):
    if clean_text(query_location) is None:
        return math.nan

    if clean_text(candidate_location) is None:
        return 0.0

    char_score = sequence_similarity(query_location, candidate_location)
    token_score = token_overlap_similarity(query_location, candidate_location)

    return max(char_score, token_score)


def year_similarity(query_year, candidate_year, max_difference = 20):
    if query_year is None:
        return math.nan

    try:
        if pd.isna(candidate_year):
            return 0.0
    except TypeError:
        pass

    try:
        candidate_year_int = int(float(candidate_year))
    except (TypeError, ValueError):
        return 0.0

    diff = abs(query_year - candidate_year_int)
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


def weighted_score(component_scores, weights):
    numerator = 0.0
    denominator = 0.0

    for component, weight in weights.items():
        score = component_scores.get(component, math.nan)
        if score is None or (isinstance(score, float) and math.isnan(score)):
            continue

        numerator += score * weight
        denominator += weight

    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 2)


def explain_matches(component_scores):
    labels = {
        "first_name_score": "first name",
        "last_name_score": "last name",
        "birth_year_score": "birth year",
        "birth_location_score": "birth location",
        "gender_score": "gender",
    }

    matched = []
    weak = []

    for key, label in labels.items():
        score = component_scores.get(key, math.nan)
        if isinstance(score, float) and math.isnan(score):
            continue
        if score >= 0.80:
            matched.append(label)
        elif score > 0:
            weak.append(label)

    if matched and weak:
        return f"Strong match on {', '.join(matched)}; partial match on {', '.join(weak)}."
    if matched:
        return f"Strong match on {', '.join(matched)}."
    if weak:
        return f"Partial match on {', '.join(weak)}."
    return "No strong field-level match."



class CandidateRetriever:

    DEFAULT_WEIGHTS = {
        "first_name_score": 0.25,
        "last_name_score": 0.25,
        "birth_year_score": 0.35,
        "birth_location_score": 0.15,
        "gender_score": 0.05,
    }

    def __init__(self, schema_dir: Path = DEFAULT_SCHEMA_DIR, weights: dict[str, float] | None = None):
        self.schema_dir = Path(schema_dir)
        self.weights = weights or self.DEFAULT_WEIGHTS

        self.person_df = self._read_required_csv(PERSON_FILE)
        self.names_df = self._read_required_csv(NAMES_FILE)
        self.event_df = self._read_required_csv(EVENT_FILE)
        self.index_df = self._build_search_index()

    def _read_required_csv(self, filename):
        path = self.schema_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}. Run transform_wikitree_to_schema.py first."
            )
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

        birth_events = (
            events[events["Event_Type"] == "birth"]
            .sort_values(by=["Person_ID_1", "Event_Year"], na_position="last")
            .drop_duplicates(subset=["Person_ID_1"], keep="first")
            .rename(
                columns={
                    "Person_ID_1": "Person_ID",
                    "Event_Raw_Date": "Birth_Raw_Date",
                    "Event_Year": "Birth_Year",
                    "Event_Location": "Birth_Location",
                    "Data Status": "Birth_Data_Status",
                }
            )
        )

        death_events = (
            events[events["Event_Type"] == "death"]
            .sort_values(by=["Person_ID_1", "Event_Year"], na_position="last")
            .drop_duplicates(subset=["Person_ID_1"], keep="first")
            .rename(
                columns={
                    "Person_ID_1": "Person_ID",
                    "Event_Raw_Date": "Death_Raw_Date",
                    "Event_Year": "Death_Year",
                    "Event_Location": "Death_Location",
                    "Data Status": "Death_Data_Status",
                }
            )
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

    @staticmethod
    def _build_full_name(row):
        parts = [
            clean_text(row.get("First_Name")),
            clean_text(row.get("Middle_Name")),
            clean_text(row.get("Last_Name_At_Birth")) or clean_text(row.get("Last_Name_Current")),
        ]
        parts = [part for part in parts if part]
        return " ".join(parts) if parts else None

    def find_candidates(
        self,
        *,
        first_name,
        last_name,
        birth_year,
        birth_location,
        gender,
        top_k = 5,
        min_score = 0.0,
    ) -> pd.DataFrame:
        query = QueryProfile(
            first_name=first_name,
            last_name=last_name,
            birth_year=birth_year,
            birth_location=birth_location,
            gender=gender,
        )

        records = []

        for _, candidate in self.index_df.iterrows():
            component_scores = {
                "first_name_score": best_string_similarity(query.first_name, [candidate.get("First_Name")]),
                "last_name_score": max(
                        best_string_similarity(query.last_name, [candidate.get("Last_Name_At_Birth")]),
                        0.5 * best_string_similarity(query.last_name, [candidate.get("Last_Name_Current")]),
                ),
                "birth_year_score": year_similarity(query.birth_year, candidate.get("Birth_Year")),
                "birth_location_score": best_location_similarity(query.birth_location, candidate.get("Birth_Location")),
                "gender_score": gender_similarity(query.gender, candidate.get("Gender")),
            }

            final_score = weighted_score(component_scores, self.weights)
            if final_score < min_score:
                continue

            records.append(
                {
                    "rank_score": final_score,
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
                    "first_name_score": round(component_scores["first_name_score"], 3)
                    if not math.isnan(component_scores["first_name_score"])
                    else None,
                    "last_name_score": round(component_scores["last_name_score"], 3)
                    if not math.isnan(component_scores["last_name_score"])
                    else None,
                    "birth_year_score": round(component_scores["birth_year_score"], 3)
                    if not math.isnan(component_scores["birth_year_score"])
                    else None,
                    "birth_location_score": round(component_scores["birth_location_score"], 3)
                    if not math.isnan(component_scores["birth_location_score"])
                    else None,
                    "gender_score": round(component_scores["gender_score"], 3)
                    if not math.isnan(component_scores["gender_score"])
                    else None,
                    "match_explanation": explain_matches(component_scores),
                }
            )

        result = pd.DataFrame(records)
        if result.empty:
            return result

        result = result.sort_values(
            by=["rank_score", "last_name_score", "first_name_score", "birth_year_score", "birth_location_score"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

        result.insert(0, "rank", range(1, len(result) + 1))
        return result.head(top_k)





def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve ranked genealogy candidates from transformed WikiTree schema CSVs.")

    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR, help="Directory containing person.csv, names.csv, and event.csv.")
    parser.add_argument("--first-name", type=str, default=None, help="Query first name, e.g. Jane")
    parser.add_argument("--last-name", type=str, default=None, help="Query surname/last name, e.g. Austen")
    parser.add_argument("--birth-year", type=int, default=None, help="Approximate or exact birth year, e.g. 1775")
    parser.add_argument("--birth-location", type=str, default=None, help="Birth location clue, e.g. Hampshire")
    parser.add_argument("--gender", type=str, default=None, help="Optional gender clue, e.g. Female")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to return.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum candidate score from 0-100.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path to save results.")

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

    display_columns = [
        "rank",
        "rank_score",
        "wikitree_id",
        "full_name",
        "birth_year",
        "birth_location",
        "death_year",
        "match_explanation",
    ]

    print(results[display_columns].to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
