import pandas as pd
import pytest

import app.retrieval.candidate_retrieval as candidate_retrieval


@pytest.fixture
def schema_dir(tmp_path):
    person = pd.DataFrame(
        [
            {
                "Person_ID": "person-jane",
                "Wikitree_ID": "Austen-489",
                "Gender": "Female",
                "Profile_URL": "https://www.wikitree.com/wiki/Austen-489",
                "Has_Children": "False",
            },
            {
                "Person_ID": "person-george",
                "Wikitree_ID": "Austen-109",
                "Gender": "Male",
                "Profile_URL": "https://www.wikitree.com/wiki/Austen-109",
                "Has_Children": "True",
            },
            {
                "Person_ID": "person-charles",
                "Wikitree_ID": "Darwin-15",
                "Gender": "Male",
                "Profile_URL": "https://www.wikitree.com/wiki/Darwin-15",
                "Has_Children": "True",
            },
        ]
    )

    names = pd.DataFrame(
        [
            {
                "Name_ID": "name-jane",
                "Person_ID": "person-jane",
                "Last_Name_Current": "Austen",
                "Last_Name_At_Birth": "Austen",
                "Middle_Name": "",
                "Middle_Initials": "",
                "First_Name": "Jane",
                "Prefix": "",
                "Suffix": "",
                "Nicknames": "",
            },
            {
                "Name_ID": "name-george",
                "Person_ID": "person-george",
                "Last_Name_Current": "Austen",
                "Last_Name_At_Birth": "Austen",
                "Middle_Name": "",
                "Middle_Initials": "",
                "First_Name": "George",
                "Prefix": "",
                "Suffix": "",
                "Nicknames": "",
            },
            {
                "Name_ID": "name-charles",
                "Person_ID": "person-charles",
                "Last_Name_Current": "Darwin",
                "Last_Name_At_Birth": "Darwin",
                "Middle_Name": "Robert",
                "Middle_Initials": "R",
                "First_Name": "Charles",
                "Prefix": "",
                "Suffix": "",
                "Nicknames": "",
            },
        ]
    )

    event = pd.DataFrame(
        [
            {
                "Marriage_ID": "event-jane-birth",
                "Person_ID_1": "person-jane",
                "Person_ID_2": "",
                "Event_Type": "birth",
                "Event_Raw_Date": "1775-12-16",
                "Event_Month": "12",
                "Event_Day": "16",
                "Event_Year": "1775",
                "Event_Location": "Steventon, Hampshire, England",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-jane-death",
                "Person_ID_1": "person-jane",
                "Person_ID_2": "",
                "Event_Type": "death",
                "Event_Raw_Date": "1817-07-18",
                "Event_Month": "7",
                "Event_Day": "18",
                "Event_Year": "1817",
                "Event_Location": "Winchester, Hampshire, England",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-george-birth",
                "Person_ID_1": "person-george",
                "Person_ID_2": "",
                "Event_Type": "birth",
                "Event_Raw_Date": "1731-05-01",
                "Event_Month": "5",
                "Event_Day": "1",
                "Event_Year": "1731",
                "Event_Location": "Royal Tunbridge Wells, Kent, England",
                "Data Status": "certain",
            },
            {
                "Marriage_ID": "event-charles-birth",
                "Person_ID_1": "person-charles",
                "Person_ID_2": "",
                "Event_Type": "birth",
                "Event_Raw_Date": "1809-02-12",
                "Event_Month": "2",
                "Event_Day": "12",
                "Event_Year": "1809",
                "Event_Location": "Shrewsbury, Shropshire, England",
                "Data Status": "certain",
            },
        ]
    )

    person.to_csv(tmp_path / candidate_retrieval.PERSON_FILE, index=False)
    names.to_csv(tmp_path / candidate_retrieval.NAMES_FILE, index=False)
    event.to_csv(tmp_path / candidate_retrieval.EVENT_FILE, index=False)

    return tmp_path

### Testing Schema Loading ###

def test_candidate_retriever(schema_dir):
    retriever = candidate_retrieval.CandidateRetriever(schema_dir=schema_dir)

    assert len(retriever.person_df) == 3
    assert len(retriever.names_df) == 3
    assert len(retriever.event_df) == 4
    assert len(retriever.index_df) == 3

def test_candidate_retriever_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run transform.py first"):
        candidate_retrieval.CandidateRetriever(schema_dir=tmp_path)

### Testing Index Creation ### 

def test_build_search_index(schema_dir):
    retriever = candidate_retrieval.CandidateRetriever(schema_dir=schema_dir)

    jane = retriever.index_df[
        retriever.index_df["Wikitree_ID"] == "Austen-489"
    ].iloc[0]

    assert jane["Full_Name"] == "Jane Austen"
    assert jane["Birth_Year"] == "1775"
    assert jane["Birth_Location"] == "Steventon, Hampshire, England"
    assert jane["Death_Year"] == "1817"

def test_build_full_name(schema_dir):
    retriever = candidate_retrieval.CandidateRetriever(schema_dir=schema_dir)

    row = pd.Series(
        {
            "First_Name": "Barbara",
            "Middle_Name": "Vernice",
            "Last_Name_At_Birth": "",
            "Last_Name_Current": "Franklin",
        }
    )

    assert retriever._build_full_name(row) == "Barbara Vernice Franklin"

### Testing Candidate Match ###

def test_find_candidates(schema_dir):
    retriever = candidate_retrieval.CandidateRetriever(schema_dir=schema_dir)

    results = retriever.find_candidates(
        first_name="Jane",
        last_name="Austen",
        birth_year=1775,
        birth_location="Steventon, Hampshire, England",
        gender="Female",
        top_k=3,
    )

    assert not results.empty

    top = results.iloc[0]

    assert top["rank"] == 1
    assert top["wikitree_id"] == "Austen-489"
    assert top["full_name"] == "Jane Austen"
    assert top["birth_year"] == "1775"
    assert top["first_name_score"] == 1.0
    assert top["last_name_score"] == 1.0
    assert top["birth_year_score"] == 1.0
    assert top["gender_score"] == 1.0

### Testing Top-K Filtering ###

def test_find_candidates_top_k(schema_dir):
    retriever = candidate_retrieval.CandidateRetriever(schema_dir=schema_dir)

    results = retriever.find_candidates(
        first_name="Jane",
        last_name="Austen",
        birth_year=1775,
        birth_location="England",
        gender="Female",
        top_k=1,
    )

    assert len(results) == 1

def test_find_candidates_min_score(schema_dir):
    retriever = candidate_retrieval.CandidateRetriever(schema_dir=schema_dir)

    results = retriever.find_candidates(
        first_name="Completely",
        last_name="Wrong",
        birth_year=2020,
        birth_location="Mars",
        gender="Female",
        top_k=5,
        min_score=90,
    )

    assert results.empty

### Testing Partial Queries ###

def test_partial_name_query(schema_dir):
    retriever = candidate_retrieval.CandidateRetriever(schema_dir=schema_dir)

    results = retriever.find_candidates(
        first_name="Charles",
        last_name="Darwin",
        top_k=3,
    )

    assert not results.empty
    assert results.iloc[0]["wikitree_id"] == "Darwin-15"
