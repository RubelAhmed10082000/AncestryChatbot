# Ancestry Chatbot

Ancestry Chatbot is an MSc research prototype for finding genealogy records and building a preliminary ancestor tree. 

A user enters a name and any known birth details through chatbot. The system ranks possible WikiTree profiles, gives each result a confidence score, and can display up to three generations of ancestors for a selected profile.

The current prototype uses record linkage rather than a generative language model. The chatbot asks a fixed questions and passes the answers to the retrieval system. This keeps the matching process traceable and makes it possible to evaluate each stage against known records.

The project is intended as a starting point for genealogy research. A high score does not prove that two records refer to the same person, and the generated tree should be checked against original sources.

## What is included

| Path | Purpose |
| --- | --- |
| `run.py` | Main entry point for running the application, pipeline, evaluations and tests |
| `app/api/` | FastAPI routes and the service layer used by the browser interface |
| `app/data_pipeline/` | WikiTree data extraction and transformation |
| `app/retrieval/` | Candidate retrieval and ranking |
| `app/scoring/` | Confidence scoring and explanations |
| `app/tree/` | Ancestor tree construction and file export |
| `app/ui/` | HTML, CSS and JavaScript for the chatbot interface |
| `scripts/` | Commands used to prepare data and run the evaluations |
| `tests/` | Unit, integration and end-to-end tests |
| `data/wikitree_test/` | WikiTree inputs used by the submitted prototype |
| `data/wikitree_schema/` | Transformed person, name and event tables |
| `data/evaluation/` | Evaluation cases and generated result tables |
| `requirements.txt` | Python packages required by the project |
| `pytest.ini` | Pytest configuration and test markers |

## Executable entry point

`run.py` is the executable entry point for the project. It is a Python launcher rather than a compiled Windows `.exe`. The application depends on Python packages listed in `requirements.txt`.

After the one-time setup below, every main task can be run as:

```text
python run.py <command>
```

Run `python run.py help` to see the available commands. The launcher runs each module from the project root, so the scripts do not need to be called directly.

## Requirements

- Python 3.10 or newer
- A current web browser
- Internet access only if the optional WikiTree extraction or live API test is run

The submitted dataset, application, pipeline and evaluation scripts can be used offline if the Python packages have been installed.

The submitted version was tested on Windows with Python 3.13.5. The package versions used for testing are fixed in `requirements.txt`.

## Setup

Extract the submitted ZIP file and open a terminal in the `AncestryChatbot` folder. Please create an isolated Python environment and install the required packages.

## Run the application

Start the local web server:

```text
python run.py serve
```

Open [http://127.0.0.1:8000] in a browser. The chatbot asks for a first name, surname, birth year, birth location and gender. Only the first name and surname are required, but the other fields usually produce a clearer result.

The application does not search every profile on WikiTree. It searches the records contained with this project in `data/wikitree_test/`. The nine seed figures below are the supported figures. Some ancestors and alternative candidate records are also present in the local data, but an arbitrary person from WikiTree will not be found unless their record is included in this dataset.

| Figure | First name | Surname | Birth year | Birth location | Gender |
| --- | --- | --- | --- | --- | --- |
| Samuel Langhorne Clemens (Mark Twain) | Samuel | Clemens | 1835 | Florida, Monroe, Missouri, United States | Male |
| Aretha Franklin | Aretha | Franklin | 1942 | Memphis, Shelby, Tennessee, United States | Female |
| Charles Darwin | Charles | Darwin | 1809 | Shrewsbury, Shropshire, England | Male |
| Jane Austen | Jane | Austen | 1775 | Steventon, Hampshire, England | Female |
| Isaac Newton | Isaac | Newton | 1642 | Woolsthorpe by Colsterworth, Lincolnshire, England | Male |
| William Shakespeare | William | Shakespeare | 1564 | Stratford-upon-Avon, Warwickshire, England | Male |
| Florence Nightingale | Florence | Nightingale | 1820 | Firenze, Firenze, Tuscany, Italy | Female |
| Winston Churchill | Winston | Churchill | 1874 | Blenheim Palace, Woodstock, Oxfordshire, England | Male |
| Isambard Kingdom Brunel | Isambard | Brunel | 1806 | Portsea, Hampshire, England | Male |

These values match the submitted dataset. Shorter locations can still work, but the full values above are useful when demonstrating the system.

For example, enter:

```text
First name: Jane
Surname: Austen
Birth year: 1775
Birth location: Hampshire, England
Gender: Female
```

The page will show ranked candidates with their matching evidence and confidence information. Select **View family tree** on a result to display its known ancestors. Press `Ctrl+C` in the terminal to stop the server.

## Run the complete pipeline and evaluations

The main command is:

```text
python run.py pipeline
```

It runs four stages in order:

1. Transforms the WikiTree records 
2. Recreates the evaluation cases
3. Runs the retrieval and confidence evaluation
4. Runs the family-tree evaluation

The pipeline stops if a stage fails. It uses the data included in `data/wikitree_test/` and does not contact WikiTree.

The main results are written to `data/evaluation/final/`:

| File | Contents |
| --- | --- |
| `evaluation_summary.csv` | Overall and per-condition retrieval metrics. |
| `evaluation_results.csv` | Result for every retrieval case. |
| `confidence_summary.csv` | Confidence-score results grouped by condition. |
| `ambiguity_cases.csv` | Tied or low-margin candidate results. |
| `failure_cases.csv` | Retrieval failures or incorrect top-ranked candidates. |
| `tree_evaluation_summary.csv` | Aggregate tree-generation metrics. |
| `tree_evaluation_results.csv` | Result for each evaluated family tree. |
| `tree_discrepancies.csv` | Missing or unexpected nodes, edges and generations. |


## Run the tests

For a repeatable test run that does not contact an external service, use:

```text
python run.py test -q --ignore=tests/extract/test_extract_e2e.py
```

To include the live WikiTree API test, run:

```text
python run.py test -q
```

The full command needs an internet connection. A failure in `tests/extract/test_extract_e2e.py` can indicate that WikiTree is unavailable or that its live data has changed. It does not necessarily mean the frozen local pipeline has failed.

### Search for candidates in the terminal

```text
python run.py search --first-name Jane --last-name Austen --birth-year 1775 --birth-location "Hampshire, England" --gender Female
```

Add `--output results.csv` to save the ranked candidates.

### Search and calculate confidence scores

```text
python run.py confidence --first-name Jane --last-name Austen --birth-year 1775 --birth-location "Hampshire, England" --gender Female
```

### Export a family tree

```text
python run.py tree --wikitree-id Austen-489 --generations 3
```

The tree is saved under `data/family_trees/Austen-489/` as CSV and JSON files.

### Run one pipeline stage

```text
python run.py transform
python run.py create-cases
python run.py evaluate-retrieval --output-dir data/evaluation/final
python run.py evaluate-tree
```

## Optional: refresh the WikiTree data

The submitted results use frozen data so that they can be reproduced. Refreshing the source data is not required to run or assess the software.

To collect current public records from the WikiTree API and rebuild the derived data, run:

```text
python run.py extract
python run.py hard-negatives
python run.py pipeline
```

This process needs an internet connection and overwrites files in `data/wikitree_test/`. WikiTree is a live service, so the downloaded profiles and later evaluation results may differ from the submitted versions.

## Command reference

| Command | Action |
| --- | --- |
| `serve` | Starts the web application. Options such as `--reload` and `--port 8001` are passed to Uvicorn. |
| `pipeline` | Runs transformation and both evaluations from start to finish |
| `test` | Runs Pytest. Extra Pytest options are passed through |
| `transform` | Rebuilds the structured schema from the frozen raw data |
| `create-cases` | Rebuilds the retrieval evaluation cases |
| `evaluate-retrieval` | Runs candidate-retrieval and confidence evaluation. |
| `evaluate-tree` | Runs family-tree evaluation |
| `search` | Searches for ranked candidate records from the terminal |
| `confidence` | Searches for candidates and adds confidence scores |
| `tree` | Generates a tree for a WikiTree ID or internal person ID |
| `extract` | Downloads current seed and ancestor records from WikiTree |
| `hard-negatives` | Adds alternative search matches to the candidate data |

