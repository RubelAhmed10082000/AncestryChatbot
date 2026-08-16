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

## Setup

Extract the submitted ZIP file, open a terminal in the `AncestryChatbot` folder, and create a virtual environment.

## Run the application

Start the local web server:

```text
python run.py serve
```

Open [http://127.0.0.1:8000] in a browser. The chabot asks for a first name, surname, birth year, birth location and gender. Only the first name and surname are required. Currently, only the below seed figures work:

A simple demonstration is:

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
2. Recreates the retrieval evaluation cases
3. Runs the retrieval and confidence evaluation
4. Runs the family-tree evaluation

The pipeline stops if a stage . It uses the data included in `data/wikitree_test/` and does not contact WikiTree.

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

Because these files are regenerated, Git may show them as changed if formatting or library behaviour differs between environments.

## Run the tests

For a repeatable test run that does not contact an external service, use:

```text
python run.py test -q --ignore=tests/extract/test_extract_e2e.py
```

To include the live WikiTree API test, run:

```text
python run.py test -q
```

The full command needs an internet connection. A failure in `tests/extract/test_extract_e2e.py` can indicate that WikiTree is unavailable or that its live data has changed it does not necessarily mean the frozen local pipeline has failed.

## Other useful commands

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
| `pipeline` | Runs transformation and both evaluations from start to finish. |
| `test` | Runs Pytest. Extra Pytest options are passed through. |
| `transform` | Rebuilds the structured schema from the frozen raw data. |
| `create-cases` | Rebuilds the retrieval evaluation cases. |
| `evaluate-retrieval` | Runs candidate-retrieval and confidence evaluation. |
| `evaluate-tree` | Runs family-tree evaluation. |
| `search` | Searches for ranked candidate records from the terminal. |
| `confidence` | Searches for candidates and adds confidence scores. |
| `tree` | Generates a tree for a WikiTree ID or internal person ID. |
| `extract` | Downloads current seed and ancestor records from WikiTree. |
| `hard-negatives` | Adds alternative search matches to the candidate data. |

## Troubleshooting

**`ModuleNotFoundError` or a missing command**  
Check that the virtual environment is active and run `python -m pip install -r requirements.txt` again.

**A script works with `python -m` but not when its file path is used**  
Use `python run.py <command>`. The launcher sets the correct working directory and module path.

**Port 8000 is already in use**  
Run `python run.py serve --port 8001`, then open `http://127.0.0.1:8001`.

**Schema CSV files are missing**  
Run `python run.py transform`, or run the complete pipeline.

**The live WikiTree test fails**  
Check the internet connection and try again later. Use the offline test command above when checking the submitted implementation.

## Scope and limitations

- The included dataset is a small research sample built from public WikiTree profiles of historical figures. It is not a general index of every WikiTree record.
- Candidate ranking uses names, birth year, location and gender. It does not establish identity or biological relationship.
- Confidence values are "rule of thumb" indicators based on the available matching evidence. They are not calibrated probabilities.
- The family tree follows parent relationships in the submitted data. Missing or incorrect source records will affect the generated tree.
- The interface runs locally and does not store user queries in a database.
