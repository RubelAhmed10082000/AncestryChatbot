"""
FastAPI interface for prototype.

Retrieval, confidence scoring and tree traversal are implemented in their
modules rather than inside the API routes.
"""

from app.api.schema import CandidateSearchRequest, CandidateSearchResponse
from app.api.services import search_candidate, tree
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="Ancestry Chatbot API",
    version = "0.1.0",
    description="API layer"
)

templates = Jinja2Templates(directory="app/ui/templates")

app.mount(
    "/static",
    StaticFiles(directory="app/ui/static"),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
def chatbot_ui(request: Request):
    """Serve interface."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )

@app.get("/health")
def health_check():
    return {"status":"ok"}

@app.post("/api/candidates/search", response_model=CandidateSearchResponse)
def search_candidates(request: CandidateSearchRequest):
    """Return ranked candidates.

    Validation is handled by Pydantic model. 

    Args -
        request(CandidateSearchRequest): Candidate search parameters.

    Returns -
        Original query, number of returned candidates and
        ranked results.
    """
    try:
        candidates = search_candidate(
            first_name=request.first_name,
            last_name=request.last_name,
            birth_year=request.birth_year,
            birth_location=request.birth_location,
            gender=request.gender,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        return {
            "query": request,
            "count": len(candidates),
            "candidates": candidates,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Candidate search failed: {exc}")
    
@app.get("/api/tree/{person_id}")
def get_tree_by_person_id(
    person_id: str,
    generations: int = Query(default=3, ge=0, le=6),
    include_missing_stubs: bool = False,
):
    """Generate an ancestor tree from Person ID.
    
    Args -
        person_id(str): identifier for root person.
        generations(int): Maximum ancestor generation to traverse.
        include_missing_stubs(bool): Decides if unresolved linked profiles may appear as
        placeholder nodes.

    Returns -
            Tree metadata together with generated nodes and edges.
    """
    try:
        return tree(
            person_id=person_id,
            generations=generations,
            include_missing_stubs=include_missing_stubs,
          )
    except ValueError as exc:
         raise HTTPException(status_code=404, detail=str(exc))
    
    except FileNotFoundError as exc:
         raise HTTPException(status_code=500, detail=str(exc))
    
    except Exception as exc:
         raise HTTPException(status_code=500, detail=f"Tree generation failed: {exc}")
    
@app.get("/api/tree/by-wikitree/{wikitree_id}")
def get_tree_by_wikitree_id(
    wikitree_id: str,
    generations: int = Query(default=3, ge=0, le=6),
    include_missing_stubs: bool = False,
):
    """
    Generate ancestor tree from a WikiTreeID.
    """
    try:
        return tree(
            wikitree_id=wikitree_id,
            generations=generations,
            include_missing_stubs=include_missing_stubs,
        )

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tree generation failed: {exc}")
