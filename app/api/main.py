from fastapi import FastAPI, HTTPException, Query

from app.api.schema import CandidateSearchRequests, CandidateSearchResponse
from app.api.services import search_candidates_services, tree

app = FastAPI(
    title="Ancestry Chatbot API",
    version = "0.1.0",
    description="API layer"
)

@app.get("/health")
def health_check():
    return {"status":"ok"}

@app.post("/api/candidates/search", response_model=CandidateSearchResponse)
def search_candidates(request: CandidateSearchRequests):
    try:
        candidates = search_candidates_services(
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
    
@app.get("api/tree/by-wikitree/{wikitree_id}")
def get_tree_by_wikitree_id(
    wikitree_id: str,
    generations: int = Query(default=3, ge=0, le=6),
    include_missing_stubs: bool = False,
):
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
     