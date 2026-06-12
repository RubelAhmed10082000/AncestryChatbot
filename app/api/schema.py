from pydantic import BaseModel, Field
from typing import Optional, List, Any

class CandidateSearchRequest(BaseModel):
    """
    Model for candidate retrieval based on user request
    """
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    birth_year: Optional[int] = None
    birth_location: Optional[str] = None
    gender: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0 le=100.0)

class CandidateResponse(BaseModel):
    """
    Model for candidate retrieval response
    """
    rank: int
    rank_score: float
    confidence_score: Optional[float] = None
    wikitree_id: Optional[str] = None
    full_name: Optional[str] = None
    birth_year: Optional[Any] = None
    birth_location: Optional[str] = None
    confidence_explanation: Optional[str] = None
    confidence_interpretation: Optional[str] = None
    profile_url: Optional[str] = None

class CandidateSearchResponse(BaseModel):
    """
    Model response to candidate search
    """
    query: CandidateSearchRequest
    count: int
    candidates: List[dict]

class TreeResponse(BaseModel):
    """
    Model for tree visualisation
    """
    root_person_id: str
    generations: int
    summary: list[dict]
    nodes: list[dict]
    edges: list[dict]