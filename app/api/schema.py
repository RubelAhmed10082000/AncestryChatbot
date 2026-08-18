from pydantic import BaseModel, Field
from typing import Optional, List, Any

class CandidateSearchRequest(BaseModel):

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    birth_year: Optional[int] = None
    birth_location: Optional[str] = None
    gender: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0, le=100.0)

class CandidateSearchResponse(BaseModel):

    query: CandidateSearchRequest
    count: int
    candidates: List[dict]

