from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional

class DocumentInput(BaseModel):
    url: HttpUrl
    # Potentially add content_type if you plan to handle raw file uploads,
    # but for this example, we assume parsing from URL.

class QueryRequest(BaseModel):
    documents: List[HttpUrl] # Using HttpUrl directly for simplicity as per example
    queries: List[str]

class MappedClause(BaseModel):
    text: str
    document_url: HttpUrl
    # You might want to add page_number, chunk_id, etc.

class QueryAnswer(BaseModel):
    query: str
    decision: str # e.g., "Yes", "No", "Conditional", "N/A"
    status: Optional[str] = None # e.g., "approved", "rejected", "pending"
    payout_amount: Optional[str] = None # Or float/Decimal if numeric
    justification: str
    mapped_clauses: List[MappedClause]

class QueryResponse(BaseModel):
    answers: List[QueryAnswer]