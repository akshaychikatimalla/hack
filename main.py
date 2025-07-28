from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List

# Ensure these imports match your project structure
from models import QueryRequest, QueryResponse, QueryAnswer
from services.query_processor import QueryProcessor
from config import settings

# This security object is used by FastAPI to generate the "Authorize" button
security = HTTPBearer()

# This is the 'app' object Uvicorn is looking for.
# If there's an error above this line, this object won't be created.
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="API for processing natural language queries against documents.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

query_processor = QueryProcessor()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the Bearer token from the Authorization header.
    """
    token = credentials.credentials
    if token != settings.API_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or unauthorized token")
    return token

@app.get("/", tags=["General"], include_in_schema=False)
def read_root():
    """
    Redirects the root URL ('/') to the interactive API documentation.
    """
    return RedirectResponse(url="/docs")

@app.post("/api/v1/hackrx/run", response_model=QueryResponse, tags=["Hackathon"])
async def run_submissions(
    request: QueryRequest,
    auth_token: str = Depends(verify_api_key)
):
    """
    Processes a list of queries against provided document URLs.
    """
    all_answers: List[QueryAnswer] = []

    try:
        await query_processor.ingest_documents_if_needed(request.documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {e}")

    for query_text in request.queries:
        try:
            answer = await query_processor.process_query(request.documents, query_text)
            all_answers.append(answer)
        except Exception as e:
            all_answers.append(
                QueryAnswer(
                    query=query_text, decision="Error",
                    justification=f"An internal error occurred: {e}", mapped_clauses=[]
                )
            )

    return QueryResponse(answers=all_answers)

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": "Service is up and running."}

# This block allows running the app directly with `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
