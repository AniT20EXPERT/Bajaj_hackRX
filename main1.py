import os
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import requests
from get_text_pdf import extract_text_from_pdf
from get_embeddings import get_embeddings_batch, split_text
from hash_text import hash_text
from llm_answering import answer_question
from pinecone_db import store_embeddings_in_pinecone, query_pinecone_for_context
from pinecone_db import clear_pinecone_index
from dotenv import load_dotenv

load_dotenv()
print("FastAPI starting up")

app = FastAPI()

print("FastAPI started")

API_KEY = os.getenv("MY_API_KEY_FOR_AUTH")
print(f" API Key loaded: {'Success' if API_KEY else 'Error'}")


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


# Method 1: Fixed header handling
def verify_token(authorization: Optional[str] = Header(None)):
    """Verify the authorization token."""
    print(f" Received authorization header: {authorization}")

    if not authorization:
        print(" No authorization header provided")
        raise HTTPException(status_code=401, detail="Authorization header missing")

    # Handle both "Bearer token" and just "token" formats
    if authorization.startswith("Bearer "):
        token = authorization[7:]  # Remove "Bearer " prefix
    else:
        token = authorization

    print(f" Extracted token: {token[:10]}..." if token else "No token")

    if token != API_KEY:
        print(
            f" Token mismatch. Expected: {API_KEY[:10] if API_KEY else 'None'}..., Got: {token[:10] if token else 'None'}...")
        raise HTTPException(status_code=401, detail="Invalid token")

    print(" Authorization successful")
    return token


@app.post("/hackrx/run")
async def run_rag(payload: QueryRequest, token: str = Depends(verify_token)):
    print(" Function entered - run_rag")
    print(f" Processing {len(payload.questions)} questions")
    print(f" Document length: {len(payload.documents)} characters")

    try:
        # Processing logic here (steps 2–5)
        print(" Extracting text from PDF...")
        text = extract_text_from_pdf(payload.documents)
        print(f" Extracted {len(text)} characters")

        print(" Generating document hash...")
        doc_id = hash_text(text)
        print(f" Document ID: {doc_id}")

        print("✂ Splitting text into chunks...")
        chunks = split_text(text)
        print(f" Created {len(chunks)} chunks")

        print(" Getting embeddings...")
        embeddings = get_embeddings_batch(chunks)
        print(f" Generated {len(embeddings)} embeddings")

        print(" Storing embeddings in Pinecone...")
        store_embeddings_in_pinecone(embeddings, doc_id)
        print(" Embeddings stored successfully")

        answers = []
        print(f" Processing {len(payload.questions)} questions...")

        for i, q in enumerate(payload.questions, 1):
            print(f" Question {i}: {q[:100]}...")

            print(" Querying Pinecone for context...")
            context = query_pinecone_for_context(q)
            print(f" Retrieved context length: {len(context)} characters")
            print(f" Context: {context[:100]}...")

            print(" Generating answer...")
            answer = answer_question(q, context)
            answers.append(answer)
            print(f" Answer {i} generated: {answer[:100]}...")

        print(f" Successfully processed all {len(answers)} questions")
        return {"answers": answers}

    except Exception as e:
        print(f" Error in run_rag: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/hackrx/clear")
async def run_db_clearing(token: str = Depends(verify_token)):
    print(" Function entered - run_db_clearing")

    try:
        print(" Clearing Pinecone index...")
        index_cleared, success = clear_pinecone_index()

        if not success:
            print(f" Failed to clear index: {index_cleared}")
            raise HTTPException(status_code=500, detail=f"Failed to clear Pinecone index {index_cleared}")

        print(f" Successfully cleared index: {index_cleared}")
        return {"status": f"PINECONE WITH INDEX: {index_cleared} CLEARED"}

    except Exception as e:
        print(f" Error in run_db_clearing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clear operation failed: {str(e)}")


# Alternative method using direct header parameter
@app.post("/v1/run-alt")
async def run_rag_alt(payload: QueryRequest, authorization: str = Header(...)):
    print(" Function entered - run_rag_alt")
    print(f" Full authorization header: '{authorization}'")

    # More flexible token extraction
    token = None
    if authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        elif authorization.startswith("bearer "):
            token = authorization[7:]
        else:
            token = authorization

    print(f"Extracted token: '{token}'")
    print(f"Expected token: '{API_KEY}'")

    if not token or token != API_KEY:
        print(" Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")

    print(" Authorized user")

    # Rest of your processing logic...
    return {"status": "success", "message": "Alternative method working"}


# Health check endpoint
@app.get("/health")
async def health_check():
    print(" Health check requested")
    return {
        "status": "healthy",
        "api_key_configured": bool(API_KEY),
        "message": "FastAPI server is running"
    }


# Debug endpoint to check environment
@app.get("/debug")
async def debug_info():
    print(" Debug info requested")
    return {
        "api_key_set": bool(API_KEY),
        "api_key_length": len(API_KEY) if API_KEY else 0,
        "environment_vars": list(os.environ.keys())
    }


if __name__ == "__main__":
    import uvicorn

    print("🌟 Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
