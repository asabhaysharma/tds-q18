import os
import math
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import uvicorn
import numpy as np

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://aipipe.org/openai/v1"

# Initialize Async Client
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

app = FastAPI(title="Semantic Re-ranker API")

# --- Pydantic Models ---
class Document(BaseModel):
    id: str | int
    text: str
    metadata: Dict[str, Any] = {}

class RerankRequest(BaseModel):
    query: str
    documents: List[Document]
    top_k: int = 5

class RerankResponse(BaseModel):
    query: str
    ranked_documents: List[Document]

# --- Helper Functions ---
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

async def get_embedding(text: str) -> List[float]:
    """Fetches embedding for a single string asynchronously."""
    try:
        # Using text-embedding-3-small as a standard, efficient model
        # Adjust model name if your specific API pipe requires a different one
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small" 
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

# --- Endpoints ---
@app.get("/")
async def root():
    return {"status": "healthy", "service": "Semantic Re-ranker"}

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    if not request.documents:
        return RerankResponse(query=request.query, ranked_documents=[])

    # 1. Get embedding for the query
    query_embedding = await get_embedding(request.query)

    # 2. Get embeddings for all documents concurrently
    # We use asyncio.gather to fire all API requests at once rather than sequentially
    doc_tasks = [get_embedding(doc.text) for doc in request.documents]
    doc_embeddings = await asyncio.gather(*doc_tasks)

    # 3. Calculate similarities
    scored_docs = []
    for doc, doc_emb in zip(request.documents, doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        scored_docs.append({
            "doc": doc,
            "score": score
        })

    # 4. Sort by score (descending)
    scored_docs.sort(key=lambda x: x["score"], reverse=True)

    # 5. Slice to top_k and extract documents
    top_docs = [item["doc"] for item in scored_docs[:request.top_k]]

    return RerankResponse(
        query=request.query,
        ranked_documents=top_docs
    )

if __name__ == "__main__":
    # This allows local debugging with `python main.py`
    # On Railway, the Procfile will handle execution
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)