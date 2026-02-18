import os
import json
import time
import asyncio
import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
import uvicorn

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://aipipe.org/openai/v1"

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

# Global Store
DOCUMENTS: List[Dict[str, Any]] = [] # Stores {"id": int, "text": str}
DOC_EMBEDDINGS: List[List[float]] = []

# --- Helper: Cosine Similarity ---
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# --- Helper: Get Embedding ---
async def get_embedding(text: str) -> List[float]:
    try:
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 1536

# --- Helper: LLM Re-ranking ---
async def get_llm_score(query: str, doc_text: str) -> float:
    """
    Asks the LLM to rate relevance 0-10 and normalizes to 0-1.
    """
    prompt = (
        f"Query: \"{query}\"\n"
        f"Document: \"{doc_text}\"\n\n"
        "Rate the relevance of this document to the query on a scale of 0-10. "
        "Respond with only the number."
    )
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # Use a fast model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        
        # Parse the number
        try:
            score = float(content)
        except ValueError:
            # Fallback if LLM is chatty (e.g., "The score is 7")
            import re
            match = re.search(r'\d+(\.\d+)?', content)
            score = float(match.group()) if match else 0.0
            
        return min(max(score, 0), 10) / 10.0  # Normalize 0-10 -> 0.0-1.0
        
    except Exception as e:
        print(f"Re-ranking error: {e}")
        return 0.0

# --- Startup: Load Docs & Cache Embeddings ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global DOCUMENTS, DOC_EMBEDDINGS
    try:
        print("Loading docs.json...")
        with open("docs.json", "r") as f:
            raw_data = json.load(f)
        
        # Normalize data format (list of strings OR list of dicts)
        DOCUMENTS = []
        for idx, item in enumerate(raw_data):
            if isinstance(item, str):
                DOCUMENTS.append({"id": idx, "text": item})
            elif isinstance(item, dict):
                # Ensure 'text' key exists, use 'id' if present or generate one
                text = item.get("text", str(item))
                doc_id = item.get("id", idx)
                DOCUMENTS.append({"id": doc_id, "text": text})

        print(f"Loaded {len(DOCUMENTS)} documents. Generating embeddings...")
        
        # Generate embeddings in batches to avoid rate limits
        batch_size = 10
        DOC_EMBEDDINGS = []
        for i in range(0, len(DOCUMENTS), batch_size):
            batch = DOCUMENTS[i:i + batch_size]
            tasks = [get_embedding(d["text"]) for d in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            DOC_EMBEDDINGS.extend(batch_embeddings)
            print(f"Embedded {len(DOC_EMBEDDINGS)}/{len(DOCUMENTS)}")
            
        print("Startup complete.")
        
    except FileNotFoundError:
        print("CRITICAL: docs.json not found. Search will fail.")
        DOCUMENTS = []
        DOC_EMBEDDINGS = []
    
    yield
    DOCUMENTS.clear()
    DOC_EMBEDDINGS.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class SearchRequest(BaseModel):
    query: str
    k: int = 7              # Initial vector retrieval count
    rerank: bool = True     # Whether to use LLM re-ranking
    rerankK: int = 4        # Final count after re-ranking

class ResultItem(BaseModel):
    id: int | str
    score: float
    content: str
    metadata: Dict[str, Any] = {}

class Metrics(BaseModel):
    latency: float
    totalDocs: int

class SearchResponse(BaseModel):
    results: List[ResultItem]
    reranked: bool
    metrics: Metrics

# --- Main Endpoint ---
@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    start_time = time.time()
    
    if not DOCUMENTS:
        raise HTTPException(status_code=500, detail="No documents loaded")

    # 1. Vector Search (Initial Retrieval)
    query_emb = await get_embedding(req.query)
    
    # Calculate all scores
    scores = []
    for idx, doc_emb in enumerate(DOC_EMBEDDINGS):
        score = cosine_similarity(query_emb, doc_emb)
        scores.append({
            "id": DOCUMENTS[idx]["id"],
            "content": DOCUMENTS[idx]["text"],
            "score": score,
            "original_index": idx
        })
    
    # Sort descending and take top K
    scores.sort(key=lambda x: x["score"], reverse=True)
    candidates = scores[:req.k]

    # 2. Re-ranking (Optional)
    is_reranked = False
    final_results = candidates

    if req.rerank:
        is_reranked = True
        # Create parallel tasks for LLM scoring
        tasks = [get_llm_score(req.query, doc["content"]) for doc in candidates]
        llm_scores = await asyncio.gather(*tasks)
        
        # Update scores with LLM results
        for i, doc in enumerate(candidates):
            doc["score"] = llm_scores[i] # Replace vector score with LLM score
            doc["metadata"] = {"method": "llm_rerank"}

        # Sort by new LLM scores
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Slice to rerankK
        final_results = candidates[:req.rerankK]

    # 3. Format Response
    response_items = [
        ResultItem(
            id=item["id"],
            score=item["score"],
            content=item["content"],
            metadata=item.get("metadata", {"method": "vector"})
        )
        for item in final_results
    ]

    latency_ms = (time.time() - start_time) * 1000

    return SearchResponse(
        results=response_items,
        reranked=is_reranked,
        metrics=Metrics(
            latency=latency_ms,
            totalDocs=len(DOCUMENTS)
        )
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)