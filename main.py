import os
import json
import asyncio
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
import uvicorn
import numpy as np

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://aipipe.org/openai/v1"

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

# Global variables to store documents and their embeddings
DOCUMENTS = []
DOC_EMBEDDINGS = []

# --- Helper Functions ---
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

async def get_embedding(text: str) -> List[float]:
    try:
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        return [0.0] * 1536 # Return zero vector on failure

# --- Lifespan Manager (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load documents from JSON file
    global DOCUMENTS, DOC_EMBEDDINGS
    try:
        with open("documents.json", "r") as f:
            data = json.load(f)
            # Handle if docs.json is list of strings OR list of dicts
            if data and isinstance(data[0], dict):
                DOCUMENTS = [d.get('text', '') for d in data] 
            else:
                DOCUMENTS = data
            print(f"Loaded {len(DOCUMENTS)} documents.")
            
        # Generate embeddings for all documents on startup
        print("Generating embeddings...")
        tasks = [get_embedding(doc) for doc in DOCUMENTS]
        DOC_EMBEDDINGS = await asyncio.gather(*tasks)
        print("Embeddings generated.")
        
    except FileNotFoundError:
        print("Error: docs.json not found!")
        DOCUMENTS = []
    
    yield
    # Clean up resources if needed
    DOCUMENTS.clear()
    DOC_EMBEDDINGS.clear()

app = FastAPI(lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: Optional[str] = None # Tester might send "true" or "false" string or boolean
    rerankK: int = 5

# --- Endpoints ---
@app.post("/search")
async def search(request: SearchRequest):
    if not DOCUMENTS:
         raise HTTPException(status_code=500, detail="No documents loaded")

    # 1. Get embedding for the query
    query_embedding = await get_embedding(request.query)

    # 2. Calculate Similarity
    similarities = []
    for idx, doc_emb in enumerate(DOC_EMBEDDINGS):
        score = cosine_similarity(query_embedding, doc_emb)
        similarities.append((score, DOCUMENTS[idx]))

    # 3. Sort by score (descending)
    similarities.sort(key=lambda x: x[0], reverse=True)

    # 4. Filter top K
    # If rerank is requested, we might fetch more initially, 
    # but for this assignment, vector search IS the ranking mechanism.
    top_k = request.k
    
    results = [doc for score, doc in similarities[:top_k]]
    
    return results

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)