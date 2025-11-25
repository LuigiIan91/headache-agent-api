import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# Load environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY or not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY or Supabase credentials in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"

app = FastAPI(title="Headache PDF Retriever")


class SearchRequest(BaseModel):
    query: str
    k: int = 8
    category: Optional[str] = None  # e.g. "guidelines", "trials", "reviews", "root"


class Chunk(BaseModel):
    id: int
    pdf_name: str
    pdf_path: str
    category: Optional[str]
    content: str
    similarity: float


class SearchResponse(BaseModel):
    results: List[Chunk]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        emb = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=req.query
        ).data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI embedding error: {e}")

    params = {
        "query_embedding": emb,
        "match_count": req.k
    }
    if req.category:
        params["category_filter"] = req.category

    try:
        resp = supabase.rpc("match_headache_chunks", params).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase RPC error: {e}")

    data = getattr(resp, "data", None) or resp.get("data")
    if data is None:
        raise HTTPException(status_code=500, detail="No data returned from Supabase")

    return {"results": data}
