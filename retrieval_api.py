import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# -------------------------------------------------
# Load environment
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing Supabase credentials in environment")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"

app = FastAPI(title="Headache PDF Retriever")


# -------------------------------------------------
# Request / response models (for documentation)
# -------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    category: Optional[str] = None  # e.g. "guidelines", "trials", "reviews", "root"


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def get_query_embedding(text: str):
    try:
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        # This will propagate as JSON instead of raw 500
        raise HTTPException(status_code=500, detail=f"OpenAI embedding error: {e}")


def call_match_headache_chunks(embedding, k: int, category: Optional[str]):
    params = {
        "query_embedding": embedding,
        "match_count": k
    }
    if category:
        params["category_filter"] = category

    try:
        resp = supabase.rpc("match_headache_chunks", params).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase RPC error: {e}")

    # supabase-py may return either .data or ['data']
    data = getattr(resp, "data", None)
    if data is None and isinstance(resp, dict):
        data = resp.get("data")

    if data is None:
        # Should not normally happen; make it explicit
        raise HTTPException(status_code=500, detail="Supabase RPC error: no data field in response")

    # Ensure it's a list
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail=f"Supabase RPC error: data is not a list, got {type(data)}")

    return data


# -------------------------------------------------
# Main endpoint
# -------------------------------------------------

@app.post("/search")
def search(req: SearchRequest):
    """
    Main search endpoint used by the Headache Expert GPT.
    Returns: {"results": [...rows from match_headache_chunks...]}
    """
    try:
        embedding = get_query_embedding(req.query)
        rows = call_match_headache_chunks(embedding, req.k, req.category)
        return {"results": rows}
    except HTTPException:
        # Already has a good message
        raise
    except Exception as e:
        # Catch any unexpected error and log it
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
