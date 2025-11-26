import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import HTMLResponse

from openai import OpenAI
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# ------------------------------------------------------------
# Load environment
# ------------------------------------------------------------
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment variables")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(
    title="Headache Agent API",
    description="Retrieval API for the Headache Expert Agent.",
    version="1.0.0"
)

# ------------------------------------------------------------
# Root endpoint (HTML Status Page)
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
      <head>
        <title>Headache Agent API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                padding: 40px;
                line-height: 1.6;
                background-color: #f8f8f8;
            }
            .box {
                background: white;
                padding: 30px;
                border-radius: 10px;
                max-width: 600px;
                margin: auto;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            p {
                color: #555;
            }
        </style>
      </head>
      <body>
        <div class="box">
          <h1>Headache Agent API</h1>
          <p>Status: <strong>OK</strong></p>
          <p>The service is live and ready to handle <code>/search</code> requests.</p>
          <p>Version: 1.0.0</p>
        </div>
      </body>
    </html>
    """


# ------------------------------------------------------------
# Request model
# ------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 8
    category: Optional[str] = None


# ------------------------------------------------------------
# Embedding function
# ------------------------------------------------------------
def get_query_embedding(text: str):
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI embedding error: {e}")


# ------------------------------------------------------------
# Supabase RPC call
# ------------------------------------------------------------
def call_match_headache_chunks(embedding, k, category: Optional[str]):
    try:
        params = {
            "query_embedding": embedding,
            "match_count": k,
            "category_filter": category
        }

        response = supabase.rpc("match_headache_chunks", params).execute()

        if not hasattr(response, "data"):
            raise HTTPException(
                status_code=500,
                detail="Supabase RPC returned no data"
            )

        if not isinstance(response.data, list):
            raise HTTPException(
                status_code=500,
                detail="Supabase RPC returned malformed data"
            )

        return response.data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase RPC error: {e}")


# ------------------------------------------------------------
# /search endpoint
# ------------------------------------------------------------
@app.post("/search")
def search(req: SearchRequest):
    try:
        embedding = get_query_embedding(req.query)
        results = call_match_headache_chunks(embedding, req.k, req.category)
        return {"results": results}

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")


# ------------------------------------------------------------
# End of file
# ------------------------------------------------------------
