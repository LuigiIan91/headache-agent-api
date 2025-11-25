import os
import sys
import json

import pdfplumber
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# ============================================================
# Load .env
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH, override=True)
else:
    print("ERROR: .env file not found.")
    sys.exit(1)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    print("ERROR: Missing OPENAI_API_KEY in .env")
    sys.exit(1)

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("ERROR: Missing Supabase credentials in .env")
    sys.exit(1)

# ============================================================
# Init clients
# ============================================================

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ============================================================
# Configuration
# ============================================================

PDF_ROOT = os.path.join(BASE_DIR, "pdfs")
BUCKET_NAME = "headache-pdfs"

# We switch to SMALL model = 1536 dimensions (required)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ============================================================
# Tokenizer
# ============================================================

tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text):
    tokens = tokenizer.encode(text)
    if len(tokens) == 0:
        return []

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(tokens), step):
        end = start + CHUNK_SIZE
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)

    return chunks

# ============================================================
# PDF text extraction
# ============================================================

def extract_pdf_text(path):
    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        print(f"WARNING: Could not read {path}. Error: {e}")
        return ""

# ============================================================
# Upload to storage
# ============================================================

def upload_pdf(local_path, storage_path):
    with open(local_path, "rb") as f:
        try:
            supabase.storage.from_(BUCKET_NAME).upload(
                storage_path,
                f,
                {"content-type": "application/pdf"}
            )
            print(f"Uploaded to storage: {storage_path}")
        except Exception as e:
            msg = str(e)
            if "resource already exists" in msg.lower():
                print(f"Already exists in storage, skipping: {storage_path}")
            else:
                print(f"WARNING: Upload failed for {storage_path}: {e}")

# ============================================================
# Embedding
# ============================================================

def embed_text(text):
    try:
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        emb = resp.data[0].embedding

        # SAFETY CHECK
        if len(emb) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(emb)}"
            )

        return emb
    except Exception as e:
        print("OpenAI embedding error:", e)
        raise

# ============================================================
# Insert into Supabase
# ============================================================

def insert_chunk(pdf_name, pdf_path, category, chunk_index, content, embedding):
    data = {
        "pdf_name": pdf_name,
        "pdf_path": pdf_path,
        "category": category,
        "chunk_index": chunk_index,
        "content": content,
        "embedding": embedding,
        "metadata": {
            "source": pdf_name,
            "path": pdf_path,
            "category": category
        }
    }
    supabase.table("headache_pdf_chunks").insert(data).execute()

# ============================================================
# Main
# ============================================================

def main():
    if not os.path.isdir(PDF_ROOT):
        print(f"PDF directory not found: {PDF_ROOT}")
        print("Create a folder named 'pdfs' and put PDFs inside it.")
        sys.exit(1)

    print(f"Scanning folder: {PDF_ROOT}")

    file_count = 0

    for root, dirs, files in os.walk(PDF_ROOT):
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue

            file_count += 1

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, PDF_ROOT)
            storage_path = rel_path.replace(os.sep, "/")

            parts = storage_path.split("/")
            category = parts[0] if len(parts) > 1 else "root"

            print(f"\n=== Processing PDF #{file_count}: {storage_path} ===")

            text = extract_pdf_text(full_path)
            if not text.strip():
                print("WARNING: PDF contains no extractable text. Skipping.")
                continue

            chunks = chunk_text(text)
            print(f"Extracted text length: {len(text)} chars → {len(chunks)} chunks.")

            upload_pdf(full_path, storage_path)

            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {filename}")):
                try:
                    emb = embed_text(chunk)
                    insert_chunk(
                        pdf_name=filename,
                        pdf_path=storage_path,
                        category=category,
                        chunk_index=i,
                        content=chunk,
                        embedding=emb
                    )
                except Exception as e:
                    print(f"WARNING: Failed chunk {i} of {filename}: {e}")

    print(f"\n=== Completed: {file_count} PDF files processed. ===")


if __name__ == "__main__":
    main()

