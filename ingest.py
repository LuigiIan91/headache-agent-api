import os
import re
import string
import unicodedata
from typing import List, Optional, Tuple

import pdfplumber
import tiktoken
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------

PDF_ROOT = "pdfs"                   # root folder
BUCKET_NAME = "headache-pdfs"       # Supabase storage bucket
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500                    # tokens
CHUNK_OVERLAP = 100                 # tokens

# ------------------------------------------------------------
# 1. Environment and clients
# ------------------------------------------------------------

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
tokenizer = tiktoken.get_encoding("cl100k_base")

# ------------------------------------------------------------
# 2. Category keywords (more specific)
# ------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "migraine_general": [
        "migraine", "migraine without aura", "chronic migraine", "episodic migraine",
        "headache disorder", "primary headache"
    ],
    "migraine_aura": [
        "migraine with aura", "visual aura", "sensory aura", "speech aura",
        "visual phenomena", "scintillating scotoma", "fortification spectra",
        "cortical spreading depression", "csd", "hemiplegic migraine",
        "brainstem aura", "typical aura"
    ],
    "migraine_mechanisms": [
        "trigeminovascular", "trigemino-vascular", "meningeal nociceptors",
        "cgrp", "calcitonin gene-related peptide", "pacap", "vip", "nitric oxide",
        "k-atp", "katp", "nav1.7", "nav1.9", "ion channel", "sensitization"
    ],
    "tension_type_headache": [
        "tension-type headache", "episodic tension-type", "chronic tension-type"
    ],
    "cluster_headache_TACs": [
        "cluster headache", "trigeminal autonomic cephalalgia", "paroxysmal hemicrania",
        "sunct", "suna", "hypothalamus", "indomethacin-responsive"
    ],
    "medication_overuse_headache": [
        "medication overuse headache", "moh", "overuse of analgesics",
        "withdrawal", "detoxification", "overuse of triptans"
    ],
    "post_traumatic_headache": [
        "post-traumatic headache", "posttraumatic headache", "mild traumatic brain injury",
        "concussion", "tbi"
    ],
    "vascular_structural_headache": [
        "reversible cerebral vasoconstriction", "rcvs",
        "intracranial hypertension", "intracranial hypotension",
        "cerebral aneurysm", "subarachnoid haemorrhage", "subarachnoid hemorrhage",
        "cervicogenic headache"
    ],
    "CGRP_mAbs": [
        "erenumab", "fremanezumab", "galcanezumab", "eptinezumab",
        "anti-cgrp monoclonal antibody", "cgrp monoclonal antibody"
    ],
    "gepants": [
        "ubrogepant", "rimegepant", "atogepant", "zavegepant", "gepant"
    ],
    "onabotulinumtoxinA": [
        "onabotulinumtoxin", "onabotulinumtoxin a", "botulinum toxin type a",
        "btx-a", "btxa", "preempt", "botox"
    ],
    "traditional_preventives": [
        "topiramate", "sodium valproate", "valproate", "amitriptyline",
        "propranolol", "metoprolol", "candesartan", "flunarizine", "beta-blocker"
    ],
    "emerging_preventives": [
        "nav1.7 blocker", "nav1.9 blocker", "trpv1", "trpa1",
        "pacap antagonist", "k-atp blocker"
    ],
    "triptans": [
        "sumatriptan", "rizatriptan", "zolmitriptan", "eletriptan",
        "almotriptan", "frovatriptan", "naratriptan"
    ],
    "ditans": [
        "lasmiditan", "5-ht1f agonist", "ditan"
    ],
    "NSAIDs_analgesics": [
        "nsaid", "non-steroidal anti-inflammatory", "ibuprofen", "naproxen",
        "ketorolac", "diclofenac", "paracetamol", "acetaminophen"
    ],
    "neuromodulation": [
        "vagus nerve stimulation", "n-vns", "nvns", "transcranial magnetic stimulation",
        "single-pulse tms", "tms", "tdcs", "occipital nerve stimulation",
        "peripheral nerve stimulation"
    ],
    "behavioral_therapy": [
        "cognitive behavioural therapy", "cognitive behavioral therapy",
        "cbt", "biofeedback", "relaxation training", "mindfulness"
    ],
    "lifestyle_interventions": [
        "sleep hygiene", "dietary trigger", "hydration", "exercise",
        "trigger management", "lifestyle modification"
    ],
    "guidelines_consensus_ICHD": [
        "guideline", "guidelines", "consensus statement", "position paper",
        "practice guideline", "ichd", "international classification of headache disorders"
    ],
    "clinical_trials": [
        "randomized controlled trial", "randomised controlled trial",
        "double-blind", "placebo-controlled", "phase ii", "phase iii", "clinical trial"
    ],
    "experimental_models": [
        "animal model", "rodent model", "nitroglycerin model", "gtn model",
        "levcromakalim", "cortical spreading depression model", "csd model"
    ],
    "other": []
}

# ------------------------------------------------------------
# 3. Metadata extraction
# ------------------------------------------------------------

def clean_text(text: str) -> str:
    return text.replace("\x00", "") if text else ""

def is_generic_header(line: str) -> bool:
    l = line.lower()
    return any(tok in l for tok in [
        "original article", "nih public access", "author manuscript",
        "copyright", "all rights reserved", "journal of", "the journal of",
        "received:", "accepted:"
    ])

def extract_pdf_metadata_and_text(pdf_path: str) -> Tuple[str, str, List[str], str, Optional[int], Optional[str], str]:
    title = ""
    abstract = ""
    keywords: List[str] = []
    first_author = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    text_pages: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            return title, abstract, keywords, first_author, year, doi, ""

        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_pages.append(page_text)

        full_text = clean_text("\n".join(text_pages))

        first_page = pdf.pages[0].extract_text() or ""
        second_page = pdf.pages[1].extract_text() or "" if len(pdf.pages) > 1 else ""
        combined_head = first_page + "\n" + second_page

        lines = [l.strip() for l in first_page.splitlines() if l.strip()]

        # 1) metadata title
        meta_title = ""
        try:
            meta = pdf.metadata or {}
            meta_title = (meta.get("Title") or "").strip()
        except Exception:
            meta_title = ""

        if meta_title and not is_generic_header(meta_title) and len(meta_title) > 10:
            title = meta_title
        else:
            # 2) longest non-generic line within the first ~15 lines
            best = ""
            for ln in lines[:15]:
                if len(ln) < 15 or len(ln) > 200:
                    continue
                if is_generic_header(ln):
                    continue
                ll = ln.lower()
                if "@" in ll or "http://" in ll or "https://" in ll or "doi:" in ll:
                    continue
                if len(ln) > len(best):
                    best = ln
            if best:
                title = best
            elif lines:
                title = lines[0]

        # first author: line after title with capitalised words, no year/affiliation
        candidate_auth_line = ""
        if lines:
            try:
                title_index = lines.index(title)
            except ValueError:
                title_index = 0
            for ln in lines[title_index + 1:title_index + 10]:
                l = ln.lower()
                if any(w in l for w in ["department", "university", "hospital", "clinic", "centre", "center"]):
                    continue
                if "@" in l or "http" in l:
                    continue
                if re.search(r"(19|20)\d{2}", l):
                    continue
                if "," in ln or " " in ln:
                    candidate_auth_line = ln
                    break

        if candidate_auth_line:
            first_piece = candidate_auth_line.split(",")[0]
            first_author = re.sub(r"[\*\d\†\‡]+$", "", first_piece).strip()

        # year
        ym = re.search(r"\b(19|20)\d{2}\b", combined_head)
        if ym:
            try:
                yy = int(ym.group(0))
                if 1900 <= yy <= 2099:
                    year = yy
            except ValueError:
                year = None

        # doi
        dm = re.search(r"\b10\.\d{4,9}/\S+\b", combined_head, flags=re.IGNORECASE)
        if dm:
            doi = dm.group(0).rstrip(" .);]")

        # abstract
        fp_lower = first_page.lower()
        m_abs = re.search(r"\babstract\b[:\s]*", fp_lower)
        if m_abs:
            start = m_abs.end()
            abstract_candidate = first_page[start:start + 1500]
            stop = re.search(r"\b(keywords?|introduction|background)\b", abstract_candidate, flags=re.IGNORECASE)
            if stop:
                abstract_candidate = abstract_candidate[:stop.start()]
            abstract = abstract_candidate.strip()

        # keywords
        km = re.search(r"\bkeywords?\b\s*[:\-]\s*(.*)", first_page, flags=re.IGNORECASE)
        if km:
            kw_line = km.group(1)
            keywords = [k.strip() for k in re.split(r"[;,]", kw_line) if k.strip()]

    return title, abstract, keywords, first_author, year, doi, full_text

# ------------------------------------------------------------
# 4. doc_key and DB helpers
# ------------------------------------------------------------

def _normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = "".join(c for c in text if c not in string.punctuation)
    return re.sub(r"\s+", " ", text).strip()

def build_doc_key(title: str, first_author: str, year: Optional[int], doi: Optional[str]) -> Optional[str]:
    if doi:
        return "doi:" + _normalize(doi)
    if not first_author or not year:
        return None
    norm_title = _normalize(title)[:80] if title else ""
    return f"meta:{_normalize(first_author)}_{year}_{norm_title}"

def get_document_by_key(doc_key: str):
    resp = (
        supabase.table("headache_documents")
        .select("id, pdf_path")
        .eq("doc_key", doc_key)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None

def upsert_document(doc_key: str,
                    pdf_name: str,
                    pdf_path: str,
                    title: str,
                    first_author: str,
                    year: Optional[int],
                    doi: Optional[str]) -> int:
    existing = get_document_by_key(doc_key)
    payload = {
        "doc_key": doc_key,
        "pdf_name": pdf_name,
        "pdf_path": pdf_path,
        "title": title,
        "first_author": first_author,
        "pub_year": year,
        "doi": doi,
    }
    if existing:
        resp = (
            supabase.table("headache_documents")
            .update(payload)
            .eq("doc_key", doc_key)
            .execute()
        )
        return existing["id"]
    else:
        resp = supabase.table("headache_documents").insert(payload).execute()
        return resp.data[0]["id"]

# ------------------------------------------------------------
# 5. Category with folder hints
# ------------------------------------------------------------

def auto_category(pdf_path: str, pdf_name: str, structured_text: str, full_text: str) -> str:
    text = structured_text.lower() if structured_text else full_text.lower()
    name_lower = pdf_name.lower()
    path_lower = pdf_path.lower()
    parts = path_lower.split("/")
    folder = parts[0] if parts else ""

    # Folder-based hints
    if "aura" in folder:
        return "migraine_aura"
    if "btx" in folder or "botox" in folder:
        return "onabotulinumtoxinA"
    if "animal" in folder or "model" in folder:
        return "experimental_models"
    if "moh" in folder or "overuse" in folder:
        return "medication_overuse_headache"

    best_category = "other"
    best_score = 0

    for cat, kws in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in kws:
            kwl = kw.lower()
            if kwl in name_lower:
                score += 3
            if kwl in text:
                score += 5
        if cat == "migraine_aura":
            score *= 1.5
        if score > best_score:
            best_score = score
            best_category = cat

    return best_category

# ------------------------------------------------------------
# 6. Chunking, embeddings, storage
# ------------------------------------------------------------

def make_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks: List[str] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(clean_text(chunk))
        if end == n:
            break
        start = end - overlap
    return chunks

def get_embedding(text: str) -> List[float]:
    r = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return r.data[0].embedding

def upload_to_storage(local_path: str, storage_path: str):
    storage = supabase.storage
    try:
        with open(local_path, "rb") as f:
            storage.from_(BUCKET_NAME).upload(storage_path, f)
        print(f"[STORAGE] Uploaded: {storage_path}")
    except Exception as e:
        msg = str(e).lower()
        if "exists" in msg:
            print(f"[STORAGE] Exists, skipping: {storage_path}")
        else:
            print(f"[STORAGE] Error for {storage_path}: {e}")

# ------------------------------------------------------------
# 7. Main ingestion (idempotent, updates on rerun)
# ------------------------------------------------------------

def ingest_all_pdfs():
    total_pdfs = 0
    ingested = 0
    per_cat = {}

    for root, _, files in os.walk(PDF_ROOT):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue

            total_pdfs += 1
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, PDF_ROOT).replace("\\", "/")

            print(f"\n=== Processing: {rel_path} ===")

            try:
                title, abstract, kw_list, first_author, year, doi, full_text = extract_pdf_metadata_and_text(full_path)
            except Exception as e:
                print(f"[ERROR] Extraction failed for {rel_path}: {e}")
                continue

            if not full_text.strip():
                print(f"[WARN] Empty or unreadable text, skipping: {rel_path}")
                continue

            doc_key = build_doc_key(title, first_author, year, doi)
            if not doc_key:
                print(f"[WARN] Could not build doc_key, using path-based key.")
                doc_key = "path:" + _normalize(rel_path)

            existing = get_document_by_key(doc_key)
            if existing and existing["pdf_path"] != rel_path:
                print(f"[DEDUP] Same article already indexed as {existing['pdf_path']}, skipping duplicate file.")
                continue

            doc_id = upsert_document(doc_key, fname, rel_path, title, first_author, year, doi)
            print(f"[DOC] Upserted document id={doc_id}, key={doc_key}")

            structured_parts = []
            if title:
                structured_parts.append(title)
            if abstract:
                structured_parts.append(abstract)
            if kw_list:
                structured_parts.append(" ".join(kw_list))
            structured_text = "\n".join(structured_parts)

            category = auto_category(rel_path, fname, structured_text, full_text)
            print(f"[CATEGORY] {category}")

            upload_to_storage(full_path, rel_path)

            chunks = make_chunks(full_text)
            print(f"[CHUNKS] {len(chunks)} chunks")

            # idempotent: delete existing chunks for this pdf_path, then insert fresh
            supabase.table("headache_pdf_chunks").delete().eq("pdf_path", rel_path).execute()

            rows = []
            for idx, ch in enumerate(chunks):
                try:
                    emb = get_embedding(ch)
                except Exception as e:
                    print(f"[ERROR] Embedding failed for chunk {idx} of {rel_path}: {e}")
                    continue

                rows.append({
                    "pdf_name": fname,
                    "pdf_path": rel_path,
                    "category": category,
                    "chunk_index": idx,
                    "content": ch,
                    "embedding": emb,
                    "doc_key": doc_key,
                    "metadata": {
                        "source": rel_path,
                        "title": title,
                        "first_author": first_author,
                        "year": year,
                        "doi": doi,
                        "category": category
                    }
                })

            if rows:
                supabase.table("headache_pdf_chunks").insert(rows).execute()
                print(f"[DB] Inserted {len(rows)} chunks for {rel_path}")
                ingested += 1
                per_cat[category] = per_cat.get(category, 0) + 1
            else:
                print(f"[WARN] No rows to insert for {rel_path}")

    print("\n================ INGESTION SUMMARY ================")
    print(f"Total PDF files seen:   {total_pdfs}")
    print(f"PDFs ingested/updated:  {ingested}")
    print("\nPer-category this run:")
    for cat, n in sorted(per_cat.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:30s} {n}")
    print("===================================================\n")

if __name__ == "__main__":
    ingest_all_pdfs()


