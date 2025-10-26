import os
import requests
import json
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# ======================
# App / Page config
# ======================
st.set_page_config(
    page_title="ICD Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🧠 ICD Coding Assistant")
st.write("Free-text → ICD suggestions → LLM validation")


# ======================
# Constants & Secrets
# ======================

ICD_CSV_PATH = "icd_clean.csv"          # must be in repo/container
FAISS_LOCAL_PATH = "icd_index.faiss"    # cached locally
EMBED_MODEL_NAME = "neuml/pubmedbert-base-embeddings"
TOP_K = 3                               # how many ICD candidates to retrieve

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", None)
FAISS_REMOTE_URL  = st.secrets.get("FAISS_REMOTE_URL", None)
# FAISS_REMOTE_URL must be like:
# https://huggingface.co/datasets/USERNAME/DATASET/resolve/main/icd_index.faiss

if not ANTHROPIC_API_KEY:
    st.error("Missing Anthropic API key in Streamlit secrets (ANTHROPIC_API_KEY). Please add it.")
    st.stop()


# ======================
# Helper functions
# ======================

def assert_valid_faiss_file(path: str):
    """
    Sanity check that FAISS file is a real index, not an HTML page.
    """
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        raise RuntimeError("FAISS index is missing or empty after download.")

    with open(path, "rb") as f:
        header = f.read(32)

    # Hugging Face 'blob' links return HTML, so catch that early.
    if header.startswith(b"<!DOCTYPE html") or header.startswith(b"<html"):
        raise RuntimeError(
            "Downloaded FAISS file looks like HTML, not a real FAISS index.\n"
            "Double-check FAISS_REMOTE_URL. It must use '/resolve/main/...', NOT '/blob/main/...'."
        )


def ensure_faiss_local():
    """
    Make sure FAISS_LOCAL_PATH exists, is non-empty, and looks valid.
    If not, download from FAISS_REMOTE_URL.
    """
    # If already exists, validate it. If valid, done.
    if os.path.exists(FAISS_LOCAL_PATH) and os.path.getsize(FAISS_LOCAL_PATH) > 0:
        try:
            assert_valid_faiss_file(FAISS_LOCAL_PATH)
            return
        except Exception as e:
            st.warning(f"Local FAISS file failed validation, will re-download. Details: {e}")

    # Need remote URL to download if local is not usable.
    if not FAISS_REMOTE_URL:
        raise RuntimeError(
            "FAISS index not available locally and FAISS_REMOTE_URL "
            "was not provided in Streamlit secrets."
        )

    with st.spinner("⬇️ Downloading FAISS index ..."):
        resp = requests.get(FAISS_REMOTE_URL)
        resp.raise_for_status()
        with open(FAISS_LOCAL_PATH, "wb") as f:
            f.write(resp.content)

    # validate after download
    assert_valid_faiss_file(FAISS_LOCAL_PATH)


@st.cache_resource(show_spinner=True)
def load_runtime_artifacts():
    """
    Loads & caches all heavyweight runtime assets:
    - ICD CSV
    - FAISS index
    - Embedding model
    - Anthropic client
    """
    # 1. ICD table
    if not os.path.exists(ICD_CSV_PATH):
        raise RuntimeError(
            f"Could not find {ICD_CSV_PATH}. "
            "Make sure icd_clean.csv is included in the repo/container."
        )
    icd_df = pd.read_csv(ICD_CSV_PATH)

    # 2. FAISS file
    ensure_faiss_local()

    # 2a. optional debug view for you
    faiss_size = os.path.getsize(FAISS_LOCAL_PATH)
    st.write(f"📦 FAISS index size: {faiss_size} bytes")
    with open(FAISS_LOCAL_PATH, "rb") as f:
        first16 = f.read(16)
    st.write(f"🔎 First 16 bytes of FAISS file: {first16}")

    # 3. Load FAISS index into memory
    try:
        index = faiss.read_index(FAISS_LOCAL_PATH)
    except Exception as e:
        raise RuntimeError(
            "faiss.read_index() failed.\n"
            "Possible causes:\n"
            "- The downloaded file is not a valid FAISS index (corrupt or HTML).\n"
            "- The FAISS index was built on GPU and this runtime uses CPU-only faiss.\n"
            f"Original error: {e}"
        )

    # 4. Embedding model for queries
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 5. Anthropic client
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    return icd_df, index, embed_model, client


def embed_query(text: str, embed_model: SentenceTransformer) -> np.ndarray:
    """
    Turn clinician text into a float32 vector for FAISS.
    normalize_embeddings=True gives cosine-like behavior
    with inner product search if index is built that way.
    """
    vec = embed_model.encode([text], normalize_embeddings=True)
    vec = np.asarray(vec, dtype="float32")
    return vec


def faiss_search(query_text: str, k: int, icd_df: pd.DataFrame, index, embed_model):
    """
    Return top-k ICD rows with FAISS scores.
    """
    qv = embed_query(query_text, embed_model)
    scores, idxs = index.search(qv, k)

    out = []
    for rank, (row_i, score) in enumerate(zip(idxs[0], scores[0]), start=1):
        row = icd_df.iloc[row_i]
        out.append({
            "rank": rank,
            "score": float(score),
            "row_index": int(row_i),
            "code": row.get("Full Code", None),
            "short_description": row.get("Short Description", None),
            "long_description": row.get("Long Description", None),
            "full_description": row.get("Full Description", None),
        })
    return out


def call_anthropic_validation(client: Anthropic, user_text: str, retrieved: list):
    """
    Ask Anthropic to:
    - extract causes of death / conditions
    - pick best ICD from retrieved list
    - warn if unsure

    You can tune this prompt however you want.
    """
    tool_icd_options = []
    for r in retrieved:
        tool_icd_options.append(
            f"- {r['code']}: {r['short_description'] or r['long_description']}"
        )
    tool_icd_options_str = "\n".join(tool_icd_options)

    prompt_text = f"""
You are an expert mortality coder in a hospital.
You will receive: (1) the doctor's free-text and
(2) top ICD code candidates from a retrieval system.

Task:
1. Extract:
   a. Immediate cause of death
   b. Time interval for immediate cause
   c. Underlying cause of death
   d. Time interval(s) for underlying cause(s)
2. Choose the single most appropriate ICD code from the provided list.
3. If none of the ICD codes fit, say "NO MATCH".

Doctor text:
{user_text}

ICD candidates:
{tool_icd_options_str}

Return valid JSON with these keys:
"immediate_cause", "immediate_interval",
"underlying_cause", "underlying_interval",
"chosen_icd_code", "reason".
"""

    # Anthropic messages API style:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",  # you can change model name if needed
        max_tokens=400,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt_text.strip()
            }
        ],
    )

    # Anthropic returns a list of content blocks; we take first text block
    try:
        llm_text = response.content[0].text
    except Exception:
        llm_text = str(response)

    # Try to parse JSON if possible
    parsed_json = None
    try:
        parsed_json = json.loads(llm_text)
    except Exception:
        # Model might have wrapped it in code fences or added extra words.
        # Very common. Let's try a crude cleanup.
        cleaned = llm_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove possible "json" language tag
            cleaned = cleaned.replace("json", "", 1).strip()
        try:
            parsed_json = json.loads(cleaned)
        except Exception:
            parsed_json = {"raw": llm_text}

    return parsed_json


# ======================
# Main App UI
# ======================

# Load heavy assets (cached)
icd_df, index, embed_model, client = load_runtime_artifacts()

st.subheader("Step 1. Enter the clinical / death certificate text")
user_text = st.text_area(
    "Example: 'Respiratory failure due to severe pneumonia, 3 days, underlying metastatic lung carcinoma 6 months'",
    height=120
)

col_a, col_b = st.columns([1,1])

with col_a:
    st.subheader("Step 2. Retrieve ICD candidates")
    if st.button("🔍 Search ICD", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some clinical text first.")
        else:
            hits = faiss_search(user_text, TOP_K, icd_df, index, embed_model)
            st.write("Top matches:")
            st.json(hits)
            st.session_state["last_hits"] = hits
            st.session_state["last_text"] = user_text

with col_b:
    st.subheader("Step 3. Ask LLM to structure + choose best code")
    if st.button("🧠 Run LLM Coding"):
        if "last_hits" not in st.session_state or "last_text" not in st.session_state:
            st.warning("Run the ICD search first.")
        else:
            llm_out = call_anthropic_validation(
                client,
                st.session_state["last_text"],
                st.session_state["last_hits"]
            )
            st.write("LLM suggestion:")
            st.json(llm_out)

st.markdown("---")
st.caption("This tool is a coding assistant. Final ICD coding decisions must follow hospital policy and human review.")
