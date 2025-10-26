import os
import json
import requests
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic


# ======================
# Streamlit page setup
# ======================
st.set_page_config(
    page_title="ICD Coding Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🧠 ICD Coding Assistant")
st.write("Free-text → ICD suggestions → LLM validation")


# ======================
# Config & Secrets
# ======================

ICD_CSV_PATH = "icd_clean.csv"       # Must be bundled in repo/container
FAISS_LOCAL_PATH = "icd_index.faiss" # Cached in container
EMBED_MODEL_NAME = "neuml/pubmedbert-base-embeddings"
TOP_K = 3                            # Number of ICD candidates to retrieve

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", None)
FAISS_REMOTE_URL  = st.secrets.get("FAISS_REMOTE_URL", None)
# IMPORTANT: FAISS_REMOTE_URL must look like:
# https://huggingface.co/datasets/<user_or_org>/<dataset>/resolve/main/icd_index.faiss
# NOT "blob/main/..."


# Hard-stop if Anthropic key missing
if not ANTHROPIC_API_KEY:
    st.error("Missing Anthropic API key in Streamlit secrets (ANTHROPIC_API_KEY). Please add it.")
    st.stop()


# ======================
# Helper functions
# ======================

def assert_valid_faiss_file(path: str):
    """
    Check that the FAISS file:
    - exists
    - is non-empty
    - does not start with HTML (which means we downloaded the webpage instead of the binary)
    """
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        raise RuntimeError("FAISS index is missing or empty after download.")

    with open(path, "rb") as f:
        header = f.read(32)

    # If it's an HTML page, header starts with <!DOCTYPE html or <html
    if header.startswith(b"<!DOCTYPE html") or header.startswith(b"<html"):
        raise RuntimeError(
            "Downloaded FAISS file looks like HTML, not a real FAISS index.\n"
            "Double-check FAISS_REMOTE_URL. It must use '/resolve/main/...', NOT '/blob/main/...'."
        )


def ensure_faiss_local():
    """
    Make sure FAISS_LOCAL_PATH points to a valid FAISS index.
    If not present or invalid, download it from FAISS_REMOTE_URL.
    """
    # If a file already exists locally, try to validate it.
    if os.path.exists(FAISS_LOCAL_PATH) and os.path.getsize(FAISS_LOCAL_PATH) > 0:
        try:
            assert_valid_faiss_file(FAISS_LOCAL_PATH)
            return
        except Exception as e:
            # It's either HTML or corrupted → try re-download.
            st.warning(f"Local FAISS file failed validation, will re-download. Details: {e}")

    # Otherwise (or validation failed), fetch from remote.
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

    # Validate after download.
    assert_valid_faiss_file(FAISS_LOCAL_PATH)


@st.cache_resource(show_spinner=True)
def load_runtime_artifacts():
    """
    Load and cache everything heavy:
      - ICD data table
      - FAISS index
      - embedding model
      - Anthropic client
    This only runs once per session (thanks to @st.cache_resource).
    """

    # 1. Load ICD dataframe
    if not os.path.exists(ICD_CSV_PATH):
        raise RuntimeError(
            f"Could not find {ICD_CSV_PATH}. "
            "Make sure icd_clean.csv is included in the repo/container."
        )
    icd_df = pd.read_csv(ICD_CSV_PATH)

    # 2. Ensure FAISS index is available locally
    ensure_faiss_local()

    # 2a. Debug info for you (good for dev visibility)
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
            "- The downloaded file is not a valid FAISS index (e.g. it's HTML instead of binary).\n"
            "- The FAISS index was built on GPU and this runtime uses CPU-only faiss.\n"
            f"Original error: {e}"
        )

    # 4. Load embedding model
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 5. Create Anthropic client
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    return icd_df, index, embed_model, client


def embed_query(text: str, embed_model: SentenceTransformer) -> np.ndarray:
    """
    Convert clinician text -> embedding vector (float32) for FAISS.
    We normalize embeddings for cosine-ish behavior.
    """
    vec = embed_model.encode([text], normalize_embeddings=True)
    vec = np.asarray(vec, dtype="float32")  # faiss prefers float32
    return vec


def faiss_search(query_text: str, k: int, icd_df: pd.DataFrame, index, embed_model):
    """
    Run FAISS search for query_text.
    Return a list of dicts describing the top-k ICD candidates.
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
      1. Extract structured cause-of-death info.
      2. Pick best ICD from retrieved candidates.
      3. Return JSON.
    We try multiple possible model IDs, catch errors, and parse JSON-ish output.
    """

    # Build a readable ICD candidate list string for the prompt
    tool_icd_options = []
    for r in retrieved:
        icd_code = r['code']
        icd_desc = r['short_description'] or r['long_description'] or ""
        tool_icd_options.append(f"- {icd_code}: {icd_desc}")
    tool_icd_options_str = "\n".join(tool_icd_options)

    prompt_text = f"""
You are an expert mortality coder in a hospital.
You will receive: (1) the doctor's free-text description and
(2) top ICD code candidates from a retrieval system.

Your tasks:
1. Extract:
   a. Immediate cause of death
   b. Time interval for immediate cause
   c. Underlying cause of death
   d. Time interval(s) for underlying cause(s)
2. Choose the single most appropriate ICD code ONLY from the provided candidates.
3. If none of the ICD codes fit, set chosen_icd_code = "NO MATCH".

Doctor text:
{user_text}

ICD candidates:
{tool_icd_options_str}

Return valid JSON with these keys exactly:
"immediate_cause", "immediate_interval",
"underlying_cause", "underlying_interval",
"chosen_icd_code", "reason".
"""

    # We'll try multiple model names so we don't hard-crash if one isn't allowed
    candidate_models = [
        "claude-3-5-sonnet-latest",
        "claude-3-sonnet-20240229",
    ]

    last_err = None

    for m in candidate_models:
        try:
            response = client.messages.create(
                model=m,
                max_tokens=400,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text.strip()
                    }
                ],
            )

            # Anthropic returns a list of content blocks
            try:
                llm_text = response.content[0].text
            except Exception:
                # If SDK shape is different, just stringify whole response
                llm_text = str(response)

            # Try to parse JSON directly
            parsed_json = None
            try:
                parsed_json = json.loads(llm_text)
            except Exception:
                # clean up common formatting like ```json ... ```
                cleaned = llm_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    cleaned = cleaned.replace("json", "", 1).strip()
                try:
                    parsed_json = json.loads(cleaned)
                except Exception:
                    parsed_json = {"raw": llm_text}

            return parsed_json

        except Exception as e:
            # Save the error and move on to next model name
            last_err = e

    # If we get here, every model failed or was not found.
    return {
        "error": "Anthropic call failed",
        "details": str(last_err),
        "note": "Your API key may not have access to the requested Anthropic models.",
    }


# ======================
# Main UI Logic
# ======================

# Load heavy things once
icd_df, index, embed_model, client = load_runtime_artifacts()

# --- UI: Step 1 input text
st.subheader("Step 1. Enter the clinical / death certificate text")
user_text = st.text_area(
    "Example: 'Respiratory failure due to severe pneumonia, 3 days, underlying metastatic lung carcinoma 6 months'",
    height=120
)

col_a, col_b = st.columns([1, 1])


# --- UI: Step 2 retrieval
with col_a:
    st.subheader("Step 2. Retrieve ICD candidates")
    if st.button("🔍 Search ICD", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some clinical/death text first.")
        else:
            hits = faiss_search(user_text, TOP_K, icd_df, index, embed_model)

            # Show user
            st.write("Top matches:")
            st.json(hits)

            # Persist for step 3
            st.session_state["last_hits"] = hits
            st.session_state["last_text"] = user_text


# --- UI: Step 3 LLM
with col_b:
    st.subheader("Step 3. Ask LLM to structure + choose best code")
    if st.button("🧠 Run LLM Coding"):
        if "last_hits" not in st.session_state or "last_text" not in st.session_state:
            st.warning("Please run 'Search ICD' first to get candidates.")
        else:
            llm_out = call_anthropic_validation(
                client,
                st.session_state["last_text"],
                st.session_state["last_hits"]
            )

            st.write("LLM suggestion:")
            st.json(llm_out)

st.markdown("---")
st.caption(
    "Disclaimer: This tool is an assistant. Final ICD assignments and cause-of-death statements "
    "must be reviewed and approved by qualified clinical coders per hospital policy."
)
