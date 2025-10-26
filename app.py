import os
import requests
import json
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from anthropic import Anthropic

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "MISSING_KEY")

ICD_CSV_PATH = "icd_clean.csv"          # must be committed in your GitHub repo
FAISS_LOCAL_PATH = "icd_index.faiss"    # downloaded each time container starts
FAISS_REMOTE_URL = st.secrets.get("FAISS_REMOTE_URL", None)

EMBED_MODEL_NAME = "neuml/pubmedbert-base-embeddings"
TOP_K = 3


def ensure_faiss_local():
    # reuse file if already there and non-empty
    if os.path.exists(FAISS_LOCAL_PATH) and os.path.getsize(FAISS_LOCAL_PATH) > 0:
        return

    if not FAISS_REMOTE_URL:
        raise RuntimeError(
            "FAISS index is not available locally and FAISS_REMOTE_URL "
            "was not provided in Streamlit secrets."
        )

    with st.spinner("Downloading ICD FAISS index from remote (Hugging Face)..."):
        resp = requests.get(FAISS_REMOTE_URL)
        resp.raise_for_status()
        with open(FAISS_LOCAL_PATH, "wb") as f:
            f.write(resp.content)

    if os.path.getsize(FAISS_LOCAL_PATH) == 0:
        raise RuntimeError("Downloaded FAISS index is 0 bytes. Download failed.")


@st.cache_resource(show_spinner=True)
def load_runtime_artifacts():
    # 1. load ICD CSV
    if not os.path.exists(ICD_CSV_PATH):
        raise RuntimeError(
            f"Could not find {ICD_CSV_PATH}. Make sure icd_clean.csv is committed in the repo."
        )
    icd_df = pd.read_csv(ICD_CSV_PATH)

    # 2. ensure FAISS is present
    ensure_faiss_local()

    # 2a. debug info
    faiss_size = os.path.getsize(FAISS_LOCAL_PATH)
    st.write(f"FAISS index file size: {faiss_size} bytes")
    with open(FAISS_LOCAL_PATH, "rb") as f:
        head = f.read(16)
    st.write(f"First 16 bytes of FAISS file: {head}")

    # 3. load FAISS index
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

    # 4. embed model and client
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    return icd_df, index, embed_model, client
