# -*- coding: utf-8 -*-
import os
import requests
import json
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from anthropic import Anthropic

# =========================
# CONFIG
# =========================
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "MISSING_KEY")

ICD_CSV_PATH = "icd_clean.csv"            # <-- keep this file in GitHub repo
FAISS_LOCAL_PATH = "icd_index.faiss"      # <-- we will download this at runtime
FAISS_REMOTE_URL = st.secrets.get(
    "FAISS_REMOTE_URL",
    None  # <-- set this in Streamlit secrets to your Drive direct download URL
)

EMBED_MODEL_NAME = "neuml/pubmedbert-base-embeddings"
TOP_K = 3


# =========================
# HELPERS
# =========================
def ensure_faiss_local():
    """
    Make sure icd_index.faiss exists locally and isn't empty.
    If not, download it using FAISS_REMOTE_URL.
    """
    if os.path.exists(FAISS_LOCAL_PATH) and os.path.getsize(FAISS_LOCAL_PATH) > 0:
        return

    if not FAISS_REMOTE_URL:
        raise RuntimeError(
            "FAISS index is not available locally and FAISS_REMOTE_URL "
            "was not provided in Streamlit secrets."
        )

    with st.spinner("Downloading ICD FAISS index from remote..."):
        resp = requests.get(FAISS_REMOTE_URL)
        resp.raise_for_status()
        with open(FAISS_LOCAL_PATH, "wb") as f:
            f.write(resp.content)

    # sanity check download
    if os.path.getsize(FAISS_LOCAL_PATH) == 0:
        raise RuntimeError("Downloaded FAISS index is 0 bytes. Download failed.")


@st.cache_resource(show_spinner=True)
def load_runtime_artifacts():
    """
    Load heavy components once per session:
    - ICD dataframe (codes + descriptions)
    - FAISS index (for retrieval)
    - Embedding model (for encoding doctor phrases)
    - Anthropic client (LLM)
    Also surfaces debug info so we can see what's wrong on Streamlit Cloud.
    """
    # 1. Load ICD code table
    if not os.path.exists(ICD_CSV_PATH):
        raise RuntimeError(
            f"Could not find {ICD_CSV_PATH}. "
            "Make sure icd_clean.csv is committed in your GitHub repo."
        )
    icd_df = pd.read_csv(ICD_CSV_PATH)

    # 2. Ensure FAISS file exists locally, download if needed
    ensure_faiss_local()

    # 2a. Debug info: show size + first bytes so we know if Drive gave HTML
    faiss_size = os.path.getsize(FAISS_LOCAL_PATH)
    st.write(f"FAISS index file size: {faiss_size} bytes")

    with open(FAISS_LOCAL_PATH, "rb") as f:
        head = f.read(16)
    st.write(f"First 16 bytes of FAISS file: {head}")

    # 3. Try to load FAISS index
    try:
        index = faiss.read_index(FAISS_LOCAL_PATH)
    except Exception as e:
        raise RuntimeError(
            "faiss.read_index() failed.\n"
            "Possible causes:\n"
            "- The downloaded file is actually HTML (Google Drive warning page), not a real .faiss file.\n"
            "- The FAISS index was built on GPU and Streamlit is CPU-only.\n"
            f"Original error: {e}"
        )

    # 4. Load embedding model
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 5. Init Anthropic client
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    return icd_df, index, embed_model, client


# =========================
# LLM STEP 1: extract causes
# =========================
def extract_causes(client: Anthropic, text: str) -> dict:
    """
    Ask Claude to break the certificate text into:
    - immediate cause + interval
    - intermediate cause + interval
    - underlying cause + interval
    Returns a dict.
    """
    prompt = f"""
You are an expert mortality coder in a hospital.
Extract the following information from this text and return ONLY valid JSON with these exact keys:
{{
  "immediate_cause": "",
  "immediate_interval": "",
  "intermediate_cause": "",
  "intermediate_interval": "",
  "underlying_cause": "",
  "underlying_interval": ""
}}
Text: {text}
Rules:
- If something is missing, leave it as an empty string "".
- Do not include any markdown or explanation, only JSON.
"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=250,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "immediate_cause": "",
            "immediate_interval": "",
            "intermediate_cause": "",
            "intermediate_interval": "",
            "underlying_cause": "",
            "underlying_interval": "",
            "_raw": raw
        }
    return data


# =========================
# ICD RETRIEVAL
# =========================
def build_search_phrases(cause_dict: dict):
    """
    Return phrases we want to embed & search in FAISS.
    """
    phrases = [
        cause_dict.get("immediate_cause", ""),
        cause_dict.get("intermediate_cause", ""),
        cause_dict.get("underlying_cause", "")
    ]
    phrases = [p.strip() for p in phrases if p and p.strip()]
    return phrases


def search_icd_single(query: str, icd_df, embed_model, index, k=3):
    """
    Embed a single cause phrase, search FAISS, and return top-k matches.
    """
    qv = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    scores, idxs = index.search(qv, k)

    out = []
    for rank, (row_i, dist) in enumerate(zip(idxs[0], scores[0]), start=1):
        row = icd_df.iloc[row_i]
        out.append({
            "rank": rank,
            "distance": float(dist),
            "code": row.get("Full Code", None),
            "description": row.get("Full Description", None),
            "interval": "",
        })
    return out


def retrieve_icd_for_phrases(phrases, cause_info, icd_df, embed_model, index, k=3):
    """
    Build the table we show the user:
    - Section (Immediate / Intermediate / Underlying)
    - cause_text (doctor wording)
    - interval
    - icd_code (top match)
    - icd_desc (description of the ICD code)
    """
    table_rows = []

    mapping = [
        ("Immediate cause of death",       "immediate_cause",    "immediate_interval"),
        ("Intermediate cause of death I",  "intermediate_cause", "intermediate_interval"),
        ("Underlying cause of death II",   "underlying_cause",   "underlying_interval"),
    ]

    for label, cause_key, int_key in mapping:
        cause_txt = cause_info.get(cause_key, "").strip()
        interval = cause_info.get(int_key, "").strip()

        if cause_txt == "":
            table_rows.append({
                "row_label": label,
                "cause_text": "",
                "interval": "",
                "icd_code": "",
                "icd_desc": "",
            })
            continue

        hits = search_icd_single(cause_txt, icd_df, embed_model, index, k=k)

        if not hits:
            table_rows.append({
                "row_label": label,
                "cause_text": cause_txt,
                "interval": interval,
                "icd_code": "N/A",
                "icd_desc": "No match",
            })
        else:
            top_hit = hits[0]
            table_rows.append({
                "row_label": label,
                "cause_text": cause_txt,
                "interval": interval,
                "icd_code": top_hit["code"],
                "icd_desc": top_hit["description"],
            })

    return table_rows


# =========================
# LLM STEP 2: validate causal chain
# =========================
def build_certificate_summary_for_validation(table_rows):
    """
    Build a summary for Claude so it can judge:
    - Are these ICD codes reasonable?
    - Is the sequence logical (underlying -> intermediate -> immediate)?
    """
    lines = []
    for row in table_rows:
        if not row["cause_text"]:
            continue
        line = (
            f"- {row['row_label']}: {row['icd_code']} "
            f"({row['cause_text']} -> {row['icd_desc']})"
        )
        lines.append(line)
    return "\n".join(lines)


def validate_chain_with_llm(client: Anthropic, certificate_summary: str) -> dict:
    validation_prompt = f"""
You are a senior mortality coder with WHO ICD-10 experience.
Validate the correctness of the coding and causal sequence below.

Certificate summary:
{certificate_summary}

Tasks:
1. Check if each ICD code correctly matches the cause phrase.
2. Identify if any cause is *ill-defined* or *unlikely to cause death*.
3. Evaluate if the causal order (underlying → intermediate → immediate) is logical.
4. Suggest corrections if needed.
5. Return your answer as structured JSON with this exact format:

{{
  "validation_status": "Valid" or "Needs review",
  "issues": [list of problems or empty if none],
  "suggested_changes": {{
     "immediate_cause": "",
     "intermediate_cause": "",
     "underlying_cause": ""
  }}
}}
"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": validation_prompt}]
    )

    raw = response.content[0].text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "validation_status": "Needs review",
            "issues": ["LLM returned non-JSON", raw],
            "suggested_changes": {
                "immediate_cause": "",
                "intermediate_cause": "",
                "underlying_cause": ""
            }
        }
    return data


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Death Certificate Assistant",
    page_icon="💀",
    layout="wide"
)

st.title("Death Certificate Assistant 🏥")
st.caption("Prototype for mortality coding support (ICD lookup + chain validation).")

left, right = st.columns([1,1])

with left:
    st.subheader("Patient info")
    patient_name = st.text_input("Name")
    patient_id = st.text_input("ID / MRN")
    gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])

    st.subheader("Cause of death (doctor free text)")
    st.write("Write the chain exactly how you'd put it on the certificate.")
    default_note = (
        "Cardiac arrest (10 minutes) due to respiratory failure (1 hour) "
        "due to COVID-19 pneumonia (5 days)"
    )
    cod_text = st.text_area(
        "Cause of death narrative:",
        value=default_note,
        height=160
    )

    run_button = st.button("Analyze")

with right:
    st.subheader("Results")

    if run_button:
        if not cod_text.strip():
            st.error("Please enter the cause of death text first.")
        else:
            # 1. Load resources (will also print FAISS debug info to the app)
            icd_df, index, embed_model, client = load_runtime_artifacts()

            # 2. Extract structured causes
            causes = extract_causes(client, cod_text)
            st.markdown("**Extracted structured causes:**")
            st.json(causes)

            # 3. ICD mapping
            table_rows = retrieve_icd_for_phrases(
                build_search_phrases(causes),
                causes,
                icd_df,
                embed_model,
                index,
                k=TOP_K
            )

            st.markdown("**ICD suggestion table:**")
            result_df = pd.DataFrame(table_rows)
            st.dataframe(
                result_df.rename(columns={
                    "row_label": "Section",
                    "cause_text": "Cause of death text",
                    "interval": "Interval",
                    "icd_code": "ICD Code",
                    "icd_desc": "ICD Description"
                }),
                use_container_width=True
            )

            # 4. Validation of sequence
            summary_for_validation = build_certificate_summary_for_validation(table_rows)
            validation = validate_chain_with_llm(client, summary_for_validation)

            st.markdown("**Validation (LLM check):**")
            st.json(validation)

            st.success("Done ✅")
    else:
        st.info("Fill the form on the left and click Analyze.")

st.markdown("---")
st.caption("This tool is experimental and does not replace certified mortality coding staff.")
