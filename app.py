#############################################
# app.py - Death Certificate Assistant
# Retrieval → ICD → Validation → Certificate Preview
#############################################

import os
import json
import requests
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic


#############################################
# STREAMLIT PAGE CONFIG
#############################################

st.set_page_config(
    page_title="Death Certificate Assistant",
    page_icon="💀",
    layout="wide"
)

st.title("Death Certificate Assistant 🏥")
st.caption("Prototype for mortality coding support (ICD lookup + chain validation).")


#############################################
# CONFIG / SECRETS
#############################################

# Local filenames expected in the container / repo
ICD_CSV_PATH     = "icd_clean.csv"
FAISS_LOCAL_PATH = "icd_index.faiss"

# Embedding model used to encode phrases for FAISS retrieval
EMBED_MODEL_NAME = "neuml/pubmedbert-base-embeddings"

# How many ICD hits to retrieve per cause
TOP_K = 3

# Anthropic key comes from Streamlit secrets in production
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", None)

# Optional: where to download FAISS from if not found locally.
# IMPORTANT: must be a direct binary link using "resolve/main", NOT "blob/main".
FAISS_REMOTE_URL = st.secrets.get("FAISS_REMOTE_URL", None)

if not ANTHROPIC_API_KEY:
    st.error("Missing Anthropic API key in Streamlit secrets (ANTHROPIC_API_KEY). Please add it.")
    st.stop()


#############################################
# FAISS INDEX HANDLING (download + validate)
#############################################

def assert_valid_faiss_file(path: str):
    """
    Check that the FAISS file:
    - exists
    - is non-empty
    - does not look like an HTML page (which happens if you downloaded the Hugging Face 'blob/' URL instead of 'resolve/')
    """
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        raise RuntimeError("FAISS index is missing or empty after download.")

    with open(path, "rb") as f:
        header = f.read(32)

    if header.startswith(b"<!DOCTYPE html") or header.startswith(b"<html"):
        raise RuntimeError(
            "Downloaded FAISS file looks like HTML, not a real FAISS index.\n"
            "Double-check FAISS_REMOTE_URL. It must use '/resolve/main/...', NOT '/blob/main/...'."
        )


def ensure_faiss_local():
    """
    Make sure we have a valid local FAISS index at FAISS_LOCAL_PATH.
    If it's invalid or missing, download from FAISS_REMOTE_URL (if provided).
    """
    # If we already have something, validate it
    if os.path.exists(FAISS_LOCAL_PATH) and os.path.getsize(FAISS_LOCAL_PATH) > 0:
        try:
            assert_valid_faiss_file(FAISS_LOCAL_PATH)
            return
        except Exception as e:
            st.warning(f"Local FAISS file failed validation, will re-download. Details: {e}")

    # Otherwise we need to download
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

    # Validate the new download
    assert_valid_faiss_file(FAISS_LOCAL_PATH)


#############################################
# RUNTIME ARTIFACTS (CACHED)
#############################################

@st.cache_resource(show_spinner=True)
def load_runtime_artifacts():
    """
    Load heavy assets exactly once per session:
      - ICD dataframe
      - FAISS index
      - Embedding model
      - Anthropic client
    """

    # 1. ICD table
    if not os.path.exists(ICD_CSV_PATH):
        raise RuntimeError(
            f"Could not find {ICD_CSV_PATH}. Make sure icd_clean.csv is available."
        )
    icd_df = pd.read_csv(ICD_CSV_PATH)

    # 2. FAISS index (ensure it's here and valid, else download)
    ensure_faiss_local()

    # Optional debug info
    faiss_size = os.path.getsize(FAISS_LOCAL_PATH)
    st.write(f"📦 FAISS index size: {faiss_size} bytes")
    with open(FAISS_LOCAL_PATH, "rb") as f:
        first16 = f.read(16)
    st.write(f"🔎 First 16 bytes of FAISS file: {first16}")

    # Load FAISS into memory
    try:
        index = faiss.read_index(FAISS_LOCAL_PATH)
    except Exception as e:
        raise RuntimeError(
            "faiss.read_index() failed.\n"
            "Possible causes:\n"
            "- The downloaded file is not a valid FAISS index (HTML instead of binary).\n"
            "- The FAISS index was built on GPU but this runtime only has CPU FAISS.\n"
            f"Original error: {e}"
        )

    # 3. Embedding model
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 4. Anthropic client
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    return icd_df, index, embed_model, client


#############################################
# STEP 1: LLM extraction of causes + intervals
#############################################

def extract_causes(client: Anthropic, text: str) -> dict:
    """
    Use Anthropic to split the doctor's narrative into:
      - immediate cause + interval
      - intermediate cause + interval
      - underlying cause + interval
    Returns a dict with those exact keys.
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
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=250,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text
    except Exception as e:
        return {
            "immediate_cause": "",
            "immediate_interval": "",
            "intermediate_cause": "",
            "intermediate_interval": "",
            "underlying_cause": "",
            "underlying_interval": "",
            "_error": f"Anthropic call failed: {e}"
        }

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


#############################################
# STEP 2: FAISS ICD retrieval
#############################################

def search_icd_single(query: str, icd_df, embed_model, index, k=3):
    """
    Embed a single cause phrase and retrieve k nearest ICD codes.
    """
    if not query.strip():
        return []

    qv = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    # make sure it's float32 for faiss
    qv = np.asarray(qv, dtype="float32")

    scores, idxs = index.search(qv, k)

    out = []
    for rank, (row_i, dist) in enumerate(zip(idxs[0], scores[0]), start=1):
        row = icd_df.iloc[row_i]
        out.append({
            "rank": rank,
            "distance": float(dist),
            "code": row.get("Full Code", None),
            "description": row.get("Full Description", None),
        })
    return out


def retrieve_icd_for_certificate_rows(cause_info, icd_df, embed_model, index, k=3):
    """
    Build table rows representing the death certificate rows:
      1. Immediate cause of death
      2. Intermediate cause of death I
      3. Underlying cause of death II

    Each row has:
      - label
      - cause text
      - interval
      - icd_code
      - icd_desc
    """
    mapping = [
        ("Immediate cause of death",      "immediate_cause",    "immediate_interval"),
        ("Intermediate cause of death I", "intermediate_cause", "intermediate_interval"),
        ("Underlying cause of death II",  "underlying_cause",   "underlying_interval"),
    ]

    table_rows = []

    for row_label, cause_key, interval_key in mapping:
        cause_txt = cause_info.get(cause_key, "").strip()
        interval  = cause_info.get(interval_key, "").strip()

        if not cause_txt:
            # nothing detected for this line
            table_rows.append({
                "row_label": row_label,
                "cause_text": "",
                "interval": "",
                "icd_code": "",
                "icd_desc": "",
            })
            continue

        hits = search_icd_single(cause_txt, icd_df, embed_model, index, k=k)
        if len(hits) == 0:
            table_rows.append({
                "row_label": row_label,
                "cause_text": cause_txt,
                "interval": interval,
                "icd_code": "N/A",
                "icd_desc": "No match",
            })
        else:
            best = hits[0]
            table_rows.append({
                "row_label": row_label,
                "cause_text": cause_txt,
                "interval": interval,
                "icd_code": best["code"],
                "icd_desc": best["description"],
            })

    return table_rows


#############################################
# STEP 3: Validation / "is valid? why?"
#############################################

def build_certificate_summary_for_validation(table_rows):
    """
    Turn ICD assignment into bullet summary for validation.
    Example:
    - Immediate cause of death: J96.0 (Acute respiratory failure -> ...)
    """
    lines = []
    for row in table_rows:
        if not row["cause_text"]:
            continue
        lines.append(
            f"- {row['row_label']}: {row['icd_code']} "
            f"({row['cause_text']} -> {row['icd_desc']})"
        )
    return "\n".join(lines)


def validate_chain_with_llm(client: Anthropic, certificate_summary: str) -> dict:
    """
    Ask Anthropic to check:
      - Are these causes acceptable medically?
      - Are any ill-defined / not true underlying causes of death?
      - Is the causal chain logical?
    """
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

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": validation_prompt}]
        )
        raw = response.content[0].text
    except Exception as e:
        return {
            "validation_status": "Needs review",
            "issues": [f"Anthropic call failed: {e}"],
            "suggested_changes": {
                "immediate_cause": "",
                "intermediate_cause": "",
                "underlying_cause": ""
            }
        }

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


#############################################
# SMALL UTILS FOR CERTIFICATE PREVIEW
#############################################

def pick_row(table_rows, label):
    """
    Find "Immediate cause of death", etc. in table_rows.
    Returns a dict with cause_text, interval, icd_code, icd_desc (or blanks).
    """
    for r in table_rows:
        if r["row_label"] == label:
            return r
    return {
        "cause_text": "",
        "interval": "",
        "icd_code": "",
        "icd_desc": ""
    }


#############################################
# STREAMLIT UI LAYOUT
#############################################

left, right = st.columns([1, 1])

with left:
    st.subheader("Patient info")
    patient_name = st.text_input("Name")
    patient_id = st.text_input("ID / MRN")
    gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])

    st.subheader("Cause of death (doctor free text)")
    st.write("Write the chain exactly how you'd put it on the certificate.")
    default_note = "Respiratory failure (3 days) due to severe bacterial pneumonia (5 days) due to COPD (10 years)"
    cod_text = st.text_area(
        "Cause of death narrative:",
        value=default_note,
        height=160
    )

    run_button = st.button("Analyze 🔍")


with right:
    st.subheader("Results")

    if run_button:
        if not cod_text.strip():
            st.error("Please enter the cause of death text first.")
        else:
            # 1. Load heavy stuff (cached)
            icd_df, index, embed_model, client = load_runtime_artifacts()

            # 2. Extract structured causes from doctor's narrative
            causes = extract_causes(client, cod_text)
            st.markdown("**Step 1. Extracted structured causes (LLM):**")
            st.json(causes)

            # 3. Map each cause to ICD via FAISS
            table_rows = retrieve_icd_for_certificate_rows(
                causes, icd_df, embed_model, index, k=TOP_K
            )
            st.markdown("**Step 2. ICD suggestion table:**")
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

            # 4. Ask LLM to validate medical logic and chain
            certificate_summary = build_certificate_summary_for_validation(table_rows)
            validation = validate_chain_with_llm(client, certificate_summary)
            st.markdown("**Step 3. Validation (LLM check):**")
            st.json(validation)

            # 5. Show a certificate-style preview (boxes)
            st.markdown("### 📝 Death Certificate Preview (auto-filled)")

            row_immediate    = pick_row(table_rows, "Immediate cause of death")
            row_intermediate = pick_row(table_rows, "Intermediate cause of death I")
            row_underlying   = pick_row(table_rows, "Underlying cause of death II")

            valid_status = validation.get("validation_status", "")
            issues_list  = validation.get("issues", [])
            if isinstance(issues_list, list):
                issues_text = "; ".join(issues_list)
            else:
                issues_text = str(issues_list)

            # Immediate cause row
            st.markdown("**Immediate cause of death**")
            c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
            with c1:
                st.text_input("Immediate cause of death",
                              value=row_immediate["cause_text"],
                              key="imm_text_box")
            with c2:
                st.text_input("Interval",
                              value=row_immediate["interval"],
                              key="imm_interval_box")
            with c3:
                st.text_input("is valid? why?",
                              value=f"{valid_status} {issues_text}",
                              key="imm_valid_box")
            with c4:
                st.text_input("ICD Code",
                              value=row_immediate["icd_code"],
                              key="imm_icd_box")

            # Intermediate cause row
            st.markdown("**Intermediate cause of death I**")
            d1, d2, d3, d4 = st.columns([2, 1, 2, 1])
            with d1:
                st.text_input("Intermediate cause of death I",
                              value=row_intermediate["cause_text"],
                              key="int_text_box")
            with d2:
                st.text_input("Interval",
                              value=row_intermediate["interval"],
                              key="int_interval_box")
            with d3:
                st.text_input("is valid? why?",
                              value=f"{valid_status} {issues_text}",
                              key="int_valid_box")
            with d4:
                st.text_input("ICD Code",
                              value=row_intermediate["icd_code"],
                              key="int_icd_box")

            # Underlying cause row
            st.markdown("**Underlying cause of death II**")
            e1, e2, e3, e4 = st.columns([2, 1, 2, 1])
            with e1:
                st.text_input("Underlying cause of death II",
                              value=row_underlying["cause_text"],
                              key="und_text_box")
            with e2:
                st.text_input("Interval",
                              value=row_underlying["interval"],
                              key="und_interval_box")
            with e3:
                st.text_input("is valid? why?",
                              value=f"{valid_status} {issues_text}",
                              key="und_valid_box")
            with e4:
                st.text_input("ICD Code",
                              value=row_underlying["icd_code"],
                              key="und_icd_box")

            st.success("Done ✅")
    else:
        st.info("Fill the form on the left and click Analyze 🔍 to populate certificate preview.")


st.markdown("---")
st.caption(
    "This tool is experimental and does not replace certified mortality coding staff. "
    "Always review ICD codes and causal chain before finalizing a death certificate."
)
