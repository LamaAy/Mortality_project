"""
Saudi Ministry of Health – Electronic Death Certificate System
نظام شهادة الوفاة الإلكترونية – وزارة الصحة السعودية
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import datetime
import io
import os
import pickle

import anthropic

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="شهادة الوفاة | وزارة الصحة",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&family=Cairo:wght@400;600;700&display=swap');

:root {
  --green:       #006940;
  --green-dark:  #004d2e;
  --green-light: #e8f5ee;
  --green-mid:   #00843D;
  --gold:        #C8A951;
  --gray-bg:     #f4f6f4;
  --text:        #1a2e1a;
  --border:      #c5d9c8;
  --muted:       #5a7060;
}

html, body, [class*="css"] {
  font-family: 'Tajawal', 'Cairo', sans-serif;
  direction: rtl;
  color: var(--text);
}

.main .block-container {
  background: var(--gray-bg);
  padding: 1.5rem 2rem;
  max-width: 1200px;
}

.moh-header {
  background: linear-gradient(135deg, var(--green) 0%, var(--green-mid) 55%, var(--green-dark) 100%);
  color: white;
  padding: 1.4rem 2rem;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 18px rgba(0,105,64,.3);
  border-bottom: 3px solid var(--gold);
}
.moh-header h1 { font-size: 1.5rem; margin: 0; font-weight: 800; }
.moh-header p  { font-size: .82rem; margin: .2rem 0 0; opacity: .8; }
.moh-emblem {
  width: 70px; height: 70px;
  border: 2px solid rgba(200,169,81,.5);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: .62rem; font-weight: 700;
  color: var(--gold);
  text-align: center;
  line-height: 1.4;
  padding: 6px;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--green) 0%, var(--green-dark) 100%);
}
section[data-testid="stSidebar"] * { color: #dceee2 !important; }
section[data-testid="stSidebar"] h3 { color: white !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.2) !important; }
section[data-testid="stSidebar"] .stTextInput input {
  background: rgba(255,255,255,.1) !important;
  border-color: rgba(255,255,255,.3) !important;
}

.section-card {
  background: white;
  border-radius: 8px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.2rem;
  border: 1px solid var(--border);
  box-shadow: 0 1px 6px rgba(0,0,0,.05);
}
.section-title {
  color: var(--green);
  font-size: .95rem;
  font-weight: 700;
  border-bottom: 2px solid var(--green-light);
  padding-bottom: .45rem;
  margin-bottom: 1rem;
  letter-spacing: .03em;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
  border: 1.5px solid var(--border) !important;
  border-radius: 6px !important;
  font-family: 'Tajawal', sans-serif !important;
  font-size: .93rem !important;
  background: #fafcfa !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--green) !important;
  box-shadow: 0 0 0 2px rgba(0,105,64,.1) !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--green), var(--green-mid));
  color: white !important;
  border: none;
  border-radius: 6px;
  font-family: 'Tajawal', sans-serif;
  font-weight: 700;
  font-size: .92rem;
  padding: .52rem 1.6rem;
  transition: box-shadow .2s, transform .15s;
  box-shadow: 0 2px 8px rgba(0,105,64,.25);
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 5px 14px rgba(0,105,64,.35);
}

.step-bar {
  display: flex;
  justify-content: center;
  gap: 6px;
  margin-bottom: 1.5rem;
}
.step { background:#d4ddd6; color:#5a7060; border-radius:20px; padding:5px 16px; font-size:.8rem; font-weight:600; }
.step.active { background:var(--green); color:white; }
.step.done   { background:var(--gold);  color:white; }

.icd-card {
  background: var(--green-light);
  border: 1px solid #9ecaad;
  border-radius: 6px;
  padding: .85rem 1.1rem;
  margin: .45rem 0;
}
.icd-code  { font-size:1.05rem; font-weight:800; color:var(--green); font-family:'Courier New',monospace; letter-spacing:.05em; }
.icd-badge { display:inline-block; background:var(--green); color:white; border-radius:20px; padding:1px 10px; font-size:.7rem; margin:0 2px; font-weight:600; }
.icd-desc  { color:#2a4a2e; font-size:.88rem; margin-top:.2rem; }

.cert-preview {
  background: white;
  border: 2px solid var(--green);
  border-radius: 8px;
  padding: 2.2rem;
  font-family: 'Tajawal', sans-serif;
  box-shadow: 0 4px 20px rgba(0,105,64,.1);
}
.cert-title { font-size:1.6rem; font-weight:800; color:var(--green); }
.cert-sub   { color:var(--muted); font-size:.88rem; }
.cert-field { display:flex; justify-content:space-between; border-bottom:1px solid #e8ede9; padding:.36rem 0; font-size:.88rem; }
.cert-label { font-weight:700; color:var(--green); min-width:155px; }
.cert-stamp {
  border:2px solid var(--green); border-radius:50%;
  width:86px; height:86px;
  display:flex; align-items:center; justify-content:center;
  color:var(--green); font-weight:700; text-align:center; font-size:.65rem; line-height:1.5;
}

.stTabs [data-baseweb="tab"] { font-family:'Tajawal',sans-serif !important; font-weight:600; font-size:.88rem; }
.stTabs [aria-selected="true"] { color:var(--green) !important; border-bottom-color:var(--green) !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="moh-header">
  <div>
    <h1>نظام شهادة الوفاة الإلكترونية</h1>
    <p>وزارة الصحة &nbsp;|&nbsp; المملكة العربية السعودية</p>
    <p style="font-size:.76rem;opacity:.65">Death Certificate Registration System &nbsp;|&nbsp; ICD-10 AI-Assisted Coding</p>
  </div>
  <div class="moh-emblem">وزارة<br>الصحة<br>MOH<br>KSA</div>
</div>
""", unsafe_allow_html=True)

# ── API key from secrets ──────────────────────────────────────────────────────
try:
    API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    st.error("مفتاح API غير موجود. يرجى إضافة ANTHROPIC_API_KEY في Streamlit Secrets.")
    st.stop()

# ── Google Drive file IDs ─────────────────────────────────────────────────────
# Set these two IDs once — the app downloads and caches the files automatically.
# Find the ID in your Drive share link: drive.google.com/file/d/FILE_ID/view
GDRIVE_EMBEDDINGS_ID = "1lSxHUBswhVtfTzDvbWHmsdmQIt0-Ne8J"   # embeddings.npy
GDRIVE_METADATA_ID   = "1JM19CbSwtLgo7Uqeu_1MAnJuz-4m-hgJ"   # metadata.pkl
CACHE_DIR            = os.path.join(os.path.expanduser("~"), ".icd10_cache")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### اعدادات النظام")
    st.markdown("---")
    st.markdown("### بيانات ICD-10")
    st.markdown(
        '<div style="font-size:.78rem;opacity:.8;padding:.3rem 0;line-height:1.7">'        'البيانات تُحمَّل تلقائياً من Google Drive'        '<br>ولا تحتاج إلى رفع أي ملف.</div>',
        unsafe_allow_html=True,
    )
    if st.button("إعادة تحميل البيانات", use_container_width=True):
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        st.session_state.df_icd      = None
        st.session_state.faiss_index = None
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### بيانات المستشفى")
    hospital_name = st.text_input("اسم المستشفى", value="مستشفى الملك فهد التخصصي")
    hospital_city = st.text_input("المدينة", value="الرياض")
    doctor_name   = st.text_input("اسم الطبيب المُصدر", value="")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:.7rem;opacity:.55;text-align:center;line-height:2">'        'وزارة الصحة – 1446هـ<br>الإصدار 2.3</div>',
        unsafe_allow_html=True,
    )

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("page", 1), ("form_data", {}), ("icd_results", None),
             ("df_icd", None), ("faiss_index", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Google Drive downloader ───────────────────────────────────────────────────
def _gdrive_download(file_id: str, dest_path: str) -> None:
    """Download a public Google Drive file, handling the virus-scan redirect."""
    import requests
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    session  = requests.Session()
    URL      = "https://drive.google.com/uc"
    response = session.get(URL, params={"export": "download", "id": file_id}, stream=True)
    # Handle large-file confirmation
    token = next((v for k, v in response.cookies.items() if k.startswith("download_warning")), None)
    if token:
        response = session.get(URL, params={"export": "download", "id": file_id, "confirm": token}, stream=True)
    if b"confirm=" in response.content[:3000]:
        import re as _re
        m = _re.search(rb'confirm=([0-9A-Za-z_\-]+)', response.content[:3000])
        if m:
            response = session.get(URL, params={"export": "download", "id": file_id,
                                                "confirm": m.group(1).decode()}, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

# ── ICD data helpers ──────────────────────────────────────────────────────────
def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    if "CodeFormatted" not in df.columns:
        try:
            df.columns = ["Id","Code","CodeFormatted","ShortDesc","LongDesc",
                          "HIPPA","Deleted","Classification","AcceptableMain",
                          "GenderRestriction","MatchSource","MatchedFromCode","Note"]
        except ValueError:
            pass
    if "Deleted" in df.columns:
        df = df[df["Deleted"] != "Yes"].copy()
    df = df.dropna(subset=["Code"]).reset_index(drop=True)
    if "EmbedText" not in df.columns:
        df["EmbedText"] = (df["CodeFormatted"].fillna("") + " | " +
                           df["LongDesc"].fillna("") + " | " +
                           df["ShortDesc"].fillna(""))
    return df

@st.cache_resource(show_spinner="جارٍ تحميل بيانات ICD-10 من Google Drive...")
def _load_icd_cached(embeddings_id: str, metadata_id: str, cache_dir: str):
    """Download from Drive once, cache locally forever."""
    os.makedirs(cache_dir, exist_ok=True)
    emb_path  = os.path.join(cache_dir, "embeddings.npy")
    meta_path = os.path.join(cache_dir, "metadata.pkl")

    if embeddings_id and not os.path.exists(emb_path):
        _gdrive_download(embeddings_id, emb_path)

    if metadata_id and not os.path.exists(meta_path):
        _gdrive_download(metadata_id, meta_path)

    # Load dataframe
    df = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            df = pickle.load(f)
        df = _normalise_df(df)

    # Build FAISS index from embeddings
    faiss_idx = None
    if os.path.exists(emb_path) and df is not None:
        try:
            import faiss
            emb       = np.load(emb_path).astype("float32")
            faiss_idx = faiss.IndexFlatIP(emb.shape[1])
            faiss_idx.add(emb)
        except Exception:
            pass

    return df, faiss_idx

def _row_to_dict(row, score: float) -> dict:
    return {
        "code":               str(row.get("Code", "")),
        "code_formatted":     str(row.get("CodeFormatted", "")),
        "short_desc":         str(row.get("ShortDesc", "")),
        "long_desc":          str(row.get("LongDesc", "")),
        "acceptable_main":    str(row.get("AcceptableMain", "")),
        "gender_restriction": str(row.get("GenderRestriction", "")),
        "classification":     str(row.get("Classification", "")),
        "similarity":         score,
    }

@st.cache_resource(show_spinner="تحميل نموذج التضمين...")
def _embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def search_icd(df, fidx, query: str, top_k: int = 6) -> list:
    if fidx is not None:
        try:
            model = _embed_model()
            q_vec = model.encode([query], normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")
            scores, indices = fidx.search(q_vec, top_k)
            return [_row_to_dict(df.iloc[idx], float(s))
                    for s, idx in zip(scores[0], indices[0]) if idx != -1]
        except Exception:
            pass
    terms = query.lower().split()
    sc    = df["EmbedText"].str.lower().apply(lambda x: sum(t in x for t in terms))
    top   = sc.nlargest(top_k).index
    return [_row_to_dict(df.iloc[i], float(sc[i])) for i in top if sc[i] > 0]

# ── Auto-load on startup ───────────────────────────────────────────────────────
if st.session_state.df_icd is None:
    if not GDRIVE_EMBEDDINGS_ID and not GDRIVE_METADATA_ID:
        st.sidebar.warning("يرجى إضافة GDRIVE_EMBEDDINGS_ID أو GDRIVE_METADATA_ID في الكود.")
    else:
        try:
            df, fidx = _load_icd_cached(GDRIVE_EMBEDDINGS_ID, GDRIVE_METADATA_ID, CACHE_DIR)
            if df is not None:
                st.session_state.df_icd      = df
                st.session_state.faiss_index = fidx
                mode = "FAISS + embeddings" if fidx is not None else "keyword"
                st.sidebar.success(f"تم تحميل {len(df):,} رمز  ({mode})")
            else:
                st.sidebar.warning(
                    "تم تحميل embeddings.npy.\n\n"
                    "أضف GDRIVE_METADATA_ID (file ID لملف metadata.pkl) في السطر 8 من الكود."
                )
        except Exception as e:
            st.sidebar.error(f"خطأ في التحميل: {e}")

# ── Claude helpers ─────────────────────────────────────────────────────────────
def call_claude(system_prompt: str, user_content: str, max_tokens: int = 1500) -> str:
    client = anthropic.Anthropic(api_key=API_KEY)
    resp   = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return resp.content[0].text.strip()

def extract_concepts(free_text: str, gender: str, age: str) -> dict:
    sys_p = (
        "You are a clinical coding assistant for Saudi MOH death certificates. "
        "Extract structured medical information. "
        'Return ONLY valid JSON: {"immediate_cause":"string","contributing_causes":["list"],'
        '"other_conditions":["list"],"intervals":{"immediate_cause":"hours/days",'
        '"contributing_causes":["years"]},"notes":"string"}. No markdown, no extra text.'
    )
    raw = call_claude(sys_p, f"Patient: {age}yo, {gender}.\n\n{free_text}", max_tokens=700)
    raw = re.sub(r"^```json?\s*|\s*```$", "", raw).strip()
    return json.loads(raw)

def get_recommendation(concepts: dict, candidates: dict, gender: str, age: str, extra: str) -> str:
    sys_p = (
        "You are a senior clinical coding specialist at Saudi MOH. "
        "Structure your Arabic response with exactly these six numbered sections:\n\n"
        "(1) السبب المباشر للوفاة\n"
        "Write one ICD code per line in this exact format: CODE - description\n"
        "Then one short justification line starting with *\n\n"
        "(2) الأسباب المساهمة\n"
        "Each cause: CODE - description, then one bullet *\n\n"
        "(3) حالات أخرى مساهمة\n"
        "Each: CODE - description\n\n"
        "(4) التحقق من صحة العمر والجنس\n"
        "One short sentence.\n\n"
        "(5) قواعد التشفير WHO/MOH\n"
        "Two or three bullets starting with *\n\n"
        "(6) الترميز النهائي للإحصائيات\n"
        "One line: السبب الأساسي: CODE\n\n"
        "STRICT RULES: No markdown bold (**), no emoji, no headers with #. "
        "ICD codes always in format LETTER+DIGITS+DOT+DIGITS (e.g. I21.0, E11.9). "
        "Keep each section brief."
    )
    user = (
        f"Patient: {age}yo, {gender}\n{extra}\n\n"
        f"Concepts:\n{json.dumps(concepts, ensure_ascii=False, indent=2)}\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )
    return call_claude(sys_p, user, max_tokens=1200)

# ── Step bar ──────────────────────────────────────────────────────────────────
def render_steps(current: int):
    labels = ["البيانات الأساسية", "السيرة المرضية", "أسباب الوفاة", "المراجعة والتوصيات"]
    html   = '<div class="step-bar">'
    for i, lbl in enumerate(labels, 1):
        cls = "step active" if i == current else ("step done" if i < current else "step")
        html += f'<div class="{cls}">{i}. {lbl}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
#  PAGE 1 — Patient Basic Information
# =============================================================================
if st.session_state.page == 1:
    render_steps(1)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">البيانات الأساسية للمتوفى / Patient Basic Information</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        full_name_ar = st.text_input("الاسم الكامل (عربي)*", placeholder="محمد بن عبدالله العتيبي")
        full_name_en = st.text_input("Full Name (English)",   placeholder="Mohammed Abdullah Al-Otaibi")
        national_id  = st.text_input("رقم الهوية / الإقامة*", placeholder="1XXXXXXXXX")
        nationality  = st.text_input("الجنسية / Nationality", placeholder="سعودي")
        dob = st.date_input("تاريخ الميلاد / Date of Birth",
                            value=datetime.date(1960, 1, 1),
                            min_value=datetime.date(1900, 1, 1),
                            max_value=datetime.date.today())
    with c2:
        sex            = st.selectbox("الجنس / Sex*", ["ذكر / Male", "أنثى / Female"])
        marital_status = st.selectbox("الحالة الاجتماعية",
                                      ["أعزب / Single","متزوج / Married","مطلق / Divorced","أرمل / Widowed"])
        education      = st.selectbox("المستوى التعليمي / Education",
                                      ["أمي / Illiterate","ابتدائي / Primary","متوسط / Intermediate",
                                       "ثانوي / Secondary","دبلوم / Diploma","بكالوريوس / Bachelor",
                                       "دراسات عليا / Postgraduate","غير معلوم / Unknown"])
        occupation     = st.text_input("المهنة / Occupation", placeholder="مهندس")
        address        = st.text_input("العنوان / Address",   placeholder="الرياض – حي العليا")

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        dod           = st.date_input("تاريخ الوفاة / Date of Death*", value=datetime.date.today())
        time_of_death = st.time_input("وقت الوفاة / Time of Death")
        place_of_death = st.selectbox("مكان الوفاة / Place of Death",
                                      ["المستشفى / Hospital","الطوارئ / Emergency",
                                       "المنزل / Home","الطريق / Road","غير معلوم / Unknown"])
    with c4:
        cert_number = st.text_input("رقم الشهادة / Certificate No.", placeholder="DC-2025-XXXXX")
        date_issued = st.date_input("تاريخ الإصدار / Issue Date", value=datetime.date.today())

    st.markdown('</div>', unsafe_allow_html=True)

    age_years = max(0, (dod - dob).days // 365) if dob and dod else 0
    st.info(f"العمر عند الوفاة: {age_years} سنة  /  Age at Death: {age_years} years")

    if st.button("التالي"):
        if not full_name_ar or not national_id:
            st.error("يرجى إدخال الاسم الكامل ورقم الهوية.")
        else:
            st.session_state.form_data.update({
                "full_name_ar": full_name_ar, "full_name_en": full_name_en,
                "national_id":  national_id,  "nationality":  nationality,
                "dob": str(dob), "dod": str(dod),
                "time_of_death": str(time_of_death), "place_of_death": place_of_death,
                "sex": sex, "marital_status": marital_status,
                "education": education, "occupation": occupation, "address": address,
                "age_years": age_years, "cert_number": cert_number, "date_issued": str(date_issued),
            })
            st.session_state.page = 2
            st.rerun()

# =============================================================================
#  PAGE 2 — Medical History
# =============================================================================
elif st.session_state.page == 2:
    render_steps(2)
    fd = st.session_state.form_data

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">السيرة المرضية / Medical History</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        had_surgery = st.radio(
            "هل أجرى المريض عملية جراحية خلال آخر شهر؟ / Surgery within last month?",
            ["لا / No", "نعم / Yes", "غير معلوم / Unknown"],
        )
        surgery_details = ""
        if had_surgery == "نعم / Yes":
            surgery_details = st.text_area("تفاصيل العملية / Surgery Details",
                                           placeholder="نوع العملية، التاريخ، المستشفى")
        autopsy_required = st.radio(
            "هل التشريح مطلوب من وجهة نظر الطبيب؟ / Autopsy required (doctor)?",
            ["لا / No", "نعم / Yes", "غير محدد / Undetermined"],
        )
        autopsy_reason = ""
        if autopsy_required == "نعم / Yes":
            autopsy_reason = st.text_input("سبب طلب التشريح / Reason for autopsy")

    with c2:
        death_type     = st.selectbox("نوع الوفاة / Type of Death",
                                      ["طبيعية / Natural","حادث / Accident",
                                       "انتحار / Suicide","قتل / Homicide","غير محدد / Undetermined"])
        inpatient_days = st.number_input("مدة الإقامة بالمستشفى (أيام) / Hospital Stay (days)",
                                         min_value=0, value=0)
        was_pregnant = "لا ينطبق / N/A"
        if "أنثى" in fd.get("sex", ""):
            was_pregnant = st.selectbox("الحالة أثناء الوفاة / Pregnancy Status",
                                        ["لا / No","حامل / Pregnant",
                                         "أثناء الولادة / During delivery",
                                         "بعد الولادة 42 يوم / Within 42 days postpartum","لا ينطبق / N/A"])
        chronic_conditions = st.multiselect(
            "الأمراض المزمنة المعروفة / Known Chronic Conditions",
            ["داء السكري / Diabetes","ارتفاع ضغط الدم / Hypertension",
             "أمراض القلب / Heart Disease","أمراض الكلى / Renal Disease",
             "أمراض الكبد / Liver Disease","السرطان / Cancer",
             "أمراض الرئة / Pulmonary Disease","السمنة / Obesity",
             "أمراض عصبية / Neurological","أخرى / Other"],
        )

    st.markdown('</div>', unsafe_allow_html=True)

    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        if st.button("السابق", use_container_width=True):
            st.session_state.page = 1; st.rerun()
    with b2:
        if st.button("التالي", use_container_width=True):
            st.session_state.form_data.update({
                "had_surgery": had_surgery, "surgery_details": surgery_details,
                "autopsy_required": autopsy_required, "autopsy_reason": autopsy_reason,
                "death_type": death_type, "inpatient_days": inpatient_days,
                "was_pregnant": was_pregnant, "chronic_conditions": chronic_conditions,
            })
            st.session_state.page = 3; st.rerun()

# =============================================================================
#  PAGE 3 — Free-text causes of death
# =============================================================================
elif st.session_state.page == 3:
    render_steps(3)
    fd = st.session_state.form_data

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">أسباب الوفاة / Causes of Death</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#fffbe6;border-right:3px solid #C8A951;padding:9px 14px;'
        'border-radius:5px;margin-bottom:1.1rem;font-size:.87rem;color:#5a4a00">'
        'اكتب وصفاً سردياً لأسباب الوفاة — السبب الفوري أولاً ثم الأسباب الكامنة والمدد الزمنية. '
        'النظام سيستخرج الأسباب والفترات تلقائياً.</div>',
        unsafe_allow_html=True)

    free_text = st.text_area(
        "وصف أسباب الوفاة / Narrative description",
        value=fd.get("free_text", ""),
        height=200,
        placeholder=(
            "مثال: توفي المريض إثر فشل قلبي حاد (27 يوماً) ناتج عن احتشاء عضلة القلب الحاد "
            "في الجدار الأمامي (15 سنة) بسبب تصلب الشرايين التاجية. "
            "كان يعاني من داء السكري من النوع الثاني وارتفاع ضغط الدم المزمن."
        ),
    )
    st.markdown('</div>', unsafe_allow_html=True)

    b1, b2, _ = st.columns([1, 1.4, 6])
    with b1:
        if st.button("السابق", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
    with b2:
        if st.button("تحليل وترميز", use_container_width=True, type="primary"):
            if not free_text.strip():
                st.error("يرجى كتابة وصف أسباب الوفاة.")
            elif st.session_state.df_icd is None:
                st.error("بيانات ICD-10 لم تُحمَّل بعد.")
            else:
                st.session_state.form_data["free_text"] = free_text
                st.session_state.icd_results = None
                st.session_state.page = 4
                st.rerun()

# =============================================================================
#  PAGE 4 — AI Analysis: 2-step pipeline then results
# =============================================================================
elif st.session_state.page == 4:
    render_steps(4)
    fd     = st.session_state.form_data
    df_icd = st.session_state.df_icd
    fidx   = st.session_state.faiss_index

    # Guard: must have ICD data and free text
    if df_icd is None:
        st.error("بيانات ICD-10 لم تُحمَّل. ارجع وانتظر تحميل البيانات من الشريط الجانبي.")
        if st.button("السابق"):
            st.session_state.page = 3; st.rerun()
        st.stop()

    if not fd.get("free_text","").strip():
        st.error("لا يوجد نص لأسباب الوفاة.")
        if st.button("السابق"):
            st.session_state.page = 3; st.rerun()
        st.stop()

    gender_str = "Male" if "ذكر" in fd.get("sex","") else "Female"
    age_str    = str(fd.get("age_years","Unknown"))
    extra_ctx  = (
        "Surgery last month: " + fd.get("had_surgery","No") + "\n"
        + "Chronic conditions: " + ", ".join(fd.get("chronic_conditions",[])) + "\n"
        + "Death type: " + fd.get("death_type","Natural")
    )

    # ── Load full Excel from Google Sheets (cached) ───────────────────────────
    SHEETS_ID = "179CylYJwn2O6AdToCY4EcbeHp-kW_wGu"

    @st.cache_resource(show_spinner="تحميل بيانات ICD-10 الكاملة من Google Sheets...")
    def _load_excel_sheets(sheet_id: str) -> pd.DataFrame:
        import requests
        url = "https://docs.google.com/spreadsheets/d/" + sheet_id + "/export?format=xlsx"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content))
        cols = ["Id","Code","CodeFormatted","ShortDesc","LongDesc",
                "HIPPA","Deleted","Classification","AcceptableMain",
                "GenderRestriction","MatchSource","MatchedFromCode","Note"]
        if len(df.columns) == len(cols):
            df.columns = cols
        if "Deleted" in df.columns:
            df = df[df["Deleted"] != "Yes"].copy()
        df = df.dropna(subset=["Code"]).reset_index(drop=True)
        for col in ["Code","CodeFormatted","ShortDesc","LongDesc",
                    "AcceptableMain","GenderRestriction","Classification","Note"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        return df

    df_excel = None
    try:
        df_excel = _load_excel_sheets(SHEETS_ID)
    except Exception as ex_err:
        st.warning("تعذّر تحميل Excel من Sheets: " + str(ex_err) + " — سيُستخدم metadata.")
        df_excel = df_icd

    def get_excel_row(df: pd.DataFrame, code_str: str) -> dict:
        if df is None or not code_str:
            return {}
        c = code_str.strip().upper().replace(" ","")
        for col in ["CodeFormatted","Code"]:
            if col not in df.columns:
                continue
            mask = df[col].str.upper().str.replace(" ","",regex=False) == c
            if mask.any():
                row = df[mask].iloc[0]
                return {k: str(v) for k, v in row.to_dict().items()}
        return {}

    # ── 2-step pipeline (runs once, result stored in session_state) ───────────
    if st.session_state.icd_results is None:
        prog = st.empty()

        # ── STEP 1: extract causes & intervals ───────────────────────────────
        prog.info("⏳ الخطوة 1 من 2 — استخراج الأسباب والفترات الزمنية من النص...")
        try:
            s1_sys = (
                "You are a clinical coding assistant for Saudi MOH death certificates. "
                "Extract ALL causes of death with their time intervals from the doctor narrative. "
                "Return ONLY valid JSON with no markdown fences:\n"
                "{"
                "\"immediate_cause\": \"string\","
                "\"contributing_causes\": [\"string\", ...],"
                "\"other_conditions\": [\"string\", ...],"
                "\"intervals\": {"
                "  \"immediate_cause\": \"e.g. 27 days\"," 
                "  \"contributing_causes\": [\"e.g. 15 years\", ...]"
                "}"
                "}"
            )
            s1_user = (
                "Patient: " + age_str + "yo, " + gender_str + ".\n"
                + extra_ctx + "\n\nDoctor narrative:\n" + fd["free_text"]
            )
            raw1 = call_claude(s1_sys, s1_user, max_tokens=700)
            raw1 = re.sub(r"^```json?\s*|\s*```$","",raw1).strip()
            concepts = json.loads(raw1)
        except Exception as e1:
            prog.empty()
            st.error("فشلت الخطوة 1 (استخراج الأسباب): " + str(e1))
            st.stop()

        # Build cause list
        ivs         = concepts.get("intervals",{}) or {}
        contrib_ivs = ivs.get("contributing_causes",[]) or []
        all_causes  = []
        if concepts.get("immediate_cause","").strip():
            all_causes.append({
                "role":"immediate","label":"السبب الفوري",
                "cause": concepts["immediate_cause"],
                "interval": ivs.get("immediate_cause","—"),
            })
        for ci,c in enumerate(concepts.get("contributing_causes") or []):
            if c.strip():
                all_causes.append({
                    "role":"contributing",
                    "label":"سبب مساهم " + str(ci+1),
                    "cause": c,
                    "interval": contrib_ivs[ci] if ci < len(contrib_ivs) else "—",
                })
        for c in (concepts.get("other_conditions") or []):
            if c.strip():
                all_causes.append({
                    "role":"other","label":"حالة أخرى",
                    "cause": c,"interval":"—",
                })

        if not all_causes:
            prog.empty()
            st.error("لم يتمكن النظام من استخراج أي أسباب وفاة من النص. حاول إعادة الصياغة.")
            st.stop()

        # ── STEP 2: per cause → search → fetch Excel row → Claude ────────────
        s2_sys = (
            "You are a senior clinical coding specialist at Saudi MOH. "
            "Select the BEST ICD-10 code for the given cause of death "
            "using the top-5 candidate rows from the official ICD-10 database. "
            "Return ONLY valid JSON with no markdown fences:\n"
            "{"
            "\"code_formatted\": \"e.g. I21.0\"," 
            "\"short_desc\": \"exact ShortDesc from the chosen row\"," 
            "\"long_desc\": \"exact LongDesc from the chosen row\"," 
            "\"acceptable_main\": \"Acceptable or Not Acceptable\"," 
            "\"notes\": \"Arabic: why this code was chosen, any warnings (unacceptable, gender mismatch, suggest better alternative)\""
            "}"
        )

        coded_causes = []
        errors       = []
        for ci, cause_item in enumerate(all_causes):
            prog.info(
                "⏳ الخطوة 2 من 2 — تقييم السبب "
                + str(ci+1) + "/" + str(len(all_causes))
                + ": " + cause_item["cause"][:50] + "..."
            )
            try:
                hits      = search_icd(df_icd, fidx, cause_item["cause"], top_k=5)
                full_rows = []
                for h in hits:
                    row_data = get_excel_row(df_excel, h.get("code_formatted",""))
                    full_rows.append(row_data if row_data else h)

                s2_user = (
                    "Patient: " + age_str + "yo, " + gender_str + "\n"
                    + "Role: " + cause_item["role"]
                    + " | Interval: " + cause_item["interval"] + "\n"
                    + "Cause of death: " + cause_item["cause"] + "\n\n"
                    + "Top-5 ICD-10 candidates from database:\n"
                    + json.dumps(full_rows, ensure_ascii=False, indent=2)
                )
                raw2 = call_claude(s2_sys, s2_user, max_tokens=500)
                raw2 = re.sub(r"^```json?\s*|\s*```$","",raw2).strip()
                result = json.loads(raw2)
            except Exception as e2:
                errors.append("سبب " + str(ci+1) + ": " + str(e2))
                result = {
                    "code_formatted":"",
                    "short_desc": cause_item["cause"],
                    "long_desc":"",
                    "acceptable_main":"Unknown",
                    "notes":"فشل التحليل: " + str(e2),
                }

            result["role"]     = cause_item["role"]
            result["label"]    = cause_item["label"]
            result["cause"]    = cause_item["cause"]
            result["interval"] = cause_item["interval"]
            coded_causes.append(result)

        prog.empty()
        if errors:
            st.warning("بعض الأسباب لم تُحلَّل بشكل كامل:\n" + "\n".join(errors))

        st.session_state.icd_results = {
            "concepts":     concepts,
            "coded_causes": coded_causes,
        }
        st.rerun()

    # ── Display results ───────────────────────────────────────────────────────
    coded_causes = st.session_state.icd_results["coded_causes"]
    concepts     = st.session_state.icd_results["concepts"]

    tab_res, tab_cert = st.tabs(["رموز ICD-10 والتوصيات", "معاينة الشهادة"])

    with tab_res:
        st.markdown(
            '<div style="font-size:.8rem;color:var(--muted);margin-bottom:1.2rem">'            'كل رمز مستخرج من قاعدة ICD-10 الرسمية عبر البحث الدلالي + Claude. '            'رمز ICD-10 قابل للتعديل — باقي الحقول للعرض فقط.</div>',
            unsafe_allow_html=True)

        role_hdr = {"immediate":"#006940","contributing":"#2d7a4f","other":"#5a7060"}

        for idx, item in enumerate(coded_causes):
            acc    = item.get("acceptable_main","")
            bg_acc = "#006940" if acc=="Acceptable" else ("#c0392b" if acc else "#888")
            acc_ar = "✓ مقبول كسبب رئيسي" if acc=="Acceptable" else                      ("✗ غير مقبول كسبب رئيسي" if acc else "غير محدد")
            rc = role_hdr.get(item["role"],"#555")

            # Header bar (dark green = role)
            st.markdown(
                '<div style="border:1.5px solid #c8dece;border-radius:8px;'                'margin-bottom:1.4rem;overflow:hidden">'                '<div style="background:' + rc + ';color:white;padding:.5rem 1rem;'                'font-size:.84rem;font-weight:700">'                + item["label"] + ' — ' + item["cause"]                + ' <span style="opacity:.75;font-weight:400"> | الفترة: '                + item["interval"] + '</span></div>',
                unsafe_allow_html=True)

            # 4 columns
            c1, c2, c3, c4 = st.columns([1.1, 1.7, 2.7, 2.5])

            with c1:
                new_code = st.text_input(
                    "رمز ICD-10",
                    value=item.get("code_formatted",""),
                    key="code_" + str(idx),
                )
                st.markdown(
                    '<span style="background:' + bg_acc + ';color:white;'                    'border-radius:4px;padding:2px 9px;font-size:.72rem;font-weight:700">'                    + acc_ar + '</span>',
                    unsafe_allow_html=True)

            with c2:
                st.text_input(
                    "اسم المرض",
                    value=item.get("short_desc",""),
                    key="short_" + str(idx),
                    disabled=True,
                )

            with c3:
                st.text_area(
                    "الوصف التفصيلي",
                    value=item.get("long_desc",""),
                    key="long_" + str(idx),
                    height=105,
                    disabled=True,
                )

            with c4:
                notes_val = item.get("notes","")
                notes_bg  = "#fff5f5" if acc != "Acceptable" else "#f0faf4"
                notes_border = "#e8b4b8" if acc != "Acceptable" else "#9ecaad"
                st.markdown(
                    '<div style="background:' + notes_bg + ';border:1px solid '                    + notes_border + ';border-radius:6px;padding:.6rem .8rem;'                    'font-size:.82rem;color:#1a2e1a;min-height:105px;line-height:1.65">'                    '<div style="font-weight:700;color:var(--green);margin-bottom:.35rem">'                    'ملاحظات الترميز</div>'                    + notes_val + '</div>',
                    unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with tab_cert:
        cert_no  = (fd.get("cert_number")
                    or "DC-" + str(datetime.date.today().year)
                    + "-" + fd.get("national_id","")[-4:])
        imm_item = next((x for x in coded_causes if x["role"]=="immediate"),{})

        cont_rows = ""
        let_map   = ["ب","ج","د","هـ"]
        for ci,x in enumerate(x for x in coded_causes if x["role"]=="contributing"):
            cont_rows += (
                '<div class="cert-field">'                '<span class="cert-label">(' + (let_map[ci] if ci<4 else "+") + ') مساهم</span>'                '<span>' + x["cause"] + ' — <b>' + x.get("code_formatted","") + '</b>'                + ' <span style="font-size:.78rem;color:#666">(' + x["interval"] + ')</span></span></div>')

        other_rows = "".join(
            '<div class="cert-field"><span class="cert-label">حالة أخرى</span>'            '<span>' + x["cause"] + ' — <b>' + x.get("code_formatted","") + '</b></span></div>'
            for x in coded_causes if x["role"]=="other")

        st.markdown(
            '<div class="cert-preview">'            '<div style="display:flex;justify-content:space-between;align-items:center;'            'border-bottom:2px solid var(--green);padding-bottom:1rem;margin-bottom:1.4rem">'            '<div>'            '<div style="font-size:.95rem;font-weight:700;color:var(--green)">المملكة العربية السعودية</div>'            '<div style="font-size:.8rem;color:var(--muted)">وزارة الصحة</div>'            '<div style="font-size:.75rem;color:#888">' + hospital_name + ' — ' + hospital_city + '</div></div>'            '<div style="text-align:center">'            '<div class="cert-title">شهادة الوفاة</div>'            '<div class="cert-sub">DEATH CERTIFICATE</div>'            '<div style="background:var(--green);color:white;border-radius:4px;padding:2px 10px;'            'font-size:.76rem;margin-top:5px;display:inline-block">رقم: ' + cert_no + '</div></div>'            '<div style="text-align:left">'            '<div style="font-size:.95rem;font-weight:700;color:var(--green)">Kingdom of Saudi Arabia</div>'            '<div style="font-size:.8rem;color:var(--muted)">Ministry of Health</div></div></div>'            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:2rem;margin-bottom:1.4rem">'            '<div>'            '<div class="cert-field"><span class="cert-label">الاسم</span><span>' + fd.get("full_name_ar","—") + '</span></div>'            '<div class="cert-field"><span class="cert-label">الهوية</span><span>' + fd.get("national_id","—") + '</span></div>'            '<div class="cert-field"><span class="cert-label">الجنس</span><span>' + fd.get("sex","—") + '</span></div>'            '<div class="cert-field"><span class="cert-label">العمر</span><span>' + str(fd.get("age_years","—")) + ' سنة</span></div>'            '</div><div>'            '<div class="cert-field"><span class="cert-label">تاريخ الوفاة</span><span>' + str(fd.get("dod","—")) + '</span></div>'            '<div class="cert-field"><span class="cert-label">مكان الوفاة</span><span>' + fd.get("place_of_death","—") + '</span></div>'            '<div class="cert-field"><span class="cert-label">نوع الوفاة</span><span>' + fd.get("death_type","—") + '</span></div>'            '</div></div>'            '<div style="background:var(--green-light);border-radius:6px;padding:1rem 1.2rem;'            'margin-bottom:1.4rem;border:1px solid #9ecaad">'            '<div style="font-weight:700;color:var(--green);margin-bottom:.6rem">أسباب الوفاة</div>'            '<div class="cert-field"><span class="cert-label">(أ) السبب الفوري</span>'            '<span>' + imm_item.get("cause","—") + ' — <b>' + imm_item.get("code_formatted","") + '</b>'            + ' <span style="font-size:.78rem;color:#666">(' + imm_item.get("interval","—") + ')</span></span></div>'            + cont_rows + other_rows            + '</div>'            '<div style="display:flex;justify-content:space-between;padding-top:1.2rem;border-top:1px solid #d0ddd2">'            '<div><div style="font-weight:700;color:var(--green);font-size:.85rem">الطبيب المُصدر</div>'            '<div>' + (doctor_name or "________________________________") + '</div>'            '<div style="font-size:.75rem;color:#888;margin-top:6px">التوقيع: _______________________</div></div>'            '<div class="cert-stamp">وزارة<br>الصحة<br>MOH<br>ختم رسمي</div>'            '<div style="text-align:left"><div style="font-weight:700;color:var(--green);font-size:.85rem">تاريخ الإصدار</div>'            '<div>' + str(fd.get("date_issued",datetime.date.today())) + '</div></div></div></div>',
            unsafe_allow_html=True)

        st.markdown("---")
        cert_lines = [
            "شهادة وفاة / DEATH CERTIFICATE", "رقم: " + cert_no, "="*55,
            "المستشفى: " + hospital_name + " – " + hospital_city, "",
            "الاسم: " + fd.get("full_name_ar",""),
            "الهوية: " + fd.get("national_id",""),
            "الجنس: " + fd.get("sex","") + "   العمر: " + str(fd.get("age_years","")) + " سنة",
            "تاريخ الوفاة: " + str(fd.get("dod","")),
            "مكان الوفاة: " + fd.get("place_of_death",""), "", "أسباب الوفاة:",
        ]
        for item in coded_causes:
            cert_lines.append(
                "  [" + item["label"] + "] " + item["cause"]
                + " | " + item.get("code_formatted","")
                + " | " + item.get("short_desc","")
                + " | الفترة: " + item["interval"])
            if item.get("notes"):
                cert_lines.append("    ملاحظة: " + item["notes"])
        cert_lines += ["","الطبيب: " + doctor_name,
                       "تاريخ الإصدار: " + str(fd.get("date_issued",""))]
        st.download_button(
            "تحميل ملخص الشهادة",
            data="\n".join(cert_lines).encode("utf-8"),
            file_name="death_cert_" + cert_no + ".txt",
            mime="text/plain",
        )

    st.markdown("---")
    b1, b2, _ = st.columns([1,1,6])
    with b1:
        if st.button("السابق", use_container_width=True):
            st.session_state.page = 3
            st.session_state.icd_results = None
            st.rerun()
    with b2:
        if st.button("شهادة جديدة", use_container_width=True):
            st.session_state.page        = 1
            st.session_state.form_data   = {}
            st.session_state.icd_results = None
            st.rerun()
