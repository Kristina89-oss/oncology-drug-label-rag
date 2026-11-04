import json
import os
import pathlib
import sys

import streamlit as st

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import rag  # noqa: E402

# local key file (not committed): GOOGLE_API_KEY=...
env_file = HERE / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SECTION_NAMES = {
    "indications_and_usage": "Indications and usage",
    "dosage_and_administration": "Dosage and administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and precautions",
    "warnings": "Warnings",
    "adverse_reactions": "Adverse reactions",
    "drug_interactions": "Drug interactions",
    "use_in_specific_populations": "Use in specific populations",
    "overdosage": "Overdosage",
    "clinical_pharmacology": "Clinical pharmacology",
}
EXAMPLES = [
    "What premedication is required before docetaxel?",
    "What is the recommended tamoxifen dose for adjuvant breast cancer treatment?",
    "What are the contraindications of carboplatin?",
    "What are the neutropenia warnings for cabazitaxel?",
]

st.set_page_config(page_title="Oncology drug-label Q&A", page_icon="💊", layout="centered")
st.markdown("""
<style>
.block-container {max-width: 820px; padding-top: 2rem;}
.answer {background:#f3f8f6; border-left:4px solid #2e8b74; padding:1rem 1.2rem; border-radius:6px;}
.refuse {background:#fff6e5; border-left:4px solid #e0a100; padding:1rem 1.2rem; border-radius:6px;}
.small {font-size:0.85rem; color:#666;}
@media (prefers-color-scheme: dark) {.answer{background:#1c2d28}.refuse{background:#33290f}.small{color:#aaa}}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the label index...")
def get_retriever():
    store = rag.open_index() if rag.DB.exists() else rag.build_index()[0]
    return rag.Retriever(store, rag.load_documents(), hybrid=True)


@st.cache_data
def drug_list():
    out = []
    for f in sorted(rag.RAW.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append((d["drug"], d["brand"], str(d["effective_time"])))
    return out


drugs = drug_list()

with st.sidebar:
    st.header("About")
    st.write("Answers **only** from official FDA drug labels and always shows where the answer comes from.")
    chosen = st.selectbox("Search within one drug", ["All drugs"] + [d[0] for d in drugs])
    k = st.slider("Passages to retrieve", 3, 8, 5)
    with st.expander(f"{len(drugs)} drugs in the index"):
        for name, brand, eff in drugs:
            st.write(f"• {name} ({brand.title() if brand else '-'}), label {eff[:4]}")
    st.caption("Demo project on public data. Not a substitute for clinical judgement.")

st.title("💊 Oncology drug-label Q&A")
st.write("Ask about dosing, indications, contraindications, warnings or interactions.")

if "q" not in st.session_state:
    st.session_state.q = ""
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.q = ex

with st.form("ask"):
    question = st.text_area("Your question", value=st.session_state.q, height=80,
                            placeholder="e.g. What is the recommended dose of abiraterone acetate?")
    submitted = st.form_submit_button("Get answer", type="primary", use_container_width=True)

if not rag.llm_available():
    st.error("GOOGLE_API_KEY is not set (put it in a .env file next to app.py).")
elif submitted and question.strip():
    st.session_state.q = question
    with st.spinner("Searching the labels..."):
        try:
            out = rag.answer(question.strip(), get_retriever(), k=k,
                             drug=None if chosen == "All drugs" else chosen)
        except Exception as e:  # network / quota problems should not look like a crash
            st.error(f"Could not get an answer ({type(e).__name__}). Please try again in a minute.")
            st.stop()
    if out["refused"]:
        st.markdown('<div class="refuse"><b>The loaded labels do not contain an answer.</b><br>'
                    'The assistant does not answer without a source. Try rephrasing or choose another drug.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="answer">{out["answer"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
    st.subheader("Sources")
    for i, h in enumerate(out["sources"], 1):
        m = h.metadata
        title = f"[{i}] {m['drug']} - {SECTION_NAMES.get(m['section'], m['section'])} (label {str(m['effective'])[:4]})"
        with st.expander(title):
            st.write(h.page_content)
    st.markdown('<p class="small">Check doses against the full label. Demo on public FDA data.</p>',
                unsafe_allow_html=True)
