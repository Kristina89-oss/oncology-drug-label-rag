# Oncology drug-label Q&A (RAG)

Question answering over public FDA drug labels (openFDA, 18 oncology drugs). Every answer carries numbered
citations to the label passages it came from, and the assistant **refuses to answer when the labels do not
contain the answer**. Demo project on public data; not medical advice.

**Stack:** LangChain, Chroma, FastEmbed (`BAAI/bge-small-en-v1.5`, runs locally), BM25 + vector hybrid search,
Gemini (free tier) or Claude for generation, Streamlit UI, FastAPI + Docker service, n8n chat workflow.

## How it works
1. `src/download_data.py` pulls label sections (indications, dosing, warnings, interactions...) from openFDA.
2. `src/rag.py` splits sections into chunks with metadata (drug, section, label date) and indexes them.
3. Retrieval: vector top-20 fused with BM25 top-20 (reciprocal rank fusion). If the question names a drug,
   the search is restricted to that drug's labels.
4. The model must cite every sentence as `[n]` or reply `NOT_IN_SOURCES`, which the UI shows as "no answer in the labels".

## Evaluation
28 hand-written questions in `eval/questions.json`: 24 answerable from the labels (each with the expected drug
and label section) and 4 that must be refused (off-topic or personal medical advice). The questions were written
by one person from the label structure, so treat the numbers as a small smoke-test benchmark, not a clinical validation.

**Retrieval** (right drug and right section among the top-k chunks; `eval/retrieval_results.json`):

| chunk size | search | k=3 | k=5 |
|---|---|---|---|
| 600 | vector | 0.708 | 0.875 |
| 600 | hybrid | 0.875 | 0.875 |
| 1200 | vector | 0.750 | **0.917** |
| 1200 | hybrid | 0.833 | 0.875 |

With 24 questions one question is 4 points, so vector and hybrid are not clearly different here.
The right drug is found in 100% of cases because every question names a drug and the search is then
restricted to it. Effect of that drug-name filter alone (`eval/compare_drug_filter.py`, hybrid, 1200-char chunks, k=5):
section hit rate 0.792 -> 0.875, right drug in top-5 0.917 -> 1.0.

**Generation** (`python eval/run_eval.py --generate`, best retrieval setting above, Gemini; `eval/generation_results.json`):

| check | result |
|---|---|
| answers with a citation marker, in-scope questions | 22 / 24 |
| wrongly refused, in-scope questions | 2 / 24 |
| correctly refused, off-topic or personal-advice questions | 3 / 4 (1 call failed on a rate limit, not counted) |

"Citation marker" only checks that `[n]` appears in the answer, not that the cited passage actually supports the
sentence; that needs manual review. The run used vector-only search (the best setting above), while the app
defaults to hybrid search. Gemini free tier rate-limits requests, so the script retries with back-off and
reports failed calls instead of hiding them.

Known limits: some expected sections do not exist in a given label (for example no contraindications list), which
the questions do not account for; labels only (no clinical guidelines); answers are not checked for factual
correctness by a clinician.

## Run locally
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py          # data/raw is already in the repo
echo GOOGLE_API_KEY=your_key > .env  # free key from Google AI Studio
streamlit run app.py
```
`python eval/run_eval.py` runs the retrieval benchmark offline; add `--generate` for the LLM checks.

## Run as an API next to n8n
`docker compose up -d --build` starts a FastAPI service (`POST /ask {"question": "..."}`) on the n8n Docker network
without a public port. `n8n/oncology-rag-chat.json` is an importable n8n workflow with a chat UI that calls it.
