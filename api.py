"""HTTP wrapper around the RAG pipeline, meant to be called from n8n.

POST /ask  {"question": "...", "k": 5}  ->  {answer, refused, text, sources[]}
GET  /health
Optional auth: set RAG_API_TOKEN and send header X-API-Key.
"""
import os
import sys
import pathlib

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import rag  # noqa: E402

app = FastAPI(title="Oncology drug-label RAG")
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        store = rag.open_index() if rag.DB.exists() else rag.build_index()[0]
        _retriever = rag.Retriever(store, rag.load_documents(), hybrid=True)
    return _retriever


def check_token(x_api_key: str = Header(default="")):
    token = os.environ.get("RAG_API_TOKEN")
    if token and x_api_key != token:
        raise HTTPException(status_code=401, detail="bad token")


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


@app.on_event("startup")
def warm_up():
    get_retriever()


@app.get("/health")
def health():
    return {"ok": True, "llm": rag.llm_available()}


@app.post("/ask", dependencies=[Depends(check_token)])
def ask(body: Ask):
    r = get_retriever()
    if not rag.llm_available():
        raise HTTPException(status_code=503, detail="no LLM key configured")
    out = rag.answer(body.question, r, k=body.k)
    sources = [{"n": i + 1, "drug": h.metadata["drug"], "section": h.metadata["section"],
                "effective": h.metadata["effective"], "text": h.page_content}
               for i, h in enumerate(out["sources"])]
    if out["refused"]:
        text = "The loaded labels do not contain an answer to this question."
    else:
        refs = "\n".join(f"[{s['n']}] {s['drug']} - {s['section']} (label {s['effective']})"
                         for s in sources)
        text = f"{out['answer']}\n\nSources:\n{refs}\n\nDemo on public FDA data, not medical advice."
    return {"answer": out["answer"], "refused": out["refused"], "text": text, "sources": sources}
