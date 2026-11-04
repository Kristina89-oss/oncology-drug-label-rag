"""Index + retrieval + cited answering over FDA drug labels."""
import json
import os
import pathlib

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "claude-sonnet-5-5")

NOT_FOUND = "NOT_IN_SOURCES"

SYSTEM_PROMPT = f"""You answer questions about drug labels using ONLY the numbered sources provided.
Rules:
- Every factual sentence must end with citation markers like [1] or [2][3].
- If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
- Never use outside knowledge. Never give personal medical advice or dosing for a specific patient.
- Be concise. Keep drug names and doses exact."""


def llm_available():
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def get_llm():
    """Gemini (free tier) if GOOGLE_API_KEY is set, otherwise Claude."""
    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Google retires/overloads free-tier models often: try several in order.
        names = os.environ.get("RAG_GEMINI_MODELS",
                               "gemini-3-flash-preview,gemini-flash-latest,gemini-3.1-flash-lite").split(",")
        llms = [ChatGoogleGenerativeAI(model=n, temperature=0, max_output_tokens=3000,  # Gemini 3 "thinking" eats part of the budget

                                       timeout=40, max_retries=1) for n in names]
        return llms[0].with_fallbacks(llms[1:])
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=LLM_MODEL, temperature=0, max_tokens=700)


def embeddings():
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


def load_documents(chunk_size=1200, overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = []
    for f in sorted(RAW.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for section, text in d["sections"].items():
            for i, chunk in enumerate(splitter.split_text(text)):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"drug": d["drug"], "brand": d["brand"], "section": section,
                              "set_id": d["set_id"], "effective": d["effective_time"], "chunk": i},
                ))
    return docs


def build_index(chunk_size=1200, overlap=150, persist=None):
    import shutil
    import uuid
    docs = load_documents(chunk_size, overlap)
    path = pathlib.Path(persist or DB)
    if path.exists():  # never append to a stale index (duplicates / locked files on Windows)
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            path = path.with_name(f"{path.name}_{uuid.uuid4().hex[:6]}")
    store = Chroma.from_documents(docs, embeddings(), persist_directory=str(path))
    return store, docs


def open_index():
    return Chroma(persist_directory=str(DB), embedding_function=embeddings())


class Retriever:
    """Vector search, optionally fused with BM25 via reciprocal rank fusion."""

    def __init__(self, store, docs, hybrid=True, auto_filter=True):
        self.store, self.docs, self.hybrid, self.auto_filter = store, docs, hybrid, auto_filter
        if hybrid:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([d.page_content.lower().split() for d in docs])

    def mentioned_drugs(self, query):
        q = query.lower()
        return {d.metadata["drug"] for d in self.docs if d.metadata["drug"] in q}

    def search(self, query, k=5, drug=None, auto_filter=None):
        if auto_filter is None:
            auto_filter = self.auto_filter
        if drug is None and auto_filter:  # a drug named in the question restricts the search to it
            named = self.mentioned_drugs(query)
            if named:
                return [d for d in self.search(query, k=200, auto_filter=False)
                        if d.metadata["drug"] in named][:k]
        if drug:  # restrict to one drug: filter a wider candidate pool
            return [d for d in self.search(query, k=200, auto_filter=False)
                    if d.metadata["drug"] == drug][:k]
        vec = [d for d, _ in self.store.similarity_search_with_score(query, k=20 if k <= 20 else 200)]
        if not self.hybrid:
            return vec[:k]
        scores = self.bm25.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:20 if k <= 20 else 200]
        fused = {}
        key = lambda d: (d.metadata["set_id"], d.metadata["section"], d.metadata["chunk"])
        for rank, d in enumerate(vec):
            fused[key(d)] = fused.get(key(d), [d, 0])
            fused[key(d)][1] += 1 / (60 + rank)
        for rank, i in enumerate(top):
            d = self.docs[i]
            fused[key(d)] = fused.get(key(d), [d, 0])
            fused[key(d)][1] += 1 / (60 + rank)
        return [d for d, _ in sorted(fused.values(), key=lambda x: -x[1])[:k]]


def _text(content):
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content)


def answer(question, retriever, k=5, drug=None):
    """Returns dict(answer, sources, refused). Needs GOOGLE_API_KEY or ANTHROPIC_API_KEY."""
    llm = get_llm()
    hits = retriever.search(question, k=k, drug=drug)
    context = "\n\n".join(
        f"[{i+1}] ({h.metadata['drug']} - {h.metadata['section']})\n{h.page_content}"
        for i, h in enumerate(hits))
    out = llm.invoke([("system", SYSTEM_PROMPT),
                      ("human", f"Sources:\n{context}\n\nQuestion: {question}")]).content
    text = _text(out)
    refused = NOT_FOUND in text
    import time
    time.sleep(float(os.environ.get("RAG_SLEEP", "0")))  # stay under free-tier rate limits
    return {"answer": text.strip(), "sources": hits, "refused": refused}
