"""Offline retrieval check: does filtering by a drug named in the question help?

Usage: python eval/compare_drug_filter.py   (no API key needed)
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import rag  # noqa: E402

QUESTIONS = [q for q in json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8")) if q["drug"]]
store, docs = rag.build_index(1200, 150, persist=ROOT / "data" / "chroma_eval_filter")

results = []
for auto in (False, True):
    r = rag.Retriever(store, docs, hybrid=True, auto_filter=auto)
    for k in (3, 5):
        drug_hit = sec_hit = 0
        for q in QUESTIONS:
            hits = r.search(q["q"], k=k)
            drug_hit += any(h.metadata["drug"] == q["drug"] for h in hits)
            sec_hit += any(h.metadata["drug"] == q["drug"] and h.metadata["section"] == q["section"] for h in hits)
        n = len(QUESTIONS)
        results.append({"auto_drug_filter": auto, "k": k, "hit_drug": round(drug_hit / n, 3),
                        "hit_drug_section": round(sec_hit / n, 3)})
        print(results[-1])
(ROOT / "eval" / "drug_filter_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
