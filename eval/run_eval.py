"""Retrieval evaluation (offline) and optional generation evaluation (needs ANTHROPIC_API_KEY).

Usage:
  python eval/run_eval.py              # retrieval: vector vs hybrid, two chunk sizes
  python eval/run_eval.py --generate   # + refusal / citation checks with the LLM
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import rag  # noqa: E402

QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
IN_SCOPE = [q for q in QUESTIONS if q["drug"]]
OUT_SCOPE = [q for q in QUESTIONS if not q["drug"]]


def retrieval_scores(retriever, k):
    drug_hit = sec_hit = 0
    for q in IN_SCOPE:
        hits = retriever.search(q["q"], k=k)
        if any(h.metadata["drug"] == q["drug"] for h in hits):
            drug_hit += 1
        if any(h.metadata["drug"] == q["drug"] and h.metadata["section"] == q["section"] for h in hits):
            sec_hit += 1
    n = len(IN_SCOPE)
    return drug_hit / n, sec_hit / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    args = ap.parse_args()

    results = []
    for chunk_size in (600, 1200):
        store, docs = rag.build_index(chunk_size, chunk_size // 8,
                                      persist=ROOT / "data" / f"chroma_eval_{chunk_size}")
        for hybrid in (False, True):
            r = rag.Retriever(store, docs, hybrid=hybrid)
            for k in (3, 5):
                d, s = retrieval_scores(r, k)
                results.append({"chunk_size": chunk_size, "hybrid": hybrid, "k": k,
                                "chunks": len(docs), "hit_drug": round(d, 3), "hit_drug_section": round(s, 3)})
                print(results[-1])
    (ROOT / "eval" / "retrieval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.generate:
        if not rag.llm_available():
            sys.exit("Set GOOGLE_API_KEY or ANTHROPIC_API_KEY for --generate")
        best = max(results, key=lambda x: (x["hit_drug_section"], x["hybrid"]))
        store, docs = rag.build_index(best["chunk_size"], best["chunk_size"] // 8,
                                      persist=ROOT / "data" / f"chroma_eval_{best['chunk_size']}")
        r = rag.Retriever(store, docs, hybrid=best["hybrid"])
        cited = 0
        for q in IN_SCOPE:
            out = rag.answer(q["q"], r, k=best["k"])
            if not out["refused"] and re.search(r"\[\d+\]", out["answer"]):
                cited += 1
        refused = sum(rag.answer(q["q"], r, k=best["k"])["refused"] for q in OUT_SCOPE)
        gen = {"config": best, "answers_with_citations": f"{cited}/{len(IN_SCOPE)}",
               "correct_refusals_out_of_scope": f"{refused}/{len(OUT_SCOPE)}"}
        print(gen)
        (ROOT / "eval" / "generation_results.json").write_text(json.dumps(gen, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
