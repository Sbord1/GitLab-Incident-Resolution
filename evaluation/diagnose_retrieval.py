"""Inspect where an expected KB document is lost in two-stage retrieval.

Place at evaluation/diagnose_retrieval.py, then run from repository root:
    python -m evaluation.diagnose_retrieval --ids eval_002 eval_003 eval_004

No OpenAI calls. Requires the local PostgreSQL/pgvector DB and embedding/reranker.
"""
import argparse
import json
from pathlib import Path

from src import hybrid_query as hq

HERE = Path(__file__).resolve().parent
EVAL_PATH = HERE / "datasets" / "evaluation_set.json"


def position(items, doc_id):
    return next((i for i, d in enumerate(items, 1) if d == doc_id), None)


def describe_evidence(items):
    return [
        {
            "doc_id": e["row"][1],
            "section": e["row"][4],
            "chunk_key": (e["row"][6] or {}).get("chunk_key"),
            "rerank_score": e.get("rerank_score"),
            "snippet": (e["row"][5] or "")[:160].replace("\n", " "),
        }
        for e in items
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", default=["eval_002", "eval_003", "eval_004"])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    chosen = [x for x in dataset if x["eval_id"] in args.ids]
    missing_eval_ids = sorted(set(args.ids) - {x["eval_id"] for x in chosen})
    if missing_eval_ids:
        raise ValueError(f"Unknown eval IDs: {missing_eval_ids}")

    report = []
    conn = hq.get_connection()
    try:
        for case in chosen:
            question = case["question"]
            expected = case["expected_source_doc_ids"]
            vector_rows, text_rows = hq.retrieve(question, limit=hq.CANDIDATES, conn=conn)
            vector_docs = list(hq._doc_arm(vector_rows))
            text_docs = list(hq._doc_arm(text_rows))
            print(f"\n=== {case['eval_id']}: {question} ===")
            case_result = {"eval_id": case["eval_id"], "question": question, "expected": expected, "modes": {}}
            for mode in hq.MODES:
                result = hq.two_stage(question, mode=mode, top_k=args.top_k,
                                      vector_rows=vector_rows, text_rows=text_rows)
                docs_stage1 = [d["doc_id"] for d in result["documents"]]
                docs_final = [d["doc_id"] for d in result["documents_final"]]
                evidence = describe_evidence(result["evidence"])
                evidence_doc_ids = [e["doc_id"] for e in evidence]
                ranks = {}
                for doc_id in expected:
                    ranks[doc_id] = {
                        "vector_doc_rank": position(vector_docs, doc_id),
                        "fulltext_doc_rank": position(text_docs, doc_id),
                        "stage1_doc_rank": position(docs_stage1, doc_id),
                        "stage1_top_n": doc_id in docs_stage1[:hq.DOC_TOP_N],
                        "final_doc_rank": position(docs_final, doc_id),
                        "in_final_evidence": doc_id in evidence_doc_ids,
                    }
                print(f"  {mode}: {json.dumps(ranks, ensure_ascii=False)}")
                print("    final evidence:", [(e["doc_id"], e["section"]) for e in evidence])
                case_result["modes"][mode] = {
                    "expected_ranks": ranks,
                    "evidence_candidates": result["evidence_candidates"],
                    "evidence": evidence,
                }
            report.append(case_result)
    finally:
        conn.close()

    out = HERE.parent / "data" / "processed" / "retrieval_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved diagnostics to {out}")


if __name__ == "__main__":
    main()
