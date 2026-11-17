"""
Compares three search methods on the evaluation set:

1. vector   : vector search only
2. fulltext : PostgreSQL full-text search only
3. hybrid   : vector + full-text combined using RRF

For each question, the two rankings (vector and full-text) are
computed only once and reused by all three methods, ensuring a fair
comparison and avoiding recalculating the query embedding.

Results are saved to evaluation_results.json.
"""

import json
from pathlib import Path

from src.hybrid_query import CANDIDATES, MODES, two_stage, get_connection, retrieve

EVAL_PATH = Path(__file__).resolve().parent / "datasets/evaluation_set.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data/processed/evaluation_results_v2.json"

K_VALUES = [1, 3, 5]

MODE_LABELS = {
    "vector": "Vector",
    "fulltext": "Full-text",
    "hybrid": "Hybrid (RRF)",
    "hybrid_reranked": "Hybrid+Rerank",
}


def dedup_doc_ids(results):
    """Deduplicate by doc_id, keeping the first occurrence."""
    seen = set()
    ranked = []
    for item in results:
        doc_id = item["row"][1]  # row: id, doc_id, doc_type, ...
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def first_rank(ranked_ids, expected):
    """Return the first (1-based) position of an expected document, or None."""
    return next(
        (i for i, d in enumerate(ranked_ids, start=1) if d in expected),
        None,
    )


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    n = len(eval_set)
    print(f"Loaded {n} evaluation questions.")
    print(f"Methods: {', '.join(MODE_LABELS[m] for m in MODES)}\n")

    hits_at_k = {m: {k: 0 for k in K_VALUES} for m in MODES}
    reciprocal_ranks = {m: [] for m in MODES}
    evidence_hits = {m: 0 for m in MODES}
    details = []

    conn = get_connection()
    try:
        for item in eval_set:
            question = item["question"]
            expected = set(item["expected_source_doc_ids"])

            vector_rows, text_rows = retrieve(
                question, limit=CANDIDATES, conn=conn
            )

            print(f"[{item['eval_id']}] {question}")
            print(f"   expected: {sorted(expected)}")

            per_mode = {}
            for mode in MODES:
                out = two_stage(
                    question,
                    mode=mode,
                    top_k=5,
                    vector_rows=vector_rows,
                    text_rows=text_rows
                )

                ranked_ids = [d["doc_id"] for d in out["documents_final"]]
                rank = first_rank(ranked_ids, expected)

                evidence_doc_ids = [e["row"][1] for e in out["evidence"]]
                evidence_hit = bool(expected.intersection(evidence_doc_ids))
                evidence_hits[mode] += evidence_hit

                reciprocal_ranks[mode].append(1 / rank if rank else 0)
                for k in K_VALUES:
                    if rank is not None and rank <= k:
                        hits_at_k[mode][k] += 1

                status = (
                    f"rank {rank}" if rank is not None
                    else "NOT FOUND"
                )
                print(f"   {MODE_LABELS[mode]:<13} {status:<16} "
                      f"top-3: {ranked_ids[:3]}")

                per_mode[mode] = {
                    "rank_found": rank,
                    "top_5": ranked_ids[:5],
                    "ranked_doc_ids": ranked_ids,
                    "evidence_hit": evidence_hit,
                }
            print()

            details.append({
                "eval_id": item["eval_id"],
                "question": question,
                "expected": sorted(expected),
                "methods": per_mode,
            })
    finally:
        conn.close()

    # --- Comparative summary ---
    print("=" * 66)
    print(f"AGGREGATE RESULTS ({n} questions, {CANDIDATES} candidate chunks)")
    print("=" * 66)

    header = f"{'Method':<14}" + "".join(
        f"{'Recall@' + str(k):>11}" for k in K_VALUES
    ) + f"{'MRR':>9}"
    print(header)
    print("-" * len(header))

    summary = {}
    for mode in MODES:
        recalls = {
            k: hits_at_k[mode][k] / n if n else 0 for k in K_VALUES
        }
        mrr = sum(reciprocal_ranks[mode]) / n if n else 0
        summary[mode] = {
            "recall_at_k": {str(k): recalls[k] for k in K_VALUES},
            "hits_at_k": {str(k): hits_at_k[mode][k] for k in K_VALUES},
            "mrr": mrr,
            "evidence_hit_at_5": evidence_hits[mode],
        }
        print(
            f"{MODE_LABELS[mode]:<14}"
            + "".join(f"{recalls[k]:>11.1%}" for k in K_VALUES)
            + f"{mrr:>9.3f}"
        )

    output = {
    "k_values": K_VALUES,
    "candidate_chunks": CANDIDATES,
    "n_questions": n,
    "summary": summary,
    "details": details,
    "config": {
        "vector_weight": two_stage.__globals__["VECTOR_WEIGHT"],
        "text_weight": two_stage.__globals__["TEXT_WEIGHT"],
        "rrf_k": two_stage.__globals__["RRF_K"],
        "doc_top_n": two_stage.__globals__["DOC_TOP_N"],
        "evidence_per_doc": two_stage.__globals__["EVIDENCE_PER_DOC"],
        "fts_mode": two_stage.__globals__["FTS_MODE"],
        "embedding_model": two_stage.__globals__["MODEL_NAME"],
        "reranker": two_stage.__globals__["RERANK_MODEL_NAME"],
    },
}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDetails saved to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()