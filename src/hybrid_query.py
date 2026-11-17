"""
Two-stage retrieval in the PostgreSQL + pgvector knowledge base.

STAGE 1 - documents: vector and full-text retrieval return chunks; each
          ranking is converted into a ranking of distinct doc_ids
          using the BEST chunk for each document (not the sum); the two
          rankings are merged with RRF. The top DOC_TOP_N documents are kept.
STAGE 2 - evidence: within those documents, up to
          EVIDENCE_PER_DOC promising chunks are selected (based on retrieval
          rankings) and, in hybrid_reranked mode, reordered with the cross-encoder.

Modes: vector | fulltext | hybrid | hybrid_reranked
(the document stage uses the selected retrieval method; reranking only
affects the evidence).

API:
    rank_documents(...)   document ranking
    two_stage(...)        {"documents", "documents_final", "evidence", ...}
    search(...)           evidence for RAG (list of dicts with "row")
    search_flat(...)      OLD chunk-level behavior (baseline)

"""

import argparse
from collections import defaultdict
from typing import Optional

import psycopg2
from pgvector.psycopg2 import register_vector

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)
TOP_K = 5                 # evidence (chunks) passed to RAG
RRF_K = 60
CANDIDATES = 100          # chunks retrieved by each retrieval method
MODES = ("vector", "fulltext", "hybrid", "hybrid_reranked")
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 50    # only for the old "flat" mode

DOC_TOP_N = 10            # documents kept after stage 1
EVIDENCE_PER_DOC = 5      # candidate chunks per document (stage 2)


FTS_MODES = ("or", "and_or")
FTS_MODE = "and_or"
FTS_AND_MIN_RESULTS = 10

# HNSW is approximate:
HNSW_EF_SEARCH = 200
VECTOR_WEIGHT = 3
TEXT_WEIGHT = 1.0

DB_CONFIG = dict(
    host="localhost",
    port=5432,
    dbname="rag_kb",
    user="rag",
    password="rag_password",
)

_model = None
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


# ---------------------------------------------------------------------------
# Chunk retrieval: two independent rankings
# Rows keep the historical layout (compatible with rag.py):
#   (id, doc_id, doc_type, title, section_path, text, metadata, score)
# metadata is a dict: url, chunk_key, section_type, origin, chunk_index.
# ---------------------------------------------------------------------------

_CHUNK_COLUMNS = """
    c.id, c.doc_id, d.type, d.title, c.section_path, c.text,
    jsonb_build_object('url', d.url, 'chunk_key', c.chunk_key,
                       'section_type', c.section_type, 'origin', c.origin,
                       'chunk_index', c.chunk_index)
"""

_OR_TERMS = ("to_tsquery('english', regexp_replace("
             "plainto_tsquery('english', %(q)s)::text, ' & ', ' | ', 'g'))")
_AND_TERMS = "plainto_tsquery('english', %(q)s)"


def _fts_query(terms_sql, doc_filter):
    return f"""
        WITH q AS (SELECT {terms_sql} AS terms)
        SELECT {_CHUNK_COLUMNS}, ts_rank_cd(c.tsv, q.terms) AS text_score
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        CROSS JOIN q
        WHERE c.tsv @@ q.terms {doc_filter}
        ORDER BY text_score DESC, c.id
        LIMIT %(limit)s
    """


def retrieve(question: str, limit: int = CANDIDATES, conn=None,
             fts_mode: str = FTS_MODE, doc_ids=None, exact: bool = False):
    """Returns (vector_rows, text_rows), the two raw chunk rankings.

    doc_ids: if provided, limits the search to those documents (used for
             subset evaluation); forces an exact scan because filtering
             after the HNSW index would return too few rows.
    exact:   exact scan (without the HNSW index).
    """
    if fts_mode not in FTS_MODES:
        raise ValueError(f"fts_mode sconosciuto: {fts_mode!r} (attesi: {FTS_MODES})")

    query_embedding = get_model().encode(
        QUERY_PREFIX + question, normalize_embeddings=True,
    )

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    params = {"emb": query_embedding, "q": question, "limit": limit,
              "docs": list(doc_ids) if doc_ids is not None else None}
    where_docs = "WHERE c.doc_id = ANY(%(docs)s)" if doc_ids is not None else ""
    and_docs = "AND c.doc_id = ANY(%(docs)s)" if doc_ids is not None else ""

    try:
        with conn.cursor() as cur:
            if exact or doc_ids is not None:
                cur.execute("SET LOCAL enable_indexscan = off")
            ef = int(min(max(HNSW_EF_SEARCH, limit), 1000))
            cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")

            # 1. Vector ranking (the CTE allows the index to be used)
            cur.execute(
                f"""
                WITH top AS (
                    SELECT c.id, c.embedding <=> %(emb)s AS dist
                    FROM chunks c {where_docs}
                    ORDER BY c.embedding <=> %(emb)s
                    LIMIT %(limit)s
                )
                SELECT {_CHUNK_COLUMNS}, 1 - top.dist AS similarity
                FROM top
                JOIN chunks c ON c.id = top.id
                JOIN documents d ON d.doc_id = c.doc_id
                ORDER BY top.dist, c.id
                """,
                params,
            )
            vector_rows = cur.fetchall()

            # 2. Full-text ranking (weighted tsvector: title A, section B, text C)
            if fts_mode == "and_or":
                cur.execute(_fts_query(_AND_TERMS, and_docs), params)
                text_rows = cur.fetchall()
                if len(text_rows) < FTS_AND_MIN_RESULTS:
                    cur.execute(_fts_query(_OR_TERMS, and_docs), params)
                    text_rows = cur.fetchall()
            else:
                cur.execute(_fts_query(_OR_TERMS, and_docs), params)
                text_rows = cur.fetchall()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()

    return vector_rows, text_rows


# ---------------------------------------------------------------------------
# CHUNK level ("flat", historical behavior): unchanged, used as baseline
# ---------------------------------------------------------------------------

def vector_results(vector_rows):
    return [
        {
            "row": row,
            "rrf_score": None,
            "vector_rank": rank,
            "text_rank": None,
            "vector_score": row[7],
            "text_score": None,
        }
        for rank, row in enumerate(vector_rows, start=1)
    ]


def fulltext_results(text_rows):
    return [
        {
            "row": row,
            "rrf_score": None,
            "vector_rank": None,
            "text_rank": rank,
            "vector_score": None,
            "text_score": row[7],
        }
        for rank, row in enumerate(text_rows, start=1)
    ]


def rrf_fuse(vector_rows, text_rows, rrf_k: int = RRF_K):
    """Reciprocal Rank Fusion of the two rankings (at chunk level)."""
    fused = {}

    for rank, row in enumerate(vector_rows, start=1):
        fused[row[0]] = {
            "row": row,
            "rrf_score": 1 / (rrf_k + rank),
            "vector_rank": rank,
            "text_rank": None,
            "vector_score": row[7],
            "text_score": None,
        }

    for rank, row in enumerate(text_rows, start=1):
        entry = fused.get(row[0])
        if entry is not None:
            entry["rrf_score"] += 1 / (rrf_k + rank)
            entry["text_rank"] = rank
            entry["text_score"] = row[7]
        else:
            fused[row[0]] = {
                "row": row,
                "rrf_score": 1 / (rrf_k + rank),
                "vector_rank": None,
                "text_rank": rank,
                "vector_score": None,
                "text_score": row[7],
            }

    # If RRF scores are tied, the order is deterministic (by chunk ID).
    return sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], item["row"][0]),
    )


def rerank(question: str, fused: list, top_n: int = RERANK_CANDIDATES):
    """Reranks the first top_n hybrid candidates with the cross-encoder.
    Candidates beyond top_n remain at the end in their original hybrid order,
    keeping the list complete and the metrics comparable."""
    head, tail = fused[:top_n], fused[top_n:]
    if not head:
        return fused

    # row: id, doc_id, doc_type, title, section_path, text, metadata, score
    pairs = [(question, f"{item['row'][3] or ''}. {item['row'][5]}")
             for item in head]
    scores = get_reranker().predict(pairs)

    reranked = [{**item, "rerank_score": float(s)}
                for item, s in zip(head, scores)]
    reranked.sort(key=lambda it: (-it["rerank_score"], it["row"][0]))
    return reranked + tail


def build_results(vector_rows, text_rows, mode: str, question: Optional[str] = None):
    """Flat chunk ranking (historical behavior)."""
    if mode == "vector":
        return vector_results(vector_rows)
    if mode == "fulltext":
        return fulltext_results(text_rows)
    if mode == "hybrid":
        return rrf_fuse(vector_rows, text_rows)
    if mode == "hybrid_reranked":
        if question is None:
            raise ValueError("hybrid_reranked richiede la domanda")
        return rerank(question, rrf_fuse(vector_rows, text_rows))
    raise ValueError(f"Modalità sconosciuta: {mode!r} (attese: {MODES})")


def search_flat(question: str, top_k: int = TOP_K, mode: str = "hybrid", **kw):
    """Legacy search(): top_k chunks from the flat ranking."""
    vector_rows, text_rows = retrieve(question, **kw)
    return build_results(vector_rows, text_rows, mode, question=question)[:top_k]


# ---------------------------------------------------------------------------
# STAGE 1: document ranking (best chunk per document + RRF)
# ---------------------------------------------------------------------------

def _doc_arm(rows):
    """Converts score-sorted chunk rows into {doc_id: {rank, score, row}}.
    The document rank is determined by its first (= best) chunk."""
    arm = {}
    for row in rows:
        if row[1] not in arm:
            arm[row[1]] = {"rank": len(arm) + 1, "score": row[7], "row": row}
    return arm


def rank_documents(vector_rows, text_rows, mode: str = "hybrid",
                   rrf_k: int = RRF_K):
    """Ranking of DISTINCT doc_ids. Each entry contains:
    doc_rank, doc_id, doc_type, title, url, rrf_score, vector_rank, text_rank,
    vector_score, text_score. In vector/fulltext mode only that retrieval
    method is used; in hybrid/hybrid_reranked mode the two rankings are merged
    with RRF."""
    if mode not in MODES:
        raise ValueError(f"Modalità sconosciuta: {mode!r} (attese: {MODES})")
    vec, txt = _doc_arm(vector_rows), _doc_arm(text_rows)

    entries = []
    for doc_id in list(vec) + [d for d in txt if d not in vec]:
        v, t = vec.get(doc_id), txt.get(doc_id)
        ref = (v or t)["row"]
        entries.append({
            "doc_id": doc_id,
            "doc_type": ref[2],
            "title": ref[3],
            "url": (ref[6] or {}).get("url"),
            "rrf_score": (
                (VECTOR_WEIGHT / (RRF_K + v["rank"]) if v else 0.0)
                + (TEXT_WEIGHT / (RRF_K + t["rank"]) if t else 0.0)),
            "vector_rank": v["rank"] if v else None,
            "text_rank": t["rank"] if t else None,
            "vector_score": v["score"] if v else None,
            "text_score": t["score"] if t else None,
        })

    if mode == "vector":
        entries = sorted((e for e in entries if e["vector_rank"]),
                         key=lambda e: e["vector_rank"])
    elif mode == "fulltext":
        entries = sorted((e for e in entries if e["text_rank"]),
                         key=lambda e: e["text_rank"])
    else:
        entries.sort(key=lambda e: (-e["rrf_score"], e["doc_id"]))
    if mode in ("vector", "fulltext"):
        for e in entries:
            e["rrf_score"] = None
    for i, e in enumerate(entries, start=1):
        e["doc_rank"] = i
    return entries


# ---------------------------------------------------------------------------
# STAGE 2: evidence (chunks) within the selected documents
# ---------------------------------------------------------------------------

def chunk_pool(vector_rows, text_rows, mode: str):
    """Candidate chunks, sorted by the retrieval score for the selected mode."""
    if mode == "vector":
        return vector_results(vector_rows)
    if mode == "fulltext":
        return fulltext_results(text_rows)
    return rrf_fuse(vector_rows, text_rows)


def select_evidence(documents, pool, per_doc: int = EVIDENCE_PER_DOC):
    """Selects the first `per_doc` chunks from the pool for each retained document.
    Order: document rank, then chunk rank within the document."""
    order = {d["doc_id"]: d["doc_rank"] for d in documents}
    taken = defaultdict(int)
    out = []
    for item in pool:
        did = item["row"][1]
        if did in order and taken[did] < per_doc:
            taken[did] += 1
            out.append({**item, "doc_rank": order[did],
                        "chunk_rank_in_doc": taken[did]})
    out.sort(key=lambda it: (it["doc_rank"], it["chunk_rank_in_doc"]))
    return out


def rerank_evidence(question: str, evidence: list):
    """Reranks the evidence with the cross-encoder (a ranking score,
    NOT a probability of correctness)."""
    if not evidence:
        return evidence
    pairs = []
    for it in evidence:
        row = it["row"]
        head = f"{row[3] or ''}. {row[4]}" if row[4] else f"{row[3] or ''}."
        pairs.append((question, f"{head}\n{row[5]}"))
    scores = get_reranker().predict(pairs)
    out = [{**it, "rerank_score": float(s)} for it, s in zip(evidence, scores)]
    out.sort(key=lambda it: (-it["rerank_score"], it["row"][0]))
    return out


def order_documents_by_evidence(documents, evidence):
    """Document order based on reranking: documents with evidence come first,
    sorted by their best rerank score; the others follow in their original
    stage 1 order. This makes it possible to compare reranking at document level."""
    best = {}
    for it in evidence:                      # evidence already sorted by score
        best.setdefault(it["row"][1], it["rerank_score"])
    head = sorted((d for d in documents if d["doc_id"] in best),
                  key=lambda d: (-best[d["doc_id"]], d["doc_rank"]))
    tail = [d for d in documents if d["doc_id"] not in best]
    return [{**d, "doc_rank": i, "doc_rank_stage1": d["doc_rank"],
             "rerank_score": best.get(d["doc_id"])}
            for i, d in enumerate(head + tail, start=1)]


def two_stage(question: str, top_k: int = TOP_K, mode: str = "hybrid_reranked",
              doc_top_n: int = DOC_TOP_N, per_doc: int = EVIDENCE_PER_DOC,
              conn=None, fts_mode: str = FTS_MODE, doc_ids=None,
              vector_rows=None, text_rows=None):
    """Two-stage retrieval. Returns:
      documents        stage 1 ranking (all retrieved documents)
      documents_final  same as above, but reordered by reranking in hybrid_reranked mode
      evidence         the top_k chunks passed to RAG
      evidence_candidates  number of candidate evidence chunks before truncation
    If vector_rows/text_rows are provided, retrieval is not run again (useful
    for evaluation, to reuse the same rankings across all modes)."""
    if mode not in MODES:
        raise ValueError(f"Modalità sconosciuta: {mode!r} (attese: {MODES})")
    if vector_rows is None or text_rows is None:
        vector_rows, text_rows = retrieve(question, conn=conn,
                                          fts_mode=fts_mode, doc_ids=doc_ids)
    documents = rank_documents(vector_rows, text_rows, mode)
    pool = chunk_pool(vector_rows, text_rows, mode)
    evidence = select_evidence(documents[:doc_top_n], pool, per_doc)
    documents_final = documents
    if mode == "hybrid_reranked":
        evidence = rerank_evidence(question, evidence)
        documents_final = order_documents_by_evidence(documents, evidence)
    return {
        "documents": documents,
        "documents_final": documents_final,
        "evidence": evidence[:top_k],
        "evidence_candidates": len(evidence),
    }


def search(question: str, top_k: int = TOP_K, mode: str = "hybrid", **kw):
    """RAG interface: list of evidence items (dicts with "row" in the
    historical layout). For the old chunk-level behavior, use search_flat()."""
    return two_stage(question, top_k=top_k, mode=mode, **kw)["evidence"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("question", nargs="+", help="la tua domanda")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--fts-mode", choices=FTS_MODES, default=FTS_MODE)
    args = parser.parse_args()

    question = " ".join(args.question)
    print(f"\nQuestion: {question}  [mode: {args.mode}, fts: {args.fts_mode}]")
    print("=" * 80)

    out = two_stage(question, top_k=args.top_k, mode=args.mode,
                    fts_mode=args.fts_mode)
    if not out["evidence"]:
        print("No results found.")
        return

    print("STAGE 1 - documents (top 10):")
    for d in out["documents_final"][:10]:
        rrf = f" RRF={d['rrf_score']:.5f}" if d["rrf_score"] is not None else ""
        print(f"  #{d['doc_rank']:<2} {d['doc_id']:<45} vec={d['vector_rank']} "
              f"txt={d['text_rank']}{rrf} | {d['title'][:50]}")

    print("\nSTAGE 2 - evidence passed to RAG:")
    for rank, item in enumerate(out["evidence"], start=1):
        (_id, doc_id, doc_type, title, section_path, text, metadata, _s) = item["row"]
        meta = metadata or {}
        print(f"\n#{rank} [{doc_type}] {meta.get('chunk_key')} "
              f"(doc #{item['doc_rank']}, chunk #{item['chunk_rank_in_doc']})")
        print(f"title: {title}" + (f" > {section_path}" if section_path else ""))
        print(f"section type: {meta.get('section_type')} ({meta.get('origin')})")
        for label, key in (("vector similarity", "vector_score"),
                           ("full-text score", "text_score"),
                           ("chunk RRF", "rrf_score"),
                           ("reranker score", "rerank_score")):
            v = item.get(key)
            if v is not None:
                print(f"{label}: {v:.4f}")
        print(f"vector position: {item.get('vector_rank')}  "
              f"text position: {item.get('text_rank')}")
        print(f"text: {text[:300]}{'...' if len(text) > 300 else ''}")
        print(f"source: {meta.get('url', 'n/d')}")
        print("-" * 80)


if __name__ == "__main__":
    main()