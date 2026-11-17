"""
Load knowledge_base.json into PostgreSQL + pgvector:
chunking based on Markdown boundaries with limited token windows, embeddings,
insertion into `documents` and `chunks`.

WARNING: each run performs DROP + CREATE on `chunks` and `documents`
(full rebuild, accepted for this iteration). The schema is
read from init.sql, so there is no need to run `docker compose down -v`.
"""

import argparse
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB_PATH = HERE / "../data/processed/knowledge_base.json"
SCHEMA_PATH = HERE / "../db/init.sql"
# CACHE_PATH = HERE / "../data/cache/embeddings_cache.npz"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DB_CONFIG = dict(host="localhost", port=5432, dbname="rag_kb",
                 user="rag", password="rag_password")

HARD_LIMIT_TOKENS = 512     # model limit (including special tokens)
TARGET_BODY_TOKENS = 380    # target chunk body size
OVERLAP_TOKENS = 40         # overlap, only when a prose block is split
MAX_PREFIX_CHARS = 200      # cap on title and section_path IN THE EMBEDDED PREFIX

FENCE_OPEN_RE = re.compile(r"^\s*(\`{3,}|\~{3,})")


# ---------------------------------------------------------------------------
# Split into atomic blocks (complete code blocks / paragraphs)
# ---------------------------------------------------------------------------

def split_blocks(text):
    """-> [(kind, text)], kind in {'code','prose'}. Code blocks
    (fences) stay intact; prose is split at blank lines."""
    blocks, para = [], []
    lines = text.splitlines()
    i = 0

    def flush_para():
        if para:
            blocks.append(("prose", "\n".join(para).strip()))
            para.clear()

    while i < len(lines):
        m = FENCE_OPEN_RE.match(lines[i])
        if m:
            flush_para()
            fence_char, fence_len = m.group(1)[0], len(m.group(1))
            code = [lines[i]]
            i += 1
            while i < len(lines):
                code.append(lines[i])
                c = re.match(r"^\s*(\`{3,}|\~{3,})\s*$", lines[i])
                i += 1
                if c and c.group(1)[0] == fence_char and len(c.group(1)) >= fence_len:
                    break
            blocks.append(("code", "\n".join(code)))
            continue
        if not lines[i].strip():
            flush_para()
        else:
            para.append(lines[i])
        i += 1
    flush_para()
    return [(k, t) for k, t in blocks if t.strip()]


# ---------------------------------------------------------------------------
# Split content that does not fit: never truncate, only split
# ---------------------------------------------------------------------------

_SPLITTERS = (
    lambda s: re.split(r"(?<=[.!?])\s+|\n+", s),   # sentences / lines
    lambda s: s.split(" "),                       # words
)


def explode(unit, fits, level=0):
    """Splits `unit` into pieces that each fit within `fits`.
    Order: sentences/lines -> words -> half the characters."""
    if fits(unit):
        return [unit]
    if len(unit) <= 1:
        raise ValueError("Impossibile far entrare un singolo carattere nel limite: "
                         "prefisso troppo lungo?")
    if level < len(_SPLITTERS):
        parts = [p for p in _SPLITTERS[level](unit) if p != ""]
        if len(parts) > 1:
            return [x for p in parts for x in explode(p, fits, level + 1)]
        return explode(unit, fits, level + 1)
    mid = len(unit) // 2
    return explode(unit[:mid], fits, level) + explode(unit[mid:], fits, level)


def pack(units, sep, fits, count, overlap_tokens=0):
    """Packs units into windows that satisfy `fits`. If
    overlap_tokens > 0, each window starts with the last units
    from the previous window (up to overlap_tokens)."""
    windows, cur = [], []
    for u in units:
        if cur and not fits(sep.join(cur + [u])):
            windows.append(sep.join(cur))
            carry, t = [], 0
            for x in reversed(cur):
                tx = count(x)
                if t + tx > overlap_tokens:
                    break
                carry.insert(0, x)
                t += tx
            cur = carry
            while cur and not fits(sep.join(cur + [u])):
                cur.pop(0)
        cur.append(u)
    if cur:
        windows.append(sep.join(cur))
    return windows


def split_prose(text, fits, count):
    units = []
    for u in re.split(r"(?<=[.!?])\s+|\n+", text):
        if u.strip():
            units.extend(explode(u, fits, level=1))
    return pack(units, " ", fits, count, OVERLAP_TOKENS)


def split_code(text, fits, count):
    """Splits a code block by lines, closing and reopening the fence
    in each part so that every part remains valid Markdown."""
    lines = text.split("\n")
    opener = lines[0]
    m = FENCE_OPEN_RE.match(opener)
    fence = m.group(1)
    closer = fence
    inner = lines[1:]
    if inner and re.match(r"^\s*(\`{3,}|\~{3,})\s*$", inner[-1]):
        closer = inner[-1].strip()
        inner = inner[:-1]

    def wrap(body):
        return f"{opener}\n{body}\n{closer}"

    def fits_wrapped(body):
        return fits(wrap(body))

    units = []
    for line in inner:
        units.extend(explode(line, fits_wrapped, level=1))
    return [wrap(w) for w in pack(units, "\n", fits_wrapped, count, 0)]


def chunk_body(text, fits, count):
    """Section text -> list of bodies, each fitting within `fits`."""
    chunks, cur = [], []

    def cur_text():
        return "\n\n".join(cur)

    for kind, btxt in split_blocks(text):
        if fits(btxt):
            if cur and not fits(cur_text() + "\n\n" + btxt):
                chunks.append(cur_text())
                cur = [btxt]
            else:
                cur.append(btxt)
        else:
            if cur:
                chunks.append(cur_text())
                cur = []
            splitter = split_code if kind == "code" else split_prose
            chunks.extend(splitter(btxt, fits, count))
    if cur:
        chunks.append(cur_text())
    return chunks


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

def make_prefix(title, section_path):
    t = (title or "")[:MAX_PREFIX_CHARS]
    p = (section_path or "")[:MAX_PREFIX_CHARS]
    return t + (f"\n{p}" if p else "") + "\n\n"


def chunk_document(doc, count, hard_limit=HARD_LIMIT_TOKENS,
                   target=TARGET_BODY_TOKENS):
    """Generates a document's chunks. `count(text)` = number of tokens
    (including special tokens). The count is performed on the EMBEDDED text
    (prefix + body): no chunk exceeds `hard_limit`."""
    chunks = []
    for sec in doc["sections"]:
        prefix = make_prefix(doc["title"], sec["section_path"])
        ptoks = count(prefix)
        if ptoks > hard_limit // 2:
            raise ValueError(f"Prefisso troppo lungo ({ptoks} token) per {doc['doc_id']}")
        effective = min(hard_limit, ptoks + target)

        def fits(body, prefix=prefix, effective=effective):
            return count(prefix + body) <= effective

        for part, body in enumerate(chunk_body(sec["text"], fits, count)):
            embed_text = prefix + body
            n = count(embed_text)
            if n > hard_limit:
                raise AssertionError(
                    f"{doc['doc_id']} sezione {sec['index']}: {n} > {hard_limit} token")
            chunks.append({
                "chunk_key": f"{doc['doc_id']}#s{sec['index']}.{part}",
                "doc_id": doc["doc_id"],
                "chunk_index": len(chunks),
                "section_index": sec["index"],
                "part_index": part,
                "section_path": sec["section_path"],
                "section_type": sec["section_type"],
                "origin": sec["origin"],
                "text": body,
                "embed_text": embed_text,
                "n_tokens": n,
            })
    return chunks


def build_all_chunks(documents, count):
    count = lru_cache(maxsize=None)(count)
    out = []
    for doc in documents:
        out.extend(chunk_document(doc, count))
    return out


# ---------------------------------------------------------------------------
# Embeddings with on-disk cache (resumable)
# ---------------------------------------------------------------------------

def embed_key(text):
    """Cache key: model + embedded text. If either changes,
    the key changes and the embedding is recalculated."""
    return hashlib.sha1((MODEL_NAME + "\n" + text).encode("utf-8")).hexdigest()


def load_cache(path):
    import numpy as np
    if not Path(path).exists():
        return {}
    z = np.load(path)
    return dict(zip(z["keys"].tolist(), z["vecs"]))


def save_cache(path, cache):
    import numpy as np
    tmp = str(path) + ".tmp.npz"
    keys = list(cache)
    np.savez(tmp, keys=np.array(keys), vecs=np.stack([cache[k] for k in keys]))
    os.replace(tmp, path)


def embed_chunks(chunks, cache_path, device=None, batch_size=32,
                 save_every=4096, model=None):
    """Returns a (n_chunk, 384) matrix of normalized embeddings.
    Only computes embeddings not already in the cache and saves the cache
    every `save_every` texts: if the process is interrupted, it can resume
    from where it left off."""
    import numpy as np
    cache = load_cache(cache_path)
    keys = [embed_key(c["embed_text"]) for c in chunks]
    todo = {}
    for k, c in zip(keys, chunks):
        if k not in cache and k not in todo:
            todo[k] = c["embed_text"]
    print(f"Embedding: {len(chunks) - sum(k in todo for k in keys)} chunks already cached, "
          f"{len(todo)} unique texts to process")

    if todo:
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"Device: {model.device}; max_seq_length: {model.max_seq_length}")
        if model.max_seq_length < max(c["n_tokens"] for c in chunks):
            raise SystemExit("Alcuni chunk superano max_seq_length: verrebbero troncati.")
        items = list(todo.items())
        for i in range(0, len(items), save_every):
            block = items[i:i + save_every]
            vecs = model.encode([t for _, t in block], batch_size=batch_size,
                                normalize_embeddings=True, show_progress_bar=True)
            for (k, _), v in zip(block, vecs):
                cache[k] = np.asarray(v, dtype="float32")
            save_cache(cache_path, cache)
            print(f"  Cache saved: {min(i + save_every, len(items))}/{len(items)}")
    return np.stack([cache[k] for k in keys])


# ---------------------------------------------------------------------------
# Tokenizer / DB (lazy imports: tests do not require these libraries)
# ---------------------------------------------------------------------------

def make_token_counter():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def count(text):
        return len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])
    return count


def print_stats(chunks, documents):
    toks = [c["n_tokens"] for c in chunks]
    multi = sum(1 for c in chunks if c["part_index"] > 0)
    print(f"{len(documents)} documents -> {len(chunks)} chunks "
          f"({multi} are subsequent parts of split sections)")
    if toks:
        print(f"Tokens per chunk: min {min(toks)}, avg {sum(toks)/len(toks):.0f}, "
              f"max {max(toks)} (limit {HARD_LIMIT_TOKENS})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default=str(KB_PATH))
    parser.add_argument("--dry-run", action="store_true",
                        help="chunking and statistics only, without embeddings or DB")
    parser.add_argument("--cache", default=str(CACHE_PATH))
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: automatic)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-only", action="store_true",
                        help="compute/update the embedding cache and exit (useful on GPU)")
    args = parser.parse_args()

    documents = json.loads(Path(args.kb).read_text(encoding="utf-8"))["documents"]
    print(f"Loaded {len(documents)} documents from {args.kb}")

    count = make_token_counter()
    chunks = build_all_chunks(documents, count)
    print_stats(chunks, documents)
    if args.dry_run:
        return

    embeddings = embed_chunks(chunks, args.cache, args.device, args.batch_size)
    if args.embed_only:
        print(f"Cache ready at {args.cache}.")
        return

    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    print("Rebuilding schema (DROP + CREATE of chunks and documents)...")
    cur.execute("DROP TABLE IF EXISTS chunks, documents CASCADE;")
    cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

    for d in documents:
        cur.execute(
            "INSERT INTO documents (doc_id, type, title, url, raw_text, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (d["doc_id"], d["type"], d["title"], d["url"], d["raw_text"],
             json.dumps(d["metadata"])),
        )

    titles = {d["doc_id"]: d["title"] for d in documents}
    for c, emb in zip(chunks, embeddings):
        cur.execute(
            """
            INSERT INTO chunks
              (chunk_key, doc_id, chunk_index, section_index, part_index,
               section_path, section_type, origin, text, n_tokens, embedding, tsv)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    setweight(to_tsvector('english', %s), 'A') ||
                    setweight(to_tsvector('english', %s), 'B') ||
                    setweight(to_tsvector('english', %s), 'C'))
            """,
            (c["chunk_key"], c["doc_id"], c["chunk_index"], c["section_index"],
             c["part_index"], c["section_path"], c["section_type"], c["origin"],
             c["text"], c["n_tokens"], emb,
             titles[c["doc_id"]], c["section_path"], c["text"]),
        )
    conn.commit()

    # Integrity checks
    cur.execute("""SELECT count(*) FROM chunks c LEFT JOIN documents d USING (doc_id)
                   WHERE d.doc_id IS NULL OR d.url IS NULL OR d.url = ''""")
    orphans = cur.fetchone()[0]
    cur.execute("SELECT count(*), max(n_tokens) FROM chunks")
    n_chunks, max_tok = cur.fetchone()
    cur.close()
    conn.close()
    print(f"Done. {n_chunks} chunks in DB, max tokens {max_tok}, "
          f"chunks without a document/URL: {orphans}")


if __name__ == "__main__":
    main()