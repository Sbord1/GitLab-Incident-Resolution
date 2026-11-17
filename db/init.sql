CREATE EXTENSION IF NOT EXISTS vector;

-- One record per source document (incident or runbook), containing the original text.
CREATE TABLE IF NOT EXISTS documents (
    doc_id   TEXT PRIMARY KEY,
    type     TEXT NOT NULL,
    title    TEXT NOT NULL,
    url      TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- One record per chunk. `text` contains the original chunk body (the text
-- seen by the LLM); title and section_path are stored in separate columns
-- and are prepended only to the text used for embedding (see ingest.py).
CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    chunk_key     TEXT NOT NULL UNIQUE,       -- e.g. incident_7288#s3.0
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index   INT  NOT NULL,              -- sequential index within the document
    section_index INT  NOT NULL,
    part_index    INT  NOT NULL,              -- part within the section
    section_path  TEXT NOT NULL DEFAULT '',
    section_type  TEXT NOT NULL,
    origin        TEXT NOT NULL,              -- section | inline_field
    text          TEXT NOT NULL,
    n_tokens      INT  NOT NULL,              -- number of tokens in the embedded text
    embedding     vector(384) NOT NULL,
    tsv           tsvector NOT NULL,          -- title A, section_path B, text C
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);