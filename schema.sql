-- Run once in your Neon console or via psql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_file TEXT NOT NULL,
    doc_type    TEXT,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id      UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT,
    content     TEXT NOT NULL,
    -- Metadata enriched by Gemini
    summary     TEXT,
    keywords    TEXT[],
    hypo_qa     JSONB,          -- [{q: "...", a: "..."}]
    -- Filtering fields
    dept        TEXT,
    doc_type    TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    -- Vector embedding (gemini-embedding-001, capped at 1536 dims for HNSW compatibility)
    embedding   VECTOR(1536)
);

CREATE TABLE IF NOT EXISTS token_usage (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id       UUID REFERENCES chunks(id) ON DELETE CASCADE,
    doc_id         UUID REFERENCES documents(id) ON DELETE CASCADE,
    stage          TEXT NOT NULL,        -- 'enrich' | 'embed'
    model          TEXT NOT NULL,
    input_tokens   INT DEFAULT 0,
    output_tokens  INT DEFAULT 0,
    total_tokens   INT GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,
    created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS token_usage_doc_id_idx   ON token_usage (doc_id);
CREATE INDEX IF NOT EXISTS token_usage_chunk_id_idx ON token_usage (chunk_id);
CREATE INDEX IF NOT EXISTS token_usage_stage_idx    ON token_usage (stage);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_dept_idx       ON chunks (dept);
CREATE INDEX IF NOT EXISTS chunks_doc_type_idx   ON chunks (doc_type);
CREATE INDEX IF NOT EXISTS chunks_created_at_idx ON chunks (created_at);