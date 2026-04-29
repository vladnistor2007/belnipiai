CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

CREATE TABLE IF NOT EXISTS conversations (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  username    TEXT NOT NULL DEFAULT '',
  title       TEXT NOT NULL DEFAULT 'Новый чат',
  model       TEXT NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role             TEXT NOT NULL CHECK (role IN ('user','assistant','summary')),
  content          TEXT NOT NULL,
  token_count      INTEGER DEFAULT 0,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attachments (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id       UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  original_name    TEXT NOT NULL,
  mime_type        TEXT,
  extracted_text   TEXT
);

-- Миграция для существующих баз (добавляет колонку если её нет):
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS username TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_messages_conv  ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conv_updated   ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conv_username  ON conversations(username);

-- ── База знаний ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS doc_folders (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT NOT NULL,
  description TEXT DEFAULT '',
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kb_documents (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  folder_id     UUID REFERENCES doc_folders(id) ON DELETE SET NULL,
  original_name TEXT NOT NULL,
  file_size     BIGINT DEFAULT 0,
  doc_type      TEXT DEFAULT '',
  status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','indexing','indexed','error')),
  chunk_count   INTEGER DEFAULT 0,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  updated_at    TIMESTAMPTZ DEFAULT NOW(),
  meta          JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS doc_chunks (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id  UUID NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
  chunk_index  INTEGER NOT NULL,
  content      TEXT NOT NULL,
  content_tsv  TSVECTOR,
  embedding    VECTOR(768),
  page_num     INTEGER,
  token_count  INTEGER DEFAULT 0,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc     ON doc_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv     ON doc_chunks USING GIN(content_tsv);
CREATE INDEX IF NOT EXISTS idx_kb_docs_folder ON kb_documents(folder_id);
CREATE INDEX IF NOT EXISTS idx_kb_docs_status ON kb_documents(status);

-- HNSW-индекс создаётся отдельно после первой загрузки данных:
-- CREATE INDEX idx_chunks_vector ON doc_chunks
--   USING hnsw(embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
