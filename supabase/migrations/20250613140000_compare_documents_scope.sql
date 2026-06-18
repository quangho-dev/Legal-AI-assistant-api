-- User-scoped compare documents, separate from admin RAG corpus

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS clerk_id TEXT REFERENCES users(clerk_id) ON DELETE CASCADE,
  ADD COLUMN IF NOT EXISTS document_scope TEXT NOT NULL DEFAULT 'corpus'
    CHECK (document_scope IN ('corpus', 'compare'));

CREATE INDEX IF NOT EXISTS documents_scope_clerk_idx
  ON documents (document_scope, clerk_id, created_at DESC);

-- Existing rows remain corpus (admin RAG) via DEFAULT
