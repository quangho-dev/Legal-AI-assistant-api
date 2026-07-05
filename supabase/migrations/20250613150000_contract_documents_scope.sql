-- User-scoped contract drafting documents, separate from compare and corpus

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_document_scope_check;

ALTER TABLE documents
  ADD CONSTRAINT documents_document_scope_check
  CHECK (document_scope IN ('corpus', 'compare', 'contract'));
