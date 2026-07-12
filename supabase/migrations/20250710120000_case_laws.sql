-- Dedicated case-law / precedent database for search (not chat corpus)

CREATE TABLE IF NOT EXISTS case_laws (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    d_doc_name TEXT NOT NULL UNIQUE,
    case_number TEXT,
    title TEXT NOT NULL,
    linh_vuc INTEGER,
    linh_vuc_label TEXT,
    adopted_date TEXT,
    published_date TEXT,
    effective_date TEXT,
    status TEXT,
    source_url TEXT NOT NULL,
    pdf_url TEXT,
    s3_key TEXT,
    file_size INTEGER DEFAULT 0,
    full_text TEXT DEFAULT '',
    attributes_text TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    processing_status TEXT NOT NULL DEFAULT 'pending',
    task_id TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS case_laws_case_number_idx
  ON case_laws (case_number);

CREATE INDEX IF NOT EXISTS case_laws_linh_vuc_idx
  ON case_laws (linh_vuc);

CREATE INDEX IF NOT EXISTS case_laws_status_idx
  ON case_laws (status);

CREATE INDEX IF NOT EXISTS case_laws_created_at_idx
  ON case_laws (created_at DESC);

-- Vietnamese-friendly simple FTS over number + title + body
ALTER TABLE case_laws
  ADD COLUMN IF NOT EXISTS fts tsvector
  GENERATED ALWAYS AS (
    to_tsvector(
      'simple',
      coalesce(case_number, '') || ' ' ||
      coalesce(title, '') || ' ' ||
      coalesce(linh_vuc_label, '') || ' ' ||
      coalesce(full_text, '')
    )
  ) STORED;

CREATE INDEX IF NOT EXISTS case_laws_fts_idx
  ON case_laws USING gin (fts);

CREATE OR REPLACE FUNCTION search_case_laws(
    query_text TEXT,
    filter_linh_vuc INTEGER DEFAULT NULL,
    match_limit INTEGER DEFAULT 20,
    match_offset INTEGER DEFAULT 0
)
RETURNS TABLE(
    id UUID,
    d_doc_name TEXT,
    case_number TEXT,
    title TEXT,
    linh_vuc INTEGER,
    linh_vuc_label TEXT,
    adopted_date TEXT,
    published_date TEXT,
    effective_date TEXT,
    status TEXT,
    source_url TEXT,
    pdf_url TEXT,
    processing_status TEXT,
    created_at TIMESTAMPTZ,
    rank REAL
)
LANGUAGE sql
STABLE
AS $$
  SELECT
    cl.id,
    cl.d_doc_name,
    cl.case_number,
    cl.title,
    cl.linh_vuc,
    cl.linh_vuc_label,
    cl.adopted_date,
    cl.published_date,
    cl.effective_date,
    cl.status,
    cl.source_url,
    cl.pdf_url,
    cl.processing_status,
    cl.created_at,
    CASE
      WHEN nullif(trim(query_text), '') IS NULL THEN 0::real
      ELSE ts_rank_cd(cl.fts, plainto_tsquery('simple', query_text))
    END AS rank
  FROM case_laws cl
  WHERE cl.processing_status = 'completed'
    AND (filter_linh_vuc IS NULL OR cl.linh_vuc = filter_linh_vuc)
    AND (
      nullif(trim(query_text), '') IS NULL
      OR cl.fts @@ plainto_tsquery('simple', query_text)
      OR cl.case_number ILIKE '%' || query_text || '%'
      OR cl.title ILIKE '%' || query_text || '%'
    )
  ORDER BY
    CASE WHEN nullif(trim(query_text), '') IS NULL THEN cl.created_at END DESC,
    rank DESC NULLS LAST,
    cl.created_at DESC
  LIMIT GREATEST(match_limit, 1)
  OFFSET GREATEST(match_offset, 0);
$$;
