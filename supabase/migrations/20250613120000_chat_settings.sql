-- Global chat settings for RAG configuration (single row)

CREATE TABLE chat_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    singleton BOOLEAN NOT NULL DEFAULT true UNIQUE CHECK (singleton = true),
    embedding_model TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    rag_strategy TEXT NOT NULL DEFAULT 'hybrid',
    agent_type TEXT NOT NULL DEFAULT 'default',
    chunks_per_search INTEGER NOT NULL DEFAULT 20,
    final_context_size INTEGER NOT NULL DEFAULT 8,
    similarity_threshold REAL NOT NULL DEFAULT 0.3,
    number_of_queries INTEGER NOT NULL DEFAULT 3,
    reranking_enabled BOOLEAN NOT NULL DEFAULT false,
    reranking_model TEXT NOT NULL DEFAULT 'cohere-rerank-3',
    vector_weight REAL NOT NULL DEFAULT 0.7,
    keyword_weight REAL NOT NULL DEFAULT 0.3,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE OR REPLACE FUNCTION update_chat_settings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chat_settings_updated_at
    BEFORE UPDATE ON chat_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_chat_settings_updated_at();

INSERT INTO chat_settings DEFAULT VALUES;
