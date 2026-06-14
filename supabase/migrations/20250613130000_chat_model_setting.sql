-- Add OpenAI chat model setting for final answer generation

ALTER TABLE chat_settings
ADD COLUMN IF NOT EXISTS chat_model TEXT NOT NULL DEFAULT 'gpt-4o';

UPDATE chat_settings
SET chat_model = 'gpt-4o'
WHERE chat_model IS NULL OR chat_model = '';
