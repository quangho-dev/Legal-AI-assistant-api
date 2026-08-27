-- User question quota for subscription billing

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'basic',
  ADD COLUMN IF NOT EXISTS question_limit INTEGER NOT NULL DEFAULT 50,
  ADD COLUMN IF NOT EXISTS questions_used INTEGER NOT NULL DEFAULT 0;

ALTER TABLE users
  DROP CONSTRAINT IF EXISTS users_questions_used_non_negative,
  DROP CONSTRAINT IF EXISTS users_question_limit_positive;

ALTER TABLE users
  ADD CONSTRAINT users_questions_used_non_negative CHECK (questions_used >= 0),
  ADD CONSTRAINT users_question_limit_positive CHECK (question_limit > 0);

UPDATE users
SET
  plan = COALESCE(plan, 'basic'),
  question_limit = COALESCE(question_limit, 50),
  questions_used = COALESCE(questions_used, 0)
WHERE plan IS NULL
   OR question_limit IS NULL
   OR questions_used IS NULL;
