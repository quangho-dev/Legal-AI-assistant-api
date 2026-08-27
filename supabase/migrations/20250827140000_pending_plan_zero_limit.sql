-- Allow question_limit = 0 for pending (not yet activated) users

ALTER TABLE users
  DROP CONSTRAINT IF EXISTS users_question_limit_positive;

ALTER TABLE users
  ADD CONSTRAINT users_question_limit_non_negative CHECK (question_limit >= 0);
