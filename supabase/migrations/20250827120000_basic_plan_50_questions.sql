-- Set basic plan default to 50 questions

ALTER TABLE users
  ALTER COLUMN question_limit SET DEFAULT 50;

UPDATE users
SET question_limit = 50
WHERE plan = 'basic';
