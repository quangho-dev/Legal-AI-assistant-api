-- New users start as pending until an admin assigns an active plan

ALTER TABLE users
  ALTER COLUMN plan SET DEFAULT 'pending';

ALTER TABLE users
  ALTER COLUMN question_limit SET DEFAULT 0;
