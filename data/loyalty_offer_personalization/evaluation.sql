USE IDENTIFIER(:database);

-- Eval payload table consumed by `notebooks/06_generate_evaluation_data.py`
-- and `notebooks/07_run_evaluation.py`. Each row is one prompt to send to the
-- agent under test, plus optional expected behaviors used by the
-- MLflow GenAI Guidelines judge.
CREATE TABLE IF NOT EXISTS evaluation (
    eval_id     STRING  COMMENT 'Stable id for the eval case',
    persona     STRING  COMMENT 'Target operator persona (CRM | MERCHANDISER)',
    request     STRING  COMMENT 'The exact user message to send to the supervisor',
    expects     STRING  COMMENT 'Free-text expectations the Guidelines judge checks against'
)
USING DELTA
COMMENT 'Eval cases for the loyalty_offer_personalization agent';
