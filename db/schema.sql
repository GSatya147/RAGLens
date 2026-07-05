
-- RAGLens — Postgres Schema (single datastore: cache + resume + vector store)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- needed for gen_random_uuid() on runs.run_id

-- 1. QUESTIONS (source of truth for question text + ground truth answer, primary key question_id
-- and the Streamlit dashboard (Phase 9) read question text/answer from here
-- populated once in Phase 1 from musique_270.jsonl. FastAPI
CREATE TABLE IF NOT EXISTS questions (
    question_id               TEXT PRIMARY KEY,
    hop_count                 SMALLINT NOT NULL CHECK (hop_count IN (2,3,4)),
    question_text             TEXT NOT NULL,
    ground_truth_answer       TEXT NOT NULL,
    created_at                TIMESTAMPTZ DEFAULT now()
);

-- 2. CORPUS / VECTOR STORE 
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        SERIAL PRIMARY KEY,
    question_id     TEXT NOT NULL REFERENCES questions(question_id),
    hop_count       SMALLINT NOT NULL,     -- denormalized from questions, avoids a join on every retrieval call during the 810-run sweep
    is_supporting   BOOLEAN NOT NULL,      -- MuSiQue ground truth label, THE deterministic ground truth Phase 5's hop-level metrics match against
    content         TEXT NOT NULL,
    token_count     SMALLINT,              
    embedding       vector(1024) NOT NULL,           
    sparse_tsv      TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED, 
                                           -- generated column: always in sync with content,
                                           
    created_at      TIMESTAMPTZ DEFAULT now(),

    UNIQUE (question_id, content)          -- idempotency backstop, reruns of the ingestion script during dev can't silently duplicate chunks
);

CREATE INDEX IF NOT EXISTS idx_chunks_sparse ON chunks USING GIN (sparse_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_question ON chunks (question_id);

-- 3. Generation RUNS 
CREATE TABLE IF NOT EXISTS runs (
    run_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_id     TEXT NOT NULL REFERENCES questions(question_id),
    config_id       TEXT NOT NULL CHECK (config_id IN ('naive', 'advanced', 'hybrid')),
    hop_count       SMALLINT NOT NULL,

    -- resume/checkpoint state, this is the batch driver's memory
    gen_status      TEXT NOT NULL DEFAULT 'pending'
                    CHECK (gen_status IN ('pending','running','done','failed')),
    locked          BOOLEAN NOT NULL DEFAULT false,   -- prevents double-processing on restart

    trace_path      TEXT,               -- JSONL backup file location 
    model_name      TEXT,              
    model_version    TEXT,              -- resolved version at call time, if anything happens to primary model
    step_count      SMALLINT,           -- actual hops taken for the model to generate/ answer the query
    error_message   TEXT,               -- populated on gen_status = 'failed', never silently dropped

    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);

-- resume queries only ever look for non-done rows, so index only those. 
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (gen_status) WHERE gen_status != 'done';

-- Blocks accidental duplicate inserts if the batch driver restarts and doesn't check properly; belt-and-suspenders on top of gen_status.
CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_question_config ON runs (question_id, config_id);

-- 4. EVAL_RESULTS 
CREATE TABLE IF NOT EXISTS eval_results (
    eval_id             SERIAL PRIMARY KEY,
    run_id              UUID NOT NULL REFERENCES runs(run_id),

    content_hash        TEXT NOT NULL,    -- hash(question + contexts + answer), this is the cache layer

    -- RAGAS pipeline-level metrics (~8-10 calls/run)
    ragas_faithfulness      NUMERIC(4,3),
    ragas_answer_relevancy  NUMERIC(4,3),
    ragas_context_recall    NUMERIC(4,3),

    -- failure classification (1 call/run): a LABEL on evidence already computed in step_results, not an independent re-judgment
    failure_class       TEXT CHECK (failure_class IN
                        ('PASS','PREMATURE_COLLAPSE','OVER_EXTENSION',
                         'RETRIEVAL_FAILURE','REASONING_FAILURE')),

    judge_model          TEXT,          
    eval_status          TEXT NOT NULL DEFAULT 'pending'
                         CHECK (eval_status IN ('pending','running','done','failed')),
    locked               BOOLEAN NOT NULL DEFAULT false,

    evaluated_at          TIMESTAMPTZ
);

-- the cache check. Before any judge API call
CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_hash ON eval_results (content_hash);

CREATE INDEX IF NOT EXISTS idx_eval_status ON eval_results (eval_status) WHERE eval_status != 'done';

-- 5. STEP_RESULTS (actual hop-level/step-level)
CREATE TABLE IF NOT EXISTS step_results (
    step_id             SERIAL PRIMARY KEY,
    run_id              UUID NOT NULL REFERENCES runs(run_id),
    step_number         SMALLINT NOT NULL,   -- 1st hop, 2nd hop, etc.

    -- deterministic hop-level metrics
    context_precision_at_3   NUMERIC(4,3),
    context_recall           NUMERIC(4,3),

    -- step-level faithfulness: LLM-as-Judge
    step_faithful             BOOLEAN,
    step_faithful_reason      TEXT,          -- one-sentence judge explanation, renders in the dashboard's red-step diagnostic view, not just the score

    -- retrieval metadata (tracer data)
    sub_query            TEXT,
    retrieved_chunk_ids   INTEGER[],          -- references chunks.chunk_id
    retrieval_scores      JSONB,              -- {dense: [...], bm25: [...], rerank: [...]}
                                              -- kept as JSONB since shape varies by config: Naive has no rerank scores, Hybrid has BM25 scores
    step_latency_ms       INTEGER,

    -- mitigation tip layer
    mitigation_tip         TEXT,              -- NULL for passed steps

    created_at            TIMESTAMPTZ DEFAULT now()
);

-- this table is what the dashboard's Trace Inspector and
-- Single Query View read directly: green/yellow/red coloring is derived from step_faithful + context_precision here,
-- never recomputed in the dashboard layer.
CREATE INDEX IF NOT EXISTS idx_step_run ON step_results (run_id, step_number);

-- 6. ERRORS 
CREATE TABLE IF NOT EXISTS errors (
    error_id        SERIAL PRIMARY KEY,
    run_id          UUID REFERENCES runs(run_id),
    stage           TEXT CHECK (stage IN ('generation','evaluation')),
    error_type      TEXT,        -- '429', '503', '504', 'context_length_exceeded', 'empty_response'
    error_message   TEXT,
    occurred_at     TIMESTAMPTZ DEFAULT now()
);
-- kept separate from runs.error_message on purpose - this table is append-only log