-- hivemind's own working schema. langgraph adds its checkpoint tables here later.
-- decision records live in support.decision_records (the business-visible audit trail).

create schema agent;

create table agent.evidence_ledger (
  evidence_id text primary key,
  case_id text not null,
  source text not null,
  tool text not null,
  payload jsonb not null,
  sha256 text not null,
  created_at timestamptz not null
);
create index on agent.evidence_ledger (case_id);

create table agent.findings (
  finding_id text primary key,
  case_id text not null,
  task_id text not null,
  type text not null,
  claim text not null,
  evidence_refs text[] not null,
  confidence text not null check (confidence in ('low', 'medium', 'high')),
  created_at timestamptz not null
);
create index on agent.findings (case_id);
