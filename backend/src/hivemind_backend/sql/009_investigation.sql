-- phase 2: what the investigation keeps per case, in hivemind's own schema.
-- a finding's facts are the numbers and ids its read returned (never a model's words); the plan of each run with
-- what became of every check; the evidence reconciler's fact sheet, one version per run.

alter table agent.findings add column facts jsonb not null default '{}';

-- one row per run of a case: the plan as the validator let it through, and each check's outcome as it is decided
create table agent.case_plans (
  case_id text not null,
  fence bigint not null,          -- the run's lease number: a later run of the case writes a new row
  case_type text not null,
  order_id text,
  customer_id text not null,
  conversation_id text not null,
  plan jsonb not null,
  outcomes jsonb not null default '{}',
  updated_at timestamptz not null,
  primary key (case_id, fence)
);

-- the reconciler's answer for a run: facts that cite ledger evidence, contradictions, gaps, follow-ups, a verdict
create table agent.fact_sheets (
  case_id text not null,
  fence bigint not null,
  round int not null,             -- 1, then 2 after the one follow-up round
  verdict text not null check (verdict in ('sufficient', 'sufficient_with_uncertainty', 'insufficient_escalate')),
  sheet jsonb not null,
  created_at timestamptz not null,
  primary key (case_id, fence, round)
);

create index on agent.case_plans (updated_at desc);

grant select, insert, update, delete on agent.case_plans, agent.fact_sheets to hm_agent;
grant select on agent.case_plans, agent.fact_sheets to hm_eval;
