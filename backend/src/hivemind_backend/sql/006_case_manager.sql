-- the full case manager: the history of every case in the business, and in hivemind's own schema one run per
-- case (a lease with a fencing number) and an evidence ledger keyed by case.

-- every open, attach and move of a case, written in the same step as the change itself
create table support.case_events (
  case_id text not null references support.cases,
  seq int not null,
  type text not null check (type in ('opened', 'attached', 'moved')),
  from_state text,
  to_state text not null,
  reason text,
  at timestamptz not null,
  primary key (case_id, seq)
);
grant select, insert, update, delete on support.case_events to hm_backend;
grant select on support.case_events to hm_eval;

-- the same read can be proof in two cases, so an entry belongs to its case
alter table agent.evidence_ledger drop constraint evidence_ledger_pkey;
alter table agent.evidence_ledger add primary key (case_id, evidence_id);

-- a finding as the blueprint's result envelope carries it (deliverable 2, section 2.3)
alter table agent.findings add column capability text not null;
alter table agent.findings add column confidence_basis text;
alter table agent.findings add column fence bigint not null;
alter table agent.findings add column seq bigint generated always as identity;  -- the order they were posted in

-- one run per case. the fencing number only grows, so a run whose lease ran out cannot write after another took
-- over; a released lease keeps its row for that reason
create table agent.case_leases (
  case_id text primary key,
  holder text not null,
  fence bigint not null,
  expires_at timestamptz not null
);
grant select, insert, update, delete on agent.case_leases to hm_agent;
grant select on agent.case_leases to hm_eval;
