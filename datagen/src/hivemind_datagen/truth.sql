-- the hidden truth: what really happened in each faulty order. only hm_eval (the grader) may read it;
-- hm_backend and hm_agent get no grant, so hivemind cannot see it.

create schema if not exists truth;

create table if not exists truth.order_faults (
  fault_id text primary key,
  order_id text not null,
  customer_id text not null,
  fault_type text not null,
  variant text not null,
  facts jsonb not null,     -- what really happened
  claim jsonb not null,     -- what the simulated customer will say
  expected jsonb not null   -- owed_paise, proven_by_records
);

-- accounts that share a device, an address or a upi id, with many claims (blueprint 7.4)
create table if not exists truth.fraud_rings (
  ring_id text primary key,
  link_type text not null,  -- device, address, upi
  shared text not null,     -- the device id, address id or upi id they share
  members jsonb not null    -- customer ids
);

create table if not exists truth.generation_runs (
  run_id text primary key,
  seed bigint not null,
  config jsonb not null,
  data_sha256 text not null,
  generated_at timestamptz not null
);

revoke all on schema truth from public;
grant usage on schema truth to hm_eval;
grant select on all tables in schema truth to hm_eval;
