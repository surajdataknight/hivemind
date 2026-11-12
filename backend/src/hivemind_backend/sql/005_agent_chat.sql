-- hivemind's chat memory: the transcript with personal data replaced by tokens, and the vault that maps each
-- token back to its value. the vault never leaves postgres; no model call reads it.

create table agent.conversations (
  conversation_id text primary key,
  customer_id text not null,
  session_id text not null,
  started_at timestamptz not null
);
create index on agent.conversations (customer_id, started_at);

create table agent.messages (
  conversation_id text not null references agent.conversations,
  seq int not null,
  role text not null check (role in ('customer', 'assistant')),
  text text not null,            -- tokens such as <PHONE_1> stand in for personal data
  kind text not null,            -- customer: 'message'; assistant: which reply it was
  case_id text,
  client_msg_id text,            -- set on customer messages; a repeat submit finds the first one
  meta jsonb not null default '{}',
  created_at timestamptz not null,
  primary key (conversation_id, seq)
);
create unique index one_submit on agent.messages (conversation_id, client_msg_id) where client_msg_id is not null;

create table agent.pii_vault (
  conversation_id text not null references agent.conversations,
  token text not null,           -- <PHONE_1>
  kind text not null,
  value text not null,
  created_at timestamptz not null,
  primary key (conversation_id, token)
);

grant select, insert, update, delete on agent.conversations, agent.messages, agent.pii_vault to hm_agent;
grant select on agent.conversations, agent.messages to hm_eval;
