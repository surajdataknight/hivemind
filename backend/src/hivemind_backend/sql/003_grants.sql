-- who may read what. the login roles hm_backend, hm_agent and hm_eval exist before this file runs.
-- the truth schema is made by the data generator, which grants it to hm_eval only.

do $$
declare
  s text;
begin
  foreach s in array array['identity', 'customer', 'catalog', 'store_inventory', 'orders', 'payments',
    'refunds_wallet', 'delivery', 'promotions', 'risk', 'support', 'actions', 'notifications']
  loop
    execute format('grant usage on schema %I to hm_backend, hm_eval', s);
    execute format('grant select, insert, update, delete on all tables in schema %I to hm_backend', s);
    execute format('grant select on all tables in schema %I to hm_eval', s);
  end loop;
end $$;

grant usage on schema agent to hm_agent, hm_eval;
grant select, insert, update, delete on all tables in schema agent to hm_agent;
grant create on schema agent to hm_agent;
grant select on all tables in schema agent to hm_eval;
