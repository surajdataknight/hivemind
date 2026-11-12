-- the simulated business's clock: the moment the made-up data ends. the data loader sets it; every "now" the
-- backend uses comes from here, so an order that was on its way when the data was made is still on its way.

create table public.sim_clock (
  only_row boolean primary key default true check (only_row),
  now timestamptz not null
);

grant select on public.sim_clock to hm_backend, hm_eval;
