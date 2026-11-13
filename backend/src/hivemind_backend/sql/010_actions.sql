-- phase 3: what the action gateway writes, beyond the tables 001 made for it.

-- what each refund paid for: an order line, an extra capture, a failed debit, or a cancelled order. a part is paid
-- at most once, whichever case asks, so a retry or a second case can never pay it twice
create table refunds_wallet.refund_parts (
  refund_id text not null references refunds_wallet.refunds,
  order_id text not null,
  part text not null check (part ~ '^(line:[0-9]+|capture:[A-Z0-9-]+|intent:[A-Z0-9-]+|order)$'),
  amount_paise bigint not null check (amount_paise > 0),
  basis text not null check (basis in ('proven', 'benefit_of_doubt', 'payment_error', 'order_cancelled')),
  primary key (refund_id, part)
);
create unique index one_refund_per_part on refunds_wallet.refund_parts (order_id, part);

-- who the refund is for, who let it be made (support alone, a person, or the business itself, as when an order is
-- cancelled in the app), and the key it was made under (none for the business's own)
alter table refunds_wallet.refunds add column customer_id text;
alter table refunds_wallet.refunds add column route text check (route in ('auto', 'person', 'business'));
alter table refunds_wallet.refunds add column idempotency_key text;
update refunds_wallet.refunds r set customer_id = o.customer_id from orders.orders o where o.order_id = r.order_id;
update refunds_wallet.refunds set route = case when reason = 'order_cancelled' then 'business' else 'auto' end;
create index on refunds_wallet.refunds (customer_id, created_at);

-- a decision record carries the hash of its own body, so the next record of the case can point at it
alter table support.decision_records add column record_hash text;
alter table support.decision_records add column seq int;
create unique index one_record_per_seq on support.decision_records (case_id, seq);

grant select, insert, update, delete on refunds_wallet.refund_parts to hm_backend;
grant select on refunds_wallet.refund_parts to hm_eval;
