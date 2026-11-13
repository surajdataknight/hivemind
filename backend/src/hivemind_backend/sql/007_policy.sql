-- the policy and sop text the business keeps, versioned (blueprint c16, l4). a version is the hash of its text, so an
-- edited text is a new version; one version of each section is active, and only the active ones are read. the text
-- has no amounts: amounts are the eligibility engine's (blueprint c18, phase 3). its own schema, so a generated
-- dataset (which empties the business schemas) never removes it.

create schema policy;
grant usage on schema policy to hm_backend, hm_eval;

create table policy.sections (
  policy_id text not null,
  version text not null check (version = left(encode(sha256(convert_to(body, 'UTF8')), 'hex'), 8)),
  title text not null,
  body text not null,
  active boolean not null,
  primary key (policy_id, version)
);
create unique index one_active_policy_version on policy.sections (policy_id) where active;

-- which sections each issue type reads; they are handed over whole (blueprint c16)
create table policy.map (
  issue_type text not null check (issue_type in ('where_is_my_order', 'item_missing_or_wrong', 'quality',
    'cancellation', 'refund', 'payment_failed', 'delivery_partner_behaviour', 'offers_coupons')),
  policy_id text not null,
  primary key (issue_type, policy_id)
);

insert into policy.sections (policy_id, version, title, body, active)
select v.policy_id, left(encode(sha256(convert_to(v.body, 'UTF8')), 'hex'), 8), v.title, v.body, v.active from (values
('refund_policy', 'Refunds', $$A refund goes back to the payment the order was paid with, or to the HiveMart wallet. It is made only for an item or an amount that the records of the order, the packing and the handover show were not given, or were charged wrongly. A refund is worked out by the rules, not by what the customer asks for. When the records do not show the problem, the case goes to a person.$$, true),
('missing_item_sop', 'Missing or wrong item', $$Compare the order lines with what was packed and handed over. An item is missing when it is on the order and was not packed or handed over. An item is wrong when another item was packed in its place. The order record, the packing record and the handover record must show this. The customer's words alone do not settle it.$$, true),
('bod_policy', 'Benefit of the doubt', $$When a small claim cannot be proven from the records, the customer may still be given the benefit of the doubt. This applies only when the customer's account is low risk and has few earlier claims. The limits per claim and per day are set by the rules, not by this text. A high-risk account does not get the benefit of the doubt; the case goes to a person.$$, true),
('damage_sop', 'Quality and damaged items', $$A perishable item that arrived damaged, spoiled or past its date can be refunded. A photo from the customer is checked against the order line. A photo that does not clearly show the problem does not count against the customer.$$, true),
('payment_failed_sop', 'Failed payment', $$A payment that failed took no money from the customer. A payment that was taken for an order that was never placed is returned under the refund policy. The payment record and the order record show which of the two happened. When they disagree, the case goes to a person.$$, true)) as v(policy_id, title, body, active);

insert into policy.map (issue_type, policy_id) values
('item_missing_or_wrong', 'refund_policy'), ('item_missing_or_wrong', 'missing_item_sop'),
('item_missing_or_wrong', 'bod_policy'),
('quality', 'refund_policy'), ('quality', 'damage_sop'), ('quality', 'bod_policy'),
('refund', 'refund_policy'),
('payment_failed', 'payment_failed_sop'), ('payment_failed', 'refund_policy');

grant select, insert, update, delete on policy.sections, policy.map to hm_backend;
grant select on policy.sections, policy.map to hm_eval;
