-- the simulated business: 13 modules, one schema each, no cross-schema foreign keys.
-- money is integer paise; times are timestamptz (utc).

create schema identity;
create schema customer;
create schema catalog;
create schema store_inventory;
create schema orders;
create schema payments;
create schema refunds_wallet;
create schema delivery;
create schema promotions;
create schema risk;
create schema support;
create schema actions;
create schema notifications;

-- identity
create table identity.customers_auth (
  customer_id text primary key,
  phone_hash text not null,
  created_at timestamptz not null
);
create table identity.sessions (
  session_id text primary key,
  customer_id text not null references identity.customers_auth,
  device_id text not null,
  created_at timestamptz not null,
  expires_at timestamptz not null
);

-- customer
create table customer.customers (
  customer_id text primary key,
  name text not null,
  phone_masked text not null,
  persona text not null,
  city text not null,
  signup_at timestamptz not null
);
create table customer.addresses (
  address_id text primary key,
  customer_id text not null references customer.customers,
  label text not null,
  line1 text not null,
  locality text not null,
  city text not null,
  pincode text not null,
  lat double precision not null,
  lng double precision not null
);
create table customer.preferences (
  customer_id text primary key references customer.customers,
  language text not null default 'en',
  substitutions_allowed boolean not null
);

-- catalog
create table catalog.products (
  product_id text primary key,
  name text not null,
  brand text not null,
  category text not null
);
create table catalog.skus (
  sku text primary key,
  product_id text not null references catalog.products,
  pack_size text not null,
  mrp_paise bigint not null check (mrp_paise > 0)
);
create table catalog.prices (
  sku text not null references catalog.skus,
  price_paise bigint not null check (price_paise > 0),
  valid_from timestamptz not null,
  primary key (sku, valid_from)
);

-- store & inventory
create table store_inventory.stores (
  store_id text primary key,
  name text not null,
  city text not null,
  locality text not null,
  lat double precision not null,
  lng double precision not null
);
create table store_inventory.stock_levels (
  store_id text not null references store_inventory.stores,
  sku text not null,
  qty int not null,
  updated_at timestamptz not null,
  primary key (store_id, sku)
);
create table store_inventory.stock_events (
  event_id text primary key,
  store_id text not null references store_inventory.stores,
  sku text not null,
  event_type text not null check (event_type in ('restock', 'stock_out', 'adjust')),
  qty_delta int not null,
  at timestamptz not null
);
create index on store_inventory.stock_events (store_id, sku, at);
create table store_inventory.pick_pack_scans (
  scan_id text primary key,
  order_id text not null,
  store_id text not null references store_inventory.stores,
  sku text not null,
  event text not null check (event in ('picked', 'packed')),
  qty int not null,
  bag int,
  picker text not null,
  at timestamptz not null
);
create index on store_inventory.pick_pack_scans (order_id);
create table store_inventory.incidents (
  incident_id text primary key,
  store_id text not null references store_inventory.stores,
  type text not null check (type in ('out_of_stock', 'damaged_stock', 'power_cut', 'staff_shortage')),
  sku text,
  description text not null,
  at timestamptz not null
);
create index on store_inventory.incidents (store_id, at);

-- orders
create table orders.orders (
  order_id text primary key,
  customer_id text not null,
  store_id text not null,
  address_id text not null,
  status text not null check (status in ('PLACED', 'CONFIRMED', 'PACKING', 'PACKED', 'OUT_FOR_DELIVERY',
    'DELIVERED', 'CANCELLED', 'PARTIALLY_FULFILLED')),
  item_total_paise bigint not null,
  discount_paise bigint not null,
  delivery_fee_paise bigint not null,
  total_paise bigint not null,
  coupon_code text,
  placed_at timestamptz not null,
  promised_by timestamptz not null,
  delivered_at timestamptz,
  cancelled_at timestamptz,
  version int not null default 1
);
create index on orders.orders (customer_id, placed_at);
create table orders.order_lines (
  order_id text not null references orders.orders,
  line_no int not null,
  sku text not null,
  name text not null,
  qty int not null check (qty > 0),
  unit_paise bigint not null check (unit_paise > 0),
  status text not null check (status in ('ORDERED', 'UNAVAILABLE', 'SUBSTITUTED')),
  primary key (order_id, line_no)
);
create table orders.order_status_history (
  order_id text not null references orders.orders,
  seq int not null,
  status text not null,
  at timestamptz not null,
  primary key (order_id, seq)
);
create table orders.substitutions (
  order_id text not null references orders.orders,
  line_no int not null,
  original_sku text not null,
  substitute_sku text not null,
  at timestamptz not null,
  primary key (order_id, line_no)
);

-- payments
create table payments.payment_intents (
  intent_id text primary key,
  order_id text not null,
  customer_id text not null,
  method text not null check (method in ('upi', 'card', 'wallet')),
  instrument_ref text not null,
  amount_paise bigint not null check (amount_paise > 0),
  status text not null check (status in ('created', 'succeeded', 'failed')),
  created_at timestamptz not null
);
create index on payments.payment_intents (order_id);
create table payments.captures (
  capture_id text primary key,
  intent_id text not null references payments.payment_intents,
  amount_paise bigint not null check (amount_paise > 0),
  gateway_ref text not null,
  captured_at timestamptz not null
);
create table payments.gateway_events (
  event_id text primary key,
  intent_id text not null references payments.payment_intents,
  type text not null check (type in ('debit_success', 'debit_failed', 'capture', 'reversal')),
  amount_paise bigint not null,
  gateway_ref text not null,
  at timestamptz not null
);
create index on payments.gateway_events (intent_id);

-- refunds & wallet
create table refunds_wallet.refunds (
  refund_id text primary key,
  order_id text not null,
  amount_paise bigint not null check (amount_paise > 0),
  destination text not null check (destination in ('source', 'wallet')),
  status text not null check (status in ('INITIATED', 'SUBMITTED_TO_GATEWAY', 'COMPLETED', 'FAILED', 'RETRYING',
    'COMPENSATED')),
  reason text not null,
  case_id text,
  decision_record_id text,
  created_at timestamptz not null
);
create index on refunds_wallet.refunds (order_id);
-- double entry: the entries of one txn_id sum to zero
create table refunds_wallet.wallet_ledger (
  entry_id text primary key,
  txn_id text not null,
  account text not null,
  amount_paise bigint not null,
  ref text not null,
  created_at timestamptz not null
);

-- delivery
create table delivery.riders (
  rider_id text primary key,
  pseudonym text not null,
  vehicle text not null,
  joined_at timestamptz not null
);
create table delivery.trips (
  trip_id text primary key,
  order_id text not null unique,
  rider_id text not null references delivery.riders,
  store_id text not null,
  status text not null check (status in ('ASSIGNED', 'PICKED_UP', 'DELIVERED', 'CANCELLED')),
  assigned_at timestamptz not null,
  picked_up_at timestamptz,
  delivered_at timestamptz,
  eta_at timestamptz not null,
  distance_m int not null
);
create table delivery.gps_points (
  trip_id text not null references delivery.trips,
  seq int not null,
  at timestamptz not null,
  lat double precision not null,
  lng double precision not null,
  primary key (trip_id, seq)
);
create table delivery.pod (
  order_id text primary key,
  trip_id text not null references delivery.trips,
  otp_verified boolean not null,
  bags_handed int not null,
  handed_to text not null check (handed_to in ('customer', 'family_member', 'security')),
  delivered_at timestamptz not null
);
create table delivery.partner_notes (
  note_id text primary key,
  trip_id text not null references delivery.trips,
  order_id text not null,
  text text not null,
  at timestamptz not null
);
create index on delivery.partner_notes (order_id);

-- promotions
create table promotions.offers (
  code text primary key,
  title text not null,
  kind text not null check (kind in ('flat', 'percent', 'free_delivery')),
  value int not null,
  max_discount_paise bigint,
  min_cart_paise bigint not null,
  first_order_only boolean not null,
  valid_from timestamptz not null,
  valid_to timestamptz not null,
  terms text not null
);
create table promotions.redemptions (
  redemption_id text primary key,
  code text not null references promotions.offers,
  order_id text not null,
  customer_id text not null,
  status text not null check (status in ('applied', 'rejected')),
  discount_paise bigint not null,
  reason text,
  at timestamptz not null
);
create index on promotions.redemptions (order_id);

-- risk
create table risk.risk_features (
  customer_id text not null,
  feature text not null,
  value double precision not null,
  computed_at timestamptz not null,
  primary key (customer_id, feature)
);
create table risk.risk_scores (
  customer_id text primary key,
  score double precision not null,
  band text not null check (band in ('low', 'medium', 'high')),
  rules_version text not null,
  computed_at timestamptz not null
);
create table risk.link_graph (
  a_type text not null,
  a_id text not null,
  b_type text not null,
  b_id text not null,
  link_type text not null,
  first_seen timestamptz not null,
  primary key (a_type, a_id, b_type, b_id, link_type)
);

-- support (business-visible case and ticket; execution checkpoints stay in the agent schema)
create table support.cases (
  case_id text primary key,
  customer_id text not null,
  order_id text,
  issue_type text not null,
  state text not null check (state in ('OPEN', 'INVESTIGATING', 'DECIDING', 'AWAITING_CUSTOMER', 'AWAITING_HUMAN',
    'PENDING_SYSTEM', 'ACTIONING', 'ESCALATED', 'RESOLVED', 'CLOSED', 'REOPENED')),
  created_at timestamptz not null,
  updated_at timestamptz not null
);
create unique index one_open_case on support.cases (customer_id, order_id, issue_type)
  where state not in ('RESOLVED', 'CLOSED');
create table support.tickets (
  ticket_id text primary key,
  case_id text not null unique references support.cases,
  status text not null check (status in ('OPEN', 'INFO_REQUESTED', 'APPROVED', 'REJECTED', 'CLOSED')),
  priority text not null,
  summary text not null,
  created_at timestamptz not null,
  updated_at timestamptz not null
);
create table support.ticket_events (
  event_id text primary key,
  ticket_id text not null references support.tickets,
  type text not null,
  payload jsonb not null,
  at timestamptz not null
);
create table support.decision_records (
  decision_record_id text primary key,
  case_id text not null references support.cases,
  record jsonb not null,
  prev_record_hash text,
  created_at timestamptz not null
);

-- actions (the gateway's own tables)
create table actions.idempotency_keys (
  key text primary key,
  request_hash text not null,
  status text not null check (status in ('IN_PROGRESS', 'COMPLETED')),
  response jsonb,
  created_at timestamptz not null,
  expires_at timestamptz not null
);
create table actions.sagas (
  saga_id text primary key,
  case_id text not null,
  steps jsonb not null,
  state text not null,
  updated_at timestamptz not null
);
create table actions.approvals (
  approval_id text primary key,
  case_id text not null,
  remedy text not null,
  max_amount_paise bigint not null,
  approved_by text not null,
  expires_at timestamptz not null,
  created_at timestamptz not null
);

-- notifications: the transactional outbox (cloudevents 1.0 envelope)
create table notifications.notifications_outbox (
  id text primary key,
  source text not null,
  type text not null,
  subject text,
  time timestamptz not null,
  data jsonb not null,
  traceparent text,
  published_at timestamptz
);
create index outbox_unpublished on notifications.notifications_outbox (time) where published_at is null;
