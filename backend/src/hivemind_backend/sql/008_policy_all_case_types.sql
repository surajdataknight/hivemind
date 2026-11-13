-- policy text for the three case types 007 left without: cancellation, delivery partner behaviour, where is my order.
-- with these, every case type has its sections; coupons keep their offer terms (blueprint c16). no amounts, as in 007.

insert into policy.sections (policy_id, version, title, body, active)
select v.policy_id, left(encode(sha256(convert_to(v.body, 'UTF8')), 'hex'), 8), v.title, v.body, true from (values
('cancellation_policy', 'Cancelling an order', $$An order can be cancelled until the store starts packing it. The money paid for a cancelled order goes back under the refund policy. The order record shows whether packing has started. Once packing has started, a request to cancel goes to a person.$$),
('delivery_partner_sop', 'Delivery partner behaviour', $$A report about how a delivery partner behaved goes to a person, with the customer's words and the records of the delivery trip. Support does not contact the partner and does not decide whether the report is true. A refund for the order itself follows the refund policy and the records, not the report.$$),
('order_status_sop', 'Order status and delivery time', $$The status of an order and its delivery time come from the order record and the delivery trip. The delivery time shown is an estimate and can change while the order is on its way. When the records do not show where the order is, the case goes to a person.$$)) as v(policy_id, title, body);

insert into policy.map (issue_type, policy_id) values
('cancellation', 'cancellation_policy'), ('cancellation', 'refund_policy'),
('delivery_partner_behaviour', 'delivery_partner_sop'), ('delivery_partner_behaviour', 'refund_policy'),
('where_is_my_order', 'order_status_sop');
