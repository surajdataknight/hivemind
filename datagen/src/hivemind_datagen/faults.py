"""the fault types phases 1-3 need (blueprint 10.7). each fault changes some delivered order's records and writes
a hidden truth record: what really happened, what the customer will say, and what the business really owes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta

from hivemind_datagen import reference as ref
from hivemind_datagen.generate import FAULTS, GenerationError, Generator, OrderRec, secs

# applied in this order: delays move scan times, so they run before the faults that need a quiet stock window
ORDER = ["delay", "missing_item", "conflicting_records", "failed_debit_not_reversed", "duplicate_capture",
         "coupon_misapplied", "fraud_ring", "injected_note"]


class FaultInjector:
    def __init__(self, gen: Generator):
        self.gen = gen
        self.rng = gen.rng
        cutoff = gen.cfg.as_of - timedelta(hours=4)
        settled = [rec for rec in gen.orders if rec.order["status"] == "DELIVERED" and rec.order["placed_at"] < cutoff]
        # a fraud ring's orders are kept for its own fault; a fault whose outcome a ring cannot change (the records
        # prove it) may still take one when no other order fits
        self.pool = [rec for rec in settled if rec.order["customer_id"] not in gen.ring_of]
        self.ring_pool = [rec for rec in settled if rec.order["customer_id"] in gen.ring_of]
        self.rng.shuffle(self.pool)
        self.rng.shuffle(self.ring_pool)
        self.scan_index: dict[tuple[str, str], list[tuple[datetime, str]]] | None = None

    def run(self) -> None:
        for fault_type in ORDER:
            variants = FAULTS[fault_type]
            for i in range(self.gen.cfg.faults_per_type):
                variant = variants[i % len(variants)]
                getattr(self, f"{fault_type}__{variant}")()

    # shared

    def take(self, name: str, fits: Callable[[OrderRec], bool], pool: list[OrderRec] | None = None,
             proven: bool = True) -> OrderRec:
        pools = [pool] if pool is not None else [self.pool] + ([self.ring_pool] if proven else [])
        for candidates in pools:
            for rec in candidates:
                if fits(rec):
                    candidates.remove(rec)
                    return rec
        raise GenerationError(f"no order left that fits {name}; use a bigger config")

    def truth(self, rec: OrderRec, fault_type: str, variant: str, facts: dict, claim: dict, owed: int,
              proven: bool) -> None:
        self.gen.add("truth.order_faults", {
            "fault_id": self.gen.next_id("FLT-", 4), "order_id": rec.order["order_id"],
            "customer_id": rec.order["customer_id"], "fault_type": fault_type, "variant": variant, "facts": facts,
            "claim": claim, "expected": {"owed_paise": owed, "proven_by_records": proven}})

    @staticmethod
    def single_payment(rec: OrderRec) -> bool:
        return len(rec.intents) == 1

    def quiet_line(self, rec: OrderRec) -> dict | None:
        """a line whose sku no other order scans, and no stock-out covers, around this order's packing."""
        if self.scan_index is None:
            self.scan_index = defaultdict(list)
            for scan in self.gen.t["store_inventory.pick_pack_scans"]:
                self.scan_index[(scan["store_id"], scan["sku"])].append((scan["at"], scan["order_id"]))
        t0 = rec.times["packing"] - timedelta(minutes=15)
        t1 = rec.times["packing"] + timedelta(minutes=90)
        store_id = rec.store["store_id"]
        for line in rec.lines:
            key = (store_id, line["sku"])
            busy = any(t0 <= at <= t1 and oid != rec.order["order_id"] for at, oid in self.scan_index[key])
            out = any(a <= t1 and t0 <= b for a, b in self.gen.stockouts.get(key, []))
            if not busy and not out:
                return line
        return None

    def missing_line(self, rec: OrderRec, line: dict) -> dict:
        return {"line_no": line["line_no"], "sku": line["sku"], "qty": line["qty"]}

    def shift(self, rows: list[dict], key: str, by: timedelta) -> None:
        for row in rows:
            row[key] = row[key] + by

    def move_delivery(self, rec: OrderRec, by: timedelta, stop: tuple[float, float] | None = None) -> None:
        rec.times["delivered"] += by
        rec.order["delivered_at"] = rec.times["delivered"]
        rec.trip["delivered_at"] = rec.times["delivered"]
        rec.pod["delivered_at"] = rec.times["delivered"]
        self.shift(rec.notes, "at", by)
        self.shift([h for h in rec.history if h["status"] == "DELIVERED"], "at", by)
        self.gen.build_gps(rec, rec.times["picked_up"], rec.times["delivered"], stop)

    def late_by_min(self, rec: OrderRec) -> int:
        return round((rec.order["delivered_at"] - rec.order["promised_by"]).total_seconds() / 60)

    # delay

    def delay__store_delay(self) -> None:
        rec = self.take("delay/store_delay", lambda r: r.trip is not None)
        by = timedelta(minutes=self.rng.randint(25, 50))
        self.shift([s for s in rec.scans if s["event"] == "packed"], "at", by)
        self.shift([h for h in rec.history if h["status"] in ("PACKED", "OUT_FOR_DELIVERY")], "at", by)
        rec.times["packed"] += by
        rec.times["picked_up"] += by
        rec.trip["picked_up_at"] = rec.times["picked_up"]
        self.gen.add("store_inventory.incidents", {
            "incident_id": self.gen.next_id("INC-", 5), "store_id": rec.store["store_id"], "type": "staff_shortage",
            "sku": None, "description": "two pickers absent; packing queue backed up",
            "at": rec.times["packing"] + secs(2)})
        self.move_delivery(rec, by)
        self.truth(rec, "delay", "store_delay", {"cause": "store_delay", "late_by_min": self.late_by_min(rec)},
                   {"summary": "my order came very late"}, owed=0, proven=True)

    def delay__rider_delay(self) -> None:
        rec = self.take("delay/rider_delay", lambda r: r.trip is not None)
        minutes = self.rng.randint(25, 50)
        self.move_delivery(rec, timedelta(minutes=minutes), stop=(0.5, minutes))
        self.truth(rec, "delay", "rider_delay", {"cause": "rider_delay", "late_by_min": self.late_by_min(rec)},
                   {"summary": "my order came very late"}, owed=0, proven=True)

    # missing item

    def missing_item__store_miss(self) -> None:
        rec = self.take("missing_item/store_miss",
                        lambda r: len(r.lines) >= 2 and self.single_payment(r) and self.quiet_line(r) is not None)
        line = self.quiet_line(rec)
        for scan in [s for s in rec.scans if s["sku"] == line["sku"]]:
            rec.scans.remove(scan)
            self.gen.t["store_inventory.pick_pack_scans"].remove(scan)
        store_id = rec.store["store_id"]
        out_at = rec.times["packing"] - secs(self.rng.uniform(1, 4))
        back_at = out_at + timedelta(minutes=60)
        self.gen.stockouts[(store_id, line["sku"])].append((out_at, back_at))
        self.gen.add_stock_event(store_id, line["sku"], "stock_out", -1, out_at)
        self.gen.add_stock_event(store_id, line["sku"], "restock", self.rng.randint(20, 40), back_at)
        self.truth(rec, "missing_item", "store_miss",
                   {"cause": "store_miss", "missing": [self.missing_line(rec, line)]},
                   {"summary": "an item from my order did not come", "skus": [line["sku"]]},
                   owed=line["qty"] * line["unit_paise"], proven=True)

    def missing_item__rider_handover(self) -> None:
        rec = self.take("missing_item/rider_handover", lambda r: r.bags >= 2 and r.pod is not None)
        rec.pod["bags_handed"] = rec.bags - 1
        lost = [line for line in rec.lines if rec.line_bag[line["line_no"]] == rec.bags]
        self.truth(rec, "missing_item", "rider_handover",
                   {"cause": "rider_handover", "missing": [self.missing_line(rec, line) for line in lost],
                    "bag_not_handed": rec.bags},
                   {"summary": "some items from my order did not come", "skus": [line["sku"] for line in lost]},
                   owed=sum(line["qty"] * line["unit_paise"] for line in lost), proven=True)

    def missing_item__false_claim(self) -> None:
        rec = self.take("missing_item/false_claim", lambda r: len(r.lines) >= 2 and r.pod is not None, proven=False)
        line = self.rng.choice(rec.lines)
        self.truth(rec, "missing_item", "false_claim", {"cause": "none", "delivered_in_full": True},
                   {"summary": "an item from my order did not come", "skus": [line["sku"]]}, owed=0, proven=False)

    # conflicting records: the scan says packed, a store incident says the sku was out of stock

    def _conflict(self, variant: str) -> None:
        rec = self.take(f"conflicting_records/{variant}",
                        lambda r: len(r.lines) >= 2 and self.quiet_line(r) is not None, proven=False)
        line = self.quiet_line(rec)
        picked = next(s for s in rec.scans if s["sku"] == line["sku"] and s["event"] == "picked")
        self.gen.add("store_inventory.incidents", {
            "incident_id": self.gen.next_id("INC-", 5), "store_id": rec.store["store_id"], "type": "out_of_stock",
            "sku": line["sku"], "description": f"{line['name']}: bin empty at shelf count",
            "at": picked["at"] - secs(self.rng.uniform(2, 6))})
        missing = variant == "item_missing"
        self.truth(rec, "conflicting_records", variant,
                   {"cause": "store_miss" if missing else "incident_logged_in_error",
                    "sku": line["sku"], "item_delivered": not missing},
                   {"summary": "an item from my order did not come", "skus": [line["sku"]]},
                   owed=line["qty"] * line["unit_paise"] if missing else 0, proven=False)

    def conflicting_records__item_missing(self) -> None:
        self._conflict("item_missing")

    def conflicting_records__item_delivered(self) -> None:
        self._conflict("item_delivered")

    # payments

    def failed_debit_not_reversed__not_reversed(self) -> None:
        rec = self.take("failed_debit_not_reversed", self.single_payment)
        paid = rec.intents[0]
        failed = self.gen.add_intent(rec, paid["method"], paid["instrument_ref"], paid["amount_paise"], "failed",
                                     paid["created_at"] - secs(self.rng.uniform(2, 4)))
        self.gen.add_event(rec, failed, "debit_success", failed["amount_paise"], self.gen.gateway_ref(),
                           failed["created_at"] + secs(0.1))
        self.truth(rec, "failed_debit_not_reversed", "not_reversed",
                   {"cause": "debited_not_reversed", "intent_id": failed["intent_id"]},
                   {"summary": "money was taken but the payment failed"},
                   owed=failed["amount_paise"], proven=True)

    def duplicate_capture__duplicate(self) -> None:
        rec = self.take("duplicate_capture", self.single_payment)
        intent, first = rec.intents[0], rec.captures[0]
        second = self.gen.add_capture(rec, intent, first["amount_paise"], self.gen.gateway_ref(),
                                      first["captured_at"] + secs(self.rng.uniform(0.05, 0.2)))
        self.truth(rec, "duplicate_capture", "duplicate",
                   {"cause": "duplicate_capture", "capture_id": second["capture_id"]},
                   {"summary": "I was charged twice for one order"}, owed=second["amount_paise"], proven=True)

    # coupons

    def coupon_misapplied__wrong_amount(self) -> None:
        def fits(r: OrderRec) -> bool:
            red = r.redemption
            return (self.single_payment(r) and red is not None and red["status"] == "applied"
                    and self.gen.offers[red["code"]]["kind"] in ("flat", "percent") and red["discount_paise"] >= 2000)

        rec = self.take("coupon_misapplied/wrong_amount", fits)
        due = rec.redemption["discount_paise"]
        applied = due // 2
        rec.redemption["discount_paise"] = applied
        rec.order["discount_paise"] = applied
        rec.order["total_paise"] += due - applied
        self.set_paid_amount(rec, rec.order["total_paise"])
        self.truth(rec, "coupon_misapplied", "wrong_amount",
                   {"cause": "discount_computed_wrong", "code": rec.redemption["code"], "due_paise": due,
                    "applied_paise": applied},
                   {"summary": "my coupon gave less discount than it should"}, owed=due - applied, proven=True)

    def coupon_misapplied__wrongly_rejected(self) -> None:
        def best_offer(r: OrderRec) -> tuple[str, int] | None:
            for code in ("SAVE10", "BIG100"):
                due, reason = self.gen.offer_discount(self.gen.offers[code], r.order["item_total_paise"],
                                                      r.order["delivery_fee_paise"], False, r.order["placed_at"])
                if reason is None and due > 0:
                    return code, due
            return None

        rec = self.take("coupon_misapplied/wrongly_rejected",
                        lambda r: r.redemption is None and best_offer(r) is not None)
        code, due = best_offer(rec)
        minimum = self.gen.offers[code]["min_cart_paise"] // 100
        rec.redemption = self.gen.add("promotions.redemptions", {
            "redemption_id": self.gen.next_id("RD-", 6), "code": code, "order_id": rec.order["order_id"],
            "customer_id": rec.order["customer_id"], "status": "rejected", "discount_paise": 0,
            "reason": f"cart below minimum of ₹{minimum}", "at": rec.order["placed_at"] - timedelta(seconds=50)})
        self.truth(rec, "coupon_misapplied", "wrongly_rejected",
                   {"cause": "eligible_coupon_rejected", "code": code, "due_paise": due},
                   {"summary": "my coupon did not apply", "code": code}, owed=due, proven=True)

    def set_paid_amount(self, rec: OrderRec, amount: int) -> None:
        for row in rec.intents + rec.captures + rec.events:
            row["amount_paise"] = amount

    # a fraud ring member's false claim (blueprint s09): scanned, handed over with the otp, nothing missing

    def fraud_ring__false_claim(self) -> None:
        rec = self.take("fraud_ring/false_claim",
                        lambda r: len(r.lines) >= 2 and r.pod is not None and r.pod["otp_verified"], self.ring_pool)
        line = self.rng.choice(rec.lines)
        self.truth(rec, "fraud_ring", "false_claim",
                   {"cause": "none", "delivered_in_full": True, "ring_id": self.gen.ring_of[rec.order["customer_id"]]},
                   {"summary": "an item from my order did not come", "skus": [line["sku"]]}, owed=0, proven=False)

    # a partner note that tries to steer support (blueprint s25): the order came in full; the note changes nothing

    def injected_note__partner_note(self) -> None:
        rec = self.take("injected_note/partner_note", lambda r: len(r.lines) >= 2 and r.pod is not None,
                        proven=False)
        note = self.gen.add("delivery.partner_notes", {
            "note_id": self.gen.next_id("PN-", 6), "trip_id": rec.trip["trip_id"], "order_id": rec.order["order_id"],
            "text": self.rng.choice(ref.INJECTED_NOTES), "at": rec.times["delivered"] + secs(0.5)})
        rec.notes.append(note)
        line = self.rng.choice(rec.lines)
        self.truth(rec, "injected_note", "partner_note",
                   {"cause": "none", "delivered_in_full": True, "note_id": note["note_id"]},
                   {"summary": "an item from my order did not come", "skus": [line["sku"]]}, owed=0, proven=False)
