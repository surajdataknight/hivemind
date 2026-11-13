"""generator v0: one made-up city of dark stores, customers, orders, scans, trips and payments, from a seed.

the same seed and config always give the same rows. faults are injected into some delivered orders, and each one
gets a hidden truth record (truth.order_faults) that only the grader may read.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from hivemind_datagen import reference as ref

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))

TABLES = [
    "identity.customers_auth",
    "identity.sessions",
    "customer.customers",
    "customer.addresses",
    "customer.preferences",
    "catalog.products",
    "catalog.skus",
    "catalog.prices",
    "store_inventory.stores",
    "store_inventory.stock_levels",
    "store_inventory.stock_events",
    "store_inventory.pick_pack_scans",
    "store_inventory.incidents",
    "orders.orders",
    "orders.order_lines",
    "orders.order_status_history",
    "payments.payment_intents",
    "payments.captures",
    "payments.gateway_events",
    "refunds_wallet.refunds",
    "refunds_wallet.refund_parts",
    "delivery.riders",
    "delivery.trips",
    "delivery.gps_points",
    "delivery.pod",
    "delivery.partner_notes",
    "promotions.offers",
    "promotions.redemptions",
    "support.cases",
    "support.case_events",
    "support.tickets",
    "truth.order_faults",
    "truth.fraud_rings",
    "truth.generation_runs",
]

FAULTS = {
    "missing_item": ("store_miss", "rider_handover", "false_claim"),
    "delay": ("store_delay", "rider_delay"),
    "failed_debit_not_reversed": ("not_reversed",),
    "duplicate_capture": ("duplicate",),
    "coupon_misapplied": ("wrong_amount", "wrongly_rejected"),
    "conflicting_records": ("item_missing", "item_delivered"),
    "fraud_ring": ("false_claim",),       # a ring member says an item is missing; the records show it delivered
    "injected_note": ("partner_note",),   # a partner note that tries to give an order; it must change nothing
}
LINK_TYPES = ("device", "address", "upi")  # what a fraud ring's accounts share, one kind per ring, in turn
CLAIM_TYPES = ("item_missing_or_wrong", "quality", "refund")

# share of the day's orders in each ist hour (dinner peak)
HOURLY = [0.5, 0.2, 0.1, 0.1, 0.1, 0.2, 1.0, 2.5, 3.5, 3.0, 2.5, 2.5, 3.0, 3.0, 2.5, 2.0, 2.5, 3.5, 5.0, 7.0, 8.0,
          7.0, 4.0, 1.5]
STORE_LOCALITIES = [0, 1, 3, 4, 5, 2, 6, 7]  # indexes into reference.LOCALITIES
DELIVERY_FEE_PAISE = 2500
FREE_DELIVERY_FROM_PAISE = 19900


@dataclass(frozen=True)
class Config:
    # ends at 20:00 ist, the dinner peak, so some orders are still on their way at as_of
    start: datetime = datetime(2026, 10, 2, 14, 30, tzinfo=UTC)  # 2026-10-02 20:00 ist
    days: int = 14
    customers: int = 300
    riders: int = 40
    stores: int = 4
    orders_per_day: int = 150
    faults_per_type: int = 10
    live_orders: int = 8  # extra orders placed in the last 18 minutes, so some are always on their way at as_of
    not_packed_orders: int = 2  # orders placed under a minute before as_of: not packed yet, so they can be cancelled
    fraud_rings: int = 3  # groups of accounts sharing a device, an address or a upi id, with many claims
    ring_size: int = 3
    claim_rate: float = 0.12  # share of other customers with an earlier claim (closed case) on a past order

    @property
    def as_of(self) -> datetime:
        return self.start + timedelta(days=self.days)

    def to_json(self) -> dict:
        data = asdict(self)
        data["start"] = iso(self.start)
        return data


# 3 faults per type = every variant at least once; 3 days x 130 orders fits all faults for seeds 0-199
SMALL = Config(days=3, customers=60, riders=10, orders_per_day=130, faults_per_type=3, live_orders=4,
               not_packed_orders=2)


class GenerationError(Exception):
    pass


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def secs(minutes: float) -> timedelta:
    return timedelta(seconds=round(minutes * 60))


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lng1, lat2, lng2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def dataset_sha256(tables: dict[str, list[dict]]) -> str:
    """hash of every table except the run record itself, in a fixed order."""
    h = hashlib.sha256()
    for name in TABLES:
        if name == "truth.generation_runs":
            continue
        h.update(name.encode())
        payload = json.dumps(tables[name], sort_keys=True, separators=(",", ":"), default=iso, ensure_ascii=False)
        h.update(payload.encode("utf-8"))
    return h.hexdigest()


@dataclass
class OrderRec:
    order: dict
    customer: dict
    store: dict
    lines: list[dict]
    history: list[dict]
    bags: int
    times: dict[str, datetime]
    line_bag: dict[int, int] = field(default_factory=dict)
    scans: list[dict] = field(default_factory=list)
    trip: dict | None = None
    gps: list[dict] = field(default_factory=list)
    pod: dict | None = None
    notes: list[dict] = field(default_factory=list)
    intents: list[dict] = field(default_factory=list)
    captures: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    redemption: dict | None = None
    address: dict | None = None


class Generator:
    def __init__(self, seed: int, cfg: Config):
        self.seed = seed
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.t: dict[str, list[dict]] = {name: [] for name in TABLES}
        self.counters: dict[str, int] = defaultdict(int)
        self.orders: list[OrderRec] = []
        self.stockouts: dict[tuple[str, str], list[tuple[datetime, datetime]]] = defaultdict(list)
        # fraud rings and earlier claims draw from a stream of their own, so the orders and faults come out as before
        self.side = random.Random(f"{seed}:claims")
        self.rings: list[dict] = []
        self.ring_of: dict[str, str] = {}  # customer id -> ring id

    # helpers

    def next_id(self, prefix: str, width: int, offset: int = 0) -> str:
        self.counters[prefix] += 1
        return f"{prefix}{offset + self.counters[prefix]:0{width}d}"

    def add(self, table: str, row: dict) -> dict:
        self.t[table].append(row)
        return row

    def poisson(self, lam: float) -> int:
        limit, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= limit:
                return k
            k += 1

    def gateway_ref(self) -> str:
        return f"gw_{self.rng.getrandbits(48):012x}"

    # world

    def build_catalog(self) -> None:
        per_code: dict[str, int] = defaultdict(int)
        self.skus = []
        valid_from = self.cfg.start - timedelta(days=30)
        for i, (code, category, name, pack, price, mrp) in enumerate(ref.PRODUCTS, start=1):
            per_code[code] += 1
            sku = f"SKU-{code}-{per_code[code]:03d}"
            product_id = f"PRD-{i:03d}"
            brand = self.rng.choice(ref.BRANDS[category])
            self.add("catalog.products", {"product_id": product_id, "name": name, "brand": brand, "category": category})
            self.add("catalog.skus", {"sku": sku, "product_id": product_id, "pack_size": pack, "mrp_paise": mrp * 100})
            self.add("catalog.prices", {"sku": sku, "price_paise": price * 100, "valid_from": valid_from})
            weight = self.rng.uniform(0.3, 3.0) * (1.8 if category in ("dairy", "fruits_vegetables") else 1.0)
            self.skus.append({"sku": sku, "name": f"{brand} {name} ({pack})", "price": price * 100, "weight": weight})

    def build_stores(self) -> None:
        if not 1 <= self.cfg.stores <= len(STORE_LOCALITIES):
            raise GenerationError(f"stores must be 1..{len(STORE_LOCALITIES)}")
        self.stores = []
        for n in range(1, self.cfg.stores + 1):
            name, _pin, lat, lng = ref.LOCALITIES[STORE_LOCALITIES[n - 1]]
            store = {"store_id": f"DS-BLR-{n:02d}", "name": f"{name} dark store", "city": "Bengaluru",
                     "locality": name, "lat": lat, "lng": lng}
            self.add("store_inventory.stores", store)
            self.stores.append({**store, "pickers": [f"pk_ano_{n}{j}" for j in range(1, 5)], "riders": []})

    def build_riders(self) -> None:
        for n in range(1, self.cfg.riders + 1):
            rider_id = f"RDR-{n:03d}"
            self.add("delivery.riders", {
                "rider_id": rider_id,
                "pseudonym": f"rd_ano_{self.rng.getrandbits(24):06x}",
                "vehicle": self.rng.choice(["bike", "ev_scooter"]),
                "joined_at": self.cfg.start - secs(self.rng.uniform(20, 700) * 1440),
            })
            self.stores[(n - 1) % len(self.stores)]["riders"].append(rider_id)

    def place_near(self, store: dict) -> tuple[str, str, float, float]:
        km = self.rng.uniform(0.4, 3.2)
        angle = self.rng.uniform(0, 2 * math.pi)
        lat = round(store["lat"] + km / 111.0 * math.cos(angle), 6)
        lng = round(store["lng"] + km / (111.0 * math.cos(math.radians(store["lat"]))) * math.sin(angle), 6)
        nearest = min(ref.LOCALITIES, key=lambda loc: haversine_km((lat, lng), (loc[2], loc[3])))
        return nearest[0], nearest[1], lat, lng

    def build_customers(self) -> None:
        self.customers = []
        for _ in range(self.cfg.customers):
            customer_id = self.next_id("CUST-", 5)
            persona = "new" if self.rng.random() < 0.15 else "regular"
            if persona == "new":
                signup = self.cfg.start + secs(self.rng.uniform(0, self.cfg.days * 0.6) * 1440)
            else:
                signup = self.cfg.start - secs(self.rng.uniform(30, 400) * 1440)
            phone = "9" + "".join(str(self.rng.randint(0, 9)) for _ in range(9))
            name = f"{self.rng.choice(ref.FIRST_NAMES)} {self.rng.choice(ref.LAST_NAMES)}"
            store = self.rng.choice(self.stores)
            addresses = []
            for label in ["Home", "Work"][: 1 if self.rng.random() < 0.8 else 2]:
                locality, pincode, lat, lng = self.place_near(store)
                line1 = (f"#{self.rng.randint(1, 350)}, {ordinal(self.rng.randint(1, 18))} Cross, "
                         f"{ordinal(self.rng.randint(1, 12))} Main")
                addresses.append({"address_id": self.next_id("ADDR-", 5), "customer_id": customer_id, "label": label,
                                  "line1": line1, "locality": locality, "city": "Bengaluru", "pincode": pincode,
                                  "lat": lat, "lng": lng})
            self.add("identity.customers_auth", {"customer_id": customer_id,
                                                 "phone_hash": hashlib.sha256(phone.encode()).hexdigest(),
                                                 "created_at": signup})
            session_at = max(signup, self.cfg.as_of - timedelta(days=2))
            self.add("identity.sessions", {"session_id": self.next_id("SES-", 6), "customer_id": customer_id,
                                           "device_id": f"dev_{self.rng.getrandbits(48):012x}",
                                           "created_at": session_at, "expires_at": session_at + timedelta(days=30)})
            self.add("customer.customers", {"customer_id": customer_id, "name": name,
                                            "phone_masked": f"+91 {phone[0]}XXXXXX{phone[-3:]}", "persona": persona,
                                            "city": "Bengaluru", "signup_at": signup})
            for address in addresses:
                self.add("customer.addresses", address)
            self.add("customer.preferences", {"customer_id": customer_id, "language": "en",
                                              "substitutions_allowed": self.rng.random() < 0.6})
            self.customers.append({
                "customer_id": customer_id, "persona": persona, "signup_at": signup, "store": store,
                "addresses": addresses, "weight": self.rng.lognormvariate(0, 0.8), "orders": 0,
                "instruments": {
                    "upi": f"upi:{name.split()[0].lower()}{self.rng.randint(10, 99)}@okbank",
                    "card": f"card:xxxx-{self.rng.randint(1000, 9999)}",
                    "wallet": "wallet:hivemart",
                },
            })

    def build_offers(self) -> None:
        start, as_of = self.cfg.start, self.cfg.as_of
        live = (start - timedelta(days=60), as_of + timedelta(days=60))
        offers = [
            ("WELCOME50", "₹50 off your first order", "flat", 5000, None, 19900, True, live,
             "₹50 off on the first order of a new account. Minimum cart ₹199. One use per account."),
            ("SAVE10", "10% off up to ₹100", "percent", 10, 10000, 49900, False, live,
             "10% off the item total, up to ₹100. Minimum cart ₹499. Delivery fee not included."),
            ("BIG100", "₹100 off on ₹999", "flat", 10000, None, 99900, False, live,
             "Flat ₹100 off when the item total is ₹999 or more."),
            ("FREEDEL", "Free delivery", "free_delivery", 0, None, 9900, False, live,
             "Delivery fee waived on carts of ₹99 or more."),
            ("MONSOON15", "15% off up to ₹75", "percent", 15, 7500, 39900, False,
             (start - timedelta(days=120), start - timedelta(days=1)),
             "15% off up to ₹75 on carts of ₹399 or more. Valid during the monsoon sale only."),
        ]
        self.offers = {}
        for code, title, kind, value, cap, min_cart, first_only, (vfrom, vto), terms in offers:
            row = {"code": code, "title": title, "kind": kind, "value": value, "max_discount_paise": cap,
                   "min_cart_paise": min_cart, "first_order_only": first_only, "valid_from": vfrom, "valid_to": vto,
                   "terms": terms}
            self.add("promotions.offers", row)
            self.offers[code] = row

    @staticmethod
    def offer_discount(offer: dict, item_total: int, fee: int, first_order: bool, at: datetime) -> tuple[int, str | None]:
        """the discount the terms give, or 0 and the reason it does not apply."""
        if not offer["valid_from"] <= at < offer["valid_to"]:
            return 0, "offer has expired"
        if offer["first_order_only"] and not first_order:
            return 0, "offer is only for the first order"
        if item_total < offer["min_cart_paise"]:
            return 0, f"cart below minimum of ₹{offer['min_cart_paise'] // 100}"
        if offer["kind"] == "flat":
            return min(offer["value"], item_total), None
        if offer["kind"] == "percent":
            return min(item_total * offer["value"] // 100, offer["max_discount_paise"]), None
        if fee == 0:
            return 0, "delivery is already free"
        return fee, None

    def build_natural_stockouts(self) -> None:
        for midnight in self.ist_midnights():
            for store in self.stores:
                sku = self.rng.choice(self.skus)["sku"]
                out_at = midnight + secs(self.rng.uniform(8 * 60, 22 * 60))  # 08:00-22:00 ist
                if not self.cfg.start <= out_at < self.cfg.as_of:
                    continue
                back_at = out_at + secs(self.rng.uniform(60, 180))
                self.stockouts[(store["store_id"], sku)].append((out_at, back_at))
                self.add_stock_event(store["store_id"], sku, "stock_out", -self.rng.randint(1, 3), out_at)
                restock_qty = self.rng.randint(20, 40)
                if back_at <= self.cfg.as_of:  # otherwise still out of stock at as_of
                    self.add_stock_event(store["store_id"], sku, "restock", restock_qty, back_at)

    def ist_midnights(self) -> list[datetime]:
        """ist midnights (as utc) of every day the window touches."""
        first = self.cfg.start.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)
        return [(first + timedelta(days=d)).astimezone(UTC) for d in range(self.cfg.days + 1)]

    def add_stock_event(self, store_id: str, sku: str, kind: str, delta: int, at: datetime) -> None:
        self.add("store_inventory.stock_events", {"event_id": self.next_id("SE-", 6), "store_id": store_id,
                                                  "sku": sku, "event_type": kind, "qty_delta": delta, "at": at})

    def in_stock(self, store_id: str, sku: str, at: datetime) -> bool:
        return not any(a <= at < b for a, b in self.stockouts.get((store_id, sku), []))

    def build_restocks(self) -> None:
        for midnight in self.ist_midnights():
            at = midnight + timedelta(hours=6)  # 06:00 ist
            if not self.cfg.start <= at < self.cfg.as_of:
                continue
            for store in self.stores:
                for sku in self.skus:
                    self.add_stock_event(store["store_id"], sku["sku"], "restock", self.rng.randint(10, 40), at)

    def build_stock_levels(self) -> None:
        for store in self.stores:
            for sku in self.skus:
                qty = 0 if not self.in_stock(store["store_id"], sku["sku"], self.cfg.as_of) else self.rng.randint(3, 60)
                self.add("store_inventory.stock_levels", {"store_id": store["store_id"], "sku": sku["sku"],
                                                          "qty": qty, "updated_at": self.cfg.as_of})

    # orders

    def build_orders(self) -> None:
        total_weight = sum(HOURLY)
        for h in range(self.cfg.days * 24):
            hour_start = self.cfg.start + timedelta(hours=h)
            ist_hour = hour_start.astimezone(IST).hour
            count = self.poisson(HOURLY[ist_hour] * self.cfg.orders_per_day / total_weight)
            offsets = [self.rng.uniform(0, 3600) for _ in range(count)]
            if h == self.cfg.days * 24 - 1:
                offsets += [3600 - self.rng.uniform(60, 18 * 60) for _ in range(self.cfg.live_orders)]
            offsets.sort()
            for offset in offsets:
                placed = hour_start + timedelta(seconds=round(offset))
                if placed < self.cfg.as_of:
                    self.build_order(placed)
        self.build_not_packed_orders()

    def build_not_packed_orders(self) -> None:
        """the last orders, placed 20 to 50 seconds before as_of: placed or confirmed, packing not started. they draw
        from a stream of their own, so every other row comes out as before (only later ids move up)."""
        main, self.rng = self.rng, random.Random(f"{self.seed}:not_packed")
        try:
            for placed in sorted(self.cfg.as_of - timedelta(seconds=round(self.rng.uniform(20, 50)))
                                 for _ in range(self.cfg.not_packed_orders)):
                self.build_order(placed)
        finally:
            self.rng = main

    def build_order(self, placed: datetime) -> None:
        rng, as_of = self.rng, self.cfg.as_of
        eligible = [c for c in self.customers if c["signup_at"] < placed]
        if not eligible:
            return
        customer = rng.choices(eligible, weights=[c["weight"] for c in eligible])[0]
        address = rng.choice(customer["addresses"])
        store = customer["store"]
        first_order = customer["persona"] == "new" and customer["orders"] == 0
        customer["orders"] += 1

        # cart
        pool = [s for s in self.skus if self.in_stock(store["store_id"], s["sku"], placed)]
        n_lines = rng.choices([1, 2, 3, 4, 5, 6], weights=[15, 22, 22, 18, 13, 10])[0]
        picked: list[dict] = []
        while len(picked) < n_lines and pool:
            choice = rng.choices(pool, weights=[s["weight"] for s in pool])[0]
            pool.remove(choice)
            picked.append(choice)
        order_id = self.next_id("ORD-", 6, offset=100000)
        lines = [{"order_id": order_id, "line_no": i, "sku": s["sku"], "name": s["name"],
                  "qty": rng.choices([1, 2, 3], weights=[75, 20, 5])[0], "unit_paise": s["price"],
                  "status": "ORDERED"} for i, s in enumerate(picked, start=1)]
        item_total = sum(line["qty"] * line["unit_paise"] for line in lines)
        fee = DELIVERY_FEE_PAISE if item_total < FREE_DELIVERY_FROM_PAISE else 0

        # coupon attempt
        coupon, discount, redemption = None, 0, None
        if rng.random() < 0.22:
            codes = ["WELCOME50", "SAVE10", "BIG100", "FREEDEL", "MONSOON15"]
            weights = [3.0 if first_order else 0.5, 3.0, 1.5, 2.0, 1.0]
            code = rng.choices(codes, weights=weights)[0]
            due, reason = self.offer_discount(self.offers[code], item_total, fee, first_order, placed)
            if reason is None:
                coupon, discount = code, due
            redemption = {"redemption_id": self.next_id("RD-", 6), "code": code, "order_id": order_id,
                          "customer_id": customer["customer_id"], "status": "applied" if reason is None else "rejected",
                          "discount_paise": due, "reason": reason, "at": placed - timedelta(seconds=50)}
        total = item_total + fee - discount

        # timeline
        km = haversine_km((store["lat"], store["lng"]), (address["lat"], address["lng"]))
        distance_m = round(km * 1300)  # road distance ~1.3x straight line
        times: dict[str, datetime] = {"placed": placed}
        times["confirmed"] = placed + secs(rng.uniform(0.3, 0.7))
        times["packing"] = times["confirmed"] + secs(rng.uniform(0.5, 2))
        times["assigned"] = times["packing"] + secs(rng.uniform(0.5, 1.5))
        pick_times, t = [], times["packing"]
        for _ in lines:
            t = t + secs(rng.uniform(0.4, 0.8))
            pick_times.append(t)
        times["packed"] = t + secs(rng.uniform(0.5, 1.5))
        times["picked_up"] = times["packed"] + secs(rng.uniform(0.5, 2))
        travel = distance_m / 1000 / rng.uniform(18, 25) * 60 + rng.uniform(1, 2)
        times["delivered"] = times["picked_up"] + secs(travel)
        promised = placed + secs(12 + km * 3.5)
        cancelled = rng.random() < 0.025
        if cancelled:
            times["cancelled"] = times["confirmed"] + secs(rng.uniform(0.5, 2))

        steps = [("PLACED", "placed"), ("CONFIRMED", "confirmed")]
        steps += [("CANCELLED", "cancelled")] if cancelled else [
            ("PACKING", "packing"), ("PACKED", "packed"), ("OUT_FOR_DELIVERY", "picked_up"), ("DELIVERED", "delivered")]
        history = [{"order_id": order_id, "seq": i, "status": status, "at": times[key]}
                   for i, (status, key) in enumerate(steps, start=1) if times[key] <= as_of]
        status = history[-1]["status"]
        delivered_at = times["delivered"] if status == "DELIVERED" else None
        cancelled_at = times["cancelled"] if status == "CANCELLED" else None

        order = {"order_id": order_id, "customer_id": customer["customer_id"], "store_id": store["store_id"],
                 "address_id": address["address_id"], "status": status, "item_total_paise": item_total,
                 "discount_paise": discount, "delivery_fee_paise": fee, "total_paise": total, "coupon_code": coupon,
                 "placed_at": placed, "promised_by": promised, "delivered_at": delivered_at,
                 "cancelled_at": cancelled_at, "version": len(history)}
        bags = 1 if len(lines) <= 3 else 2
        rec = OrderRec(order=order, customer=customer, store=store, lines=lines, history=history, bags=bags,
                       times=times, address=address)
        rec.line_bag = {line["line_no"]: 1 + (i * bags) // len(lines) for i, line in enumerate(lines)}

        self.add("orders.orders", order)
        for line in lines:
            self.add("orders.order_lines", line)
        for row in history:
            self.add("orders.order_status_history", row)
        if redemption:
            rec.redemption = self.add("promotions.redemptions", redemption)

        self.build_payment(rec, total)
        if cancelled:
            if times["cancelled"] + secs(1) <= as_of:  # refund one minute after the cancel
                self.add_refund(order, total, "source", "order_cancelled", None, times["cancelled"] + secs(1),
                                "business", "order", "order_cancelled")
        else:
            self.build_fulfilment(rec, pick_times, distance_m, promised)
        self.orders.append(rec)

    def add_refund(self, order: dict, amount: int, destination: str, reason: str, case_id: str | None, at: datetime,
                   route: str, part: str, basis: str) -> dict:
        """a refund the business already made, with the part it paid for (each part is paid at most once)."""
        row = self.add("refunds_wallet.refunds", {
            "refund_id": self.next_id("RF-", 6), "order_id": order["order_id"], "amount_paise": amount,
            "destination": destination, "status": "COMPLETED", "reason": reason, "case_id": case_id,
            "decision_record_id": None, "created_at": at, "customer_id": order["customer_id"], "route": route,
            "idempotency_key": None})
        self.add("refunds_wallet.refund_parts", {"refund_id": row["refund_id"], "order_id": order["order_id"],
                                                 "part": part, "amount_paise": amount, "basis": basis})
        return row

    def build_payment(self, rec: OrderRec, amount: int) -> None:
        rng, placed = self.rng, rec.order["placed_at"]
        method = rng.choices(["upi", "card", "wallet"], weights=[60, 25, 15])[0]
        instrument = rec.customer["instruments"][method]
        noise = rng.random()
        if noise < 0.04:
            # an earlier attempt that failed; in a quarter of these the bank debited and then reversed it
            failed = self.add_intent(rec, method, instrument, amount, "failed", placed - secs(rng.uniform(2, 4)))
            ref_ = self.gateway_ref()
            if noise < 0.01:
                self.add_event(rec, failed, "debit_success", amount, ref_, failed["created_at"] + secs(0.1))
                reversed_at = failed["created_at"] + secs(rng.uniform(10, 60))
                if reversed_at <= self.cfg.as_of:
                    self.add_event(rec, failed, "reversal", amount, ref_, reversed_at)
            else:
                self.add_event(rec, failed, "debit_failed", amount, ref_, failed["created_at"] + secs(0.1))
        intent = self.add_intent(rec, method, instrument, amount, "succeeded", placed - secs(0.7))
        ref_ = self.gateway_ref()
        self.add_event(rec, intent, "debit_success", amount, ref_, intent["created_at"] + secs(0.1))
        self.add_capture(rec, intent, amount, ref_, intent["created_at"] + secs(0.2))

    def add_intent(self, rec, method, instrument, amount, status, at) -> dict:
        row = self.add("payments.payment_intents", {
            "intent_id": self.next_id("PI-", 6), "order_id": rec.order["order_id"],
            "customer_id": rec.order["customer_id"], "method": method, "instrument_ref": instrument,
            "amount_paise": amount, "status": status, "created_at": at})
        rec.intents.append(row)
        return row

    def add_event(self, rec, intent, kind, amount, gateway_ref, at) -> dict:
        row = self.add("payments.gateway_events", {"event_id": self.next_id("GE-", 6), "intent_id": intent["intent_id"],
                                                   "type": kind, "amount_paise": amount, "gateway_ref": gateway_ref,
                                                   "at": at})
        rec.events.append(row)
        return row

    def add_capture(self, rec, intent, amount, gateway_ref, at) -> dict:
        row = self.add("payments.captures", {"capture_id": self.next_id("CAP-", 6), "intent_id": intent["intent_id"],
                                             "amount_paise": amount, "gateway_ref": gateway_ref, "captured_at": at})
        rec.captures.append(row)
        self.add_event(rec, intent, "capture", amount, gateway_ref, at)
        return row

    def build_fulfilment(self, rec: OrderRec, pick_times: list[datetime], distance_m: int, promised: datetime) -> None:
        rng, as_of, times = self.rng, self.cfg.as_of, rec.times
        order_id, store = rec.order["order_id"], rec.store
        picker = rng.choice(store["pickers"])
        for line, at in zip(rec.lines, pick_times):
            if at <= as_of:
                rec.scans.append(self.add_scan(rec, line, "picked", None, picker, at))
        for i, line in enumerate(rec.lines):
            at = times["packed"] - secs((len(rec.lines) - i) * 5 / 60)
            if at <= as_of:
                rec.scans.append(self.add_scan(rec, line, "packed", rec.line_bag[line["line_no"]], picker, at))

        if times["assigned"] > as_of:
            return
        picked_up = times["picked_up"] if times["picked_up"] <= as_of else None
        delivered = times["delivered"] if times["delivered"] <= as_of else None
        trip_id = self.next_id("TRP-", 6)
        rec.trip = self.add("delivery.trips", {
            "trip_id": trip_id, "order_id": order_id, "rider_id": rng.choice(store["riders"]),
            "store_id": store["store_id"], "status": "DELIVERED" if delivered else "PICKED_UP" if picked_up else "ASSIGNED",
            "assigned_at": times["assigned"], "picked_up_at": picked_up, "delivered_at": delivered,
            "eta_at": promised, "distance_m": distance_m})
        if picked_up:
            self.build_gps(rec, times["picked_up"], times["delivered"])
        if delivered:
            handed_to = rng.choices(["customer", "family_member", "security"], weights=[85, 10, 5])[0]
            rec.pod = self.add("delivery.pod", {"order_id": order_id, "trip_id": trip_id,
                                                "otp_verified": handed_to != "security", "bags_handed": rec.bags,
                                                "handed_to": handed_to, "delivered_at": delivered})
            if (handed_to != "customer" or rng.random() < 0.03) and delivered + secs(0.3) <= as_of:
                rec.notes.append(self.add("delivery.partner_notes", {
                    "note_id": self.next_id("PN-", 6), "trip_id": trip_id, "order_id": order_id,
                    "text": rng.choice(ref.PARTNER_NOTES[handed_to]), "at": delivered + secs(0.3)}))

    def add_scan(self, rec, line, event, bag, picker, at) -> dict:
        return self.add("store_inventory.pick_pack_scans", {
            "scan_id": self.next_id("SC-", 7), "order_id": rec.order["order_id"], "store_id": rec.store["store_id"],
            "sku": line["sku"], "event": event, "qty": line["qty"], "bag": bag, "picker": picker, "at": at})

    def build_gps(self, rec: OrderRec, start: datetime, end: datetime, stop: tuple[float, float] | None = None) -> None:
        """one point a minute from the store to the address; stop = (fraction of the way, minutes standing)."""
        for row in rec.gps:
            self.t["delivery.gps_points"].remove(row)
        rec.gps = []
        a = (rec.store["lat"], rec.store["lng"])
        b = (rec.address["lat"], rec.address["lng"])
        total = (end - start).total_seconds()
        standing = stop[1] * 60 if stop else 0.0
        moving = max(total - standing, 1.0)
        offsets = [float(s) for s in range(0, int(total) + 1, 60)]
        if offsets[-1] < total:
            offsets.append(total)
        for seq, t in enumerate(offsets, start=1):
            if start + timedelta(seconds=t) > self.cfg.as_of:
                break
            if stop and stop[0] * moving <= t <= stop[0] * moving + standing:
                frac = stop[0]
            elif stop and t > stop[0] * moving + standing:
                frac = (t - standing) / moving
            else:
                frac = t / moving
            frac = min(frac, 1.0)
            jitter = 0.0 if frac in (0.0, 1.0) else 0.0003
            rec.gps.append(self.add("delivery.gps_points", {
                "trip_id": rec.trip["trip_id"], "seq": seq, "at": start + timedelta(seconds=round(t)),
                "lat": round(a[0] + (b[0] - a[0]) * frac + self.rng.uniform(-jitter, jitter), 6),
                "lng": round(a[1] + (b[1] - a[1]) * frac + self.rng.uniform(-jitter, jitter), 6)}))

    # fraud rings and earlier claims (blueprint 7.4, 10.7)

    def delivered_by_customer(self, skip: frozenset[str] | set[str] = frozenset(),
                              settled_before: datetime | None = None) -> dict[str, list[OrderRec]]:
        found: dict[str, list[OrderRec]] = defaultdict(list)
        for rec in self.orders:
            order = rec.order
            if order["status"] != "DELIVERED" or order["order_id"] in skip:
                continue
            if settled_before is not None and order["delivered_at"] > settled_before:
                continue
            found[order["customer_id"]].append(rec)
        return found

    def build_fraud_rings(self) -> None:
        """groups of regular accounts that share a device, a delivery address or a upi id (one kind per ring). their
        claims come in build_past_claims; a member may also make a false claim (faults)."""
        rng, size = self.side, self.cfg.ring_size
        delivered = self.delivered_by_customer()
        candidates = sorted(c["customer_id"] for c in self.customers
                            if c["persona"] == "regular" and len(delivered[c["customer_id"]]) >= 4)
        rng.shuffle(candidates)
        if len(candidates) < self.cfg.fraud_rings * size:
            raise GenerationError("not enough customers with 4 delivered orders for the fraud rings; use a bigger config")
        customers = {c["customer_id"]: c for c in self.customers}
        sessions = {s["customer_id"]: s for s in self.t["identity.sessions"]}
        for n in range(self.cfg.fraud_rings):
            members = sorted(candidates[n * size:(n + 1) * size])
            link = LINK_TYPES[n % len(LINK_TYPES)]
            first = customers[members[0]]
            if link == "device":
                shared = sessions[members[0]]["device_id"]
                for member in members[1:]:
                    sessions[member]["device_id"] = shared
            elif link == "address":
                home = first["addresses"][0]
                shared = home["address_id"]
                for member in members[1:]:
                    copy = {**home, "address_id": self.next_id("ADDR-", 5), "customer_id": member, "label": "Other"}
                    self.add("customer.addresses", copy)
                    customers[member]["addresses"].append(copy)
            else:
                shared = first["instruments"]["upi"]
                for member in members:
                    intents = [i for rec in self.orders if rec.order["customer_id"] == member for i in rec.intents]
                    if not any(i["method"] == "upi" for i in intents):
                        intents[-1]["method"] = "upi"
                    for intent in intents:
                        if intent["method"] == "upi":
                            intent["instrument_ref"] = shared
                    customers[member]["instruments"]["upi"] = shared
            ring_id = f"RING-{n + 1:02d}"
            self.rings.append({"ring_id": ring_id, "link_type": link, "members": members})
            self.ring_of.update({member: ring_id for member in members})
            self.add("truth.fraud_rings", {"ring_id": ring_id, "link_type": link, "shared": shared,
                                           "members": members})

    def build_past_claims(self, fault_orders: set[str]) -> None:
        """closed support cases on delivered orders: most customers have none, some one or two, ring members many
        (blueprint 3.5: claims on most orders). never on an order that carries a fault."""
        rng, as_of = self.side, self.cfg.as_of
        delivered = self.delivered_by_customer(skip=fault_orders, settled_before=as_of - timedelta(hours=2))
        for customer in self.customers:
            recs = delivered[customer["customer_id"]]
            if not recs:
                continue
            ring = customer["customer_id"] in self.ring_of
            if ring:
                picks = rng.sample(recs, min(len(recs), max(2, round(len(recs) * 0.75))))
            else:
                draw = rng.random()
                if draw >= self.cfg.claim_rate:
                    continue
                picks = rng.sample(recs, 2 if draw < self.cfg.claim_rate / 4 and len(recs) >= 2 else 1)
            for rec in sorted(picks, key=lambda r: (r.order["placed_at"], r.order["order_id"])):
                self.add_past_claim(rec, ring)

    def add_past_claim(self, rec: OrderRec, ring: bool) -> None:
        """one closed case with its history: refunded by the rules, turned down (the records did not support it), or
        decided by a person on a ticket. a refund made for it carries the case id."""
        rng, order = self.side, rec.order
        issue = "item_missing_or_wrong" if ring else rng.choices(CLAIM_TYPES, weights=[6, 3, 1])[0]
        outcome = rng.choices(["refunded", "rejected", "escalated"], weights=[6, 1, 3] if ring else [7, 2, 1])[0]
        approved = outcome == "escalated" and rng.random() < 0.5
        path = ["INVESTIGATING", "DECIDING"]
        if outcome == "refunded":
            path += ["ACTIONING", "RESOLVED"]
        elif outcome == "rejected":
            path += ["RESOLVED"]
        else:
            path += ["ESCALATED"] + (["ACTIONING"] if approved else []) + ["RESOLVED"]
        path.append("CLOSED")
        resolved_why = {"refunded": "refunded", "rejected": "not_supported_by_records"}.get(outcome,
                                                                                         "decided_by_person")
        case_id = f"CASE-{rng.getrandbits(48):012X}"
        at = order["delivered_at"] + secs(rng.uniform(10, 90))
        opened = at
        self.add("support.case_events", {"case_id": case_id, "seq": 1, "type": "opened", "from_state": None,
                                         "to_state": "OPEN", "reason": None, "at": at})
        state, ticket = "OPEN", None
        for seq, to in enumerate(path, start=2):
            at = at + secs(rng.uniform(1, 4))
            if to == "ACTIONING":
                line = rng.choice(rec.lines)
                self.add_refund(order, line["qty"] * line["unit_paise"], rng.choice(["source", "wallet"]),
                                f"{issue}_claim", case_id, at, "person" if outcome == "escalated" else "auto",
                                f"line:{line['line_no']}", "proven")
            if to == "ESCALATED":
                ticket = self.add("support.tickets", {
                    "ticket_id": self.next_id("TKT-", 6), "case_id": case_id, "status": "OPEN", "priority": "normal",
                    "summary": "earlier claim: " + issue.replace("_", " "), "created_at": at, "updated_at": at})
            self.add("support.case_events", {"case_id": case_id, "seq": seq, "type": "moved", "from_state": state,
                                             "to_state": to, "reason": resolved_why if to == "RESOLVED" else None,
                                             "at": at})
            state = to
        if ticket is not None:
            ticket["status"] = "APPROVED" if approved else "REJECTED"
            ticket["updated_at"] = at
        self.add("support.cases", {"case_id": case_id, "customer_id": order["customer_id"],
                                   "order_id": order["order_id"], "issue_type": issue, "state": "CLOSED",
                                   "created_at": opened, "updated_at": at})
