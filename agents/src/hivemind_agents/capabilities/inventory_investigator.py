"""c11 inventory & store investigator: what the dark store's records say about packing the order. plain code over
the pick and pack scans, the stock events and the store's incident log; staff incident notes are free text, so a
quarantined reader turns each into a label.

findings: store.item_not_scanned (a line with no pick or pack scan), store.item_scanned (a named item that was
packed, and into which bag), store.bags_packed, store.stock_out_before_pack (the item ran out just before packing
and was not restocked in time), store.incident (an incident near the order's time, with its note's label).
"""

from datetime import timedelta

from hivemind_agents.capabilities import items, quarantined_reader
from hivemind_agents.capabilities.checking import at, finding, iso, minutes
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents.contracts import CheckResult, Task

BEFORE = timedelta(hours=2)   # the window of stock events and incidents read around the order
AFTER = timedelta(hours=1)
MAX_SKUS = 6                  # stock events are read per item: the named and unscanned items, at most this many


def run(task: Task, check: CheckInput) -> CheckResult:
    order_id = task.params["order_id"]
    backend = check.backend
    order = backend.order(order_id)
    scans = backend.scans(order_id)
    order_ref, scans_ref = order["evidence_id"], scans["evidence_id"]
    lines = {line["sku"]: line for line in order["lines"]}
    when = {h["status"]: at(h["at"]) for h in order["status_history"]}
    placed = at(order["placed_at"])
    packed_at = when.get("PACKED") or when.get("PACKING") or placed

    named = items.claimed_lines(check.issue.items_mentioned, order["lines"])[0] if check.issue else []
    found = []
    missing = {m["sku"]: m for m in scans["missing_scans"]}
    for sku, gap in missing.items():
        found.append(finding(task, check, "store.item_not_scanned",
                             f"{lines[sku]['name']} has no {gap['expected']} scan at the store.", [scans_ref, order_ref],
                             facts={"sku": sku, "line_no": gap["line_no"], "expected": gap["expected"]}))
    packed = {s["sku"]: s for s in scans["scans"] if s["event"] == "packed"}
    for line in named:
        scan = packed.get(line["sku"])
        if scan is not None:
            found.append(finding(task, check, "store.item_scanned",
                                 f"{line['name']} was scanned into bag {scan['bag']} at packing.", [scans_ref],
                                 facts={"sku": line["sku"], "bag": scan["bag"], "packed_at": scan["at"]}))
    bags = sorted({s["bag"] for s in packed.values() if s["bag"] is not None})
    if bags:
        found.append(finding(task, check, "store.bags_packed", f"The store packed the order into {len(bags)} bag(s).",
                             [scans_ref], facts={"bags": len(bags)}))

    since, until = iso(placed - BEFORE), iso(packed_at + AFTER)
    focus = list(dict.fromkeys([line["sku"] for line in named] + list(missing)))[:MAX_SKUS]
    for sku in focus:
        events = backend.stock_events(order["store_id"], sku, since, until)
        out = _stock_out_before(events["events"], packed_at)
        if out is not None:
            found.append(finding(
                task, check, "store.stock_out_before_pack",
                f"{lines[sku]['name']} ran out at the store {minutes(at(out['at']), packed_at)} min before the order "
                "was packed, with no restock in between.", [events["evidence_id"]],
                facts={"sku": sku, "out_at": out["at"], "minutes_before_pack": minutes(at(out["at"]), packed_at)}))

    incidents = backend.store_incidents(order["store_id"], since, until)
    for incident in incidents["incidents"]:
        if incident["sku"] is not None and incident["sku"] not in lines:
            continue  # about an item this order does not have
        label = quarantined_reader.read("store", incident["description"], check.llm)
        about = f" for {lines[incident['sku']]['name']}" if incident["sku"] else ""
        found.append(finding(
            task, check, "store.incident",
            f"The store logged {incident['type'].replace('_', ' ')}{about}; its note reads as: "
            f"{quarantined_reader.words(label)}.", [incidents["evidence_id"]],
            confidence="low" if label == "unreadable" else "medium",
            basis="incident type from the log; the note labelled by the quarantined reader",
            facts={"incident_id": incident["incident_id"], "type": incident["type"], "sku": incident["sku"],
                   "at": incident["at"], "note_label": label,
                   "minutes_from_pack": minutes(packed_at, at(incident["at"]))}))
    return CheckResult(findings=found)


def _stock_out_before(events: list[dict], packed_at) -> dict | None:
    """the last stock-out before packing, if no restock came between it and packing."""
    last = None
    for event in sorted(events, key=lambda e: e["at"]):
        if at(event["at"]) > packed_at:
            break
        if event["event_type"] == "stock_out":
            last = event
        elif event["event_type"] == "restock":
            last = None
    return last


CAPABILITY = Capability(
    name="inventory_store",
    description="the dark store's pick and pack scans, bags packed, stock-outs and incidents around the order",
    params=("order_id",), scopes=frozenset({"orders:read:self", "inventory:read"}), run=run, cost="medium")
