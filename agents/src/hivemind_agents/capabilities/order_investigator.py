"""c8 order investigator: the order, its lines and their status, and which lines the customer's words name. plain
code: the business has no free-text substitution notes, so the model the blueprint allows here has nothing to read.

findings: order.status (always), order.line_not_delivered (a line the store marked unavailable or substituted),
order.claimed_lines (the lines the customer named and what they cost), order.claim_not_on_order (named items that
match no line). the claimed value is what the risk check reads from the board.
"""

from hivemind_agents.capabilities import items
from hivemind_agents.capabilities.checking import finding, rupees
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents.contracts import CheckResult, Task


def line_value(line: dict) -> int:
    return line["qty"] * line["unit_paise"]


def run(task: Task, check: CheckInput) -> CheckResult:
    order = check.backend.order(task.params["order_id"])
    ref = order["evidence_id"]
    lines = order["lines"]
    when = {h["status"]: h["at"] for h in order["status_history"]}
    found = [finding(task, check, "order.status",
                     f"Order {order['order_id']} is {order['status']}, with {len(lines)} lines worth "
                     f"{rupees(order['item_total_paise'])}.", [ref],
                     facts={"order_id": order["order_id"], "status": order["status"], "store_id": order["store_id"],
                            "placed_at": order["placed_at"], "packed_at": when.get("PACKED"),
                            "delivered_at": order["delivered_at"], "item_total_paise": order["item_total_paise"],
                            "total_paise": order["total_paise"], "discount_paise": order["discount_paise"],
                            "lines": [{k: line[k] for k in ("line_no", "sku", "name", "qty", "unit_paise", "status")}
                                      for line in lines]})]
    for line in lines:
        if line["status"] != "ORDERED":
            found.append(finding(task, check, "order.line_not_delivered",
                                 f"{line['name']} is marked {line['status'].lower()} on the order.", [ref],
                                 facts={"line_no": line["line_no"], "sku": line["sku"], "status": line["status"],
                                        "value_paise": line_value(line)}))
    issue = check.issue
    mentioned = issue.items_mentioned if issue is not None else []
    if mentioned:
        claimed, unmatched, unclear = items.claimed_lines(mentioned, lines)
        if claimed:
            value = sum(line_value(line) for line in claimed)
            names = ", ".join(f"{line['name']} ({rupees(line_value(line))})" for line in claimed)
            found.append(finding(
                task, check, "order.claimed_lines", f"The items the customer named are on the order: {names}.",
                [ref], confidence="medium" if unclear or unmatched else "high",
                basis="the customer's words matched to line names by code" + ("; a word fit two lines" if unclear
                                                                               else ""),
                facts={"lines": [{"line_no": line["line_no"], "sku": line["sku"], "name": line["name"],
                                  "qty": line["qty"], "value_paise": line_value(line)} for line in claimed],
                       "claim_value_paise": value, "unclear": unclear, "unmatched": unmatched}))
        if unmatched:
            found.append(finding(task, check, "order.claim_not_on_order",
                                 f"{len(unmatched)} item(s) the customer named match no line of the order.", [ref],
                                 confidence="medium", basis="no line name fits the customer's words",
                                 facts={"mentions": unmatched}))
    return CheckResult(findings=found)


CAPABILITY = Capability(
    name="order", description="the order, its lines and their status, and which lines the customer named",
    params=("order_id",), scopes=frozenset({"orders:read:self"}), run=run, cost="low")
