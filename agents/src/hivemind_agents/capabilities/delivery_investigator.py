"""c9 delivery investigator: the trip (times against the promised time, where the partner stood still), the handover
(otp, bags handed over, to whom) and the partner's notes. plain code, except the notes: free text from outside, read
only by the quarantined reader, which turns each into a label.

findings: delivery.trip, delivery.late, delivery.partner_stopped, delivery.gps_missing, delivery.handover,
delivery.partner_note (the label only; the note's words are never copied).
"""

import math

from hivemind_agents.capabilities import quarantined_reader
from hivemind_agents.capabilities.checking import at, finding, minutes
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents.contracts import CheckResult, Task

LATE_MIN = 10       # minutes past the promised time that count as late
STILL_M = 60        # gps points this close to the first point of a stretch count as standing still
STILL_MIN = 8       # a stretch this long, after pickup and before delivery, is reported


def metres(a: dict, b: dict) -> float:
    lat1, lng1, lat2, lng2 = map(math.radians, (a["lat"], a["lng"], b["lat"], b["lng"]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def longest_stop(points: list[dict]) -> tuple[int, dict, dict] | None:
    """(minutes, first point, last point) of the longest stretch spent within STILL_M of where it began."""
    best = None
    i = 0
    while i < len(points):
        j = i
        while j + 1 < len(points) and metres(points[i], points[j + 1]) <= STILL_M:
            j += 1
        span = minutes(at(points[i]["at"]), at(points[j]["at"]))
        if j > i and (best is None or span > best[0]):
            best = (span, points[i], points[j])
        i = j + 1
    return best


def run(task: Task, check: CheckInput) -> CheckResult:
    order_id = task.params["order_id"]
    backend = check.backend
    trip = backend.trip(order_id)
    pod = backend.proof_of_delivery(order_id)
    notes = backend.partner_notes(order_id)
    now = at(backend.clock()["now"])
    found = []
    if trip is not None:
        ref = trip["evidence_id"]
        eta, delivered = at(trip["eta_at"]), at(trip["delivered_at"])
        late = minutes(eta, delivered or now) if trip["status"] != "CANCELLED" else 0
        points = trip["gps_points"]
        found.append(finding(task, check, "delivery.trip",
                             f"The trip is {trip['status'].lower().replace('_', ' ')}; promised by {trip['eta_at']}.",
                             [ref], facts={"trip_id": trip["trip_id"], "status": trip["status"],
                                           "assigned_at": trip["assigned_at"], "picked_up_at": trip["picked_up_at"],
                                           "delivered_at": trip["delivered_at"], "eta_at": trip["eta_at"],
                                           "late_by_min": max(0, late), "distance_m": trip["distance_m"],
                                           "gps_points": len(points)}))
        if late >= LATE_MIN:
            done = "was delivered" if delivered else "is still on its way,"
            found.append(finding(task, check, "delivery.late",
                                 f"The order {done} {late} minutes after the promised time.", [ref],
                                 facts={"late_by_min": late, "delivered": delivered is not None}))
        if trip["picked_up_at"] and len(points) < 2:
            found.append(finding(task, check, "delivery.gps_missing", "The trip has too few gps points to follow.",
                                 [ref], confidence="low", basis="fewer than 2 gps points after pickup",
                                 facts={"gps_points": len(points)}))
        stop = longest_stop(points)
        if stop is not None and stop[0] >= STILL_MIN:
            found.append(finding(task, check, "delivery.partner_stopped",
                                 f"The delivery partner stayed in one place for {stop[0]} minutes on the way.", [ref],
                                 confidence="medium", basis="gps points within 60 m of each other",
                                 facts={"minutes": stop[0], "from_at": stop[1]["at"], "to_at": stop[2]["at"]}))
    if pod is not None:
        found.append(finding(
            task, check, "delivery.handover",
            f"Handed over to the {pod['handed_to'].replace('_', ' ')}: {pod['bags_handed']} bag(s), otp "
            f"{'verified' if pod['otp_verified'] else 'not verified'}.", [pod["evidence_id"]],
            facts={"otp_verified": pod["otp_verified"], "bags_handed": pod["bags_handed"],
                   "handed_to": pod["handed_to"], "delivered_at": pod["delivered_at"]}))
    for note in notes["notes"]:
        label = quarantined_reader.read("partner", note["text"], check.llm)
        found.append(finding(task, check, "delivery.partner_note",
                             f"A note from the delivery partner reads as: {quarantined_reader.words(label)}.",
                             [notes["evidence_id"]], confidence="low" if label == "unreadable" else "medium",
                             basis="labelled by the quarantined reader; the note's words are not kept here",
                             facts={"note_id": note["note_id"], "label": label, "at": note["at"]}))
    return CheckResult(findings=found)


CAPABILITY = Capability(
    name="delivery",
    description="the delivery trip against the promised time, where the partner stood still, the handover (otp, "
                "bags) and the partner's notes",
    params=("order_id",), scopes=frozenset({"delivery:read:self"}), run=run, cost="medium")
