"""the investigators the case lead may plan and the scheduler may run (phase 2.4). each only reads, and posts its
findings with proof to the case board. promotions (c12) keeps its own short path; the media analyst (c13) comes in
phase 4; the policy provider (c16) feeds the deciding steps, not the board."""

from hivemind_agents.capabilities import (delivery_investigator, history_analyst, inventory_investigator,
                                          order_investigator, payment_investigator, risk_investigator)
from hivemind_agents.capabilities.registry import Capability

ALL: dict[str, Capability] = {c.name: c for c in (
    order_investigator.CAPABILITY,
    delivery_investigator.CAPABILITY,
    payment_investigator.CAPABILITY,
    inventory_investigator.CAPABILITY,
    history_analyst.CAPABILITY,
    risk_investigator.CAPABILITY,
)}
