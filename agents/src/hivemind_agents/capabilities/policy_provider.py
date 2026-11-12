"""c16 policy knowledge provider, first version: looks up the policy sections for a case type and hands them over
whole. no search: the case type decides which sections. so far it knows one: an offer's terms, by its code.

each section carries policy_id@version; the version is a hash of the text, so a changed text is a new version.
"""

import hashlib

from hivemind_agents.backend_client import BackendClient, NotFound
from hivemind_agents.contracts import PolicySection


def offer_terms(code: str, backend: BackendClient) -> list[PolicySection]:
    try:
        offer = backend.offer(code)
    except NotFound:
        return []
    version = hashlib.sha256(offer["terms"].encode("utf-8")).hexdigest()[:8]
    return [PolicySection(policy_id=f"offer_terms/{code}", version=version, title=offer["title"],
                          text=offer["terms"], evidence_ref=offer["evidence_id"])]


def sections_for(case_type: str, backend: BackendClient, *, coupon_code: str | None = None) -> list[PolicySection]:
    if case_type == "offers_coupons" and coupon_code:
        return offer_terms(coupon_code, backend)
    return []
