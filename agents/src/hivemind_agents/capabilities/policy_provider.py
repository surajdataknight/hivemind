"""c16 policy knowledge provider: hands over the active policy and sop sections for a case type, whole. no search:
the case type decides which sections (the business's `policy.map`). coupons keep their own terms, by offer code.

each section carries policy_id@version; the version is the hash of its text (the business checks that), so a changed
text is a new version. the sections come from one read, so one evidence id covers them all.
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


def policy_sections(case_type: str, backend: BackendClient) -> list[PolicySection]:
    read = backend.policy_sections(case_type)
    return [PolicySection(policy_id=s["policy_id"], version=s["version"], title=s["title"], text=s["body"],
                          evidence_ref=read["evidence_id"]) for s in read["sections"]]


def sections_for(case_type: str, backend: BackendClient, *, coupon_code: str | None = None) -> list[PolicySection]:
    if case_type == "offers_coupons":
        return offer_terms(coupon_code, backend) if coupon_code else []
    return policy_sections(case_type, backend)
