"""the policy knowledge provider (c16): the active sections of a case type, whole, as the backend gives them; coupons
keep their offer terms. no database: the backend is a stand-in here."""

from hivemind_agents.backend_client import NotFound
from hivemind_agents.capabilities import policy_provider

SECTION = {"policy_id": "refund_policy", "version": "1a2b3c4d", "title": "Refunds", "body": "A refund goes back."}


class FakeBackend:
    def __init__(self, sections=(), offers=None):
        self.sections = list(sections)
        self.offers = offers or {}
        self.asked: list[str] = []

    def policy_sections(self, issue_type):
        self.asked.append(issue_type)
        return {"issue_type": issue_type, "sections": self.sections, "evidence_id": "ev_policy1"}

    def offer(self, code):
        if code not in self.offers:
            raise NotFound(code)
        return self.offers[code]


def test_a_section_keeps_its_id_version_text_and_evidence():
    backend = FakeBackend([SECTION])
    (section,) = policy_provider.sections_for("refund", backend)
    assert section.ref == "refund_policy@1a2b3c4d"
    assert section.text == "A refund goes back."
    assert section.evidence_ref == "ev_policy1"
    assert backend.asked == ["refund"]


def test_every_section_for_the_case_type_is_handed_over_whole():
    backend = FakeBackend([SECTION, {**SECTION, "policy_id": "bod_policy", "version": "00000001"}])
    sections = policy_provider.sections_for("item_missing_or_wrong", backend)
    assert [s.policy_id for s in sections] == ["refund_policy", "bod_policy"]


def test_a_case_type_with_no_policy_text_has_no_sections():
    assert policy_provider.sections_for("where_is_my_order", FakeBackend([])) == []


def test_a_coupon_gives_its_offer_terms_and_reads_no_policy():
    backend = FakeBackend(offers={"SAVE10": {"title": "Save 10", "terms": "Valid on the first order.",
                                             "evidence_id": "ev_offer"}})
    (section,) = policy_provider.sections_for("offers_coupons", backend, coupon_code="SAVE10")
    assert section.policy_id == "offer_terms/SAVE10"
    assert section.evidence_ref == "ev_offer"
    assert backend.asked == []


def test_a_coupon_that_does_not_exist_has_no_terms():
    assert policy_provider.sections_for("offers_coupons", FakeBackend(), coupon_code="NOPE") == []


def test_a_coupon_case_without_a_code_has_no_terms():
    assert policy_provider.sections_for("offers_coupons", FakeBackend()) == []
