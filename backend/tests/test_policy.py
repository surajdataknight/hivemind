"""the policy and sop text: the active sections for each issue type, whole, with a version that is the hash of the text;
an edited text is a new version, and only one version of a section is active at a time."""

import hashlib
import os
import re
import typing
from datetime import datetime, timedelta, timezone

import jwt
import psycopg
import pytest
from fastapi.testclient import TestClient

from hivemind_backend import auth, db, models
from hivemind_backend.app import app


@pytest.fixture(scope="module")
def client(test_db):
    return TestClient(app)


def bearer(customer_id: str, scopes=("policy:read",)) -> dict:
    """a case token as hivemind mints it, for this read only."""
    issued = datetime.now(timezone.utc)
    claims = {"iss": auth.CASE_ISSUER, "aud": auth.CASE_AUDIENCE, "sub": customer_id, "sid": "SES-000001",
              "scopes": list(scopes), "iat": issued, "exp": issued + timedelta(minutes=10)}
    return {"Authorization": f"Bearer {jwt.encode(claims, os.environ['HM_CASE_TOKEN_SECRET'], algorithm='HS256')}"}


def version_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


@pytest.mark.db
def test_a_stored_version_must_be_the_hash_of_its_text(test_db_name):
    with db.connect("admin", dbname=test_db_name) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute("insert into policy.sections (policy_id, version, title, body, active) values "
                         "('refund_policy', '00000000', 'Refunds', 'a text', false)")
        conn.rollback()


def customer(tables) -> str:
    return tables["orders.orders"][0]["customer_id"]


def sections(client, tables, issue_type):
    response = client.get("/v1/policy/sections", params={"issue_type": issue_type},
                          headers=bearer(customer(tables)))
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.db
def test_a_missing_item_gets_its_three_sections_whole(client, test_db):
    body = sections(client, test_db, "item_missing_or_wrong")
    assert [s["policy_id"] for s in body["sections"]] == ["bod_policy", "missing_item_sop", "refund_policy"]
    assert all(s["body"].strip() for s in body["sections"])


@pytest.mark.db
def test_a_version_is_the_hash_of_the_text(client, test_db):
    for section in sections(client, test_db, "quality")["sections"]:
        assert section["version"] == hashlib.sha256(section["body"].encode("utf-8")).hexdigest()[:8]


@pytest.mark.db
def test_coupons_get_no_policy_sections_they_keep_their_offer_terms(client, test_db):
    assert sections(client, test_db, "offers_coupons")["sections"] == []


@pytest.mark.db
@pytest.mark.parametrize("issue_type", sorted(set(typing.get_args(models.IssueType)) - {"offers_coupons"}))
def test_every_case_type_has_its_policy_text(client, test_db, issue_type):
    body = sections(client, test_db, issue_type)
    assert body["sections"] and all(s["body"].strip() for s in body["sections"])


@pytest.mark.db
@pytest.mark.parametrize("issue_type, expected", [
    ("cancellation", ["cancellation_policy", "refund_policy"]),
    ("delivery_partner_behaviour", ["delivery_partner_sop", "refund_policy"]),
    ("where_is_my_order", ["order_status_sop"]),
])
def test_the_case_types_added_later_get_their_sections(client, test_db, issue_type, expected):
    assert [s["policy_id"] for s in sections(client, test_db, issue_type)["sections"]] == expected


@pytest.mark.db
def test_an_unknown_issue_type_is_refused(client, test_db):
    response = client.get("/v1/policy/sections", params={"issue_type": "nonsense"},
                          headers=bearer(customer(test_db)))
    assert response.status_code == 422


@pytest.mark.db
def test_an_edited_text_is_a_new_version_and_only_the_active_one_is_served(client, test_db, test_db_name):
    before = sections(client, test_db, "item_missing_or_wrong")
    old = next(s for s in before["sections"] if s["policy_id"] == "bod_policy")
    text = "an edited text for the test"
    with db.connect("admin", dbname=test_db_name) as conn:
        conn.execute("insert into policy.sections (policy_id, version, title, body, active) values "
                     "('bod_policy', %s, 'Benefit of the doubt', %s, false)", (version_of(text), text))
        versions = conn.execute("select version from policy.sections where policy_id = 'bod_policy'").fetchall()
        assert len({row[0] for row in versions}) == 2  # the edit is a second version of the same section
        conn.rollback()
    after = sections(client, test_db, "item_missing_or_wrong")
    assert next(s for s in after["sections"] if s["policy_id"] == "bod_policy") == old


@pytest.mark.db
def test_two_active_versions_of_one_section_are_refused(test_db_name):
    text = "a second active text"
    with db.connect("admin", dbname=test_db_name) as conn:
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute("insert into policy.sections (policy_id, version, title, body, active) values "
                         "('refund_policy', %s, 'Refunds', %s, true)", (version_of(text), text))
        conn.rollback()


# the text may only promise what hivemind can do. the remedies are refund, wallet credit and cancel (record 009:
# no "change order"; build phases 3.3); anything else in a text would reach a customer as a promise nobody keeps.
# phase 3.1 adds the full check that every rule has its text and every text its rule (blueprint a.5).
REMEDIES_HIVEMIND_CANNOT_GIVE = re.compile(
    r"\b(replac\w*|exchang\w*|re-?ship\w*|re-?deliver\w*|resend\w*|send (it|them|the \w+) again|new order|"
    r"voucher\w*|cashback|gift)\b", re.IGNORECASE)


def texts_handed_out(test_db_name) -> list[tuple[str, str]]:
    """everything the policy provider can hand over: the active sections and every offer's terms."""
    with db.connect("admin", dbname=test_db_name) as conn:
        sections = conn.execute("select policy_id, body from policy.sections where active").fetchall()
        offers = conn.execute("select 'offer_terms/' || code, terms from promotions.offers").fetchall()
    return [(name, text) for name, text in sections + offers]


@pytest.mark.db
def test_no_text_promises_a_remedy_hivemind_cannot_give(test_db_name):
    promised = [(name, match.group(0)) for name, text in texts_handed_out(test_db_name)
                for match in REMEDIES_HIVEMIND_CANNOT_GIVE.finditer(text)]
    assert promised == []


@pytest.mark.db
def test_the_refund_text_names_every_place_a_refund_can_go(test_db_name):
    destinations = set(typing.get_args(models.Refund.model_fields["destination"].annotation))
    assert destinations == {"source", "wallet"}  # a new destination needs its words below and in the text
    words = {"source": "paid with", "wallet": "wallet"}
    body = dict(texts_handed_out(test_db_name))["refund_policy"]
    assert [d for d in sorted(destinations) if words[d] not in body] == []


def test_the_remedy_check_catches_what_it_should():
    for text in ("It can be refunded or replaced.", "We will send it again.", "A replacement is sent.",
                 "You get a voucher.", "We exchange it.", "It is re-delivered."):
        assert REMEDIES_HIVEMIND_CANNOT_GIVE.search(text), text
    for text in ("A refund goes back to the wallet.", "The order can be cancelled.", "Credit to the wallet."):
        assert not REMEDIES_HIVEMIND_CANNOT_GIVE.search(text), text


@pytest.mark.db
def test_the_map_names_only_sections_that_exist(test_db_name):
    with db.connect("admin", dbname=test_db_name) as conn:
        dangling = conn.execute(
            "select m.issue_type, m.policy_id from policy.map m where not exists (select 1 from "
            "policy.sections s where s.policy_id = m.policy_id and s.active)").fetchall()
        unused = conn.execute(
            "select s.policy_id from policy.sections s where s.active and not exists (select 1 from "
            "policy.map m where m.policy_id = s.policy_id)").fetchall()
    assert dangling == []
    assert unused == []  # every active section is read by at least one issue type
