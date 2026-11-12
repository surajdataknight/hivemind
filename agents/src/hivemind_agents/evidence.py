"""evidence ids: the business names each read's answer by a hash of its body. hivemind hashes the body again, so a
read whose id does not match what came back is never used as proof."""

import hashlib
import json
from typing import Any


def body_hash(body: dict[str, Any]) -> str:
    """sha256 of the body without its evidence id, in the business's canonical json."""
    rest = {k: v for k, v in body.items() if k != "evidence_id"}
    text = json.dumps(rest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def matches(body: dict[str, Any]) -> bool:
    return body.get("evidence_id") == "ev_" + body_hash(body)[:32]
