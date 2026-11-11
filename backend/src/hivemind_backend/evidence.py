"""evidence ids: a hash of the response body, so a finding can cite exactly what a read returned."""

import hashlib
import json
from typing import Any


def canonical_json(body: Any) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def evidence_id(body: dict) -> str:
    return "ev_" + hashlib.sha256(canonical_json(body)).hexdigest()[:32]
