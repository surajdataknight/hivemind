"""every answer shape hivemind asks a model for must be one the provider accepts. gemini's sdk turns the pydantic
model into its answer schema before anything is sent, and refuses some shapes (a free map, for one); a scripted model
never sees the schema, so only this test catches that. offline: the sdk's own conversion, no request is made.
google's own limit on how complex a shape may be is checked only on its side (the reconciler's full shape was refused
there, 2026-11-13): google gets each shape without its length limits (llm.loose_schema), built here too."""

import ast
from importlib import resources

import pytest
from google import genai
from google.genai import _transformers  # the sdk's schema conversion (google-genai is pinned in pyproject)

from hivemind_agents.capabilities.intake import IntakeDraft
from hivemind_agents.capabilities.quarantined_reader import PartnerNoteLabel, StoreNoteLabel
from hivemind_agents.capabilities.reconciler import ReconcilerAnswer
from hivemind_agents.contracts import PlanV1

SHAPES = {"IntakeDraft": IntakeDraft, "PlanV1": PlanV1, "PartnerNoteLabel": PartnerNoteLabel,
          "StoreNoteLabel": StoreNoteLabel, "ReconcilerAnswer": ReconcilerAnswer}


def shapes_asked_for() -> set[str]:
    """the last argument of every `.structured(...)` call in hivemind's code: the shape the model must answer in."""
    found = set()
    for path in resources.files("hivemind_agents").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "structured" and node.args):
                last = node.args[-1]
                found.add(last.id if isinstance(last, ast.Name) else ast.unparse(last))
    return found


def test_every_shape_asked_for_is_checked_here():
    assert shapes_asked_for() == set(SHAPES)  # a new structured call needs its shape added to SHAPES


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_gemini_accepts_the_shape(name):
    client = genai.Client(api_key="offline-check")  # never used to send
    _transformers.t_schema(client._api_client, SHAPES[name])  # raises on a shape gemini refuses
    from hivemind_agents.llm import GoogleClient

    assert GoogleClient("offline-check")._config("s", 0.0, SHAPES[name], 20).response_json_schema
