"""Locale boundary adapter — the shippable product."""

from __future__ import annotations

import json

from indicquant.agent.adapter import BoundaryAdapter
from indicquant.agent.schema import magentic_agent_card, openai_tools
from indicquant.agent.vision import form_scene


def test_bind_native_city_to_iata() -> None:
    bound = BoundaryAdapter().bind_args(
        "book_ticket", {"origin": "मुंबई", "destination": "दिल्ली", "passengers": 2}
    )
    assert bound == {"origin": "BOM", "destination": "DEL", "passengers": 2}


def test_bind_native_digits_in_calculator() -> None:
    bound = BoundaryAdapter().bind_args("calculator", {"expr": "५०० + ५००*१८/१००"})
    assert "500" in bound["expr"]


def test_bind_native_digits_in_gst() -> None:
    bound = BoundaryAdapter().bind_args("inr_gst", {"amount": "५००", "rate": "१८"})
    assert bound["amount"] == 500.0
    assert bound["rate"] == 18.0


def test_rewrite_azure_tool_calls() -> None:
    raw = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "book_ticket",
                "arguments": json.dumps(
                    {"origin": "मुंबई", "destination": "Delhi", "passengers": 2},
                    ensure_ascii=False,
                ),
            },
        }
    ]
    out = BoundaryAdapter().rewrite_openai_tool_calls(raw)
    args = json.loads(out[0]["function"]["arguments"])
    assert args["origin"] == "BOM"
    assert args["destination"] == "DEL"


def test_ground_hindi_submit_not_english_alias() -> None:
    scene = form_scene("hi")
    el = BoundaryAdapter().ground(scene, "submit")
    assert el is not None
    assert el.text == "जमा करें"
    assert BoundaryAdapter().ground(scene, "submit").center() != (0, 0)


def test_parse_hindi_booking() -> None:
    intent = BoundaryAdapter().parse("मुंबई से दिल्ली के लिए 2 टिकट बुक करो")
    assert intent.kind == "book"
    assert intent.slots["origin"] == "BOM"
    assert intent.slots["destination"] == "DEL"
    assert intent.slots["passengers"] == 2


def test_openai_tools_are_azure_shaped() -> None:
    names = {t["function"]["name"] for t in openai_tools()}
    assert {"calculator", "book_ticket", "lookup"} <= names
    assert "finish" not in names
    assert all(t["type"] == "function" for t in openai_tools())


def test_magentic_card_names_microsoft_hosts() -> None:
    card = magentic_agent_card()
    blob = " ".join(card["hosts"])
    assert "Azure OpenAI" in blob
    assert "Phi" in blob
