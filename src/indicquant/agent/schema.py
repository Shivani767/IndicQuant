"""Microsoft-shaped surfaces for the boundary adapter.

Azure OpenAI, GitHub Models, and Copilot Studio all speak the OpenAI tools JSON.
Magentic-One / AutoGen speak an agent card. ONNX Runtime GenAI + Phi is the on-device
path. None of these need a new model — they need this locale layer in front.
"""

from __future__ import annotations

from typing import Any

from indicquant.agent.tools import ToolRegistry, default_tools


def openai_tools(registry: ToolRegistry | None = None) -> list[dict[str, Any]]:
    """Chat Completions `tools` array — Azure OpenAI and GitHub Models accept this as-is."""
    registry = registry or default_tools()
    json_types = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
    optional_by_tool = {
        "inr_gst": {"rate"},
        "book_ticket": {"passengers"},
        "recall": {"query"},
        "lookup": {"query"},
        "search": set(),
    }
    tools = []
    for spec in registry.schema():
        if spec["name"] in {"inspect_screen", "click", "type_text", "finish"}:
            continue
        optional = optional_by_tool.get(spec["name"], set())
        properties = {
            name: {"type": json_types.get(typ, "string")}
            for name, typ in spec["parameters"].items()
        }
        required = [name for name in properties if name not in optional]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return tools


def magentic_agent_card() -> dict[str, Any]:
    """Magentic-One / AutoGen specialist card. Other agents call this one for Indic."""
    return {
        "name": "IndicBoundary",
        "role": "locale_specialist",
        "description": (
            "Binds Indic and code-mixed user text and UI labels onto English tool schemas. "
            "Does not replace the planner — wraps it."
        ),
        "inputs": ["user_goal", "tool_calls", "ocr_boxes"],
        "outputs": ["bound_tool_calls", "click_xy", "intent"],
        "hosts": [
            "Azure OpenAI (GPT-4o / o-series function calling)",
            "GitHub Models",
            "Phi + ONNX Runtime GenAI (on-device)",
            "Copilot Studio custom connector",
        ],
    }
