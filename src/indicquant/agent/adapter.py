"""Locale boundary for English-schema tool calls.

Parse, bind, ground, rewrite. The live agent uses bind/rewrite after the model
chooses a tool.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from indicquant.agent.script import ascii_digits, in_script
from indicquant.agent.tools import CITY_INDEX, resolve_city
from indicquant.agent.types import ToolCall
from indicquant.agent.vision import LABELS, Scene, UIElement

_ASCII_INT = re.compile(r"\d+")
_GST_HINTS = ("gst", "percent", "percentage", "प्रतिशत", "சதவீத", "ପ୍ରତିଶତ", "%")


@dataclass
class Entity:
    text: str
    kind: str
    canonical: str
    script: str


@dataclass
class Intent:
    kind: str
    language: str
    slots: dict[str, Any] = field(default_factory=dict)
    entities: list[Entity] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "language": self.language,
            "slots": self.slots,
            "entities": [asdict(e) for e in self.entities],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Intent:
        entities = [Entity(**row) for row in data.get("entities") or []]
        return cls(
            kind=str(data.get("kind") or "lookup"),
            language=str(data.get("language") or "en"),
            slots=dict(data.get("slots") or {}),
            entities=entities,
        )


def detect_language(text: str) -> str:
    counts = {script: 0 for script in ("Devanagari", "Tamil", "Odia", "Latin")}
    for ch in text:
        for script in counts:
            if in_script(ch, script) and ch.isalpha():
                counts[script] += 1
                break
    native = {k: v for k, v in counts.items() if k != "Latin"}
    if native and max(native.values()) > 0:
        top = max(native, key=native.get)  # type: ignore[arg-type]
        return {"Devanagari": "hi", "Tamil": "ta", "Odia": "or"}[top]
    lowered = text.lower()
    if any(w in lowered for w in ("karo", "kitna", "pe ", "mein", "ke liye")):
        return "hi_en"
    return "en"


def nums(text: str) -> list[int]:
    return [int(n) for n in _ASCII_INT.findall(ascii_digits(text))]


def gst_pair(values: list[int]) -> tuple[int, int] | None:
    if len(values) < 2:
        return None
    a, b = values[0], values[1]
    lo, hi = (a, b) if a < b else (b, a)
    if lo <= 40:
        return hi, lo
    return a, b


def extract_cities(text: str) -> list[Entity]:
    found: list[tuple[int, str, str]] = []
    for alias in sorted(CITY_INDEX, key=len, reverse=True):
        code = CITY_INDEX[alias]
        if alias.isascii():
            for m in re.finditer(rf"\b{re.escape(alias)}\b", text, re.I):
                found.append((m.start(), alias, code))
        else:
            idx = text.find(alias)
            if idx >= 0:
                found.append((idx, alias, code))
    found.sort()
    out: list[Entity] = []
    seen: set[str] = set()
    for _, alias, code in found:
        if code in seen:
            continue
        seen.add(code)
        script = detect_language(alias)
        script_name = {
            "hi": "Devanagari",
            "ta": "Tamil",
            "or": "Odia",
        }.get(script, "Latin")
        out.append(Entity(text=alias, kind="city", canonical=code, script=script_name))
        if len(out) == 2:
            break
    return out


def _label_aliases(role: str) -> set[str]:
    row = LABELS.get(role, {})
    return {v for v in row.values() if v}


class BoundaryAdapter:
    """The shippable layer. Model-agnostic. Azure OpenAI / Phi / Copilot stay English."""

    def parse(self, goal: str, scene: Scene | None = None) -> Intent:
        language = scene.language if scene is not None else detect_language(goal)
        cities = extract_cities(goal)
        pair = gst_pair(nums(goal))
        goal_kind = scene.metadata.get("goal") if scene is not None else None

        if goal_kind == "fill_name_and_submit" or (
            scene is not None and any(el.id == "btn_submit" for el in scene.elements)
        ):
            return Intent(kind="form", language=language, slots={"name": _default_name(language)})
        if goal_kind == "read_total_and_pay" or (
            scene is not None and any(el.id == "btn_pay" for el in scene.elements)
        ):
            total = scene.metadata.get("total") if scene is not None else None
            return Intent(kind="pay", language=language, slots={"total": total})
        if goal_kind == "bill_two_samosa_one_chai" or (
            scene is not None and any(el.id == "samosa" for el in scene.elements)
        ):
            prices = menu_prices(scene) if scene is not None else None
            slots = {"qty_samosa": 2, "qty_chai": 1}
            if prices:
                slots["samosa"] = prices[0]
                slots["chai"] = prices[1]
            return Intent(kind="bill", language=language, slots=slots)

        if len(cities) >= 2:
            n = nums(goal)
            return Intent(
                kind="book",
                language=language,
                slots={
                    "origin": cities[0].canonical,
                    "destination": cities[1].canonical,
                    "passengers": n[0] if n else 1,
                },
                entities=cities,
            )
        if pair is not None and any(h in goal.lower() or h in goal for h in _GST_HINTS):
            base, rate = pair
            return Intent(
                kind="gst",
                language=language,
                slots={"base": base, "rate": rate, "total": base + base * rate / 100},
            )
        return Intent(kind="lookup", language=language, slots={"query": goal.strip()})

    def bind_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Rewrite a naive English-schema call so the underlying API can accept it.

        This is the Azure / OpenAI middleware: the model may emit मुंबई; the tool still
        receives BOM.
        """
        bound = dict(args)
        if name == "book_ticket":
            for key in ("origin", "destination"):
                if key in bound and bound[key] is not None:
                    code = resolve_city(str(bound[key]))
                    if code:
                        bound[key] = code
            if "passengers" in bound:
                bound["passengers"] = int(bound["passengers"])
        if name == "calculator" and "expr" in bound:
            bound["expr"] = ascii_digits(str(bound["expr"]))
        if name == "inr_gst":
            if "amount" in bound:
                bound["amount"] = float(ascii_digits(str(bound["amount"])))
            if "rate" in bound:
                bound["rate"] = float(ascii_digits(str(bound["rate"])))
        return bound

    def rewrite_tool_call(self, call: ToolCall) -> ToolCall:
        return ToolCall(call.name, self.bind_args(call.name, call.args), thought=call.thought)

    def rewrite_openai_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """In-place bind for Azure OpenAI / GitHub Models Chat Completions tool_calls."""
        rewritten = []
        for raw in tool_calls:
            item = json.loads(json.dumps(raw))
            fn = item.get("function") or item
            name = str(fn.get("name", ""))
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            fn["arguments"] = json.dumps(self.bind_args(name, args), ensure_ascii=False)
            rewritten.append(item)
        return rewritten

    def ground(self, scene: Scene, role: str) -> UIElement | None:
        """Match a semantic widget role against *visible* text in any script."""
        aliases = _label_aliases(role)
        if role == "name":
            roles = {"textbox"}
        elif role in {"submit", "pay"}:
            roles = {"button"}
        else:
            roles = {"label", "button", "textbox"}
        for el in scene.elements:
            if el.role not in roles:
                continue
            if el.text in aliases or any(a and a in el.text for a in aliases):
                return el
        return None


def menu_prices(scene: Scene, *, english_labels_only: bool = False) -> tuple[int, int] | None:
    samosa = chai = None
    for el in scene.elements:
        values = nums(el.text)
        if not values:
            continue
        blob = el.text.lower()
        is_samosa = "samosa" in blob or any(LABELS["samosa"][lang] in el.text for lang in LABELS["samosa"])
        is_chai = "chai" in blob or any(LABELS["chai"][lang] in el.text for lang in LABELS["chai"])
        if english_labels_only:
            is_samosa = "samosa" in blob
            is_chai = "chai" in blob
        if is_samosa:
            samosa = values[0]
        elif is_chai:
            chai = values[0]
    if samosa is None or chai is None:
        return None
    return samosa, chai


def _default_name(language: str) -> str:
    return "Shivani" if language in {"en", "hi_en"} else "शिवानी"
