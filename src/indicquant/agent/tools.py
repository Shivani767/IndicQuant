"""Tools the live agent can call.

English-schema APIs (IATA codes, ASCII math) on purpose: that is how production
agent platforms look. The locale layer binds native-script arguments *after* the
model chooses the tool. Tools themselves never translate.
"""

from __future__ import annotations

import ast
import json
import operator
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from indicquant.agent.script import ascii_digits, in_script
from indicquant.agent.types import ToolResult

# Canonical city index. Keys cover English, native script, and common romanizations.
# Values are IATA-style codes — the kind of English enum a real booking API demands.
CITY_INDEX: dict[str, str] = {
    "mumbai": "BOM",
    "bombay": "BOM",
    "mum": "BOM",
    "मुंबई": "BOM",
    "மும்பை": "BOM",
    "ମୁମ୍ବାଇ": "BOM",
    "delhi": "DEL",
    "new delhi": "DEL",
    "दिल्ली": "DEL",
    "नई दिल्ली": "DEL",
    "டெல்லி": "DEL",
    "ଦିଲ୍ଲୀ": "DEL",
    "chennai": "MAA",
    "madras": "MAA",
    "चेन्नई": "MAA",
    "சென்னை": "MAA",
    "ଚେନ୍ନାଇ": "MAA",
    "bengaluru": "BLR",
    "bangalore": "BLR",
    "बेंगलुरु": "BLR",
    "बैंगलोर": "BLR",
    "பெங்களூரு": "BLR",
    "kolkata": "CCU",
    "calcutta": "CCU",
    "कोलकाता": "CCU",
    "hyderabad": "HYD",
    "हैदराबाद": "HYD",
}

# Display names for the chat UI and for tool results the model must not reverse.
CITY_LABELS: dict[str, dict[str, str]] = {
    "BOM": {"en": "Mumbai", "hi": "मुंबई", "ta": "மும்பை", "or": "ମୁମ୍ବାଇ"},
    "DEL": {"en": "Delhi", "hi": "दिल्ली", "ta": "டெல்லி", "or": "ଦିଲ୍ଲୀ"},
    "MAA": {"en": "Chennai", "hi": "चेन्नई", "ta": "சென்னை", "or": "ଚେନ୍ନାଇ"},
    "BLR": {"en": "Bengaluru", "hi": "बेंगलुरु", "ta": "பெங்களூரு", "or": "ବେଙ୍ଗାଲୁରୁ"},
    "CCU": {"en": "Kolkata", "hi": "कोलकाता", "ta": "கொல்கத்தா", "or": "କଲିକତା"},
    "HYD": {"en": "Hyderabad", "hi": "हैदराबाद", "ta": "ஹைதராபாத்", "or": "ହାଇଦ୍ରାବାଦ"},
}

# Tiny grounded KB. Queries must match the *language of the stored fact*, not a translation.
# An agent that "helpfully" translates a Hindi question into English will miss.
KB_FACTS: list[tuple[str, str]] = [
    ("gst rate india", "18%"),
    ("भारत gst दर", "18%"),
    ("irctc", "Indian Railways catering and tourism corporation"),
    ("मुंबई", "capital of Maharashtra; IATA BOM"),
    ("mumbai", "capital of Maharashtra; IATA BOM"),
    ("delhi", "capital of India; IATA DEL"),
    ("दिल्ली", "capital of India; IATA DEL"),
    ("சென்னை", "capital of Tamil Nadu; IATA MAA"),
    ("ଭୁବନେଶ୍ୱର", "capital of Odisha; IATA BBI"),
    ("visa india domestic", "no visa required for domestic Indian rail travel"),
    ("भारत घरेलू वीजा", "घरेलू रेल यात्रा के लिए वीजा नहीं चाहिए"),
]


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calc(expr: str) -> float:
    """Arithmetic only. No names, no calls, no dunders."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"unsupported expression: {expr!r}")

    return float(_eval(tree))


def resolve_city(name: str) -> str | None:
    key = name.strip()
    if not key:
        return None
    codes = {code.upper() for code in CITY_INDEX.values()}
    if key.upper() in codes:
        return key.upper()
    direct = CITY_INDEX.get(key) or CITY_INDEX.get(key.lower())
    if direct:
        return direct
    lowered = key.casefold()
    for alias, code in CITY_INDEX.items():
        if alias.casefold() == lowered:
            return code
    return None


def city_label(value: str, lang: str = "en") -> str:
    """Human city name. Prefer the native string the model/user already used."""
    raw = str(value or "").strip()
    if raw and not raw.isascii():
        return raw
    code = resolve_city(raw) or raw.upper()
    row = CITY_LABELS.get(code) or {}
    key = "hi" if lang in {"hi", "hi_en"} else lang
    return row.get(key) or row.get("en") or raw or code


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable[..., ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def subset(self, allow: set[str]) -> ToolRegistry:
        out = ToolRegistry()
        for name in self.names():
            if name in allow:
                spec = self.get(name)
                if spec is not None:
                    out.register(spec)
        return out

    def schema(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(ok=False, output=f"unknown tool: {name}")
        try:
            return spec.handler(**args)
        except TypeError as exc:
            return ToolResult(ok=False, output=f"bad arguments for {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 — tool failures are part of the trajectory
            return ToolResult(ok=False, output=str(exc))


def _calculator(expr: str) -> ToolResult:
    value = round(safe_calc(str(expr)), 6)
    return ToolResult(ok=True, output=str(value), data={"value": value})


def _lookup(query: str) -> ToolResult:
    q = str(query).strip()
    if not q:
        return ToolResult(ok=False, output="empty query")
    hits = [fact for key, fact in KB_FACTS if key in q or q in key]
    if not hits:
        return ToolResult(
            ok=True,
            output="no local KB match — answer from your own knowledge (or call search)",
            data={"query": q, "hits": []},
        )
    return ToolResult(ok=True, output=hits[0], data={"query": q, "hits": hits})


def _wiki_lang(text: str) -> str:
    for ch in text:
        if in_script(ch, "Devanagari") and ch.isalpha():
            return "hi"
        if in_script(ch, "Tamil") and ch.isalpha():
            return "ta"
        if in_script(ch, "Odia") and ch.isalpha():
            return "or"
        if in_script(ch, "Bengali") and ch.isalpha():
            return "bn"
    return "en"


def _wikipedia(query: str, lang: str) -> str:
    params = urllib.parse.urlencode(
        {"action": "opensearch", "search": query, "limit": 3, "namespace": 0, "format": "json"}
    )
    url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "IndicQuant/0.6 (local assistant; https://github.com/)"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=6) as resp:
        payload = json.loads(resp.read().decode())
    titles = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    descs = payload[2] if isinstance(payload, list) and len(payload) > 2 else []
    lines = []
    for title, desc in zip(titles, descs, strict=False):
        if desc:
            lines.append(f"{title}: {desc}")
        else:
            lines.append(str(title))
    return "\n".join(lines).strip()


def _search(query: str) -> ToolResult:
    q = str(query).strip()
    if not q:
        return ToolResult(ok=False, output="empty query")
    lang = _wiki_lang(q)
    try:
        text = _wikipedia(q, lang)
        if not text and lang != "en":
            text = _wikipedia(q, "en")
        if not text:
            return ToolResult(
                ok=True,
                output="no web result — answer from your own knowledge",
                data={"query": q, "hits": []},
            )
        return ToolResult(ok=True, output=text[:2500], data={"query": q, "lang": lang})
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, IndexError) as exc:
        return ToolResult(
            ok=True,
            output=f"search unavailable ({exc}) — answer from your own knowledge",
            data={"query": q},
        )

def _inr_gst(amount: float, rate: float = 18.0) -> ToolResult:
    base = float(ascii_digits(str(amount)))
    pct = float(ascii_digits(str(rate)))
    gst = round(base * pct / 100.0, 2)
    total = round(base + gst, 2)
    return ToolResult(
        ok=True,
        output=f"base={base} rate={pct}% gst={gst} total={total}",
        data={"base": base, "rate": pct, "gst": gst, "total": total},
    )


def _book_ticket(origin: str, destination: str, passengers: int = 1) -> ToolResult:
    orig = resolve_city(str(origin))
    dest = resolve_city(str(destination))
    if orig is None:
        return ToolResult(
            ok=False,
            output=f"unknown origin {origin!r}; API expects an IATA code or English city",
            data={"origin": origin},
        )
    if dest is None:
        return ToolResult(
            ok=False,
            output=f"unknown destination {destination!r}; API expects an IATA code or English city",
            data={"destination": destination},
        )
    n = int(passengers)
    if n < 1:
        return ToolResult(ok=False, output="passengers must be >= 1")
    pnr = f"{orig}{dest}{n:02d}"
    origin_en = CITY_LABELS.get(orig, {}).get("en", orig)
    dest_en = CITY_LABELS.get(dest, {}).get("en", dest)
    origin_hi = CITY_LABELS.get(orig, {}).get("hi", origin_en)
    dest_hi = CITY_LABELS.get(dest, {}).get("hi", dest_en)
    return ToolResult(
        ok=True,
        output=(
            f"booked {n} {orig}->{dest} PNR={pnr} "
            f"from {origin_en}/{origin_hi} to {dest_en}/{dest_hi}"
        ),
        data={
            "origin": orig,
            "destination": dest,
            "passengers": n,
            "pnr": pnr,
            "origin_name": origin_en,
            "destination_name": dest_en,
        },
    )


def default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Exact arithmetic only. ASCII digits and + - * / ( ) . Skip for word problems you can explain without a precise number.",
            parameters={"expr": "str"},
            handler=_calculator,
        )
    )
    registry.register(
        ToolSpec(
            name="lookup",
            description="Tiny local India notes (GST rate, a few cities). Do not use for general questions.",
            parameters={"query": "str"},
            handler=_lookup,
        )
    )
    registry.register(
        ToolSpec(
            name="book_ticket",
            description="Book rail tickets only when the user explicitly asks to book. origin/destination resolve to IATA codes.",
            parameters={"origin": "str", "destination": "str", "passengers": "int"},
            handler=_book_ticket,
        )
    )
    registry.register(
        ToolSpec(
            name="inr_gst",
            description="Indian GST total only when the user asks for GST. amount is the taxable base; rate defaults to 18.",
            parameters={"amount": "float", "rate": "float"},
            handler=_inr_gst,
        )
    )
    return registry


def user_tools(store: Any | None = None) -> ToolRegistry:
    """Tools for a live user-facing agent."""
    from datetime import datetime

    from indicquant.agent.memory import EpisodicStore

    registry = default_tools()
    memory = store if store is not None else EpisodicStore()

    def _remember(note: str) -> ToolResult:
        rec = memory.remember(str(note), source="user_stated", verified=False)
        return ToolResult(
            ok=True,
            output=f"saved: {note}",
            data={"n": len(memory.records), "source": rec.source, "verified": rec.verified},
        )

    def _recall(query: str = "") -> ToolResult:
        records = memory.recall_records(str(query))
        usable = [r for r in records if r.usable]
        skipped = [r.fact for r in records if not r.usable]
        if not usable:
            msg = "(nothing remembered yet)"
            if skipped:
                msg = f"(skipped inferred/unverified: {'; '.join(skipped)})"
            return ToolResult(ok=True, output=msg, data={"hits": [], "skipped": skipped})
        lines = [f"{r.fact} [{r.source}{' verified' if r.verified else ''}]" for r in usable]
        return ToolResult(
            ok=True,
            output=" | ".join(lines),
            data={"hits": [r.fact for r in usable], "skipped": skipped},
        )

    def _now() -> ToolResult:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        return ToolResult(ok=True, output=stamp, data={"utc": stamp})

    registry.register(
        ToolSpec(
            name="search",
            description=(
                "Wikipedia search for a fact you do not know. Skip for coding, opinions, "
                "explanations you can give yourself, and casual chat."
            ),
            parameters={"query": "str"},
            handler=_search,
        )
    )

    registry.register(
        ToolSpec(
            name="remember",
            description="Store a fact the user told you, verbatim, in their language.",
            parameters={"note": "str"},
            handler=_remember,
        )
    )
    registry.register(
        ToolSpec(
            name="recall",
            description="Retrieve saved notes. Empty query returns all notes.",
            parameters={"query": "str"},
            handler=_recall,
        )
    )
    registry.register(
        ToolSpec(
            name="now",
            description="Current UTC date and time.",
            parameters={},
            handler=_now,
        )
    )
    return registry
