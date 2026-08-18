"""Hard verification layer.

Every fare, PNR, and IATA claim in a model reply must equal a *field* on a tool
result, or a whole token in the raw tool output. Substring plausibility is not
enough: 590 does not verify against 1590.

The runtime may ask the model to rewrite once. If claims are still unverified,
they are redacted before the user sees the answer. Rejection counts are session
state, not vibes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from indicquant.agent.script import ascii_digits

_NUM = re.compile(r"\d+(?:\.\d+)?")
_PNR = re.compile(r"\b[A-Z]{3}[A-Z]{3}\d{2,}\b")
_IATA_PAIR = re.compile(r"\b[A-Z]{3}\s*->\s*[A-Z]{3}\b")
_TOKEN = re.compile(r"[A-Z]{6}\d{2,}|\d+(?:\.\d+)?|[A-Z]{3}->[A-Z]{3}|[A-Z]{3}")


@dataclass
class ClaimCheck:
    claim: str
    kind: str
    verified: bool
    matched_field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    ok: bool
    checks: list[ClaimCheck] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def verified_n(self) -> int:
        return sum(1 for c in self.checks if c.verified)

    @property
    def total_n(self) -> int:
        return len(self.checks)

    @property
    def missing(self) -> list[str]:
        return self.rejected

    @property
    def claims(self) -> list[str]:
        return [c.claim for c in self.checks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "skipped": self.skipped,
            "verified": self.verified_n,
            "total": self.total_n,
            "rejected": self.rejected,
            "checks": [c.to_dict() for c in self.checks],
        }


def extract_claims(reply: str) -> list[str]:
    """Money-like numbers, PNRs, IATA pairs. Skip 1–2 digit counts and rates."""
    text = ascii_digits(reply)
    claims: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        key = token.upper()
        if key not in seen:
            seen.add(key)
            claims.append(token)

    for match in _PNR.finditer(text.upper()):
        add(match.group(0))
    for match in _IATA_PAIR.finditer(text.upper()):
        add(re.sub(r"\s+", "", match.group(0)))
    for match in _NUM.finditer(text):
        token = match.group(0)
        digits = token.replace(".", "")
        if "." not in token and len(digits.lstrip("0") or "0") <= 2:
            continue
        add(token)
    return claims


def _kind(claim: str) -> str:
    upper = claim.upper()
    if _PNR.fullmatch(upper):
        return "pnr"
    if "->" in upper:
        return "iata"
    return "fare"


def _norm_num(token: str) -> float | None:
    try:
        return float(ascii_digits(token))
    except ValueError:
        return None


def _field_evidence(results: list[dict[str, Any]]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for result in results:
        data = result.get("data") or {}
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if value is None or isinstance(value, (dict, list)):
                continue
            fields[str(key)] = str(value)
    return fields


def _output_tokens(results: list[dict[str, Any]]) -> set[str]:
    blob = ascii_digits(" ".join(str(r.get("output") or "") for r in results)).upper()
    blob = blob.replace(" ", "")
    tokens = set(_TOKEN.findall(ascii_digits(" ".join(str(r.get("output") or "") for r in results)).upper()))
    # Also keep compact IATA pairs from "BOM->DEL"
    tokens.update(_TOKEN.findall(blob.replace("->", "->")))
    return tokens


def _match_field(claim: str, fields: dict[str, str]) -> str | None:
    want_num = _norm_num(claim)
    want = ascii_digits(claim).upper()
    for key, raw in fields.items():
        if want_num is not None:
            got = _norm_num(raw)
            if got is not None and got == want_num:
                return key
        if ascii_digits(raw).upper() == want:
            return key
    return None


def _match_token(claim: str, tokens: set[str]) -> bool:
    want_num = _norm_num(claim)
    want = ascii_digits(claim).upper().replace(" ", "")
    if want in tokens:
        return True
    if want_num is not None:
        for token in tokens:
            got = _norm_num(token)
            if got is not None and got == want_num:
                return True
    return False


def verify_reply(reply: str, tool_results: list[dict[str, Any]]) -> Verdict:
    """Exact field/token match against raw tool results. No plausibility shortcuts."""
    if not tool_results:
        return Verdict(ok=True, skipped=True)
    claims = extract_claims(reply)
    if not claims:
        return Verdict(ok=True, skipped=False)
    fields = _field_evidence(tool_results)
    tokens = _output_tokens(tool_results)
    checks: list[ClaimCheck] = []
    rejected: list[str] = []
    for claim in claims:
        field = _match_field(claim, fields)
        ok = field is not None or _match_token(claim, tokens)
        checks.append(ClaimCheck(claim=claim, kind=_kind(claim), verified=ok, matched_field=field))
        if not ok:
            rejected.append(claim)
    return Verdict(ok=not rejected, checks=checks, rejected=rejected)


def redact_claims(reply: str, rejected: list[str]) -> str:
    """Strip unverified claims so they never reach the user."""
    text = reply
    for claim in sorted(rejected, key=len, reverse=True):
        text = re.sub(re.escape(claim), "[unverified]", text)
    return text


def critic_prompt(verdict: Verdict) -> str:
    cited = ", ".join(verdict.rejected) or "(unknown)"
    return (
        "[runtime verifier] Your last answer cited values that are not an exact field "
        f"or token in any tool result: {cited}. Rewrite using only tool fields. "
        "Keep the user's language. Do not invent numbers, PNRs, or times. "
        "If this was a booking, keep origin then destination in that same order."
    )
