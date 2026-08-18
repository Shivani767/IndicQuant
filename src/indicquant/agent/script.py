"""Unicode script checks for locale binding."""

from __future__ import annotations

SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Latin": [(0x0020, 0x024F)],
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Odia": [(0x0B00, 0x0B7F)],
    "Tamil": [(0x0B80, 0x0BFF), (0x11FC0, 0x11FFF)],
}


def in_script(char: str, script: str) -> bool:
    ranges = SCRIPT_RANGES.get(script)
    if not ranges:
        return False
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in ranges)


# Devanagari, Tamil, Odia digits → ASCII. Shared by bind and the grounding critic.
NATIVE_DIGITS = str.maketrans(
    "०१२३४५६७८९௦௧௨௩௪௫௬௭௮௯୦୧୨୩୪୫୬୭୮୯",
    "012345678901234567890123456789",
)


def ascii_digits(text: str) -> str:
    return text.translate(NATIVE_DIGITS)
