"""Synthetic Indic UI scenes — computer-use eval without a VLM or a GPU.

A real Operator / Computer-Use eval needs pixels. This MacBook cannot host a VLM, so the
environment exposes the *same three observation channels* those systems use, generated
from a structured scene:

  a11y    accessibility tree with role + name (what English agents overfit to)
  ocr     bounding boxes + visible text in the original script
  pixels  a monospace "screenshot" of those boxes — no element ids

The claim: English-centric agents succeed when names are English (a11y on an English UI)
and fail when the visible text is Devanagari / Tamil / Odia. That is the visual analogue
of calibration-language mismatch.

Click hit-testing is geometric. There is no hidden oracle that maps "Submit" to "जमा करें".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ObservationMode = Literal["a11y", "ocr", "pixels"]


@dataclass(frozen=True)
class UIElement:
    id: str
    role: str
    text: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    value: str = ""
    language: str = "en"

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class Scene:
    name: str
    language: str
    width: int
    height: int
    elements: list[UIElement]
    metadata: dict[str, Any] = field(default_factory=dict)

    def element(self, element_id: str) -> UIElement | None:
        for el in self.elements:
            if el.id == element_id:
                return el
        return None

    def hit(self, x: int, y: int) -> UIElement | None:
        # last drawn (highest z) wins; list order is painter's order
        hits = [el for el in self.elements if el.contains(x, y)]
        return hits[-1] if hits else None


def render_a11y(scene: Scene) -> str:
    lines = [f"A11Y tree ({scene.width}x{scene.height})"]
    for el in scene.elements:
        value = f' value="{el.value}"' if el.value else ""
        lines.append(f'  {el.role} name="{el.text}"{value} id={el.id}')
    return "\n".join(lines)


def render_ocr(scene: Scene) -> str:
    lines = [f"OCR boxes ({scene.width}x{scene.height})"]
    for el in scene.elements:
        x, y, w, h = el.bbox
        shown = el.value if el.role == "textbox" and el.value else el.text
        lines.append(f"  [{x:3d},{y:3d} {w:3d}x{h:3d}] {el.role:8s} {shown}")
    return "\n".join(lines)


def render_pixels(scene: Scene) -> str:
    """Monospace screenshot. Cell size is 1 character ≈ 8px so the grid stays small."""
    cell = 8
    cols = max(8, scene.width // cell)
    rows = max(4, scene.height // cell)
    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    for el in scene.elements:
        r = min(rows - 1, max(0, el.y // cell))
        c = min(cols - 1, max(0, el.x // cell))
        shown = el.value if el.role == "textbox" and el.value else el.text
        glyph = shown.replace("\n", " ")
        for i, ch in enumerate(glyph):
            if c + i >= cols:
                break
            grid[r][c + i] = ch
    border = "+" + "-" * cols + "+"
    body = "\n".join("|" + "".join(row) + "|" for row in grid)
    return f"{border}\n{body}\n{border}"


def observe(scene: Scene, mode: ObservationMode) -> str:
    if mode == "a11y":
        return render_a11y(scene)
    if mode == "ocr":
        return render_ocr(scene)
    if mode == "pixels":
        return render_pixels(scene)
    raise ValueError(f"unknown observation mode: {mode}")


# Visible labels. English-only policies search the English column; Indic-aware policies
# search the column that matches the scene language. There is no automatic translation.
LABELS: dict[str, dict[str, str]] = {
    "name": {"en": "Name", "hi": "नाम", "ta": "பெயர்", "or": "ନାମ", "hi_en": "Naam"},
    "email": {"en": "Email", "hi": "ईमेल", "ta": "மின்னஞ்சல்", "or": "ଇମେଲ", "hi_en": "Email"},
    "submit": {
        "en": "Submit",
        "hi": "जमा करें",
        "ta": "சமர்ப்பி",
        "or": "ଦାଖଲ କରନ୍ତୁ",
        "hi_en": "Submit karo",
    },
    "total": {"en": "Total", "hi": "कुल", "ta": "மொத்தம்", "or": "ମୋଟ", "hi_en": "Total"},
    "pay": {"en": "Pay", "hi": "भुगतान", "ta": "செலுத்து", "or": "ଦେୟ", "hi_en": "Pay karo"},
    "samosa": {"en": "Samosa", "hi": "समोसा", "ta": "சமோசா", "or": "ସମୋସା", "hi_en": "Samosa"},
    "chai": {"en": "Chai", "hi": "चाय", "ta": "தேநீர்", "or": "ଚା", "hi_en": "Chai"},
}


def label(key: str, language: str) -> str:
    row = LABELS[key]
    return row.get(language, row["en"])


def form_scene(language: str, name_value: str = "") -> Scene:
    """A two-field form whose submit button is labelled in `language`."""
    return Scene(
        name=f"form_{language}",
        language=language,
        width=320,
        height=180,
        elements=[
            UIElement("lbl_name", "label", label("name", language), (16, 24, 120, 24), language=language),
            UIElement(
                "inp_name",
                "textbox",
                label("name", language),
                (150, 20, 150, 28),
                value=name_value,
                language=language,
            ),
            UIElement("lbl_email", "label", label("email", language), (16, 64, 120, 24), language=language),
            UIElement("inp_email", "textbox", label("email", language), (150, 60, 150, 28), language=language),
            UIElement("btn_submit", "button", label("submit", language), (150, 110, 140, 32), language=language),
        ],
        metadata={"goal": "fill_name_and_submit"},
    )


def receipt_scene(language: str, total: int = 590) -> Scene:
    return Scene(
        name=f"receipt_{language}",
        language=language,
        width=280,
        height=160,
        elements=[
            UIElement("title", "label", "IRCTC", (20, 16, 80, 20), language="en"),
            UIElement(
                "total",
                "label",
                f"{label('total', language)}: ₹{total}",
                (20, 70, 200, 28),
                language=language,
            ),
            UIElement("btn_pay", "button", label("pay", language), (20, 110, 120, 32), language=language),
        ],
        metadata={"total": total, "goal": "read_total_and_pay"},
    )


def menu_scene(language: str, samosa: int = 40, chai: int = 25) -> Scene:
    return Scene(
        name=f"menu_{language}",
        language=language,
        width=280,
        height=160,
        elements=[
            UIElement(
                "samosa",
                "label",
                f"{label('samosa', language)}  ₹{samosa}",
                (20, 30, 220, 28),
                language=language,
            ),
            UIElement(
                "chai",
                "label",
                f"{label('chai', language)}  ₹{chai}",
                (20, 70, 220, 28),
                language=language,
            ),
        ],
        metadata={"samosa": samosa, "chai": chai, "goal": "bill_two_samosa_one_chai"},
    )
