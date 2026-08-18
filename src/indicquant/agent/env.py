"""Episode environment: tools + an optional visual scene.

Computer-use actions (`click`, `type_text`, `inspect_screen`) mutate the scene. Language
tools (`calculator`, `lookup`, `book_ticket`, `finish`) go through the shared registry.
The environment never translates widget text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indicquant.agent.tools import ToolRegistry, ToolSpec, default_tools
from indicquant.agent.types import ToolCall, ToolResult
from indicquant.agent.vision import ObservationMode, Scene, UIElement, observe


@dataclass
class EnvState:
    scene: Scene | None = None
    mode: ObservationMode = "ocr"
    focused_id: str | None = None
    submitted: bool = False
    paid: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class AgentEnv:
    def __init__(
        self,
        tools: ToolRegistry | None = None,
        scene: Scene | None = None,
        mode: ObservationMode = "ocr",
    ) -> None:
        self.tools = tools or default_tools()
        self.state = EnvState(scene=scene, mode=mode)
        self._bind_vision_tools()

    def _bind_vision_tools(self) -> None:
        self.tools.register(
            ToolSpec(
                name="inspect_screen",
                description="Re-read the current screen in the active observation mode.",
                parameters={},
                handler=self._inspect,
            )
        )
        self.tools.register(
            ToolSpec(
                name="click",
                description="Click a pixel. Hit-testing is geometric; there is no name oracle.",
                parameters={"x": "int", "y": "int"},
                handler=self._click,
            )
        )
        self.tools.register(
            ToolSpec(
                name="type_text",
                description="Type into the focused textbox. Does not translate.",
                parameters={"text": "str"},
                handler=self._type_text,
            )
        )

    def observe(self) -> str:
        if self.state.scene is None:
            return "(no visual scene)"
        return observe(self.state.scene, self.state.mode)

    def step(self, call: ToolCall) -> ToolResult:
        return self.tools.call(call.name, call.args)

    def _inspect(self) -> ToolResult:
        text = self.observe()
        return ToolResult(ok=True, output=text, data={"mode": self.state.mode})

    def _click(self, x: int, y: int) -> ToolResult:
        scene = self.state.scene
        if scene is None:
            return ToolResult(ok=False, output="no scene to click")
        el = scene.hit(int(x), int(y))
        if el is None:
            return ToolResult(ok=False, output=f"no element at ({x}, {y})")
        self.state.focused_id = el.id
        if el.role == "button" and el.id == "btn_submit":
            self.state.submitted = True
        if el.role == "button" and el.id == "btn_pay":
            self.state.paid = True
        return ToolResult(
            ok=True,
            output=f"clicked {el.role} {el.text!r} at ({x},{y})",
            data={"id": el.id, "role": el.role, "text": el.text},
        )

    def _type_text(self, text: str) -> ToolResult:
        scene = self.state.scene
        if scene is None or self.state.focused_id is None:
            return ToolResult(ok=False, output="no focused textbox")
        updated: list[UIElement] = []
        found = False
        for el in scene.elements:
            if el.id == self.state.focused_id and el.role == "textbox":
                updated.append(
                    UIElement(el.id, el.role, el.text, el.bbox, value=str(text), language=el.language)
                )
                found = True
            else:
                updated.append(el)
        if not found:
            return ToolResult(ok=False, output=f"{self.state.focused_id} is not a textbox")
        self.state.scene = Scene(
            name=scene.name,
            language=scene.language,
            width=scene.width,
            height=scene.height,
            elements=updated,
            metadata=scene.metadata,
        )
        return ToolResult(ok=True, output=f"typed {text!r}", data={"text": text})

    def textbox_value(self, element_id: str) -> str:
        if self.state.scene is None:
            return ""
        el = self.state.scene.element(element_id)
        return el.value if el else ""
