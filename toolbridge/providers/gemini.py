from __future__ import annotations

from typing import Iterable

from toolbridge.decorator import Tool
from toolbridge.translator import to_gemini


class GeminiProvider:
    name = "gemini"

    def __init__(self, model):
        self.model = model

    def complete(self, *, prompt: str, tools: Iterable[Tool], observations: list[dict]):
        declarations = [to_gemini(tool)["functionDeclarations"][0] for tool in tools]
        return self.model.generate_content(
            [{"role": "user", "parts": [prompt]}, *observations],
            tools=[{"functionDeclarations": declarations}],
        )
