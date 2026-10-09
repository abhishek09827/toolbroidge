from __future__ import annotations

from typing import Iterable

from toolbroidge.decorator import Tool
from toolbroidge.structured import anthropic_output_tool
from toolbroidge.translator import to_anthropic


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client, *, model: str):
        self.client = client
        self.model = model

    def complete(self, *, prompt: str, tools: Iterable[Tool], observations: list[dict]):
        return self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}, *observations],
            tools=[to_anthropic(tool) for tool in tools],
        )

    def output_tool(self, schema):
        return anthropic_output_tool(schema)
