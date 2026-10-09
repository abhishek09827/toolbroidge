from __future__ import annotations

from typing import Iterable

from toolbridge.decorator import Tool
from toolbridge.structured import openai_response_format
from toolbridge.translator import to_openai


class OpenAIProvider:
    name = "openai"

    def __init__(self, client, *, model: str):
        self.client = client
        self.model = model

    def complete(self, *, prompt: str, tools: Iterable[Tool], observations: list[dict]):
        return self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}, *observations],
            tools=[to_openai(tool) for tool in tools],
        )

    def response_format(self, schema):
        return openai_response_format(schema)
