from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from pydantic import BaseModel

from .decorator import Tool
from .parallel import ToolCall, execute_parallel
from .structured import Refusal, complete_with_retrying_errors


class AgentLoopError(RuntimeError):
    pass


class ToolNotFoundError(KeyError):
    pass


class ProviderError(RuntimeError):
    def __init__(self, provider: str, message: str):
        super().__init__(f"{provider}: {message}")
        self.provider = provider


@dataclass
class AgentRun:
    output: Any
    iterations: int
    tool_call_count: int


class Agent:
    def __init__(
        self,
        *,
        provider,
        tools: Iterable[Tool] = (),
        output_schema: type[BaseModel] | None = None,
        confirm_callback: Callable[[Tool, dict[str, Any]], bool] | None = None,
        max_turns: int = 10,
    ):
        self.provider = provider
        self.tools = {tool.name: tool for tool in tools}
        self.output_schema = output_schema
        self.confirm_callback = confirm_callback
        self.max_turns = max_turns

    def run(self, prompt: str) -> AgentRun:
        observations: list[dict[str, Any]] = []
        tool_call_count = 0

        for iteration in range(1, self.max_turns + 1):
            try:
                response = self.provider.complete(prompt=prompt, tools=list(self.tools.values()), observations=observations)
            except Exception as exc:
                raise ProviderError(getattr(self.provider, "name", self.provider.__class__.__name__), str(exc)) from exc

            calls = self._extract_calls(response)
            if not calls:
                output = self._final_output(response)
                return AgentRun(output=output, iterations=iteration, tool_call_count=tool_call_count)

            for call in calls:
                if call.name not in self.tools:
                    raise ToolNotFoundError(call.name)

            results = execute_parallel(calls, self.tools, confirm_callback=self.confirm_callback)
            tool_call_count += len(calls)
            observations.append({"tool_results": results})

        raise AgentLoopError(f"Agent exceeded max_turns={self.max_turns}")

    def _final_output(self, response: Any) -> Any:
        content = response.get("content", response) if isinstance(response, dict) else response
        if self.output_schema is None:
            return content

        calls = {"count": 0}

        def model_call(errors: list[str]):
            calls["count"] += 1
            if errors and hasattr(self.provider, "complete_structured"):
                return self.provider.complete_structured(errors=errors, schema=self.output_schema)
            return content

        return complete_with_retrying_errors(model_call, self.output_schema).value

    @staticmethod
    def _extract_calls(response: Any) -> list[ToolCall]:
        raw_calls = response.get("tool_calls", []) if isinstance(response, dict) else getattr(response, "tool_calls", [])
        return [
            ToolCall(id=str(call["id"]), name=str(call["name"]), arguments=dict(call.get("arguments", {})))
            for call in raw_calls
        ]
