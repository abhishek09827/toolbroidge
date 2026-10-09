from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .decorator import Tool


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


def execute_parallel(
    calls: list[ToolCall],
    tools: dict[str, Tool],
    *,
    confirm_callback=None,
) -> dict[str, Any]:
    if not calls:
        return {}

    def run(call: ToolCall) -> Any:
        tool = tools[call.name]
        if tool.consequential:
            if confirm_callback is None or not confirm_callback(tool, call.arguments):
                return {"skipped": True, "reason": "confirmation_required"}
        try:
            return tool.execute(call.arguments)
        except ValidationError as exc:
            return {"error": format_validation_error(exc)}

    with ThreadPoolExecutor(max_workers=len(calls)) as executor:
        futures = {executor.submit(run, call): call.id for call in calls}
        return {futures[future]: future.result() for future in as_completed(futures)}


def format_validation_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first.get("loc", ())) or "input"
    example = {field: "value"}
    if first.get("type") == "missing":
        return f"Invalid input: '{field}' is required. Example: {example}"
    return f"Invalid input: '{field}' is invalid. Example: {example}"
