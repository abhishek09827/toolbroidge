from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class Refusal:
    reason: str


@dataclass(frozen=True)
class StructuredResult:
    value: BaseModel | Refusal
    attempts: int
    failure_mode: str | None = None


def openai_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }


def anthropic_output_tool(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "name": f"emit_{schema.__name__.lower()}",
        "description": "Use when returning the final structured answer. Do not use for intermediate work.",
        "input_schema": schema.model_json_schema(),
    }


def parse_structured_response(response: Any, schema: type[BaseModel]) -> BaseModel | Refusal:
    if isinstance(response, Refusal):
        return response
    if isinstance(response, dict) and "refusal" in response:
        return Refusal(str(response["refusal"]))
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError("parse_error", str(exc)) from exc
    try:
        return schema.model_validate(response)
    except ValidationError as exc:
        raise StructuredOutputError("schema_violation", str(exc)) from exc


class StructuredOutputError(ValueError):
    def __init__(self, failure_mode: str, message: str):
        super().__init__(message)
        self.failure_mode = failure_mode


def complete_with_retries(model_call, schema: type[BaseModel], *, max_retries: int = 3) -> StructuredResult:
    errors: list[str] = []
    for attempt in range(1, max_retries + 1):
        response = model_call(errors)
        parsed = parse_structured_response(response, schema)
        if isinstance(parsed, Refusal):
            return StructuredResult(parsed, attempt, "refusal")
        return StructuredResult(parsed, attempt)
    raise AssertionError("unreachable")


def complete_with_retrying_errors(model_call, schema: type[BaseModel], *, max_retries: int = 3) -> StructuredResult:
    errors: list[str] = []
    last_mode = None
    for attempt in range(1, max_retries + 1):
        response = model_call(errors)
        try:
            parsed = parse_structured_response(response, schema)
        except StructuredOutputError as exc:
            last_mode = exc.failure_mode
            errors.append(f"{exc.failure_mode}: {exc}")
            continue
        if isinstance(parsed, Refusal):
            return StructuredResult(parsed, attempt, "refusal")
        return StructuredResult(parsed, attempt)
    raise StructuredOutputError(last_mode or "parse_error", errors[-1] if errors else "structured output failed")
