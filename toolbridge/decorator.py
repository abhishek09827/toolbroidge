from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model


_SNAKE_CASE = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    consequential: bool = False
    model: type[BaseModel] | None = None

    def validate_arguments(self, arguments: dict[str, Any]) -> BaseModel:
        if self.model is None:
            raise TypeError(f"Tool {self.name!r} has no argument model")
        return self.model.model_validate(arguments)

    def execute(self, arguments: dict[str, Any]) -> Any:
        validated = self.validate_arguments(arguments)
        return self.func(**validated.model_dump())


def tool(description: str, *, consequential: bool = False) -> Callable[[Callable[..., Any]], Tool]:
    _validate_description(description)

    def decorate(func: Callable[..., Any]) -> Tool:
        _validate_name(func.__name__)
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        fields: dict[str, tuple[Any, Any]] = {}

        for name, parameter in signature.parameters.items():
            if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise ValueError(f"Tool {func.__name__!r} cannot use *args or **kwargs")
            if name not in hints:
                raise ValueError(f"Tool argument {name!r} must be type annotated")
            default = parameter.default if parameter.default is not inspect.Parameter.empty else ...
            fields[name] = (hints[name], Field(default=default))

        model = create_model(
            f"{func.__name__.title().replace('_', '')}Input",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        schema = model.model_json_schema()
        schema["additionalProperties"] = False
        canonical = Tool(
            name=func.__name__,
            description=description,
            parameters=schema,
            func=func,
            consequential=consequential,
            model=model,
        )
        setattr(func, "__toolbridge_tool__", canonical)
        return canonical

    return decorate


def _validate_name(name: str) -> None:
    if not _SNAKE_CASE.fullmatch(name):
        raise ValueError("Tool name must be snake_case")


def _validate_description(description: str) -> None:
    if not (20 <= len(description) <= 1024):
        raise ValueError("Tool description must be 20-1024 characters")
    if "Use when " not in description or "Do not use for " not in description:
        raise ValueError('Tool description must follow "Use when X. Do not use for Y."')
