from __future__ import annotations

from .decorator import Tool


def to_openai(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": True,
        },
    }


def to_anthropic(tool: Tool) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def to_gemini(tool: Tool) -> dict:
    return {
        "functionDeclarations": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        ]
    }


def translate(tool: Tool, provider: str) -> dict:
    provider = provider.lower()
    if provider == "openai":
        return to_openai(tool)
    if provider == "anthropic":
        return to_anthropic(tool)
    if provider == "gemini":
        return to_gemini(tool)
    raise ValueError(f"Unsupported provider: {provider}")
