from __future__ import annotations

import argparse
import json
import os
from typing import Any

from toolbridge import tool
from toolbridge.parallel import ToolCall, execute_parallel
from toolbridge.translator import to_anthropic, to_gemini, to_openai


@tool("Use when adding two numbers. Do not use for non-arithmetic requests.")
def add(left: int, right: int) -> int:
    return left + right


@tool("Use when multiplying two numbers. Do not use for non-arithmetic requests.")
def multiply(left: int, right: int) -> int:
    return left * right


TOOLS = {item.name: item for item in [add, multiply]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real provider tool-call smoke test.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], required=True)
    parser.add_argument("--prompt", default="Use tools to compute (7 + 5) and (7 * 5).")
    args = parser.parse_args()

    if args.provider == "openai":
        result = run_openai(args.prompt)
    elif args.provider == "anthropic":
        result = run_anthropic(args.prompt)
    else:
        result = run_gemini(args.prompt)

    print(json.dumps(result, indent=2, sort_keys=True))


def run_openai(prompt: str) -> dict[str, Any]:
    require_env("OPENAI_API_KEY")
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        tools=[to_openai(item) for item in TOOLS.values()],
        tool_choice="auto",
    )
    calls = []
    for call in response.choices[0].message.tool_calls or []:
        calls.append(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments),
            )
        )
    return {"provider": "openai", "tool_results": execute_parallel(calls, TOOLS)}


def run_anthropic(prompt: str) -> dict[str, Any]:
    require_env("ANTHROPIC_API_KEY")
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        tools=[to_anthropic(item) for item in TOOLS.values()],
    )
    calls = []
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    return {"provider": "anthropic", "tool_results": execute_parallel(calls, TOOLS)}


def run_gemini(prompt: str) -> dict[str, Any]:
    require_env("GEMINI_API_KEY")
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    declarations = [to_gemini(item)["functionDeclarations"][0] for item in TOOLS.values()]
    model = genai.GenerativeModel(
        os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        tools=[{"function_declarations": declarations}],
    )
    response = model.generate_content(prompt)
    calls = []
    for part in response.candidates[0].content.parts:
        function_call = getattr(part, "function_call", None)
        if function_call and function_call.name:
            calls.append(
                ToolCall(
                    id=function_call.name,
                    name=function_call.name,
                    arguments=dict(function_call.args),
                )
            )
    return {"provider": "gemini", "tool_results": execute_parallel(calls, TOOLS)}


def require_env(name: str) -> None:
    if not os.getenv(name):
        raise SystemExit(f"Set {name} before running this demo.")


if __name__ == "__main__":
    main()
