from toolbridge.decorator import tool
from toolbridge.translator import to_anthropic, to_gemini, to_openai


DESCRIPTION = "Use when adding two integers. Do not use for any other arithmetic."


@tool(DESCRIPTION)
def add(left: int, right: int) -> int:
    return left + right


def test_openai_translation_exact_shape():
    assert to_openai(add) == {
        "type": "function",
        "function": {
            "name": "add",
            "description": DESCRIPTION,
            "parameters": add.parameters,
            "strict": True,
        },
    }


def test_anthropic_translation_exact_shape():
    assert to_anthropic(add) == {
        "name": "add",
        "description": DESCRIPTION,
        "input_schema": add.parameters,
    }


def test_gemini_translation_exact_shape():
    assert to_gemini(add) == {
        "functionDeclarations": [
            {
                "name": "add",
                "description": DESCRIPTION,
                "parameters": add.parameters,
            }
        ]
    }
