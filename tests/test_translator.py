from toolbroidge.decorator import tool
from toolbroidge.translator import to_anthropic, to_gemini, to_openai


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
    expected_parameters = {
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
        "type": "object",
    }
    assert to_gemini(add) == {
        "functionDeclarations": [
            {
                "name": "add",
                "description": DESCRIPTION,
                "parameters": expected_parameters,
            }
        ]
    }


def test_gemini_translation_strips_unsupported_schema_keywords_recursively():
    translated = to_gemini(add)
    parameters = translated["functionDeclarations"][0]["parameters"]

    assert "additionalProperties" not in parameters
    assert "title" not in parameters
    assert "title" not in parameters["properties"]["left"]
