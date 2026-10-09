from typing import Literal

import pytest

from toolbroidge.decorator import tool


GOOD_DESCRIPTION = "Use when adding two values. Do not use for subtracting values."


def test_decorator_generates_schema_and_literal_enum():
    @tool(GOOD_DESCRIPTION)
    def choose_color(color: Literal["red", "blue"], count: int = 1) -> str:
        return color * count

    assert choose_color.name == "choose_color"
    assert choose_color.consequential is False
    assert choose_color.parameters["properties"]["color"]["enum"] == ["red", "blue"]
    assert choose_color.execute({"color": "red", "count": 2}) == "redred"


def test_decorator_rejects_bad_names():
    with pytest.raises(ValueError, match="snake_case"):

        @tool(GOOD_DESCRIPTION)
        def BadName(value: int) -> int:
            return value


def test_decorator_rejects_short_descriptions():
    with pytest.raises(ValueError, match="20-1024"):
        tool("too short")


def test_decorator_rejects_missing_do_not_use_for():
    with pytest.raises(ValueError, match="Do not use for"):
        tool("Use when adding two numbers. Avoid every other operation.")


def test_decorator_rejects_untyped_fields():
    with pytest.raises(ValueError, match="must be type annotated"):

        @tool(GOOD_DESCRIPTION)
        def missing_type(value):
            return value


def test_consequential_flag_is_recorded():
    @tool(GOOD_DESCRIPTION, consequential=True)
    def save_value(value: int) -> int:
        return value

    assert save_value.consequential is True
