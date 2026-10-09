import pytest
from pydantic import BaseModel

from toolbridge.structured import Refusal, StructuredOutputError, complete_with_retrying_errors


class Answer(BaseModel):
    value: int


def test_structured_output_parse_error_retries_3x():
    attempts = []

    def model_call(errors):
        attempts.append(list(errors))
        return "{bad json"

    with pytest.raises(StructuredOutputError) as exc:
        complete_with_retrying_errors(model_call, Answer)

    assert exc.value.failure_mode == "parse_error"
    assert len(attempts) == 3
    assert attempts[1][0].startswith("parse_error:")


def test_structured_output_schema_violation_retries_3x():
    attempts = []

    def model_call(errors):
        attempts.append(list(errors))
        return {"missing": "value"}

    with pytest.raises(StructuredOutputError) as exc:
        complete_with_retrying_errors(model_call, Answer)

    assert exc.value.failure_mode == "schema_violation"
    assert len(attempts) == 3


def test_structured_output_refusal_returns_immediately():
    attempts = []

    def model_call(errors):
        attempts.append(list(errors))
        return {"refusal": "policy"}

    result = complete_with_retrying_errors(model_call, Answer)

    assert result.value == Refusal(reason="policy")
    assert result.failure_mode == "refusal"
    assert len(attempts) == 1
