import pytest

from toolbroidge.streaming import PartialArgumentsError, ToolCallAccumulator


def test_parse_early_trap_raises_before_end_signal():
    accumulator = ToolCallAccumulator("openai")
    accumulator.add_chunk("call_1", '{"city": "Lon')

    with pytest.raises(PartialArgumentsError):
        accumulator.parse("call_1")


def test_openai_parses_after_tool_calls_finish_reason():
    accumulator = ToolCallAccumulator("openai")
    accumulator.add_chunk("call_1", '{"city": "Lon')
    accumulator.add_chunk("call_1", 'don"}')
    accumulator.observe_openai_finish("tool_calls")

    assert accumulator.parse("call_1") == {"city": "London"}


def test_anthropic_parses_after_content_block_stop():
    accumulator = ToolCallAccumulator("anthropic")
    accumulator.add_chunk("call_1", '{"ok": true}')
    accumulator.observe_anthropic_event("content_block_stop", "call_1")

    assert accumulator.parse("call_1") == {"ok": True}


def test_gemini_parses_after_stream_end():
    accumulator = ToolCallAccumulator("gemini")
    accumulator.add_chunk("call_1", '{"ok": true}')
    accumulator.observe_gemini_stream_end()

    assert accumulator.parse("call_1") == {"ok": True}
