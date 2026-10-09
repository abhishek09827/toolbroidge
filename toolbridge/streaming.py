from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


class PartialArgumentsError(RuntimeError):
    pass


class ToolCallAccumulator:
    def __init__(self, provider: str):
        self.provider = provider
        self._buffers: dict[str, list[str]] = defaultdict(list)
        self._ended: set[str] = set()

    def add_chunk(self, call_id: str, chunk: str) -> None:
        self._buffers[call_id].append(chunk)

    def mark_done(self, call_id: str | None = None) -> None:
        if call_id is None:
            self._ended.update(self._buffers)
            return
        self._ended.add(call_id)

    def observe_openai_finish(self, finish_reason: str | None) -> None:
        if finish_reason == "tool_calls":
            self.mark_done()

    def observe_anthropic_event(self, event_type: str, call_id: str) -> None:
        if event_type == "content_block_stop":
            self.mark_done(call_id)

    def observe_gemini_stream_end(self) -> None:
        self.mark_done()

    def parse(self, call_id: str) -> dict[str, Any]:
        if call_id not in self._ended:
            raise PartialArgumentsError("Cannot parse tool arguments before provider end-of-call signal")
        return json.loads("".join(self._buffers[call_id]))

    def completed_arguments(self) -> dict[str, dict[str, Any]]:
        return {call_id: self.parse(call_id) for call_id in self._ended}
