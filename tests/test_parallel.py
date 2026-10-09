import time

from toolbridge.decorator import tool
from toolbridge.parallel import ToolCall, execute_parallel


DESCRIPTION = "Use when sleeping briefly. Do not use for long running jobs."


@tool(DESCRIPTION)
def sleeper(delay: float) -> float:
    time.sleep(delay)
    return delay


def test_parallel_calls_complete_in_max_latency_not_sum():
    calls = [
        ToolCall(id="a", name="sleeper", arguments={"delay": 0.2}),
        ToolCall(id="b", name="sleeper", arguments={"delay": 0.2}),
        ToolCall(id="c", name="sleeper", arguments={"delay": 0.2}),
    ]

    start = time.perf_counter()
    results = execute_parallel(calls, {"sleeper": sleeper})
    elapsed = time.perf_counter() - start

    assert results == {"a": 0.2, "b": 0.2, "c": 0.2}
    assert elapsed < 0.45


def test_parallel_correlates_results_by_id_not_completion_order():
    calls = [
        ToolCall(id="slow", name="sleeper", arguments={"delay": 0.2}),
        ToolCall(id="fast", name="sleeper", arguments={"delay": 0.01}),
    ]

    results = execute_parallel(calls, {"sleeper": sleeper})

    assert results["slow"] == 0.2
    assert results["fast"] == 0.01
