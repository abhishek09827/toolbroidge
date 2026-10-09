# toolbridge

toolbridge is a developer-facing Python library for defining tools once with type-annotated Python functions and Pydantic v2 validation, then translating and running those tools across OpenAI, Anthropic, or Gemini. It gives agent loops a small canonical tool model, strict provider schemas, concurrent tool execution, streaming argument reassembly, structured output retries, and explicit refusal handling without bringing in a larger orchestration framework.

## Installation

```bash
pip install toolbridge
```

Provider clients are optional:

```bash
pip install "toolbridge[openai]"
pip install "toolbridge[anthropic]"
pip install "toolbridge[gemini]"
```

## Minimal Example

```python
from toolbridge import Agent, tool

@tool("Use when adding two numbers. Do not use for non-arithmetic tasks.")
def add(left: int, right: int) -> int:
    return left + right

provider = YourProvider()
agent = Agent(provider=provider, tools=[add])
run = agent.run("What is 2 + 3?")
print(run.output)
```

## Provider Shape Comparison

| Provider | Tool wire shape |
| --- | --- |
| OpenAI | `{type:"function", function:{name, description, parameters, strict:true}}` |
| Anthropic | `{name, description, input_schema}` |
| Gemini | `{functionDeclarations:[{name, description, parameters}]}` |

## Parallel Calls

When a model response contains multiple tool calls, toolbridge fans them out with `ThreadPoolExecutor`, then correlates results by tool call id before the next model turn.

```text
model response
  |-- call id=a -> tool A --\
  |-- call id=b -> tool B ----> {a: result_a, b: result_b, c: result_c}
  |-- call id=c -> tool C --/
                         next model turn
```

## Refusal Handling

Structured output accepts a Pydantic `BaseModel`. Parse errors and schema violations retry up to three times with the typed error injected for the next attempt. Refusals return immediately as `Refusal`.

```python
from pydantic import BaseModel
from toolbridge import Agent, Refusal

class Answer(BaseModel):
    value: int

agent = Agent(provider=provider, tools=[], output_schema=Answer)
result = agent.run("Return a number")

if isinstance(result.output, Refusal):
    print(result.output.reason)
else:
    print(result.output.value)
```
