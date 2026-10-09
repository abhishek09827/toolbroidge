# toolbroidge

toolbroidge is a developer-facing Python library for defining tools once with type-annotated Python functions and Pydantic v2 validation, then translating and running those tools across OpenAI, Anthropic, or Gemini. It gives agent loops a small canonical tool model, strict provider schemas, concurrent tool execution, streaming argument reassembly, structured output retries, and explicit refusal handling without bringing in a larger orchestration framework.



https://github.com/user-attachments/assets/eb42f01a-7f50-4f5f-bf50-43f1917a1bcc



## Installation

```bash
pip install toolbroidge
```

Provider clients are optional:

```bash
pip install "toolbroidge[openai]"
pip install "toolbroidge[anthropic]"
pip install "toolbroidge[gemini]"
```

## Minimal Example

```python
from toolbroidge import Agent, tool

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

When a model response contains multiple tool calls, toolbroidge fans them out with `ThreadPoolExecutor`, then correlates results by tool call id before the next model turn.

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
from toolbroidge import Agent, Refusal

class Answer(BaseModel):
    value: int

agent = Agent(provider=provider, tools=[], output_schema=Answer)
result = agent.run("Return a number")

if isinstance(result.output, Refusal):
    print(result.output.reason)
else:
    print(result.output.value)
```

## Provider API Key Demo

The repository includes a simple smoke-test app at `examples/provider_demo.py`. It defines two tools once with `@tool`, translates them to the selected provider's tool schema, asks the model to call tools, executes returned calls in parallel, and prints correlated results by tool call id.

Install the package and the provider extras you want to test:

```bash
python3 -m pip install -e ".[openai,anthropic,gemini]"
```

Run with OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
python3 examples/provider_demo.py --provider openai
```

Run with Anthropic:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python3 examples/provider_demo.py --provider anthropic
```

Run with Gemini:

```bash
export GEMINI_API_KEY="..."
python3 examples/provider_demo.py --provider gemini
```

Optional model overrides:

```bash
export OPENAI_MODEL="gpt-6-luna"
export ANTHROPIC_MODEL="claude-3-5-haiku-latest"
export GEMINI_MODEL="gemini-3.5-flash-lite"
```
