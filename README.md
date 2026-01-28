# dj-llm

A beautiful unified LLM interface for Django. One API for OpenAI, Anthropic, Google, Azure OpenAI, and Ollama with persistence, streaming, and production-ready features.

Inspired by [RubyLLM](https://github.com/crmne/ruby_llm).

## Features

- **Unified API** - Same interface for OpenAI, Anthropic, Google, Azure OpenAI, and Ollama
- **Django Integration** - Built-in model persistence with `Conversation` and `StoredMessage`
- **Streaming** - Easy streaming with callbacks or iterators
- **Tool Calling** - Define tools as classes or functions
- **Structured Output** - JSON schema responses with dataclasses or Pydantic
- **Async Support** - Full async/await support with `aask()` and `astream()`
- **Cost Tracking** - Automatic cost estimation per request
- **Observability** - Logging hooks for metrics and monitoring
- **Retry Logic** - Exponential backoff for transient failures
- **Local Models** - Run local LLMs with Ollama
- **Fluent Interface** - Chain configuration methods for clean code
- **Minimal Dependencies** - Just Django and httpx

## Installation

```bash
pip install dj-llm
```

Add to your `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    # ...
    'django_llm',
]
```

Run migrations:

```bash
python manage.py migrate
```

## Quick Start

```python
import django_llm

# Simple chat
response = django_llm.chat().ask("What is Python?")
print(response.content)

# With specific model
from django_llm import Chat

chat = Chat(model="gpt-4o")
response = chat.ask("Hello!")

# Streaming
for chunk in chat.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

## Configuration

### Environment Variables

```bash
# Cloud providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."  # or GEMINI_API_KEY

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"  # optional

# Ollama (local)
export OLLAMA_BASE_URL="http://localhost:11434/v1"  # optional, this is the default
```

### Programmatic Configuration

```python
import django_llm

django_llm.configure(
    openai_api_key="sk-...",
    default_model="gpt-4o",
    timeout=60.0,
    max_retries=3,
)
```

## Providers

### OpenAI

```python
chat = Chat(model="gpt-4o")
response = chat.ask("Hello!")
```

### Anthropic (Claude)

```python
chat = Chat(model="claude-sonnet-4-20250514")
response = chat.ask("Hello!")
```

### Google (Gemini)

```python
chat = Chat(model="gemini-2.0-flash")
response = chat.ask("Hello!")
```

### Azure OpenAI

Use the `azure:` prefix with your deployment name:

```python
chat = Chat(model="azure:my-gpt4-deployment")
response = chat.ask("Hello!")
```

### Ollama (Local Models)

Run LLMs locally with [Ollama](https://ollama.com). Use the `ollama:` prefix:

```python
# First, pull a model: ollama pull llama3.2
chat = Chat(model="ollama:llama3.2")
response = chat.ask("Hello!")

# Other popular models
chat = Chat(model="ollama:mistral")
chat = Chat(model="ollama:codellama")
chat = Chat(model="ollama:gemma2")
```

## Fluent Interface

```python
from django_llm import Chat

response = (
    Chat()
    .with_model("claude-sonnet-4-20250514")
    .with_temperature(0.7)
    .with_instructions("You are a helpful assistant.")
    .ask("What's the weather like?")
)
```

## Multi-turn Conversations

```python
chat = Chat(system="You remember everything I tell you.")

chat.ask("My name is Alice.")
response = chat.ask("What's my name?")
# "Your name is Alice."
```

## Streaming

### With Iterator

```python
for chunk in chat.stream("Write a poem"):
    print(chunk, end="", flush=True)
```

### With Callback

```python
def handle_chunk(chunk):
    print(chunk, end="", flush=True)

chat.on_chunk(handle_chunk).ask("Write a poem", stream=True)
```

### Async Streaming

```python
async for chunk in chat.astream("Write a poem"):
    print(chunk, end="", flush=True)
```

## Async Support

Full async/await support for non-blocking operations:

```python
from django_llm import Chat

chat = Chat(model="gpt-4o")

# Async ask
response = await chat.aask("Hello!")

# Async streaming
async for chunk in chat.astream("Tell me a story"):
    print(chunk, end="", flush=True)
```

## Tool Calling

### Class-based Tools

```python
from django_llm import Tool

class WeatherTool(Tool):
    name = "get_weather"
    description = "Get the current weather for a location"

    def execute(self, location: str, unit: str = "celsius") -> str:
        # Your implementation here
        return f"Weather in {location}: 22{unit[0].upper()}"

chat = Chat().with_tools([WeatherTool()])
response = chat.ask("What's the weather in Paris?")
```

### Function-based Tools

```python
from django_llm.tool import tool

@tool(description="Add two numbers together")
def add(a: int, b: int) -> int:
    return a + b

chat = Chat().with_tools([add])
response = chat.ask("What is 2 + 3?")
```

## Structured Output

Get typed responses with JSON schema validation:

```python
from dataclasses import dataclass
from django_llm import Chat

@dataclass
class Person:
    name: str
    age: int
    city: str

chat = Chat().with_schema(Person)
response = chat.ask("Extract: John is 30 years old and lives in NYC")

person = response.parsed  # Person(name='John', age=30, city='NYC')
```

Also works with Pydantic models:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

chat = Chat().with_schema(Person)
```

## Django Model Persistence

### Basic Usage

```python
from django_llm.models import Conversation

# Create a new conversation
conv = Conversation.objects.create(
    model_id="gpt-4o",
    system_prompt="You are a helpful assistant.",
    name="Support Chat",
)

# Chat and persist
response = conv.ask("Hello!")
conv.sync_messages()  # Save to database

# Later, restore the conversation
conv = Conversation.objects.get(pk=1)
response = conv.ask("What did we discuss?")
```

### Auto-save Mode

```python
conv = Conversation.objects.create(model_id="gpt-4o")
conv.with_auto_save(True)  # Messages saved automatically

conv.ask("Hello!")  # Automatically persisted
```

### With User Association

```python
conv = Conversation.objects.create(
    model_id="gpt-4o",
    name="Support Chat",
    user=request.user,
)

# Query user's conversations
user_convos = request.user.conversations.all()
```

### With Metadata

```python
conv = Conversation.objects.create(
    model_id="gpt-4o",
    metadata={"topic": "support", "priority": "high"},
)
```

## Cost Tracking

Automatic cost estimation for cloud providers:

```python
response = chat.ask("Hello!")

# Per-message cost
print(f"Cost: ${response.cost}")  # e.g., $0.000125

# Token usage
print(f"Input tokens: {response.tokens.input_tokens}")
print(f"Output tokens: {response.tokens.output_tokens}")
```

Note: Cost is `None` for Ollama (local) and models without pricing data.

## Observability Hooks

Add custom logging, metrics, or APM integration:

```python
import django_llm

def on_request(model, messages, **kwargs):
    print(f"Calling {model} with {len(messages)} messages")
    # statsd.increment('llm.requests', tags=[f'model:{model}'])

def on_response(model, message, duration_ms, **kwargs):
    print(f"Response in {duration_ms:.0f}ms")
    # statsd.timing('llm.latency', duration_ms)
    if message.cost:
        print(f"Cost: ${message.cost}")

def on_error(model, error, duration_ms, **kwargs):
    print(f"Error after {duration_ms:.0f}ms: {error}")
    # sentry_sdk.capture_exception(error)

django_llm.add_request_hook(on_request)
django_llm.add_response_hook(on_response)
django_llm.add_error_hook(on_error)

# Remove hooks when done
django_llm.clear_hooks()
```

### Django Logging Configuration

```python
LOGGING = {
    'version': 1,
    'handlers': {
        'console': {'class': 'logging.StreamHandler'},
    },
    'loggers': {
        'django_llm': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}
```

## Retry Logic

Automatic retry with exponential backoff for transient failures:

```python
django_llm.configure(
    max_retries=3,           # Number of retries (default: 3)
    retry_base_delay=1.0,    # Initial delay in seconds (default: 1.0)
    retry_max_delay=60.0,    # Maximum delay in seconds (default: 60.0)
)
```

Retries automatically on:
- Rate limit errors (429)
- Server errors (5xx)
- Connection errors
- Timeouts

Does NOT retry on:
- Authentication errors (401)
- Invalid request errors (400)

## Error Handling

```python
from django_llm import Chat
from django_llm.exceptions import (
    DjangoLLMError,      # Base exception
    ProviderError,       # Provider-specific errors
    AuthenticationError, # Invalid API key
    RateLimitError,      # Too many requests
    InvalidRequestError, # Bad request
    ModelNotFoundError,  # Unknown model
)

try:
    chat = Chat()
    response = chat.ask("Hello")
except AuthenticationError:
    print("Check your API key")
except RateLimitError:
    print("Too many requests, slow down")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Supported Models

### OpenAI
- `gpt-4o`, `gpt-4o-mini`
- `gpt-4-turbo`, `gpt-4`
- `gpt-3.5-turbo`
- `o1`, `o1-mini`, `o3-mini`

### Anthropic
- `claude-sonnet-4-20250514`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`

### Google
- `gemini-2.0-flash`
- `gemini-1.5-pro`, `gemini-1.5-flash`

### Azure OpenAI
- Use `azure:` prefix with your deployment name
- Example: `azure:my-gpt4-deployment`

### Ollama (Local)
- Use `ollama:` prefix with any Ollama model
- Examples: `ollama:llama3.2`, `ollama:mistral`, `ollama:codellama`

## Django Admin

The admin interface is automatically registered. View and search conversations and messages at `/admin/django_llm/`.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=django_llm

# Run specific test file
pytest tests/test_chat.py
```

## Dependencies

- Python 3.10+
- Django 4.2+
- httpx

## License

MIT
