# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-01-22

### Added

- **Retry with Exponential Backoff**
  - Automatic retries for transient failures (rate limits, server errors)
  - Configurable `max_retries`, `retry_base_delay`, `retry_max_delay`
  - Jitter to prevent thundering herd
- **Cost Tracking**
  - Per-request cost estimation in USD
  - `message.cost` property returns cost as `Decimal`
  - Pricing data for OpenAI, Anthropic, and Google models
  - `TokenUsage.cost(model)` for manual calculation
- **Logging & Observability Hooks**
  - Structured logging via `django_llm` logger
  - Request/response/error hooks for custom metrics
  - `add_request_hook()`, `add_response_hook()`, `add_error_hook()`
  - `timed_request()` context manager for timing
- **Ollama Provider** (local LLM inference)
  - Support for local models via `ollama:` prefix
  - No API key required
  - Configurable base URL via `OLLAMA_BASE_URL`
  - Popular models: llama3.2, mistral, codellama, gemma2
- **Azure OpenAI Provider**
  - Enterprise Azure deployments via `azure:` prefix
  - Deployment-based routing
  - Configurable endpoint, API key, and API version
  - Environment variables: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`

### Changed

- Test suite expanded to 178+ tests
- Improved error messages with provider context

## [0.1.0] - 2025-01-22

### Added

- Initial release of django-llm
- **Core Chat API** with fluent interface inspired by RubyLLM
  - `Chat` class with `.ask()`, `.say()`, `.stream()` methods
  - Method chaining: `.with_model()`, `.with_temperature()`, `.with_instructions()`
  - Message callbacks: `.on_new_message()`, `.on_chunk()`
- **Async Support**
  - `aask()`, `asay()`, `astream()` async methods
  - Full async provider implementations
- **Provider Support**
  - OpenAI (GPT-4o, GPT-4, GPT-3.5, o1 models)
  - Anthropic (Claude Sonnet 4, Claude 3 family)
  - Google (Gemini 2.0, Gemini 1.5 family)
  - Automatic model-to-provider routing
- **Tool/Function Calling**
  - Class-based tools extending `Tool` base class
  - `@tool` decorator for function-based tools
  - Automatic parameter schema generation
  - Multi-turn tool execution handling
- **Structured Output**
  - JSON schema responses with `with_schema()`
  - Support for dataclasses, Pydantic models, and raw schemas
  - Parsed results via `message.parsed`
- **Django Model Persistence**
  - `Conversation` model with user association
  - `StoredMessage` model with full message data
  - Auto-save mode for real-time persistence
  - Conversation restoration from database
- **Django Admin Interface**
  - Browse and search conversations
  - View message history inline
  - Token usage statistics
- **Token Tracking**
  - Input, output, cached, and reasoning tokens
  - Per-message and conversation-level tracking
- **Streaming**
  - Iterator-based streaming with `stream()`
  - Callback-based streaming with `on_chunk()`
- **Error Handling**
  - Provider-specific exceptions
  - `AuthenticationError`, `RateLimitError`, `InvalidRequestError`
- **Configuration**
  - Environment variable support for API keys
  - Global configuration via `configure()`
  - Per-chat model and temperature settings
- **Testing**
  - 125+ tests with pytest
  - VCR cassettes for reproducible API tests
  - Django integration tests

### Dependencies

- Django >= 4.2
- httpx >= 0.25.0

[Unreleased]: https://github.com/khasinski/dj-llm/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/khasinski/dj-llm/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/khasinski/dj-llm/releases/tag/v0.1.0
