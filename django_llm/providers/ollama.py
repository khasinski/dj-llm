"""Ollama provider implementation for local LLM inference.

Ollama uses an OpenAI-compatible API, making integration straightforward.
Default endpoint is http://localhost:11434/v1.

See https://ollama.com for installation and model downloads.
"""

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx

from django_llm.exceptions import (
    InvalidRequestError,
    ProviderError,
)
from django_llm.message import Message, Role, TokenUsage, ToolCall
from django_llm.providers.base import Provider
from django_llm.retry import calculate_backoff, should_retry
from django_llm.tool import Tool

logger = logging.getLogger("django_llm.providers.ollama")


class OllamaProvider(Provider):
    """Provider for Ollama local LLM inference.

    Ollama runs locally and uses an OpenAI-compatible API format.
    No API key is required for local inference.

    Popular models:
        - llama3.2 (Meta's Llama 3.2)
        - mistral (Mistral 7B)
        - codellama (Code Llama)
        - gemma2 (Google's Gemma 2)
        - qwen2.5 (Alibaba's Qwen 2.5)
        - phi3 (Microsoft's Phi-3)

    Example:
        >>> from django_llm import Chat
        >>> chat = Chat(model="ollama:llama3.2")
        >>> response = chat.ask("Hello!")

    With custom host:
        >>> from django_llm import configure
        >>> configure(ollama_base_url="http://my-server:11434/v1")
    """

    name = "ollama"
    base_url = "http://localhost:11434/v1"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 300.0,  # Longer timeout for local inference
        base_url: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ):
        # Ollama doesn't require an API key, but we accept one for compatibility
        super().__init__(
            api_key=api_key or "ollama",
            timeout=timeout,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
        )
        if base_url:
            self.base_url = base_url.rstrip("/")

    def _get_headers(self) -> dict[str, str]:
        """Get request headers. No auth needed for local Ollama."""
        return {
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle API error responses."""
        try:
            error_data = response.json()
            error_msg = error_data.get("error", response.text)
        except Exception:
            error_msg = response.text

        if response.status_code == 400:
            raise InvalidRequestError(error_msg, provider=self.name, status_code=400)
        elif response.status_code == 404:
            raise InvalidRequestError(
                f"Model not found. Did you run 'ollama pull <model>'? {error_msg}",
                provider=self.name,
                status_code=404,
            )
        else:
            raise ProviderError(error_msg, provider=self.name, status_code=response.status_code)

    def _get_model_name(self, model: str) -> str:
        """Extract the clean model name without ollama: prefix."""
        if model.startswith("ollama:"):
            return model[7:]
        return model

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for Ollama API (OpenAI-compatible format)."""
        formatted = []
        for msg in messages:
            formatted_msg: dict[str, Any] = {"role": msg.role.value}

            if msg.role == Role.TOOL:
                formatted_msg["role"] = "tool"
                formatted_msg["content"] = msg.content or ""
                formatted_msg["tool_call_id"] = msg.tool_call_id
            elif msg.tool_calls:
                formatted_msg["content"] = msg.content or ""
                formatted_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            else:
                formatted_msg["content"] = msg.content or ""

            formatted.append(formatted_msg)

        return formatted

    def format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Format tools for Ollama API (OpenAI-compatible format)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_parameters(),
                },
            }
            for tool in tools
        ]

    def _build_payload(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None,
        tools: list[Tool] | None,
        stream: bool,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the request payload."""
        clean_model = self._get_model_name(model)

        payload: dict[str, Any] = {
            "model": clean_model,
            "messages": self._format_messages(messages),
            "stream": stream,
        }

        if temperature is not None:
            payload["temperature"] = temperature

        if tools:
            payload["tools"] = self.format_tools(tools)

        if stream:
            payload["stream_options"] = {"include_usage": True}

        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": response_schema,
                },
            }

        return payload

    def _parse_response(self, data: dict[str, Any], model: str) -> Message:
        """Parse a completion response into a Message."""
        choice = data["choices"][0]
        message_data = choice["message"]

        content = message_data.get("content")
        tool_calls = []

        if "tool_calls" in message_data:
            for tc in message_data["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args,
                    )
                )

        tokens = None
        if "usage" in data:
            usage = data["usage"]
            tokens = TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

        # Ensure model_id has ollama: prefix but not doubled
        model_id = model if model.startswith("ollama:") else f"ollama:{model}"

        return Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            model_id=model_id,
            tokens=tokens,
            raw=data,
        )

    def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        response_schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> Message:
        """Generate a completion from Ollama."""
        payload = self._build_payload(messages, model, temperature, tools, stream, response_schema)

        if stream:
            return self._complete_streaming(payload, model, on_chunk)
        else:
            return self._complete_sync(payload, model)

    def _complete_sync(self, payload: dict[str, Any], model: str) -> Message:
        """Synchronous completion with retry."""
        import time

        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._get_headers(),
                        json=payload,
                    )

                    if response.status_code != 200:
                        self._handle_error(response)

                    return self._parse_response(response.json(), model)

            except Exception as e:
                last_exception = e

                if not should_retry(e):
                    raise

                if attempt >= self.max_retries:
                    logger.warning(f"Max retries ({self.max_retries}) exceeded for Ollama API")
                    raise

                delay = calculate_backoff(
                    attempt, self.retry_base_delay, self.retry_max_delay
                )
                logger.info(
                    f"Retry {attempt + 1}/{self.max_retries} for Ollama API "
                    f"after {delay:.2f}s due to: {e}"
                )
                time.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    def _complete_streaming(
        self,
        payload: dict[str, Any],
        model: str,
        on_chunk: Callable[[str], None] | None,
    ) -> Message:
        """Streaming completion with callback."""
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tokens: TokenUsage | None = None

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    self._handle_error(response)

                for line in response.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        if "usage" in data:
                            usage = data["usage"]
                            tokens = TokenUsage(
                                input_tokens=usage.get("prompt_tokens", 0),
                                output_tokens=usage.get("completion_tokens", 0),
                            )

                        if "choices" not in data or not data["choices"]:
                            continue

                        delta = data["choices"][0].get("delta", {})

                        if "content" in delta and delta["content"]:
                            chunk = delta["content"]
                            content_parts.append(chunk)
                            if on_chunk:
                                on_chunk(chunk)

                        if "tool_calls" in delta:
                            for tc_delta in delta["tool_calls"]:
                                idx = tc_delta.get("index", 0)
                                while len(tool_calls) <= idx:
                                    tool_calls.append({"id": "", "name": "", "arguments": ""})

                                if "id" in tc_delta:
                                    tool_calls[idx]["id"] = tc_delta["id"]
                                if "function" in tc_delta:
                                    if "name" in tc_delta["function"]:
                                        tool_calls[idx]["name"] = tc_delta["function"]["name"]
                                    if "arguments" in tc_delta["function"]:
                                        tool_calls[idx]["arguments"] += tc_delta["function"]["arguments"]

        parsed_tool_calls = []
        for tc in tool_calls:
            if tc["id"] and tc["name"]:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                parsed_tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))

        # Ensure model_id has ollama: prefix but not doubled
        model_id = model if model.startswith("ollama:") else f"ollama:{model}"

        return Message(
            role=Role.ASSISTANT,
            content="".join(content_parts) or None,
            tool_calls=parsed_tool_calls,
            model_id=model_id,
            tokens=tokens,
        )

    def stream(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream completion as an iterator."""
        payload = self._build_payload(messages, model, temperature, tools, stream=True)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    self._handle_error(response)

                for line in response.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        if "choices" not in data or not data["choices"]:
                            continue

                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"]

    # Async methods

    async def acomplete(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        response_schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> Message:
        """Async version of complete()."""
        payload = self._build_payload(messages, model, temperature, tools, stream, response_schema)

        if stream:
            return await self._acomplete_streaming(payload, model, on_chunk)
        else:
            return await self._acomplete_sync(payload, model)

    async def _acomplete_sync(self, payload: dict[str, Any], model: str) -> Message:
        """Async synchronous completion with retry."""
        import asyncio

        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._get_headers(),
                        json=payload,
                    )

                    if response.status_code != 200:
                        self._handle_error(response)

                    return self._parse_response(response.json(), model)

            except Exception as e:
                last_exception = e

                if not should_retry(e):
                    raise

                if attempt >= self.max_retries:
                    logger.warning(f"Max retries ({self.max_retries}) exceeded for Ollama API")
                    raise

                delay = calculate_backoff(
                    attempt, self.retry_base_delay, self.retry_max_delay
                )
                logger.info(
                    f"Retry {attempt + 1}/{self.max_retries} for Ollama API "
                    f"after {delay:.2f}s due to: {e}"
                )
                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    async def _acomplete_streaming(
        self,
        payload: dict[str, Any],
        model: str,
        on_chunk: Callable[[str], None] | None,
    ) -> Message:
        """Async streaming completion with callback."""
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tokens: TokenUsage | None = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._handle_error(response)

                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        if "usage" in data:
                            usage = data["usage"]
                            tokens = TokenUsage(
                                input_tokens=usage.get("prompt_tokens", 0),
                                output_tokens=usage.get("completion_tokens", 0),
                            )

                        if "choices" not in data or not data["choices"]:
                            continue

                        delta = data["choices"][0].get("delta", {})

                        if "content" in delta and delta["content"]:
                            chunk = delta["content"]
                            content_parts.append(chunk)
                            if on_chunk:
                                on_chunk(chunk)

                        if "tool_calls" in delta:
                            for tc_delta in delta["tool_calls"]:
                                idx = tc_delta.get("index", 0)
                                while len(tool_calls) <= idx:
                                    tool_calls.append({"id": "", "name": "", "arguments": ""})

                                if "id" in tc_delta:
                                    tool_calls[idx]["id"] = tc_delta["id"]
                                if "function" in tc_delta:
                                    if "name" in tc_delta["function"]:
                                        tool_calls[idx]["name"] = tc_delta["function"]["name"]
                                    if "arguments" in tc_delta["function"]:
                                        tool_calls[idx]["arguments"] += tc_delta["function"]["arguments"]

        parsed_tool_calls = []
        for tc in tool_calls:
            if tc["id"] and tc["name"]:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                parsed_tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))

        # Ensure model_id has ollama: prefix but not doubled
        model_id = model if model.startswith("ollama:") else f"ollama:{model}"

        return Message(
            role=Role.ASSISTANT,
            content="".join(content_parts) or None,
            tool_calls=parsed_tool_calls,
            model_id=model_id,
            tokens=tokens,
        )

    async def astream(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Async stream completion."""
        payload = self._build_payload(messages, model, temperature, tools, stream=True)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._handle_error(response)

                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        if "choices" not in data or not data["choices"]:
                            continue

                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
