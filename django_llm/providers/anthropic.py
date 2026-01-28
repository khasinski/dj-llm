"""Anthropic provider implementation."""

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx

from django_llm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)
from django_llm.message import Message, Role, TokenUsage, ToolCall
from django_llm.providers.base import Provider
from django_llm.retry import calculate_backoff, should_retry
from django_llm.tool import Tool

logger = logging.getLogger("django_llm.providers.anthropic")


class AnthropicProvider(Provider):
    """Provider for Anthropic's API (Claude models)."""

    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"
    api_version = "2023-06-01"

    def _get_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": self.api_version,
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle API error responses."""
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if response.status_code == 401:
            raise AuthenticationError(error_msg, provider=self.name, status_code=401)
        elif response.status_code == 429:
            raise RateLimitError(error_msg, provider=self.name, status_code=429)
        elif response.status_code == 400:
            raise InvalidRequestError(error_msg, provider=self.name, status_code=400)
        else:
            raise ProviderError(error_msg, provider=self.name, status_code=response.status_code)

    def _format_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Format messages for Anthropic API.

        Returns:
            Tuple of (system_message, formatted_messages).
            Anthropic requires system message to be separate.
        """
        system_content: str | None = None
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content
                continue

            if msg.role == Role.TOOL:
                # Tool result in Anthropic format
                formatted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content or "",
                        }
                    ],
                })
            elif msg.tool_calls:
                # Assistant message with tool calls
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                formatted.append({"role": "assistant", "content": content})
            else:
                # Regular message
                role = "assistant" if msg.role == Role.ASSISTANT else "user"
                formatted.append({
                    "role": role,
                    "content": msg.content or "",
                })

        return system_content, formatted

    def format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Format tools for Anthropic API."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.get_parameters(),
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
        max_tokens: int = 4096,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the request payload."""
        system_content, formatted_messages = self._format_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # For structured output, prepend JSON instruction to system message
        if response_schema:
            schema_instruction = (
                f"You must respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(response_schema, indent=2)}\n```\n"
                f"Respond ONLY with the JSON object, no other text."
            )
            if system_content:
                system_content = f"{schema_instruction}\n\n{system_content}"
            else:
                system_content = schema_instruction

        if system_content:
            payload["system"] = system_content

        if temperature is not None:
            payload["temperature"] = temperature

        if tools:
            payload["tools"] = self.format_tools(tools)

        return payload

    def _parse_response(self, data: dict[str, Any], model: str) -> Message:
        """Parse a completion response into a Message."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in data.get("content", []):
            if block["type"] == "text":
                content_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        tokens = None
        if "usage" in data:
            usage = data["usage"]
            tokens = TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cached_tokens=usage.get("cache_read_input_tokens", 0),
            )

        return Message(
            role=Role.ASSISTANT,
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            model_id=model,
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
        """Generate a completion from Anthropic."""
        max_tokens = kwargs.get("max_tokens", 4096)
        payload = self._build_payload(
            messages, model, temperature, tools, stream, max_tokens, response_schema
        )

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
                        f"{self.base_url}/messages",
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
                    logger.warning(f"Max retries ({self.max_retries}) exceeded for Anthropic API")
                    raise

                delay = calculate_backoff(
                    attempt, self.retry_base_delay, self.retry_max_delay
                )
                logger.info(
                    f"Retry {attempt + 1}/{self.max_retries} for Anthropic API "
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
        tool_calls: dict[int, dict[str, Any]] = {}
        tokens: TokenUsage | None = None
        current_tool_index = -1

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    self._handle_error(response)

                for line in response.iter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type")

                        if event_type == "content_block_start":
                            block = data.get("content_block", {})
                            if block.get("type") == "tool_use":
                                current_tool_index = data.get("index", 0)
                                tool_calls[current_tool_index] = {
                                    "id": block.get("id", ""),
                                    "name": block.get("name", ""),
                                    "arguments": "",
                                }

                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                chunk = delta.get("text", "")
                                content_parts.append(chunk)
                                if on_chunk:
                                    on_chunk(chunk)
                            elif delta.get("type") == "input_json_delta":
                                if current_tool_index in tool_calls:
                                    tool_calls[current_tool_index]["arguments"] += delta.get("partial_json", "")

                        elif event_type == "message_delta":
                            usage = data.get("usage", {})
                            if usage:
                                tokens = TokenUsage(
                                    input_tokens=usage.get("input_tokens", 0),
                                    output_tokens=usage.get("output_tokens", 0),
                                )

                        elif event_type == "message_start":
                            message = data.get("message", {})
                            usage = message.get("usage", {})
                            if usage:
                                tokens = TokenUsage(
                                    input_tokens=usage.get("input_tokens", 0),
                                    output_tokens=usage.get("output_tokens", 0),
                                )

        # Build final message
        parsed_tool_calls = []
        for tc in tool_calls.values():
            if tc["id"] and tc["name"]:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                parsed_tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))

        return Message(
            role=Role.ASSISTANT,
            content="".join(content_parts) or None,
            tool_calls=parsed_tool_calls,
            model_id=model,
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
        max_tokens = kwargs.get("max_tokens", 4096)
        payload = self._build_payload(messages, model, temperature, tools, stream=True, max_tokens=max_tokens)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    self._handle_error(response)

                for line in response.iter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")

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
        max_tokens = kwargs.get("max_tokens", 4096)
        payload = self._build_payload(
            messages, model, temperature, tools, stream, max_tokens, response_schema
        )

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
                        f"{self.base_url}/messages",
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
                    logger.warning(f"Max retries ({self.max_retries}) exceeded for Anthropic API")
                    raise

                delay = calculate_backoff(
                    attempt, self.retry_base_delay, self.retry_max_delay
                )
                logger.info(
                    f"Retry {attempt + 1}/{self.max_retries} for Anthropic API "
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
        tool_calls: dict[int, dict[str, Any]] = {}
        tokens: TokenUsage | None = None
        current_tool_index = -1

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._handle_error(response)

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type")

                        if event_type == "content_block_start":
                            block = data.get("content_block", {})
                            if block.get("type") == "tool_use":
                                current_tool_index = data.get("index", 0)
                                tool_calls[current_tool_index] = {
                                    "id": block.get("id", ""),
                                    "name": block.get("name", ""),
                                    "arguments": "",
                                }

                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                chunk = delta.get("text", "")
                                content_parts.append(chunk)
                                if on_chunk:
                                    on_chunk(chunk)
                            elif delta.get("type") == "input_json_delta":
                                if current_tool_index in tool_calls:
                                    tool_calls[current_tool_index]["arguments"] += delta.get("partial_json", "")

                        elif event_type == "message_delta":
                            usage = data.get("usage", {})
                            if usage:
                                tokens = TokenUsage(
                                    input_tokens=usage.get("input_tokens", 0),
                                    output_tokens=usage.get("output_tokens", 0),
                                )

                        elif event_type == "message_start":
                            message = data.get("message", {})
                            usage = message.get("usage", {})
                            if usage:
                                tokens = TokenUsage(
                                    input_tokens=usage.get("input_tokens", 0),
                                    output_tokens=usage.get("output_tokens", 0),
                                )

        parsed_tool_calls = []
        for tc in tool_calls.values():
            if tc["id"] and tc["name"]:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                parsed_tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))

        return Message(
            role=Role.ASSISTANT,
            content="".join(content_parts) or None,
            tool_calls=parsed_tool_calls,
            model_id=model,
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
        max_tokens = kwargs.get("max_tokens", 4096)
        payload = self._build_payload(messages, model, temperature, tools, stream=True, max_tokens=max_tokens)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._handle_error(response)

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
