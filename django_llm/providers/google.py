"""Google (Gemini) provider implementation."""

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

logger = logging.getLogger("django_llm.providers.google")


class GoogleProvider(Provider):
    """Provider for Google's Gemini API."""

    name = "google"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle API error responses."""
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if response.status_code == 401 or response.status_code == 403:
            raise AuthenticationError(error_msg, provider=self.name, status_code=response.status_code)
        elif response.status_code == 429:
            raise RateLimitError(error_msg, provider=self.name, status_code=429)
        elif response.status_code == 400:
            raise InvalidRequestError(error_msg, provider=self.name, status_code=400)
        else:
            raise ProviderError(error_msg, provider=self.name, status_code=response.status_code)

    def _format_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Format messages for Gemini API.

        Returns:
            Tuple of (system_instruction, formatted_contents).
        """
        system_instruction: str | None = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
                continue

            if msg.role == Role.TOOL:
                # Tool result in Gemini format
                contents.append({
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": msg.tool_call_id,  # Gemini uses function name, not ID
                                "response": {"result": msg.content or ""},
                            }
                        }
                    ],
                })
            elif msg.tool_calls:
                # Assistant message with tool calls (function calls in Gemini)
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    })
                contents.append({"role": "model", "parts": parts})
            else:
                # Regular message
                role = "model" if msg.role == Role.ASSISTANT else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content or ""}],
                })

        return system_instruction, contents

    def format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Format tools for Gemini API."""
        function_declarations = []
        for tool in tools:
            params = tool.get_parameters()
            function_declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": params,
            })

        return [{"functionDeclarations": function_declarations}]

    def _build_payload(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None,
        tools: list[Tool] | None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the request payload."""
        system_instruction, contents = self._format_messages(messages)

        payload: dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature

        # Gemini supports native JSON schema response
        if response_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = response_schema

        if generation_config:
            payload["generationConfig"] = generation_config

        if tools:
            payload["tools"] = self.format_tools(tools)

        return payload

    def _parse_response(self, data: dict[str, Any], model: str) -> Message:
        """Parse a completion response into a Message."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            content = candidate.get("content", {})

            for part in content.get("parts", []):
                if "text" in part:
                    content_parts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=fc["name"],  # Gemini uses name as ID
                            name=fc["name"],
                            arguments=fc.get("args", {}),
                        )
                    )

        tokens = None
        if "usageMetadata" in data:
            usage = data["usageMetadata"]
            tokens = TokenUsage(
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
                cached_tokens=usage.get("cachedContentTokenCount", 0),
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
        """Generate a completion from Gemini."""
        payload = self._build_payload(messages, model, temperature, tools, response_schema)

        if stream:
            return self._complete_streaming(payload, model, on_chunk)
        else:
            return self._complete_sync(payload, model)

    def _complete_sync(self, payload: dict[str, Any], model: str) -> Message:
        """Synchronous completion with retry."""
        import time

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        url,
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
                    logger.warning(f"Max retries ({self.max_retries}) exceeded for Google API")
                    raise

                delay = calculate_backoff(
                    attempt, self.retry_base_delay, self.retry_max_delay
                )
                logger.info(
                    f"Retry {attempt + 1}/{self.max_retries} for Google API "
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
        url = f"{self.base_url}/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tokens: TokenUsage | None = None

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                url,
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

                        candidates = data.get("candidates", [])
                        if candidates:
                            candidate = candidates[0]
                            content = candidate.get("content", {})

                            for part in content.get("parts", []):
                                if "text" in part:
                                    chunk = part["text"]
                                    content_parts.append(chunk)
                                    if on_chunk:
                                        on_chunk(chunk)
                                elif "functionCall" in part:
                                    fc = part["functionCall"]
                                    tool_calls.append(
                                        ToolCall(
                                            id=fc["name"],
                                            name=fc["name"],
                                            arguments=fc.get("args", {}),
                                        )
                                    )

                        if "usageMetadata" in data:
                            usage = data["usageMetadata"]
                            tokens = TokenUsage(
                                input_tokens=usage.get("promptTokenCount", 0),
                                output_tokens=usage.get("candidatesTokenCount", 0),
                            )

        return Message(
            role=Role.ASSISTANT,
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
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
        payload = self._build_payload(messages, model, temperature, tools)
        url = f"{self.base_url}/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                url,
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

                        candidates = data.get("candidates", [])
                        if candidates:
                            candidate = candidates[0]
                            content = candidate.get("content", {})

                            for part in content.get("parts", []):
                                if "text" in part:
                                    yield part["text"]

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
        payload = self._build_payload(messages, model, temperature, tools, response_schema)

        if stream:
            return await self._acomplete_streaming(payload, model, on_chunk)
        else:
            return await self._acomplete_sync(payload, model)

    async def _acomplete_sync(self, payload: dict[str, Any], model: str) -> Message:
        """Async synchronous completion with retry."""
        import asyncio

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url,
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
                    logger.warning(f"Max retries ({self.max_retries}) exceeded for Google API")
                    raise

                delay = calculate_backoff(
                    attempt, self.retry_base_delay, self.retry_max_delay
                )
                logger.info(
                    f"Retry {attempt + 1}/{self.max_retries} for Google API "
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
        url = f"{self.base_url}/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tokens: TokenUsage | None = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                url,
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

                        candidates = data.get("candidates", [])
                        if candidates:
                            candidate = candidates[0]
                            content = candidate.get("content", {})

                            for part in content.get("parts", []):
                                if "text" in part:
                                    chunk = part["text"]
                                    content_parts.append(chunk)
                                    if on_chunk:
                                        on_chunk(chunk)
                                elif "functionCall" in part:
                                    fc = part["functionCall"]
                                    tool_calls.append(
                                        ToolCall(
                                            id=fc["name"],
                                            name=fc["name"],
                                            arguments=fc.get("args", {}),
                                        )
                                    )

                        if "usageMetadata" in data:
                            usage = data["usageMetadata"]
                            tokens = TokenUsage(
                                input_tokens=usage.get("promptTokenCount", 0),
                                output_tokens=usage.get("candidatesTokenCount", 0),
                            )

        return Message(
            role=Role.ASSISTANT,
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
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
        payload = self._build_payload(messages, model, temperature, tools)
        url = f"{self.base_url}/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                url,
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

                        candidates = data.get("candidates", [])
                        if candidates:
                            candidate = candidates[0]
                            content = candidate.get("content", {})

                            for part in content.get("parts", []):
                                if "text" in part:
                                    yield part["text"]
