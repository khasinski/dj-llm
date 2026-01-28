"""Configuration management for Django LLM."""

import os
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Config:
    """Global configuration for Django LLM.

    API keys are read from environment variables by default:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY (or GEMINI_API_KEY)
    - OLLAMA_BASE_URL (optional, defaults to http://localhost:11434/v1)
    - AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for Azure OpenAI
    """

    # API Keys
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    google_api_key: str | None = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )

    # Ollama settings (local LLM inference)
    ollama_base_url: str | None = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL")
    )

    # Azure OpenAI settings
    azure_openai_endpoint: str | None = field(
        default_factory=lambda: os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    azure_openai_api_key: str | None = field(
        default_factory=lambda: os.environ.get("AZURE_OPENAI_API_KEY")
    )
    azure_openai_api_version: str = field(
        default_factory=lambda: os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    )

    # Default model (auto-selects based on available keys if None)
    default_model: str | None = None

    # Request settings
    timeout: float = 120.0

    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0

    # Streaming chunk callback (for global handling)
    on_chunk: Callable[[str], None] | None = None

    def get_default_model(self) -> str:
        """Get the default model, auto-selecting based on available API keys."""
        if self.default_model:
            return self.default_model

        # Auto-select based on available keys
        if self.openai_api_key:
            return "gpt-4o"
        elif self.anthropic_api_key:
            return "claude-sonnet-4-20250514"
        elif self.google_api_key:
            return "gemini-2.0-flash"
        else:
            raise ValueError(
                "No API keys configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or GOOGLE_API_KEY environment variable."
            )


# Global configuration instance
_config = Config()


def configure(**kwargs) -> Config:
    """Configure Django LLM settings.

    Args:
        **kwargs: Configuration options to set.

    Returns:
        The updated configuration instance.

    Example:
        >>> import django_llm
        >>> django_llm.configure(
        ...     openai_api_key="sk-...",
        ...     default_model="gpt-4o",
        ...     timeout=60.0
        ... )
    """
    global _config
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            raise ValueError(f"Unknown configuration option: {key}")
    return _config


def get_config() -> Config:
    """Get the current configuration."""
    return _config
