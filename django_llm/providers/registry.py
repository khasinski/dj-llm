"""Provider registry and model routing."""

from django_llm.configuration import get_config
from django_llm.exceptions import AuthenticationError, ModelNotFoundError
from django_llm.providers.base import Provider

# Model prefix to provider mapping
MODEL_PREFIXES = {
    # OpenAI models
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "chatgpt-": "openai",
    # Anthropic models
    "claude-": "anthropic",
    # Google models
    "gemini-": "google",
    "gemma-": "google",
    # Ollama (local) models
    "ollama:": "ollama",
    # Azure OpenAI
    "azure:": "azure_openai",
}


def get_provider_name_for_model(model: str) -> str:
    """Determine the provider for a given model identifier."""
    model_lower = model.lower()

    for prefix, provider in MODEL_PREFIXES.items():
        if model_lower.startswith(prefix):
            return provider

    raise ModelNotFoundError(
        f"Unknown model: {model}. Cannot determine provider. "
        f"Supported prefixes: {list(MODEL_PREFIXES.keys())}"
    )


def get_provider(provider_name: str) -> Provider:
    """Get a provider instance by name.

    Args:
        provider_name: One of 'openai', 'anthropic', or 'google'.

    Returns:
        An initialized provider instance.

    Raises:
        ValueError: If provider name is unknown.
        AuthenticationError: If API key is not configured.
    """
    config = get_config()

    if provider_name == "openai":
        if not config.openai_api_key:
            raise AuthenticationError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable.",
                provider="openai"
            )
        from django_llm.providers.openai import OpenAIProvider
        return OpenAIProvider(
            api_key=config.openai_api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_base_delay,
            retry_max_delay=config.retry_max_delay,
        )

    elif provider_name == "anthropic":
        if not config.anthropic_api_key:
            raise AuthenticationError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable.",
                provider="anthropic"
            )
        from django_llm.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            api_key=config.anthropic_api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_base_delay,
            retry_max_delay=config.retry_max_delay,
        )

    elif provider_name == "google":
        if not config.google_api_key:
            raise AuthenticationError(
                "Google API key not configured. Set GOOGLE_API_KEY environment variable.",
                provider="google"
            )
        from django_llm.providers.google import GoogleProvider
        return GoogleProvider(
            api_key=config.google_api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_base_delay,
            retry_max_delay=config.retry_max_delay,
        )

    elif provider_name == "ollama":
        from django_llm.providers.ollama import OllamaProvider
        return OllamaProvider(
            base_url=config.ollama_base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_base_delay,
            retry_max_delay=config.retry_max_delay,
        )

    elif provider_name == "azure_openai":
        if not config.azure_openai_api_key or not config.azure_openai_endpoint:
            raise AuthenticationError(
                "Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT and "
                "AZURE_OPENAI_API_KEY environment variables.",
                provider="azure_openai"
            )
        from django_llm.providers.azure_openai import AzureOpenAIProvider
        return AzureOpenAIProvider(
            api_key=config.azure_openai_api_key,
            endpoint=config.azure_openai_endpoint,
            api_version=config.azure_openai_api_version,
            timeout=config.timeout,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_base_delay,
            retry_max_delay=config.retry_max_delay,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            "Use 'openai', 'anthropic', 'google', 'ollama', or 'azure_openai'."
        )


def get_provider_for_model(model: str) -> Provider:
    """Get a provider instance for a given model.

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514', 'gemini-2.0-flash').

    Returns:
        An initialized provider instance for the model.
    """
    provider_name = get_provider_name_for_model(model)
    return get_provider(provider_name)
