"""Exception classes for Django LLM."""


class DjangoLLMError(Exception):
    """Base exception for all Django LLM errors."""
    pass


class ProviderError(DjangoLLMError):
    """Error from an LLM provider."""

    def __init__(self, message: str, provider: str | None = None, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Authentication failed with the provider."""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded with the provider."""
    pass


class InvalidRequestError(ProviderError):
    """Invalid request sent to the provider."""
    pass


class ModelNotFoundError(DjangoLLMError):
    """Requested model was not found."""
    pass


class ToolError(DjangoLLMError):
    """Error executing a tool."""
    pass
