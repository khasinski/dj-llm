"""Model pricing for cost estimation.

Prices are in USD per 1M tokens.
Last updated: January 2025

Note: Prices may change. Check provider documentation for current rates.
"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class ModelPricing:
    """Pricing for a model in USD per 1M tokens."""

    input_price: Decimal  # USD per 1M input tokens
    output_price: Decimal  # USD per 1M output tokens
    cached_input_price: Decimal | None = None  # USD per 1M cached input tokens


# OpenAI pricing (as of January 2025)
# https://openai.com/pricing
OPENAI_PRICING: dict[str, ModelPricing] = {
    # GPT-4o
    "gpt-4o": ModelPricing(
        input_price=Decimal("2.50"),
        output_price=Decimal("10.00"),
        cached_input_price=Decimal("1.25"),
    ),
    "gpt-4o-2024-11-20": ModelPricing(
        input_price=Decimal("2.50"),
        output_price=Decimal("10.00"),
        cached_input_price=Decimal("1.25"),
    ),
    "gpt-4o-2024-08-06": ModelPricing(
        input_price=Decimal("2.50"),
        output_price=Decimal("10.00"),
        cached_input_price=Decimal("1.25"),
    ),
    # GPT-4o mini
    "gpt-4o-mini": ModelPricing(
        input_price=Decimal("0.15"),
        output_price=Decimal("0.60"),
        cached_input_price=Decimal("0.075"),
    ),
    "gpt-4o-mini-2024-07-18": ModelPricing(
        input_price=Decimal("0.15"),
        output_price=Decimal("0.60"),
        cached_input_price=Decimal("0.075"),
    ),
    # GPT-4 Turbo
    "gpt-4-turbo": ModelPricing(
        input_price=Decimal("10.00"),
        output_price=Decimal("30.00"),
    ),
    "gpt-4-turbo-2024-04-09": ModelPricing(
        input_price=Decimal("10.00"),
        output_price=Decimal("30.00"),
    ),
    # GPT-4
    "gpt-4": ModelPricing(
        input_price=Decimal("30.00"),
        output_price=Decimal("60.00"),
    ),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": ModelPricing(
        input_price=Decimal("0.50"),
        output_price=Decimal("1.50"),
    ),
    "gpt-3.5-turbo-0125": ModelPricing(
        input_price=Decimal("0.50"),
        output_price=Decimal("1.50"),
    ),
    # o1 models
    "o1": ModelPricing(
        input_price=Decimal("15.00"),
        output_price=Decimal("60.00"),
        cached_input_price=Decimal("7.50"),
    ),
    "o1-2024-12-17": ModelPricing(
        input_price=Decimal("15.00"),
        output_price=Decimal("60.00"),
        cached_input_price=Decimal("7.50"),
    ),
    "o1-preview": ModelPricing(
        input_price=Decimal("15.00"),
        output_price=Decimal("60.00"),
    ),
    "o1-mini": ModelPricing(
        input_price=Decimal("3.00"),
        output_price=Decimal("12.00"),
        cached_input_price=Decimal("1.50"),
    ),
    # o3-mini (preview pricing)
    "o3-mini": ModelPricing(
        input_price=Decimal("1.10"),
        output_price=Decimal("4.40"),
        cached_input_price=Decimal("0.55"),
    ),
}

# Anthropic pricing (as of January 2025)
# https://www.anthropic.com/pricing
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # Claude Sonnet 4
    "claude-sonnet-4-20250514": ModelPricing(
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        cached_input_price=Decimal("0.30"),
    ),
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": ModelPricing(
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        cached_input_price=Decimal("0.30"),
    ),
    "claude-3-5-sonnet-20240620": ModelPricing(
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
    ),
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": ModelPricing(
        input_price=Decimal("0.80"),
        output_price=Decimal("4.00"),
        cached_input_price=Decimal("0.08"),
    ),
    # Claude 3 Opus
    "claude-3-opus-20240229": ModelPricing(
        input_price=Decimal("15.00"),
        output_price=Decimal("75.00"),
        cached_input_price=Decimal("1.50"),
    ),
    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": ModelPricing(
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
    ),
    # Claude 3 Haiku
    "claude-3-haiku-20240307": ModelPricing(
        input_price=Decimal("0.25"),
        output_price=Decimal("1.25"),
        cached_input_price=Decimal("0.03"),
    ),
}

# Google pricing (as of January 2025)
# https://ai.google.dev/pricing
GOOGLE_PRICING: dict[str, ModelPricing] = {
    # Gemini 2.0 Flash
    "gemini-2.0-flash": ModelPricing(
        input_price=Decimal("0.10"),
        output_price=Decimal("0.40"),
    ),
    "gemini-2.0-flash-exp": ModelPricing(
        input_price=Decimal("0.00"),  # Free during preview
        output_price=Decimal("0.00"),
    ),
    # Gemini 1.5 Pro
    "gemini-1.5-pro": ModelPricing(
        input_price=Decimal("1.25"),  # Up to 128K
        output_price=Decimal("5.00"),
        cached_input_price=Decimal("0.3125"),
    ),
    "gemini-1.5-pro-latest": ModelPricing(
        input_price=Decimal("1.25"),
        output_price=Decimal("5.00"),
        cached_input_price=Decimal("0.3125"),
    ),
    # Gemini 1.5 Flash
    "gemini-1.5-flash": ModelPricing(
        input_price=Decimal("0.075"),  # Up to 128K
        output_price=Decimal("0.30"),
        cached_input_price=Decimal("0.01875"),
    ),
    "gemini-1.5-flash-latest": ModelPricing(
        input_price=Decimal("0.075"),
        output_price=Decimal("0.30"),
        cached_input_price=Decimal("0.01875"),
    ),
    # Gemini 1.0 Pro (legacy)
    "gemini-pro": ModelPricing(
        input_price=Decimal("0.50"),
        output_price=Decimal("1.50"),
    ),
}

# Combined pricing dictionary
ALL_PRICING: dict[str, ModelPricing] = {
    **OPENAI_PRICING,
    **ANTHROPIC_PRICING,
    **GOOGLE_PRICING,
}


def get_model_pricing(model: str) -> ModelPricing | None:
    """Get pricing for a model.

    Args:
        model: Model identifier.

    Returns:
        ModelPricing if found, None otherwise.
    """
    # Try exact match first
    if model in ALL_PRICING:
        return ALL_PRICING[model]

    # Try prefix matching for versioned models
    model_lower = model.lower()
    for model_id, pricing in ALL_PRICING.items():
        if model_lower.startswith(model_id.lower()):
            return pricing

    return None


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> Decimal | None:
    """Calculate the cost of a request in USD.

    Args:
        model: Model identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cached_tokens: Number of cached input tokens.

    Returns:
        Cost in USD, or None if pricing is unknown.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    # Calculate cost (prices are per 1M tokens)
    million = Decimal("1000000")

    # Non-cached input tokens
    non_cached_input = input_tokens - cached_tokens
    input_cost = (Decimal(non_cached_input) / million) * pricing.input_price

    # Cached input tokens (if pricing available)
    cached_cost = Decimal("0")
    if cached_tokens > 0 and pricing.cached_input_price:
        cached_cost = (Decimal(cached_tokens) / million) * pricing.cached_input_price

    # Output tokens
    output_cost = (Decimal(output_tokens) / million) * pricing.output_price

    return input_cost + cached_cost + output_cost
