"""Tests for pricing and cost calculation."""

from decimal import Decimal

import pytest

from django_llm.pricing import (
    ALL_PRICING,
    ANTHROPIC_PRICING,
    GOOGLE_PRICING,
    OPENAI_PRICING,
    ModelPricing,
    calculate_cost,
    get_model_pricing,
)


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_pricing_structure(self):
        pricing = ModelPricing(
            input_price=Decimal("2.50"),
            output_price=Decimal("10.00"),
            cached_input_price=Decimal("1.25"),
        )
        assert pricing.input_price == Decimal("2.50")
        assert pricing.output_price == Decimal("10.00")
        assert pricing.cached_input_price == Decimal("1.25")

    def test_cached_price_optional(self):
        pricing = ModelPricing(
            input_price=Decimal("2.50"),
            output_price=Decimal("10.00"),
        )
        assert pricing.cached_input_price is None


class TestGetModelPricing:
    """Tests for model pricing lookup."""

    def test_exact_match_openai(self):
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.input_price == Decimal("2.50")

    def test_exact_match_anthropic(self):
        pricing = get_model_pricing("claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing.input_price == Decimal("3.00")

    def test_exact_match_google(self):
        pricing = get_model_pricing("gemini-2.0-flash")
        assert pricing is not None
        assert pricing.input_price == Decimal("0.10")

    def test_prefix_match(self):
        # Should match gpt-4o even with suffix
        pricing = get_model_pricing("gpt-4o-2024-11-20")
        assert pricing is not None

    def test_unknown_model_returns_none(self):
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None

    def test_case_insensitive_prefix_match(self):
        pricing = get_model_pricing("GPT-4O")
        assert pricing is not None


class TestCalculateCost:
    """Tests for cost calculation."""

    def test_basic_cost_calculation(self):
        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost = calculate_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=0,
        )
        assert cost is not None
        # 1000 input tokens = $0.0025
        # 500 output tokens = $0.005
        # Total = $0.0075
        assert cost == Decimal("0.0075")

    def test_cost_with_cached_tokens(self):
        # gpt-4o: $2.50/1M input, $1.25/1M cached
        cost = calculate_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=0,
            cached_tokens=500,
        )
        assert cost is not None
        # 500 non-cached input = $0.00125
        # 500 cached input = $0.000625
        # Total = $0.001875
        assert cost == Decimal("0.001875")

    def test_unknown_model_returns_none(self):
        cost = calculate_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost is None

    def test_zero_tokens(self):
        cost = calculate_cost(
            model="gpt-4o",
            input_tokens=0,
            output_tokens=0,
        )
        assert cost == Decimal("0")

    def test_large_token_count(self):
        # 1 million tokens
        cost = calculate_cost(
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert cost is not None
        # 1M input = $2.50, 1M output = $10.00
        assert cost == Decimal("12.50")


class TestPricingData:
    """Tests for pricing data completeness."""

    def test_openai_models_have_pricing(self):
        expected_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini"]
        for model in expected_models:
            assert model in OPENAI_PRICING, f"Missing pricing for {model}"

    def test_anthropic_models_have_pricing(self):
        expected_models = [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
        for model in expected_models:
            assert model in ANTHROPIC_PRICING, f"Missing pricing for {model}"

    def test_google_models_have_pricing(self):
        expected_models = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        for model in expected_models:
            assert model in GOOGLE_PRICING, f"Missing pricing for {model}"

    def test_all_pricing_combined(self):
        # ALL_PRICING should contain all providers
        assert len(ALL_PRICING) == len(OPENAI_PRICING) + len(ANTHROPIC_PRICING) + len(GOOGLE_PRICING)

    def test_prices_are_positive(self):
        for model, pricing in ALL_PRICING.items():
            assert pricing.input_price >= 0, f"{model} has negative input price"
            assert pricing.output_price >= 0, f"{model} has negative output price"
            if pricing.cached_input_price is not None:
                assert pricing.cached_input_price >= 0, f"{model} has negative cached price"
