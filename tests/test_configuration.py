"""Tests for configuration management."""

import os
from unittest.mock import patch

import pytest

from django_llm.configuration import Config, configure, get_config


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_timeout(self):
        config = Config()
        assert config.timeout == 120.0

    def test_default_retry_settings(self):
        config = Config()
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 60.0

    def test_api_keys_from_environment(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test-openai",
            "ANTHROPIC_API_KEY": "sk-test-anthropic",
            "GOOGLE_API_KEY": "test-google-key",
        }):
            config = Config()
            assert config.openai_api_key == "sk-test-openai"
            assert config.anthropic_api_key == "sk-test-anthropic"
            assert config.google_api_key == "test-google-key"

    def test_gemini_api_key_fallback(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini"}, clear=True):
            # Clear GOOGLE_API_KEY to test fallback
            os.environ.pop("GOOGLE_API_KEY", None)
            config = Config()
            assert config.google_api_key == "test-gemini"

    def test_ollama_base_url_from_environment(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:11434/v1"}):
            config = Config()
            assert config.ollama_base_url == "http://custom:11434/v1"

    def test_azure_settings_from_environment(self):
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://my-resource.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "azure-key-123",
            "AZURE_OPENAI_API_VERSION": "2024-01-01",
        }):
            config = Config()
            assert config.azure_openai_endpoint == "https://my-resource.openai.azure.com"
            assert config.azure_openai_api_key == "azure-key-123"
            assert config.azure_openai_api_version == "2024-01-01"

    def test_default_azure_api_version(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing env var
            os.environ.pop("AZURE_OPENAI_API_VERSION", None)
            config = Config()
            assert config.azure_openai_api_version == "2024-08-01-preview"


class TestGetDefaultModel:
    """Tests for automatic model selection."""

    def test_openai_default_when_key_present(self):
        config = Config(openai_api_key="sk-test")
        assert config.get_default_model() == "gpt-4o"

    def test_anthropic_default_when_only_anthropic_key(self):
        config = Config(
            openai_api_key=None,
            anthropic_api_key="sk-ant-test",
        )
        assert config.get_default_model() == "claude-sonnet-4-20250514"

    def test_google_default_when_only_google_key(self):
        config = Config(
            openai_api_key=None,
            anthropic_api_key=None,
            google_api_key="test-google",
        )
        assert config.get_default_model() == "gemini-2.0-flash"

    def test_explicit_default_model_overrides(self):
        config = Config(
            openai_api_key="sk-test",
            default_model="gpt-4-turbo",
        )
        assert config.get_default_model() == "gpt-4-turbo"

    def test_no_keys_raises_error(self):
        config = Config(
            openai_api_key=None,
            anthropic_api_key=None,
            google_api_key=None,
        )
        with pytest.raises(ValueError, match="No API keys configured"):
            config.get_default_model()


class TestConfigure:
    """Tests for configure() function."""

    def test_configure_updates_global_config(self):
        original = get_config().timeout
        try:
            configure(timeout=30.0)
            assert get_config().timeout == 30.0
        finally:
            configure(timeout=original)

    def test_configure_returns_config(self):
        result = configure(timeout=60.0)
        assert isinstance(result, Config)
        # Reset
        configure(timeout=120.0)

    def test_configure_unknown_option_raises(self):
        with pytest.raises(ValueError, match="Unknown configuration option"):
            configure(unknown_option="value")

    def test_configure_multiple_options(self):
        original_timeout = get_config().timeout
        original_retries = get_config().max_retries
        try:
            configure(timeout=45.0, max_retries=5)
            config = get_config()
            assert config.timeout == 45.0
            assert config.max_retries == 5
        finally:
            configure(timeout=original_timeout, max_retries=original_retries)


class TestGetConfig:
    """Tests for get_config() function."""

    def test_returns_config_instance(self):
        config = get_config()
        assert isinstance(config, Config)

    def test_returns_same_instance(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
