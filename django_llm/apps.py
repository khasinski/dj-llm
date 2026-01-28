"""Django app configuration for django_llm."""

from django.apps import AppConfig


class DjangoLLMConfig(AppConfig):
    """Django app configuration for the LLM integration."""

    name = "django_llm"
    verbose_name = "Django LLM"
    default_auto_field = "django.db.models.BigAutoField"
