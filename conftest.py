"""Pytest configuration for django_llm tests."""

import os
import sys
from pathlib import Path

import django
import pytest

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Configure Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "example_project.settings")
django.setup()


@pytest.fixture(scope="session")
def django_db_setup(django_db_blocker):
    """Set up the test database with migrations."""
    from django.core.management import call_command

    with django_db_blocker.unblock():
        call_command("migrate", "--run-syncdb", verbosity=0)


@pytest.fixture
def vcr_config():
    """VCR configuration for recording API responses."""
    return {
        "cassette_library_dir": "tests/cassettes",
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path"],
        "filter_headers": [
            "authorization",
            "x-api-key",
            "api-key",
        ],
        "filter_query_parameters": ["key"],
        "decode_compressed_response": True,
    }
