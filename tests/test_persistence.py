"""Tests for Django model persistence."""

from unittest.mock import Mock, patch

import pytest

from django_llm.message import Message
from django_llm.models import Conversation, StoredMessage


@pytest.mark.django_db
class TestConversation:
    """Tests for Conversation model."""

    def test_create_conversation(self):
        """Test creating a new conversation."""
        conv = Conversation.objects.create(model_id="gpt-4o")
        assert conv.pk is not None
        assert conv.model_id == "gpt-4o"

    def test_conversation_with_system_prompt(self):
        """Test conversation with system prompt."""
        conv = Conversation.objects.create(
            model_id="gpt-4o",
            system_prompt="You are helpful.",
        )
        assert conv.system_prompt == "You are helpful."

    def test_get_chat_instance(self):
        """Test getting chat instance from conversation."""
        conv = Conversation.objects.create(model_id="gpt-4o")
        chat = conv.chat

        assert chat is not None
        assert chat.model == "gpt-4o"

    def test_chat_instance_cached(self):
        """Test that chat instance is cached."""
        conv = Conversation.objects.create(model_id="gpt-4o")
        chat1 = conv.chat
        chat2 = conv.chat

        assert chat1 is chat2

    def test_conversation_with_metadata(self):
        """Test conversation with metadata."""
        conv = Conversation.objects.create(
            model_id="gpt-4o",
            metadata={"user_id": 123, "tags": ["test"]},
        )

        assert conv.metadata["user_id"] == 123
        assert "test" in conv.metadata["tags"]

    @patch("django_llm.chat.get_provider_for_model")
    def test_ask_persists_messages(self, mock_get_provider):
        """Test that ask persists messages."""
        mock_provider = Mock()
        mock_response = Message.assistant("Response")
        mock_provider.complete.return_value = mock_response
        mock_get_provider.return_value = mock_provider

        conv = Conversation.objects.create(model_id="gpt-4o")
        conv.ask("Hello")
        conv.sync_messages()

        assert conv.stored_messages.count() == 2
        messages = list(conv.stored_messages.all())
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Response"

    @patch("django_llm.chat.get_provider_for_model")
    def test_auto_save_enabled(self, mock_get_provider):
        """Test auto-save mode."""
        mock_provider = Mock()
        mock_response = Message.assistant("Response")
        mock_provider.complete.return_value = mock_response
        mock_get_provider.return_value = mock_provider

        conv = Conversation.objects.create(model_id="gpt-4o")
        conv.with_auto_save(True)
        conv.ask("Hello")

        # Messages should be saved automatically
        assert conv.stored_messages.count() == 2

    def test_restore_conversation(self):
        """Test restoring a conversation from database."""
        # Create and save a conversation with messages
        conv = Conversation.objects.create(model_id="gpt-4o")
        StoredMessage.objects.create(
            conversation=conv,
            role="user",
            content="Hello",
        )
        StoredMessage.objects.create(
            conversation=conv,
            role="assistant",
            content="Hi there!",
        )

        # Reload from database
        conv = Conversation.objects.get(pk=conv.pk)
        chat = conv.chat

        assert len(chat.messages) == 2
        assert chat.messages[0].content == "Hello"
        assert chat.messages[1].content == "Hi there!"

    def test_clear_messages(self):
        """Test clearing all messages."""
        conv = Conversation.objects.create(model_id="gpt-4o")
        StoredMessage.objects.create(
            conversation=conv,
            role="user",
            content="Hello",
        )

        conv.clear_messages()

        assert conv.stored_messages.count() == 0

    def test_conversation_with_name(self):
        """Test conversation with name field."""
        conv = Conversation.objects.create(
            model_id="gpt-4o",
            name="Customer Support Chat",
        )
        assert conv.name == "Customer Support Chat"

    def test_conversation_name_nullable(self):
        """Test that name field is nullable."""
        conv = Conversation.objects.create(model_id="gpt-4o")
        assert conv.name is None

    def test_conversation_with_user(self):
        """Test conversation with user foreign key."""
        from django.contrib.auth import get_user_model

        User = get_user_model()
        user = User.objects.create_user(username="testuser", password="testpass")

        conv = Conversation.objects.create(
            model_id="gpt-4o",
            user=user,
            name="User's Chat",
        )

        assert conv.user == user
        assert conv.user.username == "testuser"

    def test_conversation_user_nullable(self):
        """Test that user field is nullable."""
        conv = Conversation.objects.create(model_id="gpt-4o")
        assert conv.user is None

    def test_user_conversations_related_name(self):
        """Test accessing user's conversations via related name."""
        from django.contrib.auth import get_user_model

        User = get_user_model()
        user = User.objects.create_user(username="testuser2", password="testpass")

        Conversation.objects.create(model_id="gpt-4o", user=user, name="Chat 1")
        Conversation.objects.create(model_id="gpt-4o", user=user, name="Chat 2")

        assert user.llm_conversations.count() == 2


@pytest.mark.django_db
class TestStoredMessage:
    """Tests for StoredMessage model."""

    def test_create_stored_message(self):
        """Test creating a stored message."""
        conv = Conversation.objects.create()
        msg = StoredMessage.objects.create(
            conversation=conv,
            role="user",
            content="Test message",
        )

        assert msg.pk is not None
        assert msg.role == "user"
        assert msg.content == "Test message"

    def test_stored_message_with_tokens(self):
        """Test stored message with token counts."""
        conv = Conversation.objects.create()
        msg = StoredMessage.objects.create(
            conversation=conv,
            role="assistant",
            content="Response",
            input_tokens=10,
            output_tokens=5,
        )

        assert msg.input_tokens == 10
        assert msg.output_tokens == 5

    def test_stored_message_with_tool_calls(self):
        """Test stored message with tool calls."""
        import json

        conv = Conversation.objects.create()
        tool_calls = [
            {"id": "call_1", "name": "get_weather", "arguments": {"location": "NYC"}},
        ]
        msg = StoredMessage.objects.create(
            conversation=conv,
            role="assistant",
            tool_calls=json.dumps(tool_calls),
        )

        loaded = json.loads(msg.tool_calls)
        assert len(loaded) == 1
        assert loaded[0]["name"] == "get_weather"

    def test_message_ordering(self):
        """Test that messages are ordered by creation time."""
        conv = Conversation.objects.create()
        StoredMessage.objects.create(
            conversation=conv,
            role="user",
            content="First",
        )
        StoredMessage.objects.create(
            conversation=conv,
            role="assistant",
            content="Second",
        )

        messages = list(conv.stored_messages.all())
        assert messages[0].content == "First"
        assert messages[1].content == "Second"
