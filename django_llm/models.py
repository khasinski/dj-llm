"""Django models for persisting LLM conversations."""

import json
from typing import TYPE_CHECKING

from django.conf import settings
from django.db import models
from django.utils import timezone

from django_llm.chat import Chat
from django_llm.message import Message, Role, TokenUsage, ToolCall
from django_llm.tool import Tool

if TYPE_CHECKING:
    pass


class Conversation(models.Model):
    """A persisted conversation with an LLM.

    This model stores the entire conversation history and provides
    a convenient interface for continuing conversations across requests.

    Example:
        # Create a new conversation
        conversation = Conversation.objects.create(
            name="Support Chat",
            model_id="gpt-4o",
            user=request.user,
        )

        # Get the chat interface
        response = conversation.chat.ask("Hello!")

        # Save after each interaction (or use auto_save=True)
        conversation.save()

        # Later, restore the conversation
        conversation = Conversation.objects.get(pk=1)
        response = conversation.chat.ask("What did we discuss?")
    """

    # Thread identification
    name = models.CharField(max_length=255, blank=True, null=True, help_text="Optional name for this conversation")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="llm_conversations",
        help_text="User who owns this conversation",
    )

    # Model configuration
    model_id = models.CharField(max_length=100, blank=True, null=True)
    system_prompt = models.TextField(blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Cached chat instance
    _chat: Chat | None = None
    _auto_save: bool = False
    _tools: list[Tool] = []

    class Meta:
        app_label = "django_llm"
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        if self.name:
            return f"{self.name} (#{self.pk})"
        return f"Conversation {self.pk} ({self.model_id or 'default'})"

    @property
    def chat(self) -> Chat:
        """Get the Chat instance for this conversation.

        The chat instance is cached and automatically synced with
        the database messages.
        """
        if self._chat is None:
            self._chat = self._build_chat()
        return self._chat

    def _build_chat(self) -> Chat:
        """Build a Chat instance from stored messages."""
        messages = [self._db_message_to_message(m) for m in self.stored_messages.order_by("created_at")]

        chat = Chat(
            model=self.model_id,
            system=self.system_prompt,
            messages=messages,
        )

        # Wire up auto-save callback if enabled
        if self._auto_save:
            original_callback = chat._on_new_message
            def on_new_message(msg: Message) -> None:
                if original_callback:
                    original_callback(msg)
                self._persist_message(msg)
            chat._on_new_message = on_new_message

        # Add tools if set
        if self._tools:
            chat.with_tools(self._tools)

        return chat

    def with_auto_save(self, enabled: bool = True) -> "Conversation":
        """Enable or disable auto-saving messages.

        When enabled, messages are automatically persisted after each
        interaction without needing to call save().

        Args:
            enabled: Whether to enable auto-save.

        Returns:
            Self for method chaining.
        """
        self._auto_save = enabled
        self._chat = None  # Reset chat to rewire callbacks
        return self

    def with_tools(self, tools: list[Tool]) -> "Conversation":
        """Add tools to this conversation.

        Args:
            tools: List of Tool instances.

        Returns:
            Self for method chaining.
        """
        self._tools = tools
        if self._chat:
            self._chat.with_tools(tools)
        return self

    def ask(self, content: str, stream: bool = False) -> Message:
        """Send a message and get a response.

        This is a convenience method that delegates to the chat instance.

        Args:
            content: The user's message.
            stream: Whether to stream the response.

        Returns:
            The assistant's response message.
        """
        response = self.chat.ask(content, stream=stream)

        # Persist if not auto-saving
        if not self._auto_save:
            self.sync_messages()

        return response

    def sync_messages(self) -> None:
        """Sync the chat messages to the database.

        Call this after interactions if auto_save is not enabled.
        """
        if self._chat is None:
            return

        # Get existing message count
        existing_count = self.stored_messages.count()

        # Only persist new messages
        for msg in self._chat.messages[existing_count:]:
            self._persist_message(msg)

        self.updated_at = timezone.now()
        self.save(update_fields=["updated_at"])

    def _persist_message(self, msg: Message) -> "StoredMessage":
        """Persist a single message to the database."""
        return StoredMessage.objects.create(
            conversation=self,
            role=msg.role.value,
            content=msg.content,
            model_id=msg.model_id,
            tool_calls=json.dumps([tc.to_dict() for tc in msg.tool_calls]) if msg.tool_calls else None,
            tool_call_id=msg.tool_call_id,
            input_tokens=msg.tokens.input_tokens if msg.tokens else None,
            output_tokens=msg.tokens.output_tokens if msg.tokens else None,
            thinking=msg.thinking,
        )

    def _db_message_to_message(self, stored: "StoredMessage") -> Message:
        """Convert a StoredMessage to a Message."""
        tool_calls = []
        if stored.tool_calls:
            for tc_data in json.loads(stored.tool_calls):
                tool_calls.append(ToolCall.from_dict(tc_data))

        tokens = None
        if stored.input_tokens is not None or stored.output_tokens is not None:
            tokens = TokenUsage(
                input_tokens=stored.input_tokens or 0,
                output_tokens=stored.output_tokens or 0,
            )

        return Message(
            role=Role(stored.role),
            content=stored.content,
            model_id=stored.model_id,
            tool_calls=tool_calls,
            tool_call_id=stored.tool_call_id,
            tokens=tokens,
            thinking=stored.thinking,
        )

    def clear_messages(self) -> None:
        """Delete all messages in this conversation."""
        self.stored_messages.all().delete()
        self._chat = None


class StoredMessage(models.Model):
    """A persisted message in a conversation."""

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="stored_messages",
    )
    role = models.CharField(max_length=20)
    content = models.TextField(blank=True, null=True)
    model_id = models.CharField(max_length=100, blank=True, null=True)
    tool_calls = models.TextField(blank=True, null=True)  # JSON
    tool_call_id = models.CharField(max_length=100, blank=True, null=True)
    input_tokens = models.IntegerField(blank=True, null=True)
    output_tokens = models.IntegerField(blank=True, null=True)
    thinking = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "django_llm"
        ordering = ["created_at"]

    def __str__(self) -> str:
        content_preview = (self.content or "")[:50]
        return f"{self.role}: {content_preview}..."
