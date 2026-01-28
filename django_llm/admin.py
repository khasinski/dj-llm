"""Django admin configuration for django-llm models."""

from django.contrib import admin
from django.utils.html import format_html

from django_llm.models import Conversation, StoredMessage


class StoredMessageInline(admin.TabularInline):
    """Inline display of messages within a conversation."""

    model = StoredMessage
    extra = 0
    readonly_fields = (
        "role",
        "content_preview",
        "model_id",
        "input_tokens",
        "output_tokens",
        "created_at",
    )
    fields = ("role", "content_preview", "model_id", "input_tokens", "output_tokens", "created_at")
    ordering = ("created_at",)
    can_delete = False

    def content_preview(self, obj):
        """Show truncated content."""
        if obj.content:
            preview = obj.content[:100]
            if len(obj.content) > 100:
                preview += "..."
            return preview
        return "-"

    content_preview.short_description = "Content"

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    """Admin interface for Conversation model."""

    list_display = (
        "id",
        "name",
        "user",
        "model_id",
        "message_count",
        "total_tokens",
        "created_at",
    )
    list_filter = ("model_id", "created_at")
    search_fields = ("name", "user__username", "user__email", "system_prompt")
    readonly_fields = ("created_at", "updated_at", "message_count", "total_tokens")
    raw_id_fields = ("user",)
    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    fieldsets = (
        (None, {"fields": ("name", "user", "model_id")}),
        ("System Prompt", {"fields": ("system_prompt",), "classes": ("collapse",)}),
        ("Metadata", {"fields": ("metadata",), "classes": ("collapse",)}),
        (
            "Statistics",
            {
                "fields": ("message_count", "total_tokens", "created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    inlines = [StoredMessageInline]

    def message_count(self, obj):
        """Return the number of messages in this conversation."""
        return obj.stored_messages.count()

    message_count.short_description = "Messages"

    def total_tokens(self, obj):
        """Return total tokens used in this conversation."""
        from django.db.models import Sum

        totals = obj.stored_messages.aggregate(
            input=Sum("input_tokens"),
            output=Sum("output_tokens"),
        )
        input_tokens = totals["input"] or 0
        output_tokens = totals["output"] or 0
        total = input_tokens + output_tokens
        if total > 0:
            return format_html(
                '<span title="Input: {}, Output: {}">{:,}</span>',
                input_tokens,
                output_tokens,
                total,
            )
        return "-"

    total_tokens.short_description = "Tokens"


@admin.register(StoredMessage)
class StoredMessageAdmin(admin.ModelAdmin):
    """Admin interface for StoredMessage model."""

    list_display = (
        "id",
        "conversation_link",
        "role",
        "content_preview",
        "model_id",
        "tokens_display",
        "created_at",
    )
    list_filter = ("role", "model_id", "created_at")
    search_fields = ("content", "conversation__name")
    readonly_fields = ("created_at",)
    raw_id_fields = ("conversation",)
    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    fieldsets = (
        (None, {"fields": ("conversation", "role", "content")}),
        ("Model Info", {"fields": ("model_id", "tool_call_id", "tool_calls")}),
        (
            "Token Usage",
            {"fields": ("input_tokens", "output_tokens"), "classes": ("collapse",)},
        ),
        ("Thinking", {"fields": ("thinking",), "classes": ("collapse",)}),
        ("Timestamps", {"fields": ("created_at",)}),
    )

    def conversation_link(self, obj):
        """Link to parent conversation."""
        from django.urls import reverse

        url = reverse("admin:django_llm_conversation_change", args=[obj.conversation_id])
        return format_html('<a href="{}">{}</a>', url, obj.conversation)

    conversation_link.short_description = "Conversation"

    def content_preview(self, obj):
        """Show truncated content."""
        if obj.content:
            preview = obj.content[:80]
            if len(obj.content) > 80:
                preview += "..."
            return preview
        return "-"

    content_preview.short_description = "Content"

    def tokens_display(self, obj):
        """Display token counts."""
        if obj.input_tokens or obj.output_tokens:
            return f"{obj.input_tokens or 0} / {obj.output_tokens or 0}"
        return "-"

    tokens_display.short_description = "In/Out Tokens"
