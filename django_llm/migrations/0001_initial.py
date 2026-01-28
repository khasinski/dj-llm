"""Initial migration for django_llm models."""

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    """Create Conversation and StoredMessage models."""

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Conversation",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("model_id", models.CharField(blank=True, max_length=100, null=True)),
                ("system_prompt", models.TextField(blank=True, null=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["-updated_at"],
            },
        ),
        migrations.CreateModel(
            name="StoredMessage",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("role", models.CharField(max_length=20)),
                ("content", models.TextField(blank=True, null=True)),
                ("model_id", models.CharField(blank=True, max_length=100, null=True)),
                ("tool_calls", models.TextField(blank=True, null=True)),
                ("tool_call_id", models.CharField(blank=True, max_length=100, null=True)),
                ("input_tokens", models.IntegerField(blank=True, null=True)),
                ("output_tokens", models.IntegerField(blank=True, null=True)),
                ("thinking", models.TextField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "conversation",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="stored_messages",
                        to="django_llm.conversation",
                    ),
                ),
            ],
            options={
                "ordering": ["created_at"],
            },
        ),
    ]
