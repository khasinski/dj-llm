"""Add name and user fields to Conversation model."""

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    """Add name and user fields to support thread management."""

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("django_llm", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversation",
            name="name",
            field=models.CharField(
                blank=True,
                help_text="Optional name for this conversation",
                max_length=255,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="conversation",
            name="user",
            field=models.ForeignKey(
                blank=True,
                help_text="User who owns this conversation",
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="llm_conversations",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
