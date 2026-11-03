from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "documents"

    def ready(self):
        # Register the post_save signal that auto-updates the Odia glossary.
        from . import signals  # noqa: F401
