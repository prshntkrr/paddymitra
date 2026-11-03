import secrets
import uuid

from django.conf import settings
from django.db import models


def _new_token() -> str:
    return secrets.token_urlsafe(32)


class GeneratedAPI(models.Model):
    """
    A user-generated public API endpoint that runs the RAG pipeline against
    a fixed catalogue/language/persona. Callers POST to /api/v1/q/<public_id>/
    with an Authorization: Token <token> header and a JSON or multipart body.
    """

    USER_TYPE_FARMER = "farmer"
    USER_TYPE_RESEARCHER = "researcher"
    USER_TYPE_CHOICES = [
        (USER_TYPE_FARMER, "Farmer (short, plain-language)"),
        (USER_TYPE_RESEARCHER, "Researcher (detailed, technical)"),
    ]

    LANGUAGE_CHOICES = [
        ("English", "English"),
        ("Hindi", "Hindi"),
        ("Odia", "Odia"),
        ("Bengali", "Bengali"),
        ("Telugu", "Telugu"),
        ("Tamil", "Tamil"),
    ]

    public_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    token = models.CharField(max_length=64, default=_new_token, unique=True, db_index=True)

    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)

    catalogue = models.ForeignKey(
        "catalogues.Catalogue",
        on_delete=models.PROTECT,
        related_name="generated_apis",
    )
    language = models.CharField(max_length=20, choices=LANGUAGE_CHOICES, default="English")
    user_type = models.CharField(max_length=20, choices=USER_TYPE_CHOICES, default=USER_TYPE_FARMER)

    sample_query = models.TextField(blank=True)
    source_query = models.ForeignKey(
        "queries.Query", null=True, blank=True,
        on_delete=models.SET_NULL, related_name="generated_apis",
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="generated_apis",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    is_active = models.BooleanField(default=True)
    request_count = models.IntegerField(default=0)
    last_called_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Generated API"
        verbose_name_plural = "Generated APIs"

    def __str__(self):
        return f"{self.name} ({self.public_id})"

    @property
    def relative_url(self) -> str:
        return f"/api/v1/q/{self.public_id}/"
