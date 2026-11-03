import re

from django.core.exceptions import ValidationError
from django.db import models
from django.utils.text import slugify


CHROMA_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")


class Catalogue(models.Model):
    VDB_NOT_BUILT = "NotBuilt"
    VDB_BUILDING = "Building"
    VDB_READY = "Ready"
    VDB_FAILED = "Failed"
    VDB_CHOICES = [
        (VDB_NOT_BUILT, "Not Built"),
        (VDB_BUILDING, "Building"),
        (VDB_READY, "Ready"),
        (VDB_FAILED, "Failed"),
    ]

    STATUS_DRAFT = "Draft"
    STATUS_PUBLISHED = "Published"
    STATUS_CHOICES = [(STATUS_DRAFT, "Draft"), (STATUS_PUBLISHED, "Published")]

    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=63, unique=True)
    description = models.TextField(blank=True)
    geography = models.CharField(max_length=120, blank=True, default="All India")
    season_year = models.CharField(max_length=40, blank=True, help_text="e.g. Kharif 2026")

    documents = models.ManyToManyField(
        "documents.Document", blank=True, related_name="catalogues"
    )

    vector_db_status = models.CharField(
        max_length=20, choices=VDB_CHOICES, default=VDB_NOT_BUILT
    )
    vector_db_generated_at = models.DateTimeField(null=True, blank=True)
    vector_db_chunks_count = models.IntegerField(default=0)
    vector_db_error = models.TextField(blank=True)
    embedding_model = models.CharField(max_length=120, blank=True)
    vector_store_collection_name = models.CharField(max_length=64, blank=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_DRAFT)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def clean(self):
        if self.slug and not CHROMA_SLUG_RE.match(self.slug):
            raise ValidationError(
                {"slug": "Slug must be 3–63 chars, ASCII letters/digits/._- and start/end with alphanumeric."}
            )

    def save(self, *args, **kwargs):
        if not self.slug:
            base = slugify(self.name)[:60] or "catalogue"
            slug = base
            i = 1
            while Catalogue.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                i += 1
                slug = f"{base}-{i}"[:63]
            self.slug = slug
        if not self.vector_store_collection_name:
            self.vector_store_collection_name = self.slug
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    @property
    def api_endpoint(self):
        return f"/v1/catalogues/{self.slug}"

    @property
    def documents_count(self):
        return self.documents.count()
