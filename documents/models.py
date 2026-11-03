from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.text import slugify


def _key(value: str) -> str:
    return (value or "").strip().lower()


class Category(models.Model):
    name = models.CharField(max_length=255)
    name_key = models.CharField(max_length=255, unique=True, db_index=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "Categories"

    def save(self, *args, **kwargs):
        self.name = (self.name or "").strip()
        self.name_key = _key(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name


class Subcategory(models.Model):
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name="subcategories")
    name = models.CharField(max_length=255)
    name_key = models.CharField(max_length=255, db_index=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["category__name", "name"]
        unique_together = (("category", "name_key"),)
        verbose_name_plural = "Subcategories"

    def save(self, *args, **kwargs):
        self.name = (self.name or "").strip()
        self.name_key = _key(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.category} / {self.name}"


class Document(models.Model):
    CROP_CHOICES = [
        ("Rice", "Rice"),
        ("Wheat", "Wheat"),
        ("Maize", "Maize"),
        ("Pulses", "Pulses"),
        ("Oilseeds", "Oilseeds"),
        ("Vegetables", "Vegetables"),
        ("Other", "Other"),
    ]
    CONTENT_TYPE_CHOICES = [
        ("Advisory", "Advisory"),
        ("Research Paper", "Research Paper"),
        ("SOP / Practice", "SOP / Practice"),
        ("FAQ", "FAQ"),
        ("Policy", "Policy"),
    ]
    DOC_TYPE_CHOICES = [
        ("Factsheet", "Factsheet"),
        ("Research Article", "Research Article"),
        ("Bulletin", "Bulletin"),
        ("Training Manual", "Training Manual"),
        ("Technical Manual", "Technical Manual"),
        ("Booklet", "Booklet"),
        ("PPT", "PPT"),
        ("Other", "Other"),
    ]
    STATUS_DRAFT = "Draft"
    STATUS_PENDING = "PendingReview"
    STATUS_PUBLISHED = "Published"
    STATUS_UNPUBLISHED = "Unpublished"
    STATUS_CHOICES = [
        (STATUS_DRAFT, "Draft"),
        (STATUS_PENDING, "Pending Review"),
        (STATUS_PUBLISHED, "Published"),
        (STATUS_UNPUBLISHED, "Unpublished"),
    ]

    title = models.CharField(max_length=500)
    title_key = models.CharField(max_length=510, unique=True, db_index=True, editable=False)
    file = models.FileField(upload_to="documents/", blank=True, null=True)
    file_path = models.CharField(
        max_length=1000, blank=True,
        help_text="Original filename/path from CSV if file not uploaded",
    )
    source_url = models.URLField(max_length=1000, blank=True)

    crop = models.CharField(max_length=40, choices=CROP_CHOICES, default="Rice")
    content_type = models.CharField(max_length=40, choices=CONTENT_TYPE_CHOICES, default="Advisory")
    doc_type = models.CharField(max_length=60, choices=DOC_TYPE_CHOICES, default="Factsheet")

    category = models.ForeignKey(
        Category, null=True, blank=True, on_delete=models.SET_NULL, related_name="documents"
    )
    subcategory = models.ForeignKey(
        Subcategory, null=True, blank=True, on_delete=models.SET_NULL, related_name="documents"
    )
    keywords = models.ManyToManyField(
        "keywords.Keyword", blank=True, related_name="documents"
    )

    geography = models.CharField(max_length=120, blank=True, default="All India")
    year = models.IntegerField(null=True, blank=True)
    source = models.CharField(max_length=255, blank=True)
    summary = models.TextField(blank=True)

    # CSV-specific metadata
    authors = models.TextField(blank=True)
    organizations = models.TextField(blank=True)
    journal_or_book = models.CharField(max_length=500, blank=True)
    countries = models.CharField(max_length=255, blank=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_DRAFT)
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True, blank=True, on_delete=models.SET_NULL, related_name="uploaded_documents",
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_on = models.DateTimeField(null=True, blank=True)

    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True, blank=True, on_delete=models.SET_NULL, related_name="reviewed_documents",
    )
    review_notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def save(self, *args, **kwargs):
        self.title = (self.title or "").strip()
        base_key = slugify(self.title)[:500] or "document"
        # ensure uniqueness when editing
        if not self.title_key:
            key = base_key
            i = 1
            while Document.objects.filter(title_key=key).exclude(pk=self.pk).exists():
                i += 1
                key = f"{base_key}-{i}"
            self.title_key = key
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title

    @property
    def doc_code(self):
        return f"DOC-{self.pk:04d}"

    @property
    def status_display_short(self):
        return "Pending Review" if self.status == self.STATUS_PENDING else self.status

    def publish(self, by=None):
        self.status = self.STATUS_PUBLISHED
        self.published_on = timezone.now()
        if by:
            self.reviewed_by = by
        self.save()

    def unpublish(self):
        self.status = self.STATUS_UNPUBLISHED
        self.save()

    def text_for_embedding(self) -> str:
        """Fallback body used when no real PDF text exists."""
        parts = [
            self.title,
            self.authors,
            self.journal_or_book,
            self.summary,
            ", ".join(k.keyword for k in self.keywords.all()),
            self.category.name if self.category else "",
            self.subcategory.name if self.subcategory else "",
            self.countries,
            self.geography,
            self.crop,
            self.content_type,
        ]
        return "\n\n".join(p for p in parts if p)
