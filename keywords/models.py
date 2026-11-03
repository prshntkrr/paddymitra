from django.db import models


class Keyword(models.Model):
    PARENT_CROP = "Crop"
    PARENT_DISEASE = "Disease"
    PARENT_SEASON = "Season"
    PARENT_REGION = "Region"
    PARENT_PRACTICE = "Practice"
    PARENT_FERTILISER = "Fertiliser"
    PARENT_OTHER = "Other"
    PARENT_CHOICES = [
        (PARENT_CROP, "Crop"),
        (PARENT_DISEASE, "Disease"),
        (PARENT_SEASON, "Season"),
        (PARENT_REGION, "Region"),
        (PARENT_PRACTICE, "Practice"),
        (PARENT_FERTILISER, "Fertiliser"),
        (PARENT_OTHER, "Other"),
    ]

    LANGUAGE_CHOICES = [
        ("Odia", "Odia"),
        ("Hindi", "Hindi"),
        ("Bengali", "Bengali"),
        ("Telugu", "Telugu"),
        ("Tamil", "Tamil"),
        ("Kannada", "Kannada"),
        ("Marathi", "Marathi"),
        ("Gujarati", "Gujarati"),
        ("Punjabi", "Punjabi"),
        ("English", "English"),
    ]

    REGION_CHOICES = [
        ("All India", "All India"),
        ("Odisha", "Odisha"),
        ("Bihar", "Bihar"),
        ("West Bengal", "West Bengal"),
        ("Uttar Pradesh", "Uttar Pradesh"),
        ("Jharkhand", "Jharkhand"),
        ("Chhattisgarh", "Chhattisgarh"),
        ("Andhra Pradesh", "Andhra Pradesh"),
    ]

    VALIDATION_SOURCE_CHOICES = [
        ("IRRI", "IRRI"),
        ("Govt", "Govt"),
        ("Research Paper", "Research Paper"),
        ("Other", "Other"),
    ]

    STATUS_PUBLISHED = "Published"
    STATUS_DRAFT = "Draft"
    STATUS_DISABLED = "Disabled"
    STATUS_CHOICES = [
        (STATUS_PUBLISHED, "Published"),
        (STATUS_DRAFT, "Draft"),
        (STATUS_DISABLED, "Disabled"),
    ]

    keyword = models.CharField(max_length=255)
    keyword_key = models.CharField(max_length=255, unique=True, db_index=True, editable=False)

    parent_category = models.CharField(max_length=40, choices=PARENT_CHOICES, default=PARENT_CROP)
    subcategory = models.CharField(max_length=255, blank=True)
    region = models.CharField(max_length=80, choices=REGION_CHOICES, default="All India")

    translation = models.CharField(max_length=255, blank=True)
    language = models.CharField(max_length=30, choices=LANGUAGE_CHOICES, default="Odia")
    local_names = models.TextField(
        blank=True, help_text="Comma-separated synonyms / local names / spellings."
    )

    expert_validated = models.BooleanField(default=False)
    validation_source = models.CharField(
        max_length=40, choices=VALIDATION_SOURCE_CHOICES, blank=True
    )
    image = models.ImageField(upload_to="keywords/", blank=True, null=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PUBLISHED)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["keyword"]

    def save(self, *args, **kwargs):
        self.keyword = (self.keyword or "").strip()
        self.keyword_key = self.keyword.lower()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.keyword

    @property
    def kw_code(self):
        return f"KW-{self.pk:03d}"

    @property
    def linked_docs_count(self):
        return self.documents.count()
