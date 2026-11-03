from django.db import models


class ImageGroup(models.Model):
    """
    A visual group within a catalogue: a shared filename prefix
    (e.g. 'bacterial_leaf_blight') plus the visual description that applies
    to every image in the group. Mirrors AgriModel's <prefix>.txt sidecar
    convention but scoped to a catalogue.
    """

    catalogue = models.ForeignKey(
        "catalogues.Catalogue",
        on_delete=models.CASCADE,
        related_name="image_groups",
    )
    prefix = models.CharField(max_length=120, db_index=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["catalogue", "prefix"]
        unique_together = (("catalogue", "prefix"),)

    def __str__(self):
        return f"{self.prefix} ({self.catalogue.slug})"

    @property
    def image_count(self):
        return self.images.count()

    @property
    def cover_image(self):
        return self.images.first()


class KnowledgeImage(models.Model):
    """One image inside an ImageGroup."""

    group = models.ForeignKey(
        ImageGroup,
        on_delete=models.CASCADE,
        related_name="images",
        null=True, blank=True,
    )
    catalogue = models.ForeignKey(
        "catalogues.Catalogue",
        on_delete=models.CASCADE,
        related_name="knowledge_images",
        null=True, blank=True,
    )
    # Kept for backwards compatibility; mirrors group.prefix.
    prefix = models.CharField(max_length=120, db_index=True, blank=True)
    image = models.ImageField(upload_to="image_bank/")
    original_filename = models.CharField(max_length=255, blank=True)
    # Kept for backwards compatibility; mirrors group.description at index time.
    description = models.TextField(blank=True)
    embedding_dim = models.IntegerField(default=0)
    indexed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["prefix", "id"]

    def __str__(self):
        return f"{self.prefix or '?'} #{self.pk}"
