from django.contrib import admin

from .models import ImageGroup, KnowledgeImage


@admin.register(ImageGroup)
class ImageGroupAdmin(admin.ModelAdmin):
    list_display = ("id", "prefix", "catalogue", "image_count", "updated_at")
    list_filter = ("catalogue",)
    search_fields = ("prefix", "description")


@admin.register(KnowledgeImage)
class KnowledgeImageAdmin(admin.ModelAdmin):
    list_display = ("id", "prefix", "catalogue", "group", "indexed", "created_at")
    list_filter = ("catalogue", "indexed")
    search_fields = ("prefix", "original_filename", "description")
