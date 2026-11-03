from django.contrib import admin

from .models import Catalogue


@admin.register(Catalogue)
class CatalogueAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "status", "vector_db_status", "vector_db_chunks_count")
    list_filter = ("status", "vector_db_status")
    search_fields = ("name", "slug", "description")
    filter_horizontal = ("documents",)
    prepopulated_fields = {"slug": ("name",)}
