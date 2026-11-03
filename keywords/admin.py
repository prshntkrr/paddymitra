from django.contrib import admin

from .models import Keyword


@admin.register(Keyword)
class KeywordAdmin(admin.ModelAdmin):
    list_display = ("keyword", "parent_category", "subcategory", "region", "language", "status")
    list_filter = ("parent_category", "region", "language", "status", "expert_validated")
    search_fields = ("keyword", "local_names", "translation")
    readonly_fields = ("keyword_key", "created_at", "updated_at")
