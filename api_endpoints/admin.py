from django.contrib import admin

from .models import GeneratedAPI


@admin.register(GeneratedAPI)
class GeneratedAPIAdmin(admin.ModelAdmin):
    list_display = ("name", "public_id", "catalogue", "user_type", "language",
                    "request_count", "is_active", "created_at")
    list_filter = ("is_active", "user_type", "language", "catalogue")
    search_fields = ("name", "public_id", "description")
    readonly_fields = ("public_id", "token", "request_count", "last_called_at",
                       "created_at", "updated_at")
