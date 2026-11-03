from django.contrib import admin

from .models import Category, Document, Subcategory


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)


@admin.register(Subcategory)
class SubcategoryAdmin(admin.ModelAdmin):
    list_display = ("name", "category")
    list_filter = ("category",)
    search_fields = ("name",)


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "crop", "content_type", "doc_type", "status", "uploaded_at")
    list_filter = ("status", "crop", "content_type", "doc_type", "category")
    search_fields = ("title", "summary", "authors", "organizations")
    autocomplete_fields = ("category", "subcategory", "keywords")
    readonly_fields = ("title_key", "uploaded_at", "updated_at")
