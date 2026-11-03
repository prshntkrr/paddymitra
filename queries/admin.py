from django.contrib import admin

from .models import FAQ, ChatQuery, Query


@admin.register(FAQ)
class FAQAdmin(admin.ModelAdmin):
    list_display = ("id", "question_en", "is_active", "catalogue", "hit_count",
                    "source_query", "updated_at")
    list_filter = ("is_active", "language", "catalogue")
    search_fields = ("question_en", "question_local", "answer_en", "answer_local")
    # Derived from question_en by services/faq.save_entry — editing them here
    # would only put the match keys out of step with the question.
    readonly_fields = ("question_norm", "embedding_model", "hit_count",
                       "last_used_at", "created_at", "updated_at")
    exclude = ("question_embedding",)

    def save_model(self, request, obj, form, change):
        """Go through the service so the match keys are rebuilt with the entry."""
        from .services.faq import save_entry
        if not change and obj.created_by_id is None:
            obj.created_by = request.user
        save_entry(obj)


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    list_display = ("code", "timestamp", "district", "category", "farmer_feedback", "status")
    list_filter = ("status", "category", "farmer_feedback", "season", "crop_stage")
    search_fields = ("translated_query_text", "original_query_text", "district", "problem_entity")
    readonly_fields = ("timestamp",)


@admin.register(ChatQuery)
class ChatQueryAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "language", "feedback", "feedback_local", "question", "response_time_seconds")
    list_filter = ("feedback", "language")
    search_fields = ("question", "response", "feedback_local")
    readonly_fields = ("timestamp",)
