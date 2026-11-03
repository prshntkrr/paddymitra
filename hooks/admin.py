from django.contrib import admin, messages

from .models import Hook, InboundMessage, WhatsAppApiKey


@admin.register(Hook)
class HookAdmin(admin.ModelAdmin):
    list_display = ("name", "trigger_keyword", "gateway_provider", "catalogue", "status")
    list_filter = ("gateway_provider", "status", "primary_language")
    search_fields = ("name", "trigger_keyword", "whatsapp_number")


@admin.register(WhatsAppApiKey)
class WhatsAppApiKeyAdmin(admin.ModelAdmin):
    """Keys, with the plaintext deliberately absent.

    Only a hash is stored, so nothing here can reveal a key — including to us.
    A key that has been lost is rotated, not recovered, and that is the whole
    reason rotation is one click.
    """
    list_display = ("masked", "hook", "state", "name", "request_count",
                    "last_used_at", "created_at")
    list_filter = ("hook", "created_at")
    search_fields = ("prefix", "name", "hook__trigger_keyword")
    readonly_fields = ("prefix", "key_hash", "created_at", "last_used_at",
                       "request_count", "rotated_from", "masked", "state")
    actions = ("rotate_keys", "revoke_keys")

    @admin.display(description="key")
    def masked(self, obj):
        return obj.masked

    @admin.display(description="state")
    def state(self, obj):
        return obj.state

    def has_add_permission(self, request):
        # Adding through the form would store a key nobody ever sees the
        # plaintext of. Issuing is done where the plaintext can be shown once:
        #   python manage.py whatsapp_key --create --hook DHAN
        return False

    @admin.action(description="Rotate (new key now, old one valid 24h)")
    def rotate_keys(self, request, queryset):
        for key in queryset:
            new, plaintext = key.rotate()
            # Shown once, in the message frame — never stored anywhere readable.
            self.message_user(
                request,
                f"{key.hook.trigger_keyword}: new key {plaintext} — copy it now, it "
                f"cannot be shown again. The previous key works until "
                f"{key.expires_at:%Y-%m-%d %H:%M}.",
                level=messages.WARNING)

    @admin.action(description="Revoke immediately")
    def revoke_keys(self, request, queryset):
        for key in queryset:
            key.revoke()
        self.message_user(request, f"revoked {queryset.count()} key(s).")


@admin.register(InboundMessage)
class InboundMessageAdmin(admin.ModelAdmin):
    """Every message the gateway sent, with the payload it sent.

    The only place an unanswered message survives, so it is searchable by number
    and readable in full — that is what it is for.
    """
    list_display = ("created_at", "provider", "from_number", "status",
                    "message_type", "hook", "query", "attempts")
    list_filter = ("status", "provider", "message_type", "hook")
    search_fields = ("message_id", "from_number", "text", "error")
    readonly_fields = ("provider", "message_id", "hook", "query", "from_number",
                       "message_type", "text", "media_url", "raw_payload",
                       "created_at", "processed_at", "attempts", "error")
    date_hierarchy = "created_at"
