import hashlib
import hmac
import secrets
from datetime import timedelta

from django.conf import settings
from django.db import models
from django.utils import timezone


class Hook(models.Model):
    GATEWAY_PICKY = "Picky Assist"
    GATEWAY_META = "Meta WhatsApp"
    GATEWAY_OTHER = "Other"
    GATEWAY_CHOICES = [
        (GATEWAY_PICKY, "Picky Assist"),
        (GATEWAY_META, "Meta WhatsApp"),
        (GATEWAY_OTHER, "Other"),
    ]

    LANGUAGE_CHOICES = [
        ("Odia", "Odia"),
        ("Hindi", "Hindi"),
        ("Bengali", "Bengali"),
        ("Telugu", "Telugu"),
        ("Tamil", "Tamil"),
        ("English", "English"),
    ]

    STATUS_ACTIVE = "Active"
    STATUS_PAUSED = "Paused"
    STATUS_DISABLED = "Disabled"
    STATUS_CHOICES = [
        (STATUS_ACTIVE, "Active"),
        (STATUS_PAUSED, "Paused"),
        (STATUS_DISABLED, "Disabled"),
    ]

    name = models.CharField(max_length=200)
    whatsapp_number = models.CharField(max_length=30)
    gateway_provider = models.CharField(
        max_length=40, choices=GATEWAY_CHOICES, default=GATEWAY_PICKY
    )
    trigger_keyword = models.CharField(max_length=40)
    catalogue = models.ForeignKey(
        "catalogues.Catalogue", on_delete=models.PROTECT, related_name="hooks"
    )
    primary_language = models.CharField(
        max_length=20, choices=LANGUAGE_CHOICES, default="Odia"
    )
    secondary_language = models.CharField(
        max_length=20, choices=LANGUAGE_CHOICES, default="English", blank=True
    )

    stt_provider = models.CharField(max_length=50, default="OpenAI Whisper")
    tts_provider = models.CharField(max_length=50, default="OpenAI TTS")
    translation_model = models.CharField(max_length=80, default="gpt-4o-mini")
    rag_model = models.CharField(max_length=80, default="gpt-4o-mini")

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_ACTIVE)
    welcome_message = models.TextField(blank=True)
    webhook_url = models.CharField(max_length=500, blank=True)

    messages_processed_count = models.IntegerField(default=0)
    avg_response_time = models.FloatField(default=0.0)
    success_rate = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        unique_together = (("gateway_provider", "trigger_keyword"),)

    def save(self, *args, **kwargs):
        self.trigger_keyword = (self.trigger_keyword or "").upper().strip()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.trigger_keyword})"

    @property
    def api_endpoint(self):
        """The URL the gateway actually posts to.

        One path serves every hook: the keyword in the message (or in the
        payload) is what selects this one. It used to advertise
        "/v1/hooks/<keyword>", which was never routed — anyone who configured a
        gateway from this screen got 404s.
        """
        return "/api/v1/whatsapp/pickyassist/"


class WhatsAppApiKey(models.Model):
    """An API key that authenticates a gateway AND says which hook answers.

    WHY THIS EXISTS ALONGSIDE PICKY_ASSIST_WEBHOOK_SECRET
    -----------------------------------------------------
    The env secret is one value for the whole deployment. That makes three things
    awkward and this makes all three easy:

      * ROTATION means editing .env and restarting, with every gateway broken in
        between. A key here rotates in one command, and the old key keeps working
        for a grace period so there is no window where messages are refused.
      * TWO GATEWAYS (or a staging one) cannot be told apart, so a leak from
        either is a leak of everything. One key per integration, revoked alone.
      * ROUTING had to be guessed from a trigger keyword in the message. The key
        already identifies the hook, so no keyword is needed and "no active hook
        matched" stops being a failure mode.

    THE KEY IS NOT STORED. Only its prefix (to find the row) and a SHA-256 of its
    secret half (to check it). A database dump therefore contains nothing that can
    send or receive a message, and the plaintext is shown exactly once, when it is
    issued — which is also why `rotate()` exists rather than a "show me the key
    again" button.
    """

    PREFIX_BYTES = 6            # -> 12 hex chars, the public half
    SECRET_BYTES = 24           # -> 32 url-safe chars, the half that is hashed
    LABEL = "pmw"               # PaddyMitra WhatsApp, so a leaked key is greppable
    DEFAULT_GRACE_HOURS = 24

    # NULL means a GENERIC key: it authenticates the provider-independent
    # /api/v1/whatsapp/message/ endpoint, which has no hook and needs none. The
    # hashing, rotation, revocation and usage tracking are identical for both, so
    # they share this model rather than duplicating a second credential system —
    # what differs is only what the key is allowed to reach, and `scope` says so.
    hook = models.ForeignKey(Hook, on_delete=models.CASCADE, null=True, blank=True,
                             related_name="api_keys")
    name = models.CharField(max_length=120, blank=True,
                            help_text="Which integration uses this key.")

    prefix = models.CharField(max_length=32, unique=True, db_index=True)
    key_hash = models.CharField(max_length=64)

    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="whatsapp_api_keys",
    )
    # Set when this key is superseded: it keeps working until then.
    expires_at = models.DateTimeField(null=True, blank=True)
    revoked_at = models.DateTimeField(null=True, blank=True)
    rotated_from = models.OneToOneField(
        "self", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="rotated_to",
    )

    last_used_at = models.DateTimeField(null=True, blank=True)
    request_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "WhatsApp API key"

    SCOPE_GENERIC = "generic"

    def __str__(self):
        return f"{self.LABEL}_{self.prefix}… ({self.scope}, {self.state})"

    @property
    def scope(self) -> str:
        """"generic", or the keyword of the hook this key is bound to."""
        return self.hook.trigger_keyword if self.hook_id else self.SCOPE_GENERIC

    @property
    def is_generic(self) -> bool:
        return self.hook_id is None

    # ── issuing ───────────────────────────────────────────────────────────
    @classmethod
    def issue(cls, hook=None, name: str = "", created_by=None) -> tuple:
        """(key_row, plaintext). The plaintext is never recoverable after this."""
        prefix = secrets.token_hex(cls.PREFIX_BYTES)
        secret = secrets.token_urlsafe(cls.SECRET_BYTES)
        row = cls.objects.create(
            hook=hook, name=name, prefix=prefix,
            key_hash=cls._hash(secret), created_by=created_by,
        )
        return row, f"{cls.LABEL}_{prefix}_{secret}"

    def rotate(self, grace_hours: int = None, name: str = "") -> tuple:
        """Issue a replacement and start this key's countdown.

        The old key stays valid for the grace period ON PURPOSE. Rotating a
        webhook credential means editing the configuration at the gateway, and
        anything the farmer sends between the two edits would otherwise be
        refused — a silent gap that looks exactly like an outage.
        """
        grace = self.DEFAULT_GRACE_HOURS if grace_hours is None else grace_hours
        new, plaintext = self.issue(self.hook, name or self.name, self.created_by)
        new.rotated_from = self
        new.save(update_fields=["rotated_from"])
        self.expires_at = timezone.now() + timedelta(hours=max(0, grace))
        self.save(update_fields=["expires_at"])
        return new, plaintext

    def revoke(self) -> None:
        """Kill this key now. Nothing about it can be used again."""
        self.revoked_at = timezone.now()
        self.save(update_fields=["revoked_at"])

    # ── state ─────────────────────────────────────────────────────────────
    @staticmethod
    def _hash(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    @property
    def state(self) -> str:
        now = timezone.now()
        if self.revoked_at:
            return "revoked"
        if self.expires_at and self.expires_at <= now:
            return "expired"
        if self.expires_at:
            return "expiring"
        return "active"

    @property
    def is_usable(self) -> bool:
        return self.state in ("active", "expiring")

    @property
    def masked(self) -> str:
        return f"{self.LABEL}_{self.prefix}_{'•' * 8}"

    # ── checking ──────────────────────────────────────────────────────────
    @classmethod
    def looks_like_key(cls, presented: str) -> bool:
        """True for anything shaped like one of our keys.

        Used to tell an API key from the shared webhook secret when both arrive
        in the same Authorization header — so a mistyped key is reported as a bad
        key rather than silently falling through to the other check.
        """
        return bool(presented) and presented.startswith(f"{cls.LABEL}_")

    @classmethod
    def authenticate(cls, presented: str):
        """The key row for `presented`, or None. Records the use."""
        if not cls.looks_like_key(presented):
            return None
        parts = presented.split("_", 2)
        if len(parts) != 3:
            return None
        _label, prefix, secret = parts
        row = cls.objects.filter(prefix=prefix).select_related("hook").first()
        if row is None:
            return None
        # compare_digest, so a wrong key cannot be narrowed down by timing.
        if not hmac.compare_digest(row.key_hash, cls._hash(secret)):
            return None
        if not row.is_usable:
            return None
        cls._record_use(row)
        return row

    @classmethod
    def authenticate_generic(cls, presented):
        """The GENERIC key for `presented`, or None.

        A hook-bound key is deliberately NOT accepted here, and vice versa: the
        two endpoints are separate surfaces, so revoking the credential of one
        integration must not be able to shut off the other.
        """
        row = cls.authenticate(presented)
        if row is None or not row.is_generic:
            return None
        return row

    @classmethod
    def _record_use(cls, row) -> None:
        # Written with an UPDATE so two concurrent messages cannot lose a count.
        cls.objects.filter(pk=row.pk).update(
            last_used_at=timezone.now(),
            request_count=models.F("request_count") + 1,
        )


class InboundMessage(models.Model):
    """One message received from a WhatsApp gateway.

    WHY IT EXISTS
    -------------
    Two jobs, both of which need a row of their own rather than a field on Query:

    1. IDEMPOTENCY. The gateway retries when it does not get a fast 200, and the
       old webhook ran the entire pipeline (retrieval, LLM, TTS — tens of
       seconds) inside the request. Every retry therefore started a second
       identical run, and the farmer got the same advice two or three times over,
       each answer costing another set of API calls. The unique constraint on
       (provider, message_id) is what makes a retry cheap and silent: the second
       insert fails, and that failure IS the dedupe.

    2. RECOVERY. The pipeline now runs on a background thread, so a message in
       flight when the process restarts would otherwise be lost with no trace of
       having arrived. A row in `received`/`processing` older than a few minutes
       is exactly the list `manage.py whatsapp_retry` works through.

    The raw payload is kept because gateway payload shapes are not stable and
    are not fully documented: when a farmer reports a message that got no reply,
    this is the only place the original request survives.
    """

    PROVIDER_GENERIC = "generic"

    STATUS_RECEIVED = "received"
    STATUS_PROCESSING = "processing"
    STATUS_ANSWERED = "answered"
    STATUS_FAILED = "failed"
    # The answer was produced and the gateway would not take it. A separate state
    # because the two failures need opposite handling: a failed ANSWER should be
    # retried through the pipeline, and a failed DELIVERY must never be — the
    # answer is already paid for and sitting on the Query row, so retrying the
    # pipeline would spend the money again to produce what we already have.
    STATUS_DELIVERY_FAILED = "delivery_failed"
    STATUS_IGNORED = "ignored"
    STATUS_CHOICES = [
        (STATUS_RECEIVED, "Received"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_ANSWERED, "Answered"),
        (STATUS_FAILED, "Failed"),
        (STATUS_DELIVERY_FAILED, "Answered, delivery failed"),
        (STATUS_IGNORED, "Ignored"),
    ]

    # "Picky Assist", "generic", or whatever gateway sent it. Combined with
    # message_id this is the idempotency key, so two providers can issue the same
    # id without colliding.
    provider = models.CharField(max_length=40, default=Hook.GATEWAY_PICKY)
    # Not unique on its own: two gateways can issue the same id.
    message_id = models.CharField(max_length=200, db_index=True)

    hook = models.ForeignKey(
        Hook, null=True, blank=True, on_delete=models.SET_NULL,
        related_name="inbound_messages",
    )
    query = models.ForeignKey(
        "queries.Query", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="inbound_messages",
    )

    from_number = models.CharField(max_length=30, blank=True)
    message_type = models.CharField(max_length=20, default="text")
    text = models.TextField(blank=True)
    media_url = models.CharField(max_length=1000, blank=True)
    raw_payload = models.JSONField(default=dict, blank=True)
    # The farmer's language, when the caller states it. Blank means "decide the
    # way this deployment already decides" — for a hook, its primary_language.
    language = models.CharField(max_length=20, blank=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES,
                              default=STATUS_RECEIVED)
    error = models.TextField(blank=True)
    attempts = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["provider", "message_id"],
                                    name="uniq_gateway_message"),
        ]
        indexes = [models.Index(fields=["status", "created_at"])]

    def __str__(self):
        return f"{self.provider} {self.message_id} ({self.status})"
