from django.conf import settings
from django.db import models


class Query(models.Model):
    CATEGORY_CHOICES = [
        ("Rice Variety", "Rice Variety"),
        ("Disease", "Disease"),
        ("Fertiliser", "Fertiliser"),
        ("Water Mgmt", "Water Mgmt"),
        ("Others", "Others"),
    ]
    TASK_TYPE_CHOICES = [
        ("Diagnosis", "Diagnosis"),
        ("Variety Suitability", "Variety Suitability"),
        ("Fertiliser Rec.", "Fertiliser Recommendation"),
        ("Management Advice", "Management Advice"),
        ("Other", "Other"),
    ]
    SEASON_CHOICES = [
        ("Kharif", "Kharif"),
        ("Rabi", "Rabi"),
        ("Zaid", "Zaid"),
    ]
    CROP_STAGE_CHOICES = [
        ("Nursery", "Nursery"),
        ("Tillering", "Tillering"),
        ("Flowering", "Flowering"),
        ("Mature Grain", "Mature Grain"),
    ]
    FEEDBACK_CHOICES = [
        ("Good", "Good"),
        ("Average", "Average"),
        ("Bad", "Bad"),
    ]
    STATUS_CHOICES = [
        ("Received", "Received"),
        ("Processing", "Processing"),
        ("Responded", "Responded"),
        ("Failed", "Failed"),
        ("Escalated", "Escalated"),
        ("Flagged", "Flagged"),
    ]

    timestamp = models.DateTimeField(auto_now_add=True)
    whatsapp_number = models.CharField(max_length=30, blank=True)
    district = models.CharField(max_length=80, blank=True)

    hook = models.ForeignKey(
        "hooks.Hook", null=True, blank=True, on_delete=models.SET_NULL, related_name="queries"
    )
    trigger_keyword = models.CharField(max_length=40, blank=True)
    catalogue = models.ForeignKey(
        "catalogues.Catalogue", null=True, blank=True,
        on_delete=models.SET_NULL, related_name="queries",
    )

    category = models.CharField(max_length=40, choices=CATEGORY_CHOICES, blank=True)
    task_type = models.CharField(max_length=40, choices=TASK_TYPE_CHOICES, blank=True)
    problem_entity = models.CharField(max_length=200, blank=True)
    season = models.CharField(max_length=20, choices=SEASON_CHOICES, blank=True)
    crop_stage = models.CharField(max_length=30, choices=CROP_STAGE_CHOICES, blank=True)
    missing_context = models.CharField(max_length=500, blank=True)

    original_query_text = models.TextField(blank=True)
    original_query_language = models.CharField(max_length=30, blank=True, default="Odia")
    original_query_voice_file = models.FileField(upload_to="audio/in/", blank=True, null=True)

    translated_query_text = models.TextField(blank=True)
    ai_response_text = models.TextField(blank=True)
    ai_response_text_local = models.TextField(blank=True)
    ai_response_voice_file = models.FileField(upload_to="audio/out/", blank=True, null=True)

    response_time_seconds = models.FloatField(default=0.0)
    farmer_feedback = models.CharField(max_length=10, choices=FEEDBACK_CHOICES, blank=True)
    source_documents = models.ManyToManyField(
        "documents.Document", blank=True, related_name="used_in_queries"
    )
    pipeline_trace = models.JSONField(default=dict, blank=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="Received")
    flagged_for_review = models.BooleanField(default=False)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="reviewed_queries",
    )
    review_notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-timestamp"]
        verbose_name_plural = "Queries"

    def __str__(self):
        return f"Q-{self.pk} [{self.district}] {self.translated_query_text[:40]}"

    @property
    def code(self):
        return f"Q-{self.pk:04d}"

    # ── The recordings, for anything that has to display them ──────────────
    #
    # A question asked by voice over WhatsApp is stored twice: as a transcript
    # and as the farmer's own recording. The transcript is what the pipeline
    # answers, and it is the transcript that goes wrong — a district name heard
    # as a crop name, a dose heard as a duration. Whoever reviews the answer has
    # to be able to hear what was actually said, so the recording belongs
    # wherever the query is shown, not only on the detail page.
    #
    # A property rather than `voice_file.url` in the template: the field is
    # nullable AND the file can be missing from storage after a restore, and
    # FileField raises ValueError for the first while the template silently
    # renders a broken player for the second. Both come back as "" here.
    @staticmethod
    def _url_of(field) -> str:
        if not field:
            return ""
        try:
            return field.url
        except (ValueError, NotImplementedError):
            return ""

    @property
    def question_audio_url(self) -> str:
        """The farmer's own recording — a WhatsApp voice note, or one recorded here."""
        return self._url_of(self.original_query_voice_file)

    @property
    def answer_audio_url(self) -> str:
        """The spoken answer, the one the farmer actually hears."""
        return self._url_of(self.ai_response_voice_file)

    @property
    def has_audio(self) -> bool:
        return bool(self.question_audio_url or self.answer_audio_url)

    @property
    def came_from_whatsapp(self) -> bool:
        """Asked through a hook, so the recording is the farmer's WhatsApp voice note."""
        return bool(self.hook_id or self.whatsapp_number)


class FAQ(models.Model):
    """A question with an answer already settled, served instead of running RAG.

    WHY IT EXISTS
    -------------
    The same handful of questions come back every season ("how much DAP per
    acre?"). Re-running retrieval and generation for them costs money and, worse,
    is non-deterministic: the same farmer can ask the same thing twice and get two
    differently-worded answers. An entry here fixes the answer once.

    WHERE THE ENTRIES COME FROM
    ---------------------------
    Either typed on the FAQ page, or promoted automatically from a Query whose
    feedback was set to Good (see `services/faq.promote_from_query`).

    THE ANSWER IS SERVED VERBATIM. Nothing re-reads the documents behind it, so an
    entry is only as good as the answer that was approved. `is_active` is the
    switch to take one out of service without losing it.
    """
    question_en = models.TextField(
        help_text="The question in English. This is what a new question is matched against.")
    question_local = models.TextField(
        blank=True, help_text="The question as the farmer asked it, if it was not in English.")
    language = models.CharField(max_length=30, blank=True, default="Odia")

    answer_en = models.TextField(blank=True)
    answer_local = models.TextField(
        blank=True, help_text="The answer in the farmer's language. Used as-is when it is present, "
                              "so the stored wording — and its audio — never drift.")

    catalogue = models.ForeignKey(
        "catalogues.Catalogue", null=True, blank=True,
        on_delete=models.SET_NULL, related_name="faqs",
        help_text="Limit this answer to one catalogue. Leave empty to use it for all.",
    )

    # Matching keys, both maintained by services/faq.py.
    # `question_norm` is the cheap exact-match key: lowercased, punctuation and
    # spacing removed. `question_embedding` catches the paraphrases it cannot.
    question_norm = models.CharField(max_length=500, blank=True, db_index=True)
    question_embedding = models.JSONField(default=list, blank=True)
    embedding_model = models.CharField(max_length=80, blank=True)

    is_active = models.BooleanField(
        default=True, help_text="Uncheck to stop serving this answer without deleting it.")
    source_query = models.ForeignKey(
        Query, null=True, blank=True, on_delete=models.SET_NULL, related_name="faqs",
        help_text="The query this was promoted from, if it was not typed by hand.",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="faqs_created",
    )
    hit_count = models.PositiveIntegerField(
        default=0, help_text="How many times this answer has been served.")
    last_used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "FAQ"
        verbose_name_plural = "FAQs"
        indexes = [models.Index(fields=["is_active", "question_norm"])]

    def __str__(self):
        return f"FAQ-{self.pk}: {self.question_en[:60]}"

    @property
    def answer_for_display(self):
        return self.answer_local or self.answer_en


class ChatQuery(models.Model):
    FEEDBACK_CHOICES = [
        ("Correct", "Correct"),
        ("Need Improvement", "Need Improvement"),
        ("Incorrect", "Incorrect"),
    ]

    timestamp = models.DateTimeField(auto_now_add=True)
    query = models.ForeignKey(
        Query,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="chat_entries",
    )
    language = models.CharField(max_length=30, default="English")
    question = models.TextField()
    response = models.TextField(blank=True)
    response_time_seconds = models.FloatField(default=0.0)
    feedback = models.CharField(max_length=20, choices=FEEDBACK_CHOICES, blank=True, default="")
    feedback_local = models.CharField(max_length=60, blank=True, default="")

    class Meta:
        ordering = ["-timestamp"]
        verbose_name = "Chat Query"
        verbose_name_plural = "Chat Queries"

    def __str__(self):
        return f"ChatQuery-{self.pk} [{self.language}]"
