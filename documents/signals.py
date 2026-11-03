"""
Auto-update the Odia glossary whenever a Document is added or edited.

On post_save we wait for the transaction to commit, then (in a daemon thread, so
the request returns immediately) re-read the document text, extract agricultural
terms, and learn any that are not already in odia_glossary.json. Only genuinely
new terms are translated, so repeat saves are cheap.
"""
from __future__ import annotations

import logging
import threading

from django.db import transaction
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Document

log = logging.getLogger("irri")


def _learn_async(document_pk: int) -> None:
    def _run() -> None:
        try:
            from django.conf import settings as _settings
            if not getattr(_settings, "GLOSSARY_AUTO_LEARN", True):
                return

            # Re-fetch after commit so we see committed data and avoid passing a
            # possibly-detached instance across threads.
            from .models import Document as _Document
            from catalogues.services.chunking import document_text
            from queries.services.glossary_learn import learn_from_texts

            doc = _Document.objects.get(pk=document_pk)
            text = document_text(doc)
            if not (text and text.strip()):
                return
            added = learn_from_texts([text], label=f"doc:{doc.pk}")
            if added:
                log.info("Document %s glossary auto-learn added %d term(s)", doc.pk, len(added))
        except Exception as exc:
            log.warning("Document glossary auto-learn failed (pk=%s): %s", document_pk, exc)

    threading.Thread(target=_run, name="doc-glossary-learn", daemon=True).start()


@receiver(post_save, sender=Document)
def learn_glossary_from_document(sender, instance, created, raw=False, **kwargs):
    # Skip fixture loading (loaddata) — raw saves bypass normal data integrity.
    if raw:
        return
    transaction.on_commit(lambda: _learn_async(instance.pk))
