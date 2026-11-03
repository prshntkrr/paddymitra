"""
migrate_session_content
=======================
Recreate, on any environment, the Documents and FAQ entries added in the
2026-11 content session, so production can be brought in line with local.

It is ADDITIVE and IDEMPOTENT:
  * Documents are matched by title (get_or_create) and re-attached to the
    catalogue; their files must already be copied into MEDIA_ROOT/documents/.
  * FAQ rows are loaded from deploy/faq_fixture.json and created ONLY when a row
    with the same normalised question does not already exist — so a production
    FAQ is never overwritten. Embeddings ship in the fixture; a row without one
    is re-embedded.

It does NOT build the vector DB and does NOT create the "Rice Knowledge Base FAQ"
document — run `sync_faq_vector --replace-sheets` and then `build_vector_db`
after this. See DEPLOY-RUNBOOK.md.

    python manage.py migrate_session_content [--catalogue rice-knowledge-base]
                                             [--force-update-faqs]
"""
from __future__ import annotations

import json
import os

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from catalogues.models import Catalogue
from documents.models import Document
from queries.models import FAQ
from queries.services.faq import normalise_question, refresh_embedding

# Documents added this session that production needs (87/88 are intentionally
# excluded — their Q&As live in the FAQ table and the sync_faq_vector document).
DOC_SPECS = [
    {"title": "Odisha and India Paddy Variety Catalogue (dataset)",
     "file": "documents/Odisha_Paddy_varieties_filled.xlsx",
     "content_type": "Advisory", "source": "Project spreadsheet",
     "summary": "Per-variety dataset of ~230 paddy varieties (duration, category, "
                "yield, source institute, suitable land, stress tolerance)."},
    {"title": "Rice Fertilizer and Manure Dose Quick Reference",
     "file": "documents/Rice-Fertilizer-Manure-Dose-Quick-Reference.md",
     "content_type": "Advisory", "source": "Compiled dose reference",
     "summary": "Focused per-input fertilizer and manure doses for rice."},
    {"title": "All Varieties Nutrient Advisories (duration-wise fertilizer schedule)",
     "file": "documents/All varieties NUTRIENT ADVISORIES.pdf",
     "content_type": "Advisory", "source": "Project advisory document",
     "summary": "Duration-wise (short 95-115 d / medium 120-135 d / long 140-160 d) "
                "per-acre fertilizer (DAP, Urea, MOP, Zinc sulphate) advisory for "
                "~168 paddy varieties."},
]


class Command(BaseCommand):
    help = "Recreate the 2026-11 session's Documents + FAQ entries on this environment."

    def add_arguments(self, parser):
        parser.add_argument("--catalogue", default="rice-knowledge-base")
        parser.add_argument("--force-update-faqs", action="store_true",
                            help="Overwrite an existing FAQ with the same question "
                                 "(default: skip it, never clobber production data).")

    def handle(self, *args, **options):
        try:
            cat = Catalogue.objects.get(slug=options["catalogue"])
        except Catalogue.DoesNotExist:
            raise CommandError(f"Catalogue '{options['catalogue']}' not found.")

        media_docs = os.path.join(settings.MEDIA_ROOT, "documents")

        # ---- 1. Documents ----
        created_docs = 0
        for spec in DOC_SPECS:
            fname = spec["file"].split("/", 1)[1]
            if not os.path.exists(os.path.join(media_docs, fname)):
                self.stdout.write(self.style.WARNING(
                    f"  ! media file missing: {spec['file']} — copy it before "
                    "build_vector_db or this document will be empty."))
            doc, created = Document.objects.get_or_create(
                title=spec["title"],
                defaults={"crop": "Rice", "content_type": spec["content_type"],
                          "doc_type": "Other", "source": spec["source"],
                          "geography": "Odisha / India", "summary": spec["summary"],
                          "status": Document.STATUS_PUBLISHED})
            doc.file.name = spec["file"]
            doc.content_type = spec["content_type"]
            doc.status = Document.STATUS_PUBLISHED
            doc.save()
            cat.documents.add(doc)
            created_docs += int(created)
            self.stdout.write(f"  {'created' if created else 'updated'} doc id={doc.id}  {spec['title'][:48]}")

        # ---- 2. FAQ rows from the fixture ----
        fixture = os.path.join(settings.BASE_DIR, "deploy", "faq_fixture.json")
        if not os.path.exists(fixture):
            raise CommandError(f"FAQ fixture not found: {fixture} (copy the deploy/ folder).")
        data = json.loads(open(fixture, encoding="utf-8").read())
        faqs = data.get("faqs") or []
        created_faqs = updated_faqs = skipped_faqs = reembedded = 0
        for r in faqs:
            norm = r.get("question_norm") or normalise_question(r.get("question_en", ""))
            if not norm:
                continue
            existing = FAQ.objects.filter(question_norm=norm).first()
            if existing and not options["force_update_faqs"]:
                skipped_faqs += 1
                continue
            faq = existing or FAQ()
            faq.question_en = r.get("question_en", "")
            faq.answer_en = r.get("answer_en", "")
            faq.question_local = r.get("question_local", "")
            faq.answer_local = r.get("answer_local", "")
            faq.language = r.get("language", "Odia")
            faq.catalogue = cat
            faq.is_active = True
            faq.question_norm = norm
            faq.question_embedding = r.get("question_embedding") or []
            faq.embedding_model = r.get("embedding_model", "")
            faq.save()
            if not faq.question_embedding:
                if refresh_embedding(faq):
                    reembedded += 1
            if existing:
                updated_faqs += 1
            else:
                created_faqs += 1

        self.stdout.write(self.style.SUCCESS(
            f"\nDocuments: {created_docs} created, {len(DOC_SPECS) - created_docs} updated. "
            f"catalogue now has {cat.documents.count()} documents."))
        self.stdout.write(self.style.SUCCESS(
            f"FAQs: {created_faqs} created, {updated_faqs} updated, "
            f"{skipped_faqs} skipped (already present), {reembedded} re-embedded. "
            f"active FAQ total: {FAQ.objects.filter(is_active=True).count()}."))
        self.stdout.write("Next: `sync_faq_vector --replace-sheets` then `build_vector_db %s`." % cat.slug)
