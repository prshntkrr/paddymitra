"""Answer WhatsApp messages that were recorded but never replied to.

    python manage.py whatsapp_retry                 # stuck + failed, last 24h
    python manage.py whatsapp_retry --status failed
    python manage.py whatsapp_retry --hours 72 --limit 20
    python manage.py whatsapp_retry --dry-run

WHY IT EXISTS
-------------
The webhook acks in milliseconds and runs the pipeline on a worker thread, which
is what stops the gateway retrying and answering a farmer twice. The cost of
that choice is that a message in flight when the process restarts — a deploy, a
crash, an OOM kill — is left in `processing` with nobody working on it.

Because every inbound message is recorded BEFORE the work starts, that message
is not lost, just unfinished, and this is how it gets finished. Run it from cron
every few minutes and a restart costs a farmer a delay rather than an answer.
Messages already `answered` are never touched, so running it twice is safe.
"""
from __future__ import annotations

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from hooks import service
from hooks.models import InboundMessage

# A message younger than this may still be running on its own thread; picking it
# up now is how you send the same answer twice.
_MIN_AGE_MINUTES = 10


class Command(BaseCommand):
    help = "Reprocess WhatsApp messages that never produced an answer."

    def add_arguments(self, parser):
        parser.add_argument("--status", default="stuck",
                            choices=["stuck", "failed", "undelivered", "all"],
                            help="stuck = received/processing (default), failed, "
                                 "undelivered (answered but not delivered), or all")
        parser.add_argument("--hours", type=int, default=24)
        parser.add_argument("--limit", type=int, default=50)
        parser.add_argument("--max-attempts", type=int, default=3,
                            help="skip messages already tried this many times")
        parser.add_argument("--min-age-minutes", type=int, default=_MIN_AGE_MINUTES)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        now = timezone.now()
        statuses = {
            "stuck": [InboundMessage.STATUS_RECEIVED, InboundMessage.STATUS_PROCESSING],
            "failed": [InboundMessage.STATUS_FAILED],
            # Answered, but the gateway would not take it. These are the cheapest
            # of all to retry: the answer already exists, so answer_message() only
            # re-sends and no model is called again.
            "undelivered": [InboundMessage.STATUS_DELIVERY_FAILED],
        }
        wanted = (sum(statuses.values(), []) if options["status"] == "all"
                  else statuses[options["status"]])

        pending = InboundMessage.objects.filter(
            status__in=wanted,
            # NOT filtered by hook any more: a message from the generic API has
            # no hook and needs recovering just the same.
            created_at__gte=now - timedelta(hours=options["hours"]),
            created_at__lte=now - timedelta(minutes=options["min_age_minutes"]),
            attempts__lt=options["max_attempts"],
        ).order_by("created_at")[:options["limit"]]

        if not pending:
            self.stdout.write("nothing to retry")
            return

        self.stdout.write(f"{len(pending)} message(s) to retry")
        for message in pending:
            kind = "re-deliver" if message.query_id else "answer"
            label = (f"#{message.pk} {message.status} ({message.provider}) "
                     f"from {message.from_number or '?'} "
                     f"attempt {message.attempts + 1}: {message.text[:60]!r}")
            if options["dry_run"]:
                self.stdout.write(f"  would {kind} {label}")
                continue
            self.stdout.write(f"  {kind} {label}")
            # Synchronously: a retry run has nobody waiting on it, and running
            # them one at a time keeps the API bill predictable. answer_message()
            # re-delivers rather than re-answering when a Query already exists,
            # so a delivery failure never costs a second pipeline run.
            service.answer_message(message.pk)
            message.refresh_from_db()
            if message.status == InboundMessage.STATUS_ANSWERED:
                self.stdout.write(self.style.SUCCESS(
                    f"    answered (query {message.query_id})"))
            else:
                self.stdout.write(self.style.ERROR(
                    f"    {message.status}: {message.error[:160]}"))
