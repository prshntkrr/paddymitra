"""Measure the Odia -> English semantic validator against real stored queries.

Two modes, because they cost very different amounts:

  OFFLINE (default) — re-validates the (Odia, English) pairs already stored on
  Query rows. No API calls, no cost, runs in a second. Answers "what would the
  validator flag?" but not "what would the repair do about it?", because the
  stored English is the FINAL text and the repair works on the pre-canonical
  translation.

  LIVE (--live) — puts each question back through translate() with the real
  Google and OpenAI, and reads the pipeline's own log to find out which engine
  was accepted, whether a repair ran, and whether it was kept. This is the only
  mode that can report the repair columns, and it bills per question, so it
  takes --limit and defaults to the flagged ones only (--flagged-only).

A FLAG IS NOT A FAULT. The "potential false positives" column is the honest
one: it counts flagged queries whose English a reviewer has NOT confirmed as
wrong. Nothing here decides that automatically — the point of the column is to
make a human look. See --show.

    python manage.py translation_eval
    python manage.py translation_eval --show
    python manage.py translation_eval --live --flagged-only
"""
import logging
import re

from django.core.management.base import BaseCommand

from queries.models import Query
from queries.services import pipeline
from queries.services.odia_semantics import validate_odia_to_english


class _StageCapture(logging.Handler):
    """Collect the translate_in stage lines emitted for one question."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list = []

    def emit(self, record):
        try:
            message = record.getMessage()
        except Exception:                                   # noqa: BLE001
            return
        if "translate_in |" in message:
            self.lines.append(message)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class Command(BaseCommand):
    help = "Evaluate Odia->English semantic validation against stored queries."

    def add_arguments(self, parser):
        parser.add_argument("--live", action="store_true",
                            help="re-translate through the real APIs (costs money)")
        parser.add_argument("--flagged-only", action="store_true",
                            help="with --live, only re-run the queries the "
                                 "offline pass flagged")
        parser.add_argument("--limit", type=int, default=0,
                            help="stop after N queries (0 = no limit)")
        parser.add_argument("--show", action="store_true",
                            help="print every flagged query in full for review")

    def handle(self, *args, **options):
        rows = list(Query.objects
                    .filter(original_query_language="Odia")
                    .exclude(translated_query_text="")
                    .exclude(original_query_text="")
                    .order_by("id")
                    .values_list("id", "original_query_text",
                                 "translated_query_text"))

        flagged = [(pk, od, en, validate_odia_to_english(od, en))
                   for pk, od, en in rows]
        flagged = [row for row in flagged if row[3]]

        self.stdout.write(self.style.MIGRATE_HEADING("\nOFFLINE — stored pairs"))
        self.stdout.write(f"  total Odia queries        : {len(rows)}")
        self.stdout.write(f"  queries flagged           : {len(flagged)}"
                          f"  ({100 * len(flagged) / max(1, len(rows)):.1f}%)")
        by_rule: dict = {}
        for _pk, _od, _en, faults in flagged:
            for fault in faults:
                by_rule[fault.code] = by_rule.get(fault.code, 0) + 1
        for code, count in sorted(by_rule.items(), key=lambda kv: -kv[1]):
            self.stdout.write(f"      {code:34} {count}")
        self.stdout.write(
            f"  potential false positives : {len(flagged)} until reviewed "
            f"(nothing here confirms a fault automatically)")

        if options["show"]:
            self.stdout.write(self.style.MIGRATE_HEADING("\n  flagged, in full"))
            for pk, od, en, faults in flagged:
                self.stdout.write(f"\n  #{pk}  {', '.join(f.code for f in faults)}")
                self.stdout.write(f"      odia   : {od}")
                self.stdout.write(f"      english: {en}")

        if not options["live"]:
            self.stdout.write(
                "\n  (run with --live to measure the repair columns; it calls "
                "the real APIs)\n")
            return

        targets = [(pk, od) for pk, od, _en, _f in flagged] if options["flagged_only"] \
            else [(pk, od) for pk, od, _en in rows]
        if options["limit"]:
            targets = targets[:options["limit"]]

        counts = {"run": 0, "google_accepted": 0, "flagged": 0, "attempted": 0,
                  "accepted": 0, "rejected": 0, "faulty_after_repair": 0,
                  "errors": 0}
        details = []

        logger = logging.getLogger("irri")
        for pk, odia in targets:
            capture = _StageCapture()
            logger.addHandler(capture)
            try:
                english = pipeline.translate(odia, "Odia", "English")
            except Exception as exc:                        # noqa: BLE001
                counts["errors"] += 1
                self.stderr.write(f"  #{pk} failed: {exc}")
                continue
            finally:
                logger.removeHandler(capture)

            log_text = capture.text
            counts["run"] += 1
            if "backend = google" in log_text:
                counts["google_accepted"] += 1
            if "8b semantic validation" in log_text:
                counts["flagged"] += 1
            attempted = "8c semantic repair accepted" in log_text or \
                        "8d repair still faulty" in log_text
            if attempted:
                counts["attempted"] += 1
            if "8c semantic repair accepted" in log_text:
                counts["accepted"] += 1
            if "8d repair still faulty" in log_text:
                counts["rejected"] += 1
                counts["faulty_after_repair"] += 1

            google_raw = re.search(r"7 raw google output = (.+)", log_text)
            details.append({
                "pk": pk, "odia": odia,
                "google": google_raw.group(1) if google_raw else "",
                "final": english,
                "attempted": attempted,
                "accepted": "8c semantic repair accepted" in log_text,
            })

        self.stdout.write(self.style.MIGRATE_HEADING("\nLIVE — real APIs"))
        self.stdout.write(f"  queries re-translated     : {counts['run']}")
        self.stdout.write(f"  google output accepted    : {counts['google_accepted']}")
        self.stdout.write(f"  queries flagged           : {counts['flagged']}")
        self.stdout.write(f"  repairs attempted         : {counts['attempted']}")
        self.stdout.write(f"  repairs accepted          : {counts['accepted']}")
        self.stdout.write(f"  repairs rejected          : {counts['rejected']}")
        self.stdout.write(f"  still faulty after repair : {counts['faulty_after_repair']}")
        self.stdout.write(f"  errors                    : {counts['errors']}")

        self.stdout.write(self.style.MIGRATE_HEADING(
            "\n  BEFORE -> AFTER (a human must read these)"))
        for row in details:
            if not row["attempted"] and not options["show"]:
                continue
            verdict = "REPAIRED" if row["accepted"] else "kept original"
            self.stdout.write(f"\n  #{row['pk']}  {verdict}")
            self.stdout.write(f"      odia   : {row['odia']}")
            self.stdout.write(f"      google : {row['google']}")
            self.stdout.write(f"      final  : {row['final']}")
        self.stdout.write("")
