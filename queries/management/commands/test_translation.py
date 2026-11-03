"""
test_translation
================
Re-run ONLY the English -> Odia step of the pipeline, on text you already have.

Why this exists: when the Odia comes out wrong the fix is usually a glossary
entry, and the only thing that needs re-checking is the translation. Asking the
whole question again re-runs retrieval, generation and the LLM bill for an
answer that was already fine, and the new English is never quite the same text,
so you cannot tell whether the glossary edit worked. This command feeds the
SAME English through the SAME pipeline.translate() call the pipeline makes at
stage 5 - same prompt, same glossary forcing, same post-passes - so what it
prints is what the farmer would have heard.

The glossary is re-read from disk on every run, so edit odia_glossary.json (or
run glossary_import) and re-run; no server restart needed.

Usage:
    # the English answer of a query that already ran
    python manage.py test_translation --query 412
    python manage.py test_translation --last

    # any text
    python manage.py test_translation --text "Apply 20 kg of MOP per acre."
    python manage.py test_translation --file answer.txt
    cat answer.txt | python manage.py test_translation

    # what changed since the glossary edit
    python manage.py test_translation --query 412 --diff

    # the other direction (farmer's question -> English)
    python manage.py test_translation --text "..." --from Odia --to English

    # is it stable? translate the same text three times
    python manage.py test_translation --last --repeat 3
"""
from __future__ import annotations

import difflib
import sys
import time
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Re-run the pipeline's translation step on existing text (no retrieval, no generation)."

    def add_arguments(self, parser):
        src = parser.add_mutually_exclusive_group()
        src.add_argument("--text", help="Text to translate.")
        src.add_argument("--file", help="Read the text from this file (UTF-8).")
        src.add_argument("--query", type=int, metavar="ID",
                         help="Use the stored English answer of Query <ID>.")
        src.add_argument("--last", action="store_true",
                         help="Use the English answer of the most recent answered query.")
        src.add_argument("--question", type=int, metavar="ID",
                         help="Use the farmer's ORIGINAL question text of Query <ID> "
                              "(pair with --from Odia --to English).")

        parser.add_argument("--from", dest="source", default="English",
                            help="Source language (default: English).")
        parser.add_argument("--to", dest="target", default="Odia",
                            help="Target language (default: Odia).")
        parser.add_argument("--repeat", type=int, default=1,
                            help="Translate N times to see whether the output is stable.")
        parser.add_argument("--diff", action="store_true",
                            help="Show a word diff against the Odia stored on the query "
                                 "(--query/--last only).")
        parser.add_argument("--glossary", action="store_true",
                            help="List the glossary terms that apply to this text and exit.")
        parser.add_argument("--save", action="store_true",
                            help="Write the new translation back to the query's "
                                 "ai_response_text_local (--query/--last only).")

    # ------------------------------------------------------------------ input

    def _load_text(self, opts) -> tuple[str, object]:
        """Return (text, query_or_None)."""
        from queries.models import Query

        if opts.get("text"):
            return opts["text"], None

        if opts.get("file"):
            path = Path(opts["file"])
            if not path.exists():
                raise CommandError(f"No such file: {path}")
            return path.read_text(encoding="utf-8"), None

        if opts.get("query") or opts.get("question"):
            pk = opts.get("query") or opts.get("question")
            try:
                q = Query.objects.get(pk=pk)
            except Query.DoesNotExist:
                raise CommandError(f"Query {pk} does not exist.")
            text = q.original_query_text if opts.get("question") else q.ai_response_text
            if not (text or "").strip():
                field = "original_query_text" if opts.get("question") else "ai_response_text"
                raise CommandError(f"Query {pk} has an empty {field}.")
            return text, q

        if opts.get("last"):
            q = (Query.objects
                 .exclude(ai_response_text="")
                 .order_by("-timestamp")
                 .first())
            if not q:
                raise CommandError("No query with an English answer found.")
            return q.ai_response_text, q

        if not sys.stdin.isatty():
            text = sys.stdin.read()
            if text.strip():
                return text, None

        raise CommandError(
            "Give the text with --text/--file/--query/--last, or pipe it on stdin.")

    # ----------------------------------------------------------------- output

    def _rule(self, label: str):
        self.stdout.write(self.style.MIGRATE_HEADING(f"\n{label}\n" + "-" * len(label)))

    def _word_diff(self, old: str, new: str):
        """Word-level diff - an Odia fix is usually one or two words."""
        old_w, new_w = (old or "").split(), (new or "").split()
        sm = difflib.SequenceMatcher(None, old_w, new_w)
        parts, changed = [], False
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                parts.append(" ".join(old_w[i1:i2]))
                continue
            changed = True
            if tag in ("replace", "delete"):
                parts.append(self.style.ERROR("[-" + " ".join(old_w[i1:i2]) + "-]"))
            if tag in ("replace", "insert"):
                parts.append(self.style.SUCCESS("{+" + " ".join(new_w[j1:j2]) + "+}"))
        self.stdout.write(" ".join(parts) if changed
                          else "(identical to the stored translation)")

    # ----------------------------------------------------------------- handle

    def handle(self, *args, **opts):
        # Imported here so the glossary module is loaded inside this process,
        # after Django has finished setting up.
        from queries.services import pipeline
        from queries.services.odia_glossary import reload_glossary, relevant_glossary

        # The Windows console defaults to cp1252, which cannot encode a single
        # Odia letter — printing the result would die with UnicodeEncodeError.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

        text, query = self._load_text(opts)
        text = text.strip()
        source, target = opts["source"], opts["target"]

        # Pick up glossary edits made since the last run.
        reload_glossary()

        if opts["glossary"]:
            rel = relevant_glossary(text)
            self._rule(f"glossary terms in this text ({len(rel)})")
            for en, od in sorted(rel.items()):
                self.stdout.write(f"  {en:<40} {od}")
            if not rel:
                self.stdout.write("  (none)")
            return

        if query:
            self.stdout.write(self.style.HTTP_INFO(
                f"Query {query.pk} - {query.timestamp:%Y-%m-%d %H:%M} - "
                f"{(query.translated_query_text or query.original_query_text or '')[:70]}"))

        self._rule(f"{source} in")
        self.stdout.write(text)

        rel = relevant_glossary(text) if target == "Odia" else {}
        if rel:
            self.stdout.write(self.style.HTTP_INFO(
                f"\n{len(rel)} glossary term(s) apply: "
                + ", ".join(f"{en} -> {od}" for en, od in sorted(rel.items())[:12])
                + (" ..." if len(rel) > 12 else "")))

        runs = []
        for i in range(max(1, opts["repeat"])):
            t0 = time.monotonic()
            out = pipeline.translate(text, source, target)
            ms = int((time.monotonic() - t0) * 1000)
            runs.append(out)
            label = f"{target} out" if opts["repeat"] == 1 else f"{target} out - run {i + 1}"
            self._rule(f"{label}  ({ms} ms)")
            self.stdout.write(out)

        if len(runs) > 1:
            self._rule("stability")
            if len(set(runs)) == 1:
                self.stdout.write(self.style.SUCCESS("identical across all runs"))
            else:
                self.stdout.write(self.style.WARNING(
                    f"{len(set(runs))} different outputs across {len(runs)} runs"))
                self._word_diff(runs[0], runs[-1])

        latest = runs[-1]

        if opts["diff"]:
            if not query:
                raise CommandError("--diff needs --query or --last (there is nothing to compare to).")
            self._rule("diff vs the stored translation")
            self._word_diff(query.ai_response_text_local, latest)

        if opts["save"]:
            if not query:
                raise CommandError("--save needs --query or --last.")
            query.ai_response_text_local = latest
            query.save(update_fields=["ai_response_text_local"])
            self.stdout.write(self.style.SUCCESS(
                f"\nSaved to Query {query.pk}.ai_response_text_local"))
