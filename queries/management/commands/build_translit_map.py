"""
build_translit_map
==================
Grow the deterministic Odia transliteration-fix map (odia_translit_map.json)
FROM the glossary, so translator output that spells an English term phonetically
in Odia script (e.g. tillering -> ଟିଲେରିଂ) is corrected without any LLM call at
request time.

For each TRANSLATABLE glossary term (English word with a real Odia translation,
not a product/variety code kept in Latin), the LLM is asked for the Odia
TRANSLITERATION a translator is likely to write by mistake. We then record
    {transliteration -> canonical Odia}
into a DRAFT file for human review (mirroring the odia_glossary_draft.json flow).

Usage:
    # 1. Propose entries into odia_translit_draft.json (review with an Odia speaker)
    python manage.py build_translit_map --limit 300

    # 2. After review, merge the draft into the live map
    python manage.py build_translit_map --commit
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

SERVICES_DIR = Path(settings.BASE_DIR) / "queries" / "services"
DRAFT_PATH = SERVICES_DIR / "odia_translit_draft.json"

_ODIA_RE = re.compile(r"[଀-୿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_PROMPT = (
    "You are an expert Odia agricultural linguist. You will receive a JSON array "
    "of English agricultural terms.\n"
    "For EACH term, return the Odia phonetic TRANSLITERATION that a translator "
    "would most likely write by mistake instead of translating it (i.e. the "
    "English word spelled out in Odia letters).\n"
    "Return ONLY a JSON object mapping each English term to that Odia "
    "transliteration. The values must be Odia script with NO Latin letters. "
    "Keep the keys exactly as given. No markdown, no explanation."
)


def _translatable(en: str, od: str) -> bool:
    """A glossary term worth a transliteration fix: multi-letter English word,
    with a genuine Odia (non-Latin) translation value."""
    return (
        len(en) >= 3
        and bool(_ODIA_RE.search(od))
        and not _LATIN_RE.search(od)
    )


class Command(BaseCommand):
    help = "Propose/commit Odia transliteration fixes built from the glossary."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=300,
                            help="Max glossary terms to process this run (default 300).")
        parser.add_argument("--batch", type=int, default=40,
                            help="Terms per LLM call (default 40).")
        parser.add_argument("--commit", action="store_true",
                            help="Merge the reviewed draft into odia_translit_map.json.")

    def handle(self, *args, **opts):
        from queries.services.odia_glossary import get_glossary
        from queries.services.odia_translit import add_translit_fixes, get_translit_map

        if opts["commit"]:
            if not DRAFT_PATH.exists():
                raise CommandError(f"No draft to commit: {DRAFT_PATH}")
            draft = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))
            added = add_translit_fixes(draft)
            self.stdout.write(self.style.SUCCESS(
                f"Committed {len(added)} new fix(es) into the live map."))
            return

        glossary = get_glossary()
        existing_targets = set(get_translit_map().keys())
        terms = [en for en, od in glossary.items() if _translatable(en, od)]
        terms = terms[: opts["limit"]]
        if not terms:
            raise CommandError("No translatable glossary terms found.")

        self.stdout.write(f"Proposing transliterations for {len(terms)} term(s)...")
        proposals: dict[str, str] = {}
        for i in range(0, len(terms), opts["batch"]):
            batch = terms[i:i + opts["batch"]]
            try:
                mapping = self._ask_llm(batch)
            except Exception as exc:
                self.stderr.write(self.style.WARNING(f"batch {i} failed: {exc}"))
                continue
            for en, translit in mapping.items():
                canonical = glossary.get(en.lower())
                translit = (translit or "").strip()
                # Keep only usable, non-redundant, not-already-known fixes.
                if (canonical and translit and translit != canonical
                        and _ODIA_RE.search(translit)
                        and not _LATIN_RE.search(translit)
                        and translit not in existing_targets):
                    proposals[translit] = canonical

        DRAFT_PATH.write_text(
            json.dumps(proposals, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.stdout.write(self.style.SUCCESS(
            f"Wrote {len(proposals)} proposed fix(es) to {DRAFT_PATH}\n"
            f"Review with an Odia speaker, then: python manage.py build_translit_map --commit"))

    def _ask_llm(self, terms: list[str]) -> dict[str, str]:
        payload = json.dumps(terms, ensure_ascii=False)
        if settings.GEMINI_API_KEY:
            try:
                from queries.services.pipeline import _gemini_text
                raw = _gemini_text(_PROMPT, payload)
                raw = raw.replace("```json", "").replace("```", "").strip()
                return {str(k): str(v) for k, v in json.loads(raw).items()}
            except Exception:
                pass
        from queries.services.pipeline import _chat
        raw = _chat(
            [{"role": "system", "content": _PROMPT},
             {"role": "user", "content": payload}],
            json_mode=True,
        )
        return {str(k): str(v) for k, v in json.loads(raw).items()}
