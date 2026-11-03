"""
glossary_clean
==============
Remove glossary entries that today's auto-learn guard would refuse.

WHY THESE EXIST
---------------
The answer path learns: any English word still in Latin script when an answer is
prepared for speech is translated and persisted, so the voice stops reading
English. `is_auto_learn_blocked()` keeps ordinary words out of that — but it was
added AFTER terms had already been learned, and nothing filters by key at load
time (deliberately: a human's entry has to be authoritative). So the words
learned before the guard are still live, in BOTH directions:

    "high" -> ଅଧିକ     inverted, ଅଧିକ ("more") now means "high" in a farmer's
                       question, so "if I apply MORE fertiliser" can be read as
                       "HIGH", and the question lands in the high-land documents
    "in"   -> ରେ       ରେ is a case ENDING, not a word
    "is"   -> ହେଉଛି
    "it"   -> ଏହା

That is a translation the operator never asked for and cannot see, and it is why
the same question can retrieve different documents on different days.

Reports by default and changes nothing. `--apply` writes, after a backup.

Usage:
    python manage.py glossary_clean
    python manage.py glossary_clean --apply
"""
from __future__ import annotations

import json
import shutil

from django.core.management.base import BaseCommand

from queries.services import odia_glossary as g


class Command(BaseCommand):
    help = "Report (or remove) glossary entries the auto-learn guard would refuse."

    def add_arguments(self, parser):
        parser.add_argument(
            "--apply", action="store_true",
            help="Actually remove them. A backup is written first.")

    def handle(self, *args, **opts):
        path = g._GLOSSARY_PATH
        ok, bad, warn = self.style.SUCCESS, self.style.ERROR, self.style.WARNING

        if not path.exists():
            self.stdout.write(bad(f"No glossary file at {path}"))
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        blocked = sorted(k for k in data if g.is_auto_learn_blocked(k))

        self.stdout.write(f"file  : {path}")
        self.stdout.write(f"terms : {len(data)}")
        self.stdout.write(f"refused by today's guard : {len(blocked)}")

        if not blocked:
            self.stdout.write(ok("Nothing to clean."))
            return

        self.stdout.write("")
        for key in blocked:
            value = data[key]
            values = value if isinstance(value, (list, tuple)) else [value]
            self.stdout.write(f"  {key!r} -> {', '.join(values)}")
        self.stdout.write("")

        if not opts["apply"]:
            self.stdout.write(warn(
                "Nothing was changed. Re-run with --apply to remove these.\n"
                "Anything in this list that you WANT — a real term the guard "
                "happens to refuse — add it back through the glossary page "
                "afterwards: a human entry is kept whatever the guard thinks."))
            return

        backup = path.with_name("odia_glossary.backup.json")
        shutil.copy2(path, backup)
        self.stdout.write(f"backup written : {backup}")

        for key in blocked:
            data.pop(key, None)
        g._atomic_write_glossary(data)
        g.reload_glossary()

        self.stdout.write(ok(f"Removed {len(blocked)} entr(ies); "
                             f"{len(data)} remain and the glossary is reloaded."))
